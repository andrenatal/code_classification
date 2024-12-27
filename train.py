import time
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cross_entropy
from customdataset import CustomDataSet
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from huggingface_hub import HfApi, Repository, PyTorchModelHubMixin

class MetaClassifier(nn.Module,
                         PyTorchModelHubMixin,
                        repo_url="https://huggingface.co/anatal/",
                        license="mit"):
        def __init__(self, input_dim, hidden_dim=128, lstm_hidden_dim=64, dropout=0.3):
            super(MetaClassifier, self).__init__()
            self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(lstm_hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            # Assuming x has shape [batch_size, seq_len, input_dim]
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use the last hidden state of the LSTM
            lstm_out = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]
            return self.fc(lstm_out)

if __name__ == "__main__":

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpus}")
    else:
        print("No GPUs available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CustomDataSet("/media/4tbdrive/corpora/code_classification/code/",
                            "/media/4tbdrive/corpora/code_classification/text/train_text_cleaned.csv",
                            10000,
                            10000)
    dataset.load_data()
    num_epochs = 1000
    batch_size =16
    dropout = 0.5
    hidden_dim = 256

    class EarlyStopping:
        def __init__(self, patience=5, verbose=False, delta=0):
            self.patience = patience
            self.verbose = verbose
            self.delta = delta
            self.best_score = None
            self.early_stop = False
            self.counter = 0
            self.best_loss = float('inf')

        def __call__(self, val_loss, model):
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            if self.verbose:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving best model to checkpoints/checkpoint.pth')
            torch.save(model.state_dict(), 'checkpoints/checkpoint.pth')
            self.best_loss = val_loss

    def collate_fn(batch):
        training_examples = torch.stack([torch.tensor(item["training_examples"]) for item in batch]).to(device)
        training_labels = torch.stack([torch.tensor(item["training_labels"]) for item in batch]).to(device)
        return {"training_examples": training_examples, "training_labels": training_labels}

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    writer = SummaryWriter(log_dir=f'runs/code_classification/{time.strftime("%Y%m%d-%H%M%S")}/')

    # Initialize classifier
    meta_classifier = MetaClassifier(input_dim=768 + 768)
    meta_classifier.to(device)
    if torch.cuda.device_count() > 1:
        meta_classifier = nn.DataParallel(meta_classifier)
    optimizer = optim.Adam(meta_classifier.parameters(), lr=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()

    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(num_epochs):
        meta_classifier.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
        # Move data to device
            inputs = batch["training_examples"]
            labels = batch["training_labels"]

            # Ensure inputs have the correct shape [batch_size, seq_len, input_dim]
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)  # Add sequence dimension

            logits = meta_classifier(inputs)
            loss = loss_fn(logits, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
        writer.add_scalar('Average Training Loss', avg_loss, epoch)

        # Validation
        meta_classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:

                inputs = batch["training_examples"]
                labels = batch["training_labels"]

                # Ensure inputs have the correct shape [batch_size, seq_len, input_dim]
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(1)  # Add sequence dimension

                logits = meta_classifier(inputs)
                loss = loss_fn(logits, labels)


                val_loss += loss.item()
                probabilities = torch.sigmoid(logits)
                # Calculate accuracy
                predicted = (probabilities > 0.5).float()
                correct += (predicted == batch["training_labels"]).sum().item()
                total += batch["training_labels"].size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        print(f"Validation Loss: {avg_val_loss}, Accuracy: {accuracy}")
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation Accuracy', accuracy, epoch)

        # Check early stopping
        early_stopping(avg_val_loss, meta_classifier)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    meta_classifier.load_state_dict(torch.load('checkpoints/checkpoint.pth', weights_only=True))
    if torch.cuda.device_count() > 1:
        meta_classifier = meta_classifier.module
    # torch format save local
    torch.save(meta_classifier.state_dict(), 'final_checkpoint.pth')
    writer.close()

