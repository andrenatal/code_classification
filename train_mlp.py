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

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_start_method("spawn", force=True)

    class MetaClassifier(nn.Module):
        def __init__(self, input_dim):
            super(MetaClassifier, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.fc(x)

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpus}")
    else:
        print("No GPUs available")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = 100

    dataset = CustomDataSet()
    dataset.load_data()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    writer = SummaryWriter(log_dir=f'runs/code_classification/{time.strftime("%Y%m%d-%H%M%S")}/')

    # Initialize classifier
    meta_classifier = MetaClassifier(input_dim=768 + 768)
    meta_classifier.to(device)
    meta_classifier = nn.DataParallel(meta_classifier)
    optimizer = optim.Adam(meta_classifier.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        meta_classifier.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            logits = meta_classifier(batch["training_examples"].to(device))
            outputs = logits.squeeze(1)
            loss = loss_fn(outputs, batch["training_labels"].to(device))

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
                logits = meta_classifier(batch["training_examples"].to(device))
                outputs = logits.squeeze(1)
                loss = loss_fn(outputs, batch["training_labels"].to(device))
                val_loss += loss.item()
                probabilities = torch.sigmoid(outputs)
                # Calculate accuracy
                predicted = (probabilities > 0.5).float()
                correct += (predicted == batch["training_labels"].to(device)).sum().item()
                total += batch["training_labels"].size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        print(f"Validation Loss: {avg_val_loss}, Accuracy: {accuracy}")
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation Accuracy', accuracy, epoch)

    meta_classifier = meta_classifier.module
    torch.save(meta_classifier.state_dict(), 'meta_classifier.pth')
    writer.close()

