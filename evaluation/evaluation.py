import sys
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from customdataset import CustomDataSet
import torch
import torch.nn as nn

model_name = 'models/final_checkpoint_lstm_full.pth'

class MetaClassifier(nn.Module):
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

meta_classifier = MetaClassifier(input_dim=768 + 768)
meta_classifier.load_state_dict(torch.load(model_name, weights_only=True))
meta_classifier.eval()
meta_classifier.to("cuda:3")

def compute_f1():
    y_true = []
    y_pred = []
    total_preds = 0
    for text, code in  zip(dataset.text_dataset["training_examples"], dataset.code_dataset["training_examples"]):
        # Ensure inputs have the correct shape [batch_size, seq_len, input_dim]
        text = np.array(text)
        text = np.reshape(text, (1, 1, text.shape[0]))
        text = torch.tensor(text, dtype=torch.float).to("cuda:3")
        logits = meta_classifier(text)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
        print("Classifier text output: with a score of", predictions, "%", "for")
        total_preds += 1
        y_true.append(0)
        y_pred.append(predictions.item())

        code = np.array(code)
        code = np.reshape(code, (1, 1, code.shape[0]))
        code = torch.tensor(code, dtype=torch.float).to("cuda:3")
        logits = meta_classifier(code)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
        print("Classifier code output: with a score of", predictions, "%", "for")
        total_preds += 1
        y_true.append(1)
        y_pred.append(predictions.item())


    print("Total predictions:", total_preds)
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:\n", cm)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix. F1 Score: {:.2f}, Recall {:.2f}, Precision {:.2f}, Accuracy: {:.2f}%'.format(f1, recall, precision, accuracy * 100))
    plt.savefig(f'confusion_matrix_{model_name.replace("/","_")}.png', dpi=300)  # Save as PNG with high resolution
    plt.close()  # Close the plot to free up memory
    return f1, accuracy

# we use only f1 score as the dataset is imbalanced
dataset = CustomDataSet("/media/4tbdrive/corpora/code_classification/code_test/",
                        "/media/4tbdrive/corpora/code_classification/text/test_text_cleaned.csv",
                        10000,
                        10000)
dataset.load_data()
print("F1 Score and accuracy", compute_f1())
