import torch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cross_entropy
from transformers import pipeline

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

# Load pre-trained models and tokenizer
text_model = AutoModel.from_pretrained("distilbert-base-uncased")
code_model = AutoModel.from_pretrained("microsoft/codebert-base")
text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Load trained meta-classifier
meta_classifier = MetaClassifier(input_dim=768 + 768)
meta_classifier.load_state_dict(torch.load(model_name, weights_only=True))
meta_classifier.eval()
meta_classifier.to("cuda:3")

def run_inference(input_text):
    # Tokenize and get embeddings
    text_inputs = text_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    code_inputs = code_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    text_embeddings = text_model(**text_inputs).last_hidden_state.mean(dim=1).to("cuda:3")  # [1, hidden_dim_text]
    code_embeddings = code_model(**code_inputs).last_hidden_state.mean(dim=1).to("cuda:3")  # [1, hidden_dim_code]

    #print(text_inputs)
    #print("text_embeddings shape:", text_model(**text_inputs).last_hidden_state.shape)
    # Combine embeddings and classify
    combined_embeddings = torch.cat((text_embeddings, code_embeddings), dim=1) # [1, hidden_dim_text + hidden_dim_code]

    # Ensure inputs have the correct shape [batch_size, seq_len, input_dim]
    if combined_embeddings.dim() == 2:
        combined_embeddings = combined_embeddings.unsqueeze(1)  # Add sequence dimension

    logits = meta_classifier(combined_embeddings)  # [1, num_classes]
    #print("logits:", logits)
    probabilities = torch.sigmoid(logits)
    #print("probabilities:", probabilities)
    predictions = (probabilities > 0.5).float()
    if predictions == 0:
        return input_text, "human language", probabilities.item()
    else:
        return input_text, "computer language", probabilities.item()

print(run_inference("hello, this is a test."))
print(run_inference("<html><head><title>Test</title></head><body><p>Hello World</p></body></html>"))
print(run_inference("The server is broken. Can you fix it?"))
print(run_inference("just doing `os.Open(filename)`"))
print(run_inference("text_tokenizer(input_text, return_tensors="))
print(run_inference("I am asking for it to be exposed, so that we dont have to use MC. ."))
