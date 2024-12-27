from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

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

onnx_file_path = "onnx/meta_classifier.onnx"

# Save embeddings models and tokenizers
text_model = AutoModel.from_pretrained("distilbert-base-uncased")
text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
code_model = AutoModel.from_pretrained("microsoft/codebert-base")
code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Save the onnx model
input_text = "This is a test to convert the model to onnx"

text_inputs = text_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
text_embeddings = text_model(**text_inputs).last_hidden_state.mean(dim=1)
code_inputs = code_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
code_embeddings = code_model(**code_inputs).last_hidden_state.mean(dim=1)
combined_embeddings = torch.cat((text_embeddings, code_embeddings), dim=1)
if combined_embeddings.dim() == 2:
    combined_embeddings = combined_embeddings.unsqueeze(1)  # Add sequence dimension

meta_classifier = MetaClassifier(input_dim=768 + 768)
meta_classifier.load_state_dict(torch.load('models/final_checkpoint_lstm_full.pth', weights_only=True))
meta_classifier.eval()
torch.onnx.export(
    meta_classifier,
    combined_embeddings,
    onnx_file_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {1: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=18
)
print(f"Model has been exported to {onnx_file_path}")
