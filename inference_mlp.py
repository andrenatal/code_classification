import torch
from transformers import AutoTokenizer, AutoModel
#from train_mlp import MetaClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cross_entropy

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


# Load pre-trained models and tokenizer
text_model = AutoModel.from_pretrained("distilbert-base-uncased")
code_model = AutoModel.from_pretrained("microsoft/codebert-base")
text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Load trained meta-classifier
meta_classifier = MetaClassifier(input_dim=768 + 768)
meta_classifier.load_state_dict(torch.load('meta_classifier.pth'))
meta_classifier.eval()

# Inference input
input_text ="text_tokenizer(input_text, return_tensors="

# Tokenize and get embeddings
text_inputs = text_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
code_inputs = code_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
text_embeddings = text_model(**text_inputs).last_hidden_state.mean(dim=1)  # [1, hidden_dim_text]
code_embeddings = code_model(**code_inputs).last_hidden_state.mean(dim=1)  # [1, hidden_dim_code]

# Combine embeddings and classify
combined_embeddings = torch.cat((text_embeddings, code_embeddings), dim=1) # [1, hidden_dim_text + hidden_dim_code]
logits = meta_classifier(combined_embeddings)  # [1, num_classes]
probabilities = torch.sigmoid(logits)
print("probabilities:", probabilities, 1-probabilities)
predictions = (probabilities > 0.5).float()
print("Predictions:", predictions)
