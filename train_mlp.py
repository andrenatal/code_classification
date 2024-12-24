from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cross_entropy

num_epochs = 100

samples = [
    {"text": "This is an English sentence.", "label": 0},
    {"text": "def add(a, b): return a + b", "label": 1},
    # Add more examples...
]

# Initialize models
text_model = AutoModel.from_pretrained("distilbert-base-uncased")
code_model = AutoModel.from_pretrained("microsoft/codebert-base")
text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def generate_combined_embeddings(text):
    # Encode English text
    text_inputs = text_tokenizer(text, return_tensors="pt")
    text_embeddings = text_model(**text_inputs).last_hidden_state.mean(dim=1)  # [batch_size, hidden_dim]

    # Encode code snippet
    code_inputs = code_tokenizer(text, return_tensors="pt")
    code_embeddings = code_model(**code_inputs).last_hidden_state.mean(dim=1)  # [batch_size, hidden_dim]

    return torch.cat((text_embeddings, code_embeddings), dim=1)  # [batch_size, hidden_dim * 2]

class MetaClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MetaClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(x)

# Initialize classifier
num_classes = 2  # English or Code
meta_classifier = MetaClassifier(input_dim=768 + 768, num_classes=2)  # Example dimensions
meta_classifier.to("cuda")
optimizer = optim.Adam(meta_classifier.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    meta_classifier.train()
    running_loss = 0.0
    for sample in samples:
        combined_embeddings = generate_combined_embeddings(sample["text"])
        labels = torch.tensor([sample["label"]])

        # Forward pass
        logits = meta_classifier(combined_embeddings)
        loss = loss_fn(logits, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

meta_classifier.eval()
torch.save(meta_classifier, 'meta_classifier.pth')
