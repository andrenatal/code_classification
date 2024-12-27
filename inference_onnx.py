import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cross_entropy
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="anatal/code_english_model", revision="bf8ccd4", device="cuda:3")

class MetaClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MetaClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.fc(x)

# Load pre-trained models and tokenizer
text_model = ORTModel.from_pretrained("onnx/distilbert-base-uncased/model.onnx")
code_model = ORTModel.from_pretrained("microsoft/codebert-base")
text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Load trained meta-classifier
meta_classifier = MetaClassifier(input_dim=768 + 768)
meta_classifier.load_state_dict(torch.load('final_checkpoint.pth', weights_only=True))
meta_classifier.eval()
meta_classifier.to("cuda:3")

def run_inference(input_text):
    # Tokenize and get embeddings
    text_inputs = text_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    code_inputs = code_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    text_embeddings = text_model(**text_inputs).last_hidden_state.mean(dim=1).to("cuda:3")  # [1, hidden_dim_text]
    code_embeddings = code_model(**code_inputs).last_hidden_state.mean(dim=1).to("cuda:3")  # [1, hidden_dim_code]

    # Combine embeddings and classify
    combined_embeddings = torch.cat((text_embeddings, code_embeddings), dim=1) # [1, hidden_dim_text + hidden_dim_code]
    logits = meta_classifier(combined_embeddings)  # [1, num_classes]
    probabilities = torch.sigmoid(logits)
    #print("probabilities:", probabilities, 1-probabilities)
    predictions = (probabilities > 0.5).float()
    if predictions == 0:
        return "human language"
    else:
        return "computer language"


text = "hello, this is a test."
print(text, classifier(text),   run_inference(text))
text = "<html><head><title>Test</title></head><body><p>Hello World</p></body></html>"
print(text, classifier(text),   run_inference(text))
text = "The server is broken. Can you fix it?"
print(text, classifier(text), run_inference(text))
text = "https://pkg.go.dev/github.com/minio/madmin-go#AdminClient.GetConfigKV"
print(text, classifier(text), run_inference(text))
text = "just doing `os.Open(filename)`"
print(text, classifier(text), run_inference(text))
text = "text_tokenizer(input_text, return_tensors="

