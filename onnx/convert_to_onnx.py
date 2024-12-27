from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

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

meta_classifier = MetaClassifier(input_dim=768 + 768)
meta_classifier.load_state_dict(torch.load('/media/4tbdrive/engines/code_classification/final_checkpoint.pth', weights_only=True))
meta_classifier.eval()
torch.onnx.export(
    meta_classifier,
    combined_embeddings,
    onnx_file_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print(f"Model has been exported to {onnx_file_path}")
