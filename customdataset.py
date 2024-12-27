import random
from torch.utils.data import Dataset
#from datasets import Dataset
import datasets
from transformers import AutoTokenizer
import glob
import os
#from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer, AutoModel
import torch
from torch.multiprocessing import Pool, current_process
import resource
import multiprocessing
import time

# Set the limit of open files to unlimited
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, hard), hard))

# 0 is human language, 1 is computer code
class CustomDataSet(Dataset):
    def __init__(self, code_path, text_path, MAX_TEXT_EXAMPLES, MAX_CODE_EXAMPLES):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.code_path = code_path
        self.text_path = text_path
        self.MAX_TEXT_EXAMPLES = MAX_TEXT_EXAMPLES
        self.MAX_CODE_EXAMPLES = MAX_CODE_EXAMPLES

        self.text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device)
        self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)
        self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    def generate_combined_embeddings(self, text):
        # Encode English text
        text_inputs = self.text_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        text_embeddings = self.text_model(**text_inputs).last_hidden_state.mean(dim=1)

        # Encode code snippet
        code_inputs = self.code_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        code_embeddings = self.code_model(**code_inputs).last_hidden_state.mean(dim=1)

        combined_embeddings = torch.cat((text_embeddings, code_embeddings), dim=1).cpu().detach()
        return combined_embeddings[0]  # Move to CPU and detach

    def __len__(self):
        return len(self.training_examples)

    def __getitem__(self, idx):
        return {"training_examples": self.training_examples[idx], "training_labels": self.training_labels[idx]}

    # code generation functions
    def read_and_process_file(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                code = file.read()
                codeembs = self.generate_combined_embeddings(code)
                return {"training_examples": codeembs, "training_labels": torch.tensor([1.0], dtype=torch.float).detach()}

    def code_generator(self, code_path, max_code_examples):
        code_files = glob.glob(os.path.join(code_path, '*'))[:max_code_examples]
        for file_path in code_files:
            example = self.read_and_process_file(file_path)
            if example is not None:
                yield example

    def load_code_files(self):
        return datasets.Dataset.from_generator(
            lambda: self.code_generator(self.code_path, self.MAX_CODE_EXAMPLES),
            num_proc=8
        )

    # text generation functions
    def text_generator(self, text_path, max_text_examples):
        with open(text_path, 'r') as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                if i > max_text_examples:
                    break
                textembs = self.generate_combined_embeddings(line)
                yield {"training_examples": textembs, "training_labels": torch.tensor([0.0], dtype=torch.float).detach()}

    def load_text_files(self):
        return datasets.Dataset.from_generator(
            lambda: self.text_generator(self.text_path, self.MAX_CODE_EXAMPLES),
            num_proc=8
        )

    def load_data(self):
        # read code
        self.code_dataset = self.load_code_files()

        # read text
        self.text_dataset = self.load_text_files()

        # split data
        self.training_examples = self.text_dataset["training_examples"] + self.code_dataset["training_examples"]
        self.training_labels = self.text_dataset["training_labels"] + self.code_dataset["training_labels"]

        print(f"Loaded {len(self.training_examples)} training examples")