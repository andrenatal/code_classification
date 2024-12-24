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

class CustomDataSet(Dataset):
    def __init__(self):
        self.code_path = "/media/4tbdrive/corpora/code_classification/code/"
        self.MAX_TEXT_EXAMPLES = 1000
        self.MAX_CODE_EXAMPLES = 1000
        self.text_models = {}
        self.code_models = {}
        for i in range(4):  # Corrected the for loop syntax
            self.text_models[i] = AutoModel.from_pretrained("distilbert-base-uncased")
            self.code_models[i] = AutoModel.from_pretrained("microsoft/codebert-base")


    def generate_combined_embeddings(self, text):
        process_name = current_process().name
        worker_number = (int(process_name.split('-')[-1]) - 1) % 4
        device = torch.device(f'cuda:{worker_number}' if torch.cuda.is_available() else 'cpu')
        text_model = self.text_models[worker_number].to(device)
        code_model = self.code_models[worker_number].to(device)
        text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

        # Encode English text
        text_inputs = text_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        text_embeddings = text_model(**text_inputs).last_hidden_state.mean(dim=1)

        # Encode code snippet
        code_inputs = code_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        code_embeddings = code_model(**code_inputs).last_hidden_state.mean(dim=1)

        combined_embeddings = torch.cat((text_embeddings, code_embeddings), dim=1)

        # Unload models from CUDA
        text_model.to('cpu')
        code_model.to('cpu')

        return combined_embeddings.cpu().detach()  # Move to CPU and detach

    def __len__(self):
        return len(self.training_examples)

    def __getitem__(self, idx):
        return {"training_examples": self.training_examples[idx], "training_labels": self.training_labels[idx]}

    def read_file(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                code = file.read()
                codeembs = self.generate_combined_embeddings(code)
                return codeembs
        return None

    def load_code_files(self, code_path, max_code_examples):
        code_files = glob.glob(os.path.join(code_path, '*'))[:max_code_examples]
        total_code_files = 0
        training_examples = []
        label = []

        with Pool(processes=16) as pool:
            results = pool.map(self.read_file, code_files)

        for code in results:
            if code is not None:
                training_examples.append(code)
                label.append(torch.tensor([1.0], dtype=torch.float).detach())
                total_code_files += 1
                if total_code_files == max_code_examples:  # we want a balanced dataset
                    break
        return training_examples, label

    def load_data(self):
        # read text
        dataset = datasets.Dataset.from_text("/media/4tbdrive/corpora/code_classification/text/train_text_cleaned.csv")
        text_training_examples = dataset["text"][:self.MAX_TEXT_EXAMPLES]
        text_labels = [torch.tensor([0.0], dtype=torch.float) for i in range(len(text_training_examples))]
        with Pool(processes=16) as pool:
            results_embs = pool.map(self.generate_combined_embeddings, text_training_examples)
        text_training_examples = results_embs

        # read code
        code_training_examples, code_label = self.load_code_files(self.code_path, self.MAX_CODE_EXAMPLES)

        # split data
        self.training_examples = text_training_examples + code_training_examples
        self.training_labels = text_labels + code_label

        print(f"Loaded {len(self.training_examples)} training examples")