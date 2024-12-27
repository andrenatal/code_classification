import re
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np
from datasets import load_dataset

code_train_df = pd.read_csv("/media/4tbdrive/corpora/ubuntu-ranking-dataset-creator/src/test.csv")

path_code_files = "/media/4tbdrive/corpora/code_classification/code_test/"
output_textfile = "/media/4tbdrive/corpora/code_classification/text/test_text_cleaned.csv"

final_list = []
data = code_train_df["Context"].tolist() + code_train_df["Ground Truth Utterance"].tolist()
print(f"Total entries: {len(data)}")

def process_entry(entry):
    contexts_list = re.split(r'__eou__|__eot__', entry)
    cleaned_list = [context.strip() for context in contexts_list if len(context.strip()) > 0]
    return cleaned_list

if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        results = pool.map(process_entry, data)

    final_list = [item for sublist in results for item in sublist]
    final_list = list(set(final_list))
    with open(output_textfile, 'w') as file:
        for item in final_list:
            file.write(f"{item}\n")
    total_text = len(final_list)

    print(f"Processing english complete. Total unique entries: {total_text}. Generating code dataset...")

"""
    max_length = max(len(t) for t in final_list)
    code_ds_train = load_dataset("codeparrot/github-code", streaming=True, split="test", trust_remote_code=False)
    total = 0
    for entry in iter(code_ds_train):
        code = entry["code"]
        if len(code) > max_length:
            code = code[:max_length]
        total += 1
        with open(path_code_files + f"/{total}.txt" , 'w') as file:
            file.write(f"{code}")
        if total == (total_text): # we want a balanced dataset
            break

    print(f"Processing code complete. Total unique entries: {total}.")
"""