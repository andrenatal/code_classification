from transformers import pipeline
from datasets import load_dataset
from datasets import Dataset
from datasets import DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


imdb = load_dataset("imdb")

print(imdb)

trainset_dict = {"text":[],"label":[]}
for entry in imdb["train"]:
    trainset_dict["text"].append(entry["text"])
    trainset_dict["label"].append(0)

testset_dict = {"text":[],"label":[]}
for entry in imdb["test"]:
    testset_dict["text"].append(entry["text"])
    testset_dict["label"].append(0)

# now we get the code dataset and add 25000 to the respective datasets
code_ds_train = load_dataset("codeparrot/github-code", streaming=True, split="train", trust_remote_code=True)
total = 0
for entry in iter(code_ds_train):
    if total % 2 == 0:
        trainset_dict["text"].append(entry["code"])
        trainset_dict["label"].append(1)
    else:
        testset_dict["text"].append(entry["code"])
        testset_dict["label"].append(1)
    total += 1
    if total == (50000):
        break


trainset = Dataset.from_dict(trainset_dict)
testset = Dataset.from_dict(testset_dict)
print(testset)
print(trainset)

dataset_dict = {"train":trainset, "test":testset}
print(dataset_dict)

print(imdb)

dataset_dict = DatasetDict(dataset_dict)
print(dataset_dict)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
tokenized_imdb = dataset_dict.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

id2label = {0: "ENGLISH", 1: "CODE"}
label2id = {"ENGLISH": 0, "CODE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="code_english_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to=["tensorboard"],
    logging_steps=1,  # how often to log to W&B
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

classifier = pipeline("sentiment-analysis", model="code_english_model/checkpoint-1564", device="cuda:0")

text = "hello, I love you a lot"
classifier(text)