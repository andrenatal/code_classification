from transformers import pipeline
from datasets import load_dataset
from datasets import Dataset
from datasets import DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import EarlyStoppingCallback, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_computed = accuracy.compute(predictions=predictions, references=labels)
    accuracy_ = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Precision {:.2f}, Recall {:.2f} F1 Score: {:.2f}, Accuracy: {:.2f}%'.format(precision, recall, f1, accuracy_ * 100))
    plt.savefig(f'confusion_matrix.png', dpi=300)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    return accuracy_computed

imdb = load_dataset("imdb")
test_size = 0.1
text = imdb["train"]["text"] + imdb["test"]["text"]
label = np.full(len(text), 0).tolist()
code_ds_train = load_dataset("codeparrot/github-code", streaming=True, split="train", trust_remote_code=True)
total = 0
for entry in iter(code_ds_train):
    text.append(entry["code"])
    label.append(1)
    total += 1
    if total == (50000):
        break

train_texts, test_texts, train_labels, test_labels = train_test_split(
    text, label, test_size=0.1, stratify=label, random_state=42
)

trainset_dict = {"text": train_texts, "label": train_labels}
testset_dict = {"text": test_texts, "label": test_labels}
dataset_dict = DatasetDict( {"train":Dataset.from_dict(trainset_dict), "test":Dataset.from_dict(testset_dict)})

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
tokenized_imdb = dataset_dict.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

id2label = {0: "ENGLISH", 1: "CODE"}
label2id = {"ENGLISH": 0, "CODE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=100,
    early_stopping_threshold=0.001
)

training_args = TrainingArguments(
    output_dir="code_english_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.001,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    push_to_hub=True,
    report_to=["tensorboard"],
    logging_steps=50,
    eval_steps=100,
    save_steps=100,
    metric_for_best_model="eval_loss",
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)

trainer.train()

print(f"Last saved checkpoint: {trainer.state.best_model_checkpoint}")
print(trainer.evaluate())
print("Finished training")

classifier = pipeline("sentiment-analysis", model=trainer.state.best_model_checkpoint, device="cuda:0")
text = "The server is broken. Can you fix it?"
print(text)
classifier(text)
text = "<html><head><title>Test</title></head><body><p>Hello World</p></body></html>"
print(text)
classifier(text)