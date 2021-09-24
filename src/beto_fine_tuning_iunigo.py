import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer,
                          AutoTokenizer
                          )
import pandas as pd
df = pd.read_csv('../Iunigo/dataset/auto-clean.csv')
df = df.dropna().reset_index(drop=True)
df = df.sample(frac=1)
train = df[:int(.7*len(df))]
test = df[int(.7*len(df)):]
train.to_csv('../Iunigo/dataset/iunigo_train.csv', index=False)
test.to_csv('../Iunigo/dataset/iunigo_test.csv', index=False)

raw_datasets = datasets.load_dataset(
    'csv', data_files={
        'train': '../Iunigo/dataset/iunigo_train.csv',
        'test': '../Iunigo/dataset/iunigo_test.csv'
    })


tokenizer = AutoTokenizer.from_pretrained(
    'dccuchile/bert-base-spanish-wwm-uncased')


def tokenize_function(examples):
    return tokenizer(examples['descripcion'], padding="max_length", truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(
    seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(
    seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

model = AutoModelForSequenceClassification.from_pretrained(
    'dccuchile/bert-base-spanish-wwm-uncased',
    num_labels=2)
training_args = TrainingArguments('test_trainer')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_score(labels, predictions)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer
