# Program that uses Trainer API to train the Data for Sequence Classification.

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    AutoModelForSequenceClassification,
    Trainer,
)

# <--------------------------(Initaillizing-Model+Tokenizer)--------------------------->
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# <--------------------------(Setting-Datasets)--------------------------->
raw_datasets = load_dataset("glue", "mrpc")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# <--------------------------(Traning-The-Model)--------------------------->
# Setting up all the parameter for the trainer:
# Default value for epochs are 3.
training_args = TrainingArguments(
    "test-trainer", eval_strategy="epoch"
)  # For saving Model Checkpoints.
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
)
trainer.train()


# <--------------------------(Model-Making-Pridictions)--------------------------->
predictions = trainer.predict(tokenized_datasets["validation"])
preds = np.argmax(
    predictions.predictions, axis=-1
)  # Picks the Prediction with the highest score between 0 & 1.


# <--------------------------(Giving-Metrics-To-Model)--------------------------->
metric = evaluate.load("glue", "mrpc")
results = metric.compute(
    predictions=preds, references=predictions.label_ids
)  # This returns Accuracy + F1 Score(Precision+Recall{Harmonic Mean})
print("Validation Results:", results)
