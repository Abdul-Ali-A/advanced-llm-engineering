# Tracking loss during training with the Trainer.
import wandb
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer)
metrics = evaluate.load("glue", "mrpc")


def tokenizer_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)  # Sofmax
    return metrics.compute(predictions=predictions, references=labels)


raw_datasets = load_dataset("glue", "mrpc")
tokenized_datasets = raw_datasets.map(tokenizer_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")


# Initialize Weights & Biases for experiment tracking:
wandb.init(project="transformer-fine-tuning", name="bert-mrpc-analysis")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    logging_steps=10,  # Log metrics every 10 steps
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    report_to="wandb",  # Send logs to Weights & Biases
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
