import torch
import evaluate
from tqdm.auto import tqdm
from torch.optim import AdamW
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (
    get_scheduler,
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)


# <--------------------------(Initiallizing-Model+Tokenizer)--------------------------->
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# <--------------------------(Setting-Datasets)--------------------------->
raw_datasets = load_dataset("glue", "mrpc")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# <--------------------------(Initiallizing-Dataloaders)--------------------------->
train_dataloader = DataLoader(
    tokenized_datasets["train"], batch_size=8, collate_fn=data_collator, shuffle=True
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)


# <--------------------------(Optimizer+Traning-Parameters-Setup)--------------------------->
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
num_epochs = 3
# len(train_dataloader) = Number of traning batches.
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,  # Starts directly at lr = 5
    num_training_steps=num_training_steps,
)
# print(num_training_steps)


# <--------------------------(Setting-Up-GPU-For-Traning-Speed)--------------------------->
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# <--------------------------(Optimizing-Steps-For-Traning)--------------------------->
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {
            k: v.to(device) for k, v in batch.items()
        }  # k:v.to(device) moves the traning batch to GPU/CPU.
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # Gradient Clipping before next step.
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# <--------------------------(Optimizing-Steps-For-Evaluation)--------------------------->
metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {
        k: v.to(device) for k, v in batch.items()
    }  # k:v.to(device) moves the traning batch to GPU/CPU.
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()


# <--------------------------(Adding-Accelerator)--------------------------->
accelerator = Accelerator()
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)
