# Program that takes batched samples in form of pair sentences and tell if they a paraphrases of each other or not.
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from datasets import load_dataset

raw_dataset = load_dataset(
    "glue", "mrpc"
)  # glue = NLP functions / mrpc = microsoft dataset

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# Applying Batching:
tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

# Dynamic Padding + Cleaning unnessassary coulums:
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:10]  # Taking 10 samples for traning.
sample = {
    k: v
    for k, v in samples.items()
    if k not in ["idx", "sentence1", "sentence2"]  # Removing not needed columns.
}

# Padding lenght for current sample batch:
# This will add paddings to all sentences equal to the lenght of longest sentence in the batch.
padding_len = [len(p) for p in samples["input_ids"]]

batch = data_collator(sample)
print({k: v.shape for k, v in batch.items()})
