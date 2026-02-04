import itertools
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets

# --------------------------Dataset URLs--------------------------:
base_url = "https://the-eye.eu/public/AI/pile/"
data_files_pile = {
    "train": [base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
}
freelaw_url = "https://the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst"
pubmed_url = ""

# --------------------------Loading Datasets in Streaming Mode--------------------------:
pubmed_dataset_streamed = load_dataset(
    "json", data_files=pubmed_url, split="train", streaming=True
)

law_dataset_streamed = load_dataset(
    "json", data_files=freelaw_url, split="train", streaming=True
)

# --------------------------Interleaving the Datasets--------------------------:
combined_dataset = interleave_datasets([pubmed_dataset_streamed, law_dataset_streamed])

# --------------------------Setting-Up Tokenizer--------------------------:
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)

# --------------------------Dispalying the Result--------------------------:
iterator = iter(tokenized_dataset)

print("--- Example 1 ---")
print(next(iterator))

print("\n--- Example 2 ---")
print(next(iterator))
