from transformers import AutoTokenizer


# --------------------------Loading base tokenizer--------------------------
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")


# --------------------------Training Data(Corpus)--------------------------
training_corpus = [
    "def merge_sort(list):",
    "    if len(list) < 2:",
    "        return list",
    "    mid = len(list) // 2",
    "    left = merge_sort(list[:mid])",
    "    right = merge_sort(list[mid:])",
    "    return merge(left, right)",
]


# --------------------------Initializing a Generator--------------------------
def batch_iterator(batch_size=1000):
    for i in range(0, len(training_corpus), batch_size):
        yield training_corpus[i : i + batch_size]


# --------------------------Training the New Tokenizer--------------------------
new_tokenizer = old_tokenizer.train_new_from_iterator(
    batch_iterator(), vocab_size=52000
)

print(new_tokenizer.tokenize("def merge_sort(list):"))
