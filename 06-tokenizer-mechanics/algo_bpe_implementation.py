# Normalizing and Pre-Tokenizing using the BPE Algorithm.

from collections import defaultdict
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

word_frequencies = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        text
    )
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_frequencies[word] += 1


alphabet = []

for word in word_frequencies.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

# required token "["<|endoftext|>"]" at the start for the gpt model
vocab = ["<|endoftext|>"] + alphabet.copy()

# assigning all the chracters to the words they belong to in a dict comprehension
splits = {word: [character for character in word] for word in word_frequencies.keys()}


# computing the frequency of each pair:
def compute_pair_frequencies(splits):
    pair_frequencies = defaultdict(int)
    for word, frequency in word_frequencies.items():
        split = splits[word]
        split_len = len(split)
        if split_len == 1:
            continue
        for i in range(split_len - 1):
            pair = (split[i], split[i + 1])
            pair_frequencies[pair] += frequency
    return pair_frequencies


pair_frequencies = compute_pair_frequencies(splits)


# finding the most frequent pair
most_frequent_pair = ""
max_frequency = None

for pair, frequency in pair_frequencies.items():
    if max_frequency is None or max_frequency < frequency:
        max_frequency = frequency
        most_frequent_pair = pair


# merging the most frequent pairs
def merging_pairs(first, second, splits):
    for word in word_frequencies:
        split = splits[word]
        split_len = len(split)
        if split_len == 1:
            continue
        i = 0
        while i < split_len - 1:
            if split[i] == first and split[i + 1] == second:
                split = split[:i] + [first + second] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
