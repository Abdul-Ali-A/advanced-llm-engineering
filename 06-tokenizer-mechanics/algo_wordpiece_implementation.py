# Normalizing and Pre-Tokenizing using the Word-Piece Algorithm.

from collections import defaultdict
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

word_frequencies = defaultdict(int)


def computing_frequencies():

    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
            text
        )

        new_words = [words for words, offsets in words_with_offsets]
        for word in new_words:
            word_frequencies[word] += 1
    return word_frequencies


word_frequencies = computing_frequencies()


alphabet = []
for word in word_frequencies.keys():
    first_letter = word[0]
    if first_letter not in alphabet:
        alphabet.append(first_letter)
    for letter in word[1:]:
        related_letter = f"##{letter}"
        if related_letter not in alphabet:
            alphabet.append(related_letter)

alphabet.sort()
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

splits = {
    word: [
        character if index == 0 else f"##{character}"
        for index, character in enumerate(word)
    ]
    for word in word_frequencies.keys()
}


def computing_pair_scores(splits):
    letter_frequencies = defaultdict(int)
    pair_frequencies = defaultdict(int)
    for word, frequency in word_frequencies.items():
        split = splits[word]
        if len(split) == 1:
            letter_frequencies[split[0]] += frequency
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_frequencies[split[i]] += frequency
            pair_frequencies[pair] += frequency
        letter_frequencies[split[-1]] += frequency

    scores = {
        pair: freq / (letter_frequencies[pair[0]] * letter_frequencies[pair[1]])
        for pair, freq in pair_frequencies.items()
    }
    return scores


pair_scores = computing_pair_scores(splits)

most_frequent_pair = ""
max_score = None
for pair, score in pair_scores.items():
    if max_score is None or max_score < score:
        most_frequent_pair = pair
        max_score = score

print(most_frequent_pair, max_score)


def merge_pair(first, second, splits):
    for word in word_frequencies:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == first and split[i + 1] == second:
                merge = (
                    first + second[2:] if second.startswith("##") else first + second
                )
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
