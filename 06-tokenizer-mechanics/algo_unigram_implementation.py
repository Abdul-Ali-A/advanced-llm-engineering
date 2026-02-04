from math import log
from collections import defaultdict
from transformers import AutoTokenizer

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

word_frequencies = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        text
    )
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_frequencies[word] += 1


character_frequencies = defaultdict(int)
subwords_frequencies = defaultdict(int)
for word, frequency in word_frequencies.items():
    word_lenght = len(word)
    for i in range(word_lenght):
        character_frequencies[word[i]] += frequency
        for j in range(i + 2, word_lenght + 1):
            subwords_frequencies[word[i:j]] += frequency


sorted_subwords = sorted(subwords_frequencies.items(), key=lambda x: x[1], reverse=True)


token_frequencies = (
    list(character_frequencies.items())
    + sorted_subwords[: 300 - len(character_frequencies)]
)
token_frequencies = {token: frequency for token, frequency in token_frequencies}


total_sum = sum([frequency for token, frequency in token_frequencies.items()])
model = {
    token: -log(frequency / total_sum) for token, frequency in token_frequencies.items()
}


def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)

    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score
