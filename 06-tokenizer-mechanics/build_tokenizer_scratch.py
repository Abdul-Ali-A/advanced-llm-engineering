from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


# 1. <-----------------------(Getting Corpus)----------------------->
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
corpus_lenght = len(dataset)


def get_training_corpus():
    for i in range(0, corpus_lenght, 1000):
        yield dataset[i : i + 1000]["text"]  # 0-1000(examples) then 1001-2000


# Creating local text file to store all the inputs and text from wikitext-2(for local use):
with open("wikitext-2.txt", "w", encoding="utf-8") as f:
    for i in range(corpus_lenght):
        sentence = dataset[i]["text"] + "\n"
        f.write(sentence)


# A. <-----------------------(Building WordPiece Tokenizer)----------------------->
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# Normalizing using BERT Normalizer:
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

# Creating Normalizer by hand:
# <------------------------------------------------------------------------------------->
# tokenizer.normalizer = normalizers.Sequence(
#     [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
# )
# <------------------------------------------------------------------------------------->


# Pre-tokenizing using BERT pre-tokenizer:
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()


# Creating Pre-tokenizer by hand:
# <------------------------------------------------------------------------------------->
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# OR
# pre_tokenizer = pre_tokenizers.WhitespaceSplit()

# pre_tokenizer = pre_tokenizers.Sequence(
#     [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
# )
# <------------------------------------------------------------------------------------->


# Initializingg the WordPiece Trainer:
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)


# Traning the tokenizer on the local text files created:
tokenizer.model = models.WordPiece(unk_token="[UNK]")  # (must have to reinitialize)
tokenizer.train(["wikitext-2.txt"], trainer=trainer)


# Adding the special post-proccessing tokens([CLS],[SEP]):
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
# Passing the special tokens to the tokenizer:
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)


# Execution Phase:
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
encoding = tokenizer.encode("Let's test this tokenizer.")
# print(encoding.tokens)


# Initializing a Decoder and Saving the Tokenizer:
tokenizer.decoder = decoders.WordPiece(prefix="##")
tokenizer.decode(encoding.ids)
tokenizer.save("tokenizer.json")

# Loading the saved Tokenizer:
new_tokenizer = Tokenizer.from_file("tokenizer.json")
