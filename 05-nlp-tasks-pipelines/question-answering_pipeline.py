import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

question = "Which deep learning libraries back 🤗 Transformers?"
context = """🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration between them."""

inputs = tokenizer(question, context, return_tensors="pt", return_offsets_mapping=True)

offset_mapping = inputs.pop("offset_mapping")

sequence_ids = inputs.sequence_ids(0)

outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# We mask everything that is NOT the context (1).
# Crucially, we do NOT unmask [CLS] (Index 0).
mask = [i != 1 for i in sequence_ids]
mask = torch.tensor(mask)[None]

start_logits[mask] = -10000
end_logits[mask] = -10000


start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]

scores = start_probabilities[:, None] * end_probabilities[None, :]
scores = torch.triu(scores)

max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]

# Text Slicing
# offset_mapping is a tensor [Batch, Seq, 2], so we access batch [0]
start_char, _ = offset_mapping[0][start_index]
_, end_char = offset_mapping[0][end_index]

answer = context[start_char:end_char]
print(f"Answer: {answer}")
