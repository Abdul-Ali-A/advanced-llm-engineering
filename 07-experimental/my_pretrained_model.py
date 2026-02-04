from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"  # Small French BERT (~110M params)
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.push_to_hub("Abdul-Ali/Test-Model")
tokenizer.push_to_hub("Abdul-Ali/Test-Model")
