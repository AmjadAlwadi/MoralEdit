from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
from datasets import load_dataset


# Load the tokenizer and model
model_name = "openai-community/gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# # Tokenize the prompt and response
# inputs = tokenizer(prompt,truncation=True, return_tensors="pt")
# labels = tokenizer(response,truncation=True,  return_tensors="pt").input_ids

ds = load_dataset("json", data_files="norms_edit_propmts_dataset.json")

print(ds)
print(ds["train"][0])
print(ds["train"]["prompt"])
