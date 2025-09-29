import os
from transformers import AutoTokenizer, AutoModel

model_name = "klue/roberta-base"
save_dir = "model"

os.makedirs(save_dir, exist_ok=True)

print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(f"Saving to {save_dir}...")
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print("Download complete!")