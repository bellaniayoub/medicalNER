import torch
import os
import json
import pdfplumber
from transformers import BertTokenizerFast, BertForTokenClassification

# Load model
model_dir = r"C:\Users\PC\Downloads\modell\bert_ner_model_version1"
print(f"Loading model from: {model_dir}")
tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
model = BertForTokenClassification.from_pretrained(model_dir, local_files_only=True)

# Load label mappings
with open(os.path.join(model_dir, "label_mappings.json"), "r") as f:
    label_mappings = json.load(f)
id2label = {int(k): v for k, v in label_mappings["id2label"].items()}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load text from PDF
pdf_path = r"C:/Users/PC/Downloads/modell/15708928_.pdf"  # Change path as needed
with pdfplumber.open(pdf_path) as pdf:
    full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Prepare output dictionary
results = {}

# Split text into sentences (simple split on ".")
sentences = full_text.split(".")

for i, sentence in enumerate(sentences):
    words = sentence.strip().split()
    if not words:
        continue

    encoding = tokenizer(
        words,
        return_tensors="pt",
        is_split_into_words=True,
        truncation=True,
        padding=True
    )
    tokens = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()

    sentence_results = []
    previous_word_idx = None
    for idx, word_idx in enumerate(encoding.word_ids(batch_index=0)):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        token = words[word_idx]
        label = id2label.get(predictions[idx], "O")
        sentence_results.append({"token": token, "label": label})
        previous_word_idx = word_idx

    results[f"sentence_{i+1}"] = sentence_results

# Save results to JSON file
output_path = "ner_results.json"
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(results, json_file, indent=4, ensure_ascii=False)

print(f"\n✅ NER results saved to {output_path}")
