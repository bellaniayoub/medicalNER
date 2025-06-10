import torch
import os
import json
import pdfplumber
from transformers import BertTokenizerFast, BertForTokenClassification
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

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
pdf_path = r"C:/Users/PC/Downloads/modell/15708928_.pdf"
with pdfplumber.open(pdf_path) as pdf:
    full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Split into sentences
sentences = full_text.split(".")

# Collect tokens and labels for new PDF
highlighted_words = []

for sentence in sentences:
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

    previous_word_idx = None
    for idx, word_idx in enumerate(encoding.word_ids(batch_index=0)):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        token = words[word_idx]
        label = id2label.get(predictions[idx], "O")
        highlighted_words.append((token, label))
        previous_word_idx = word_idx

# Create PDF with highlights
output_pdf = "ner_highlighted_outputt.pdf"
c = canvas.Canvas(output_pdf, pagesize=letter)
width, height = letter
x, y = 40, height - 50
line_height = 14

# Color map
label_colors = {
    "B-DISEASE": colors.red,
    "I-DISEASE": colors.red,
    "B-SYMPTOM": colors.orange,
    "I-SYMPTOM": colors.orange,
    "B-GENE": colors.green,
    "I-GENE": colors.green,
    "B-PROTEIN": colors.blue,
    "I-PROTEIN": colors.blue,
}


for word, label in highlighted_words:
    word_color = label_colors.get(label, colors.black)
    c.setFillColor(word_color)
    if x + len(word)*6 > width - 40:
        x = 40
        y -= line_height
    c.drawString(x, y, word)
    x += (len(word) + 1) * 6

    if y < 50:  # Create new page if space runs out
        c.showPage()
        x, y = 40, height - 50

c.save()
print(f"\n✅ Highlighted PDF saved to: {output_pdf}")
