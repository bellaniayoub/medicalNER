import torch
import os
import json
import pdfplumber
from transformers import BertTokenizerFast, BertForTokenClassification
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
import nltk
from nltk.tokenize import sent_tokenize



# 📁 Model path
model_dir = r"C:\Users\PC\Downloads\modell\bert_ner_model_version1"
pdf_path = r"C:\Users\PC\Downloads\modell\15708928_.pdf"
output_pdf_path = r"C:\Users\PC\Downloads\modell\ner_highlighted_output.pdf"

# 🎨 Color mapping for entity labels
entity_colors = {
    "B-DISEASE": "#FFB3B3",
    "I-DISEASE": "#FFB3B3",
    "B-SYMPTOM": "#FFD580",
    "I-SYMPTOM": "#FFD580",
    "B-GENE": "#C3FDB8",
    "I-GENE": "#C3FDB8",
    "B-PROTEIN": "#B0E0E6",
    "I-PROTEIN": "#B0E0E6",
    "O": None
}

# 📦 Load model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
model = BertForTokenClassification.from_pretrained(model_dir, local_files_only=True)
with open(os.path.join(model_dir, "label_mappings.json")) as f:
    label_mappings = json.load(f)
id2label = {int(k): v for k, v in label_mappings["id2label"].items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 🧾 Extract text from PDF
with pdfplumber.open(pdf_path) as pdf:
    full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# 🧠 Sentence tokenization with nltk for better splitting
sentences = sent_tokenize(full_text)

results = []

for sentence in sentences:
    words = sentence.strip().split()
    if not words:
        continue

    # Get token encoding with word_ids for alignment
    encoding = tokenizer(words, is_split_into_words=True, truncation=True, padding=True)
    inputs = tokenizer(words, return_tensors="pt", is_split_into_words=True, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    word_map = encoding.word_ids()

    seen = set()
    formatted_sentence = ""

    for idx, word_idx in enumerate(word_map):
        if word_idx is None or word_idx in seen:
            continue
        seen.add(word_idx)
        token = words[word_idx]
        label = id2label[predictions[idx]]

        color = entity_colors.get(label, None)
        if color:
            formatted_sentence += f"[{token}]({label}) "
        else:
            formatted_sentence += token + " "

    results.append(formatted_sentence.strip())

# 🖨️ Write results to PDF with highlights
c = canvas.Canvas(output_pdf_path, pagesize=A4)
width, height = A4
text_object = c.beginText(40, height - 40)
text_object.setFont("Helvetica", 12)

line_height = 16
max_width = width - 80  # margins

for line in results:
    words = line.split(" ")
    for word in words:
        if word.startswith("[") and "](" in word:
            word_text = word.split("](")[0][1:]
            label = word.split("](")[1][:-1]
            color = entity_colors.get(label, "#DDDDDD")

            word_width = stringWidth(word_text + " ", "Helvetica", 12)
            x = text_object.getX()
            y = text_object.getY()

            # Wrap line if needed
            if x + word_width > max_width:
                text_object.textLine("")
                x = text_object.getX()
                y = text_object.getY()

            # Draw highlight rectangle
            c.setFillColor(HexColor(color))
            c.rect(x - 2, y - 2, word_width + 4, line_height, fill=True, stroke=0)
            c.setFillColor(HexColor("#000000"))
            text_object.textOut(word_text + " ")

        else:
            word_width = stringWidth(word + " ", "Helvetica", 12)
            x = text_object.getX()
            if x + word_width > max_width:
                text_object.textLine("")
            text_object.textOut(word + " ")

    text_object.textLine("")

c.drawText(text_object)
c.save()

print("✅ PDF with NER highlights saved at:", output_pdf_path)
