import os
import json
import torch
import pdfplumber
from flask import Flask, request, render_template, send_file, redirect, url_for
from transformers import BertTokenizerFast, BertForTokenClassification
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16 MB upload

# Paths and device setup
MODEL_DIR = r"C:\Users\PC\Downloads\modell\bert_ner_model_version1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
model = BertForTokenClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.to(DEVICE)
model.eval()

# Load label mapping
with open(os.path.join(MODEL_DIR, "label_mappings.json"), "r") as f:
    label_mappings = json.load(f)
id2label = {int(k): v for k, v in label_mappings["id2label"].items()}

# Color map for entities
ENTITY_COLORS = {
    "B-DISEASE": "#FFB3B3",
    "I-DISEASE": "#FFB3B3",
    "B-SYMPTOM": "#FFD580",
    "I-SYMPTOM": "#FFD580",
    "B-GENE": "#C3FDB8",
    "I-GENE": "#C3FDB8",
    "B-PROTEIN": "#B0E0E6",
    "I-PROTEIN": "#B0E0E6",
    "O": None,
}

def ner_and_highlight(text):
    """
    Run NER on input text, return:
    - formatted sentences with entity labels for display,
    - dictionary of entity counts,
    - list of extracted entities (type and text).
    """
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)

    formatted_sentences = []
    entities_count = {}
    collected_entities = []

    for sentence in sentences:
        words = sentence.strip().split()
        if not words:
            continue

        # Tokenize words, keep BatchEncoding for word_ids
        inputs = tokenizer(words, return_tensors="pt", is_split_into_words=True, truncation=True, padding=True)

        word_map = inputs.word_ids(batch_index=0)  # <-- get word_ids BEFORE converting to dict or moving

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

        seen = set()
        formatted_sentence = ""
        current_entity = None
        current_entity_tokens = []

        for idx, word_idx in enumerate(word_map):
            if word_idx is None or word_idx in seen:
                continue
            seen.add(word_idx)

            token = words[word_idx]
            label = id2label[predictions[idx]]

            if label.startswith("B-"):
                # Close previous entity if any
                if current_entity:
                    entity_text = " ".join(current_entity_tokens)
                    collected_entities.append((current_entity, entity_text))
                    entities_count[current_entity] = entities_count.get(current_entity, 0) + 1
                current_entity = label[2:]
                current_entity_tokens = [token]
                # Add colored token
                color = ENTITY_COLORS.get(label, None)
                if color:
                    formatted_sentence += f'<span style="background-color:{color}">{token}</span> '
                else:
                    formatted_sentence += token + " "

            elif label.startswith("I-") and current_entity == label[2:]:
                current_entity_tokens.append(token)
                color = ENTITY_COLORS.get(label, None)
                if color:
                    formatted_sentence += f'<span style="background-color:{color}">{token}</span> '
                else:
                    formatted_sentence += token + " "
            else:
                # Close previous entity if any
                if current_entity:
                    entity_text = " ".join(current_entity_tokens)
                    collected_entities.append((current_entity, entity_text))
                    entities_count[current_entity] = entities_count.get(current_entity, 0) + 1
                current_entity = None
                current_entity_tokens = []
                formatted_sentence += token + " "

        # Close any entity left open at end of sentence
        if current_entity:
            entity_text = " ".join(current_entity_tokens)
            collected_entities.append((current_entity, entity_text))
            entities_count[current_entity] = entities_count.get(current_entity, 0) + 1

        formatted_sentences.append(formatted_sentence.strip())

    return formatted_sentences, entities_count, collected_entities

def create_highlighted_pdf(text, output_path, collected_entities):
    """Generate PDF with highlighted entity background boxes."""

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin
    line_height = 16
    c.setFont("Helvetica", 12)

    lines = text.split('\n')
    for line in lines:
        if y < margin:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 12)

        words = line.split()
        x = margin

        for word in words:
            # check if word in any entity to highlight
            highlight_color = None
            for ent_type, ent_text in collected_entities:
                if ent_text in word:
                    highlight_color = ENTITY_COLORS.get("B-" + ent_type, None)
                    break

            if highlight_color:
                c.setFillColor(HexColor(highlight_color))
                c.rect(x - 2, y - 4, len(word) * 6, line_height, fill=True, stroke=0)
                c.setFillColor(HexColor("#000000"))

            c.drawString(x, y, word)
            x += (len(word) + 1) * 6

        y -= line_height

    c.save()

@app.route("/", methods=["GET"])
def welcome():
    return render_template("welcome.html")

@app.route("/analyze", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "pdf_file" not in request.files:
            return "No file part", 400

        pdf_file = request.files["pdf_file"]
        if pdf_file.filename == "":
            return "No selected file", 400

        if not pdf_file.filename.lower().endswith(".pdf"):
            return "Only PDF files allowed", 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf_file.save(tmp_pdf.name)
            tmp_pdf_path = tmp_pdf.name

        # Extract text from PDF
        with pdfplumber.open(tmp_pdf_path) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        # Run NER and prepare highlighted html
        formatted_sentences, entities_count, collected_entities = ner_and_highlight(full_text)
        highlighted_html = "<br>".join(formatted_sentences)

        # Generate highlighted PDF
        tmp_out_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        create_highlighted_pdf(full_text, tmp_out_pdf.name, collected_entities)

        return render_template(
            "result.html",
            highlighted_html=highlighted_html,
            entities_count=entities_count,
            pdf_url=url_for("download_file", path=os.path.basename(tmp_out_pdf.name)),
        )

    return render_template("index.html")


@app.route("/download/<path:path>")
def download_file(path):
    # Download generated PDF from temp folder
    full_path = os.path.join(tempfile.gettempdir(), path)
    if os.path.exists(full_path):
        return send_file(full_path, as_attachment=True, download_name="ner_highlighted_output.pdf")
    else:
        return "File not found", 404


if __name__ == "__main__":
    app.run(debug=True)
