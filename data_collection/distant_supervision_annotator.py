import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
import fitz  # PyMuPDF
import os
import json
import re
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt")

# Charger le modèle spaCy
nlp = spacy.load("en_core_web_sm")

# Nettoyer le texte
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Charger une KB sous forme de set de termes
def load_terms(file_path, column='Term', min_words=1):
    df = pd.read_csv(file_path)
    terms = set()
    for term in df[column].dropna():
        term_clean = term.lower().strip()
        if len(term_clean.split()) >= min_words:
            terms.add(term_clean)
    return terms

# Créer un PhraseMatcher à partir des termes et d'un label
def build_phrase_matcher(kb_terms, label):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(term) for term in kb_terms]
    matcher.add(label, patterns)
    return matcher

# Extraire les phrases à partir d’un PDF
def extract_sentences_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    text = clean_text(text)
    # Save to a text file
    with open("text.txt", 'w', encoding='utf-8') as f:
        f.write(text)
    sentences = sent_tokenize(text)
    return [s for s in sentences if len(s.split()) > 3]

# Annoter une phrase avec les matchers
def annotate_with_spacy(sentence, matchers):
    doc = nlp(sentence)
    results = []
    for label, matcher in matchers.items():
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            results.append({"entity": span.text, "label": label})
    return results

# Annoter un article PDF complet
def annotate_article(pdf_path, matchers):
    annotations = []
    sentences = extract_sentences_from_pdf(pdf_path)
    for sent in sentences:
        matches = annotate_with_spacy(sent, matchers)
        if matches:
            annotations.append({"sentence": sent, "annotations": matches})
    return annotations

# === MAIN POUR TESTER UN SEUL PDF === #
if __name__ == "__main__":
    # Charger les KBs
    symptoms = load_terms("Knowledge_base/data/BioPortal/CSSO_cleaned.csv")
    diseases = load_terms("Knowledge_base/data/BioPortal/DOID_cleaned.csv")
    genes = load_terms("Knowledge_base/data/BioPortal/go_terms.csv", column="Name")
    proteins = load_terms("Knowledge_base/data/BioPortal/PR_cleaned.csv")

    # Créer les matchers
    matchers = {
        "SYMPTOM": build_phrase_matcher(symptoms, "SYMPTOM"),
        "DISEASE": build_phrase_matcher(diseases, "DISEASE"),
        "GENE": build_phrase_matcher(genes, "GENE"),
        "PROTEIN": build_phrase_matcher(proteins, "PROTEIN"),
    }

    # Chemin vers le fichier PDF à tester
    test_pdf = r"data_scrapping\pubmed_scraper\pmc_pdfs\2517695_Risk factors for.pdf"

    annotated_data = annotate_article(test_pdf, matchers)

    # Sauvegarde
    output_path = "annotated_data/test1_annotationfile_spacy.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated_data, f, indent=4)

    print(f"✅ Annotated {len(annotated_data)} sentences. Saved to {output_path}")