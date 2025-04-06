import pandas as pd
import spacy
from spacy.tokens import DocBin
import re
from pathlib import Path
import json
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from collections import Counter
import sys

# Load the knowledge base
def load_knowledge_base():
    print("Loading knowledge base...")
    try:
        # Load diseases
        diseases = pd.read_csv('Knowledge_base/data/extracted_diseases.csv')
        disease_terms = set(diseases['Name'].str.lower())
        print(f"Loaded {len(disease_terms)} disease terms")
        
        # Load symptoms
        symptoms = pd.read_csv('Knowledge_base/data/extracted_symptoms.csv')
        symptom_terms = set(symptoms['Name'].str.lower())
        print(f"Loaded {len(symptom_terms)} symptom terms")
        
        return {
            'DISEASE': disease_terms,
            'SYMPTOM': symptom_terms
        }
    except Exception as e:
        print(f"Error loading knowledge base: {str(e)}")
        sys.exit(1)

def preprocess_term(term):
    # Remove special characters and extra spaces
    term = re.sub(r'[^\w\s]', ' ', term)
    term = ' '.join(term.split())
    return term.lower()

def similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a, b).ratio() >= threshold

def create_annotations(text, knowledge_base):
    try:
        nlp = spacy.blank('en')
        doc = nlp(text)
        
        # Initialize spans list
        spans = []
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        
        # Find matches for each entity type
        for entity_type, terms in knowledge_base.items():
            for term in terms:
                # Preprocess term
                term = preprocess_term(term)
                
                # Skip very short terms that might cause false positives
                if len(term.split()) < 2 and len(term) < 4:
                    continue
                
                # Find exact matches
                for match in re.finditer(r'\b' + re.escape(term) + r'\b', text_lower):
                    start, end = match.span()
                    original_text = text[start:end]
                    spans.append((start, end, entity_type, original_text))
                
                # Find fuzzy matches
                words = text_lower.split()
                for i in range(len(words) - len(term.split()) + 1):
                    candidate = ' '.join(words[i:i + len(term.split())])
                    if similar(candidate, term):
                        # Find the exact position in original text
                        start = text_lower.find(candidate)
                        if start != -1:
                            end = start + len(candidate)
                            original_text = text[start:end]
                            spans.append((start, end, entity_type, original_text))
        
        # Sort spans by start position
        spans.sort(key=lambda x: x[0])
        
        # Remove overlapping spans by keeping the longest one
        non_overlapping_spans = []
        if spans:
            current_span = spans[0]
            for next_span in spans[1:]:
                if next_span[0] < current_span[1]:  # Overlapping spans
                    # Keep the longer span
                    if (next_span[1] - next_span[0]) > (current_span[1] - current_span[0]):
                        current_span = next_span
                else:
                    non_overlapping_spans.append(current_span)
                    current_span = next_span
            non_overlapping_spans.append(current_span)
        
        # Create doc with entities
        ents = []
        for start, end, label, _ in non_overlapping_spans:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        
        doc.ents = ents
        return doc
    except Exception as e:
        print(f"Error creating annotations: {str(e)}")
        return None

def visualize_annotations(annotations, output_dir):
    try:
        print("Creating visualizations...")
        # Count entity types
        entity_counts = Counter()
        for doc in annotations:
            for ent in doc['entities']:
                entity_counts[ent[2]] += 1
        
        if not entity_counts:
            print("No entities found to visualize")
            return
        
        print(f"Found entity counts: {dict(entity_counts)}")
        
        # Create pie chart
        plt.figure(figsize=(10, 6))
        plt.pie(entity_counts.values(), labels=entity_counts.keys(), autopct='%1.1f%%')
        plt.title('Distribution of Entity Types')
        plt.savefig(output_dir / 'entity_distribution.png')
        plt.close()
        print("Created pie chart")
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(entity_counts.keys(), entity_counts.values())
        plt.title('Number of Entities by Type')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'entity_counts.png')
        plt.close()
        print("Created bar chart")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

def main():
    try:
        print("Starting annotation process...")
        # Create spaCy blank model
        nlp = spacy.blank('en')
        
        # Load knowledge base
        knowledge_base = load_knowledge_base()
        
        # Load PubMed reports
        print("Loading PubMed reports...")
        reports = pd.read_csv('report_data/data/pubmed_medical_reports.csv')
        print(f"Loaded {len(reports)} reports")
        
        # Create DocBin for storing annotated documents
        doc_bin = DocBin()
        
        # Process each report
        print("Processing reports...")
        for i, (_, row) in enumerate(reports.iterrows(), 1):
            # Combine title and abstract
            text = f"{row['title']}. {row['abstract']}"
            
            # Create annotations
            doc = create_annotations(text, knowledge_base)
            if doc is not None:
                # Add to DocBin
                doc_bin.add(doc)
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(reports)} reports")
        
        # Save annotated data
        print("Saving annotated data...")
        output_dir = Path('annotated_data')
        output_dir.mkdir(exist_ok=True)
        
        # Save as spaCy binary format
        doc_bin.to_disk(output_dir / 'annotated_reports.spacy')
        print("Saved spaCy binary format")
        
        # Also save as JSON for easier inspection
        print("Creating JSON format...")
        annotations = []
        for doc in doc_bin.get_docs(nlp.vocab):
            annotations.append({
                'text': doc.text,
                'entities': [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            })
        
        with open(output_dir / 'annotated_reports2.json', 'w') as f:
            json.dump(annotations, f, indent=2)
        print("Saved JSON format")
        
        # Create visualizations
        visualize_annotations(annotations, output_dir)
        print("Process completed successfully")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 