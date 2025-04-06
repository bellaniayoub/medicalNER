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
import time

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

def create_iob_annotations(text, knowledge_base):
    try:
        # Tokenize the text
        nlp = spacy.blank('en')
        doc = nlp(text)
        tokens = [token.text for token in doc]
        
        # Initialize IOB tags
        iob_tags = ['O'] * len(tokens)
        
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
                    # Find which tokens are covered by this span
                    token_start = 0
                    for i, token in enumerate(tokens):
                        token_end = token_start + len(token)
                        if token_start <= start < token_end:
                            # This is the first token of the entity
                            iob_tags[i] = f'B-{entity_type}'
                            # Mark subsequent tokens as inside the entity
                            j = i + 1
                            while j < len(tokens) and token_start + len(token) <= end:
                                iob_tags[j] = f'I-{entity_type}'
                                token_start += len(tokens[j]) + 1  # +1 for space
                                j += 1
                            break
                        token_start += len(token) + 1  # +1 for space
                
                # Find fuzzy matches
                words = text_lower.split()
                for i in range(len(words) - len(term.split()) + 1):
                    candidate = ' '.join(words[i:i + len(term.split())])
                    if similar(candidate, term):
                        # Find the exact position in original text
                        start = text_lower.find(candidate)
                        if start != -1:
                            end = start + len(candidate)
                            # Find which tokens are covered by this span
                            token_start = 0
                            for i, token in enumerate(tokens):
                                token_end = token_start + len(token)
                                if token_start <= start < token_end:
                                    # This is the first token of the entity
                                    iob_tags[i] = f'B-{entity_type}'
                                    # Mark subsequent tokens as inside the entity
                                    j = i + 1
                                    while j < len(tokens) and token_start + len(token) <= end:
                                        iob_tags[j] = f'I-{entity_type}'
                                        token_start += len(tokens[j]) + 1  # +1 for space
                                        j += 1
                                    break
                                token_start += len(token) + 1  # +1 for space
        
        return tokens, iob_tags
    except Exception as e:
        print(f"Error creating IOB annotations: {str(e)}")
        return None, None

def visualize_iob_annotations(annotations, output_dir):
    try:
        print("Creating visualizations...")
        # Count entity types
        entity_counts = Counter()
        for _, tags in annotations:
            for tag in tags:
                if tag != 'O':
                    entity_type = tag.split('-')[1]
                    entity_counts[entity_type] += 1
        
        if not entity_counts:
            print("No entities found to visualize")
            return
        
        print(f"Found entity counts: {dict(entity_counts)}")
        
        # Create pie chart
        plt.figure(figsize=(10, 6))
        plt.pie(entity_counts.values(), labels=entity_counts.keys(), autopct='%1.1f%%')
        plt.title('Distribution of Entity Types (IOB Format)')
        plt.savefig(output_dir / 'iob_entity_distribution.png')
        plt.close()
        print("Created pie chart")
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(entity_counts.keys(), entity_counts.values())
        plt.title('Number of Entities by Type (IOB Format)')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'iob_entity_counts.png')
        plt.close()
        print("Created bar chart")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

# New main function that processes only 5 articles with timing information
def main():
    try:
        print("Starting IOB annotation process for 5 articles...")
        start_time_total = time.time()
        
        # Load knowledge base
        knowledge_base = load_knowledge_base()
        
        # Load PubMed reports
        print("Loading PubMed reports...")
        reports = pd.read_csv('report_data/data/pubmed_medical_reports.csv')
        print(f"Loaded {len(reports)} reports")
        
        # Limit to 5 articles
        reports = reports.head(5)
        print(f"Processing only {len(reports)} articles")
        
        # Process each report
        print("Processing reports...")
        iob_annotations = []
        
        for i, (_, row) in enumerate(reports.iterrows(), 1):
            article_start_time = time.time()
            print(f"\nProcessing article {i}/{len(reports)}: {row['title'][:50]}...")
            
            # Combine title and abstract
            text = f"{row['title']}. {row['abstract']}"
            
            # Create IOB annotations
            tokens, tags = create_iob_annotations(text, knowledge_base)
            if tokens is not None and tags is not None:
                iob_annotations.append((tokens, tags))
                
                # Count entities found
                entity_count = sum(1 for tag in tags if tag != 'O')
                print(f"Found {entity_count} entities in this article")
            
            article_end_time = time.time()
            article_duration = article_end_time - article_start_time
            print(f"Article {i} processing time: {article_duration:.2f} seconds")
        
        # Save annotated data
        print("\nSaving annotated data...")
        output_dir = Path('annotated_data')
        output_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        json_start_time = time.time()
        with open(output_dir / 'iob_annotations_5_articles.json', 'w', encoding='utf-8') as f:
            json.dump([{'tokens': tokens, 'tags': tags} for tokens, tags in iob_annotations], f, indent=2)
        json_end_time = time.time()
        print(f"Saved IOB annotations in JSON format (took {json_end_time - json_start_time:.2f} seconds)")
        
        # Save as CSV (CoNLL format)
        tsv_start_time = time.time()
        with open(output_dir / 'iob_annotations_5_articles.csv', 'w', encoding='utf-8') as f:
            for tokens, tags in iob_annotations:
                for token, tag in zip(tokens, tags):
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")  # Empty line between documents
        tsv_end_time = time.time()
        print(f"Saved IOB annotations in CSV format (took {tsv_end_time - tsv_start_time:.2f} seconds)")
        
        # Create visualizations
        viz_start_time = time.time()
        visualize_iob_annotations(iob_annotations, output_dir)
        viz_end_time = time.time()
        print(f"Created visualizations (took {viz_end_time - viz_start_time:.2f} seconds)")
        
        end_time_total = time.time()
        total_duration = end_time_total - start_time_total
        print(f"\nProcess completed successfully in {total_duration:.2f} seconds")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 