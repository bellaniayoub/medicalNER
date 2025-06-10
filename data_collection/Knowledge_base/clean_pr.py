import pandas as pd
import re
import json
import spacy
from spacy.tokens import DocBin
from collections import Counter
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import rapidfuzz.process as rfuzz

def contains_japanese(text):
    # Check if the text contains Japanese characters (Hiragana, Katakana, or Kanji)
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    return bool(japanese_pattern.search(text))

def clean_pr_data():
    # Read the original CSV file
    df = pd.read_csv('PR.csv')
    
    # Create a list to store all labels and synonyms
    all_terms = []
    
    # Process each row
    for _, row in df.iterrows():
        # Add the preferred label if it exists and is not empty
        if pd.notna(row['Preferred Label']) and row['Preferred Label'].strip():
            all_terms.append(row['Preferred Label'])
        
        # Add synonyms if they exist and are not empty
        if pd.notna(row['Synonyms']) and row['Synonyms'].strip():
            # Split synonyms by pipe character and add each as a separate term
            synonyms = row['Synonyms'].split('|')
            for synonym in synonyms:
                synonym = synonym.strip()
                # Only add non-empty synonyms that don't contain Japanese characters
                if synonym and not contains_japanese(synonym):
                    all_terms.append(synonym)
    
    # Create a new dataframe with a single column
    df_cleaned = pd.DataFrame(all_terms, columns=['Term'])
    
    # Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Save the cleaned data to a new CSV file
    output_path = 'PR_cleaned.csv'
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original rows: {len(df)}")
    print(f"Total terms (including synonyms): {len(df_cleaned)}")

class EntityMatcher:
    def __init__(self, knowledge):
        self.knowledge = knowledge
        
        # Fall back to standard spaCy model if scientific model isn't available
        try:
            self.nlp = spacy.load("en_core_sci_sm", disable=["parser", "ner"])
            print("Using scientific spaCy model")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                print("Using standard spaCy model (scientific model not available)")
            except OSError:
                print("Warning: No spaCy models available, using blank model")
                self.nlp = spacy.blank("en")
        
        # Define stop words and common terms to ignore
        self.stop_words = {
            'patient', 'patients', 'study', 'studies', 'research', 'researchers',
            'method', 'methods', 'result', 'results', 'conclusion', 'conclusions',
            'introduction', 'background', 'discussion', 'abstract', 'summary'
        }
        
        # Pre-compute exact match patterns for each entity type
        self.exact_match_patterns = {}
        for ent_type, terms in self.knowledge.items():
            # Only precompile patterns for terms that are likely to be effective
            good_terms = {t for t in terms if len(t) > 3 and t not in self.stop_words}
            
            # Limit the number of patterns to avoid memory issues
            if len(good_terms) > 10000:
                # Keep longer terms which are more specific
                good_terms = sorted(good_terms, key=len, reverse=True)[:10000]
                
            self.exact_match_patterns[ent_type] = good_terms
            
        print(f"Compiled patterns for {sum(len(p) for p in self.exact_match_patterns.values())} terms")
        
        # Set up tokenizer for BIO tagging
        self.tokenizer = self.nlp.tokenizer
    
    def preprocess(self, text):
        """Enhanced text normalization"""
        # Convert to lowercase and normalize whitespace
        text = re.sub(r'[^\w\s-]', ' ', text.lower())
        return ' '.join(text.split())
    
    def find_matches(self, text, fuzzy_threshold=90):  # Increased threshold
        """Optimized hybrid exact + fuzzy matching"""
        text_clean = self.preprocess(text)
        entities = []
        
        # First pass: Exact matches (very fast)
        for ent_type, terms in self.exact_match_patterns.items():
            # For each entity type, look for its terms in the text
            for term in terms:
                # Skip if term is in stop words
                if term in self.stop_words:
                    continue
                    
                # Find all occurrences of the term
                start_idx = 0
                while True:
                    start = text_clean.find(term, start_idx)
                    if start == -1:
                        break
                    end = start + len(term)
                    entities.append((start, end, ent_type))
                    start_idx = start + 1
        
        # Second pass: Selective fuzzy matching on sections
        if len(text_clean) < 100000:  # ~100KB limit for fuzzy matching
            # Break text into chunks for fuzzy matching
            chunk_size = 5000
            chunks = [(i, text_clean[i:i+chunk_size]) 
                      for i in range(0, len(text_clean), chunk_size)]
            
            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for offset, chunk in chunks:
                    for ent_type, terms in self.knowledge.items():
                        # Randomly sample terms to avoid processing all terms
                        sample_size = min(500, len(terms))
                        sample_terms = list(terms)[:sample_size]
                        
                        # Filter out stop words and very short terms
                        sample_terms = [t for t in sample_terms if t not in self.stop_words and len(t) > 3]
                        
                        future = executor.submit(
                            self._fuzzy_match_chunk,
                            chunk, offset, sample_terms, ent_type, fuzzy_threshold
                        )
                        futures.append(future)
                
                # Gather results
                for future in futures:
                    chunk_entities = future.result()
                    entities.extend(chunk_entities)
        
        return self._merge_overlaps(entities)
    
    def _fuzzy_match_chunk(self, chunk, offset, terms, ent_type, threshold):
        """Process fuzzy matching for a single chunk"""
        chunk_entities = []
        
        # Use RapidFuzz for better performance
        matches = rfuzz.extract(
            chunk, 
            terms,
            scorer=rfuzz.token_set_ratio,
            score_cutoff=threshold,
            limit=10  # Only take top matches
        )
        
        for term, score, _ in matches:
            # Skip if term is in stop words
            if term in self.stop_words:
                continue
                
            start = chunk.find(term)
            if start != -1:
                chunk_entities.append((
                    offset + start,
                    offset + start + len(term),
                    ent_type
                ))
        
        return chunk_entities

def process_pdf(pdf_path, matcher=None):
    """Process single PDF with error handling and BIO tagging"""
    try:
        # Create matcher if not provided (for parallel processing)
        if matcher is None:
            # Load knowledge base (cached)
            knowledge = load_knowledge_base()
            matcher = EntityMatcher(knowledge)
        
        # Step 1: Extract text
        print(f"Processing {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print(f"Warning: No text extracted from {pdf_path.name}")
            return None
        
        # Step 2: Find entities
        entities = matcher.find_matches(text)
        
        # Step 3: Convert to BIO format
        tokens, bio_tags = matcher.convert_to_bio(text, entities)
        
        return {
            'source': pdf_path.name,
            'text': text,  # Store full text for all PDFs
            'entities': entities,
            'tokens': tokens,  # Store all tokens
            'bio_tags': bio_tags,  # Store all tags
            'entity_count': len(entities),
            'entity_types': Counter([e[2] for e in entities])
        }
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {str(e)}")
        return None

def batch_process_pdfs(pdf_paths, batch_size=5):  # Reduced batch size for better memory management
    """Process PDFs in smaller batches to avoid memory issues"""
    knowledge = load_knowledge_base()
    matcher = EntityMatcher(knowledge)
    
    results = []
    for i in range(0, len(pdf_paths), batch_size):
        batch = pdf_paths[i:i+batch_size]
        
        # Process this batch
        print(f"Processing batch {i//batch_size + 1}/{len(pdf_paths)//batch_size + 1}")
        
        batch_results = []
        for pdf_path in tqdm(batch):
            result = process_pdf(pdf_path, matcher)
            if result:
                batch_results.append(result)
        
        # Save intermediate results
        results.extend(batch_results)
        
        # Save checkpoint after each batch
        with open(OUTPUT_DIR/f'annotations_batch_{i//batch_size}.json', 'w') as f:
            json.dump(batch_results, f)
            
        print(f"Completed batch {i//batch_size + 1}: {len(batch_results)}/{len(batch)} successful")
        
        # Clear some memory
        batch_results = None
    
    return results

def main():
    # Get PDF files
    pdf_files = list(PDF_DIR.glob('*.pdf'))
    print(f"Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        print("No PDF files found! Please check the path.")
        return
    
    # Process in smaller batches
    results = batch_process_pdfs(pdf_files, batch_size=5)
    
    # Combine all batches and save final results
    valid_results = [r for r in results if r]
    print(f"Successfully processed {len(valid_results)}/{len(pdf_files)} files")
    
    # Save as JSON
    with open(OUTPUT_DIR/'annotations.json', 'w') as f:
        json.dump(valid_results, f, indent=2)
    
    # Create spaCy binary format
    try:
        nlp = spacy.blank("en")
        doc_bin = DocBin()
        
        for res in valid_results:
            doc = nlp.make_doc(res.get('text', ''))
            ents = []
            for start, end, label in res.get('entities', []):
                if start < len(res.get('text', '')) and end <= len(res.get('text', '')):
                    span = doc.char_span(start, end, label=label)
                    if span is not None:
                        ents.append(span)
            doc.ents = ents
            doc_bin.add(doc)
        
        doc_bin.to_disk(OUTPUT_DIR/'annotations.spacy')
        print("Created spaCy binary format")
        
        # Export BIO format for NER training
        print("Exporting BIO formatted data...")
        with open(OUTPUT_DIR/'bio_annotations.txt', 'w') as f:
            for res in valid_results:
                if not res.get('tokens') or not res.get('bio_tags'):
                    continue
                    
                for token, tag in zip(res.get('tokens', []), res.get('bio_tags', [])):
                    f.write(f"{token} {tag}\n")
                f.write("\n")
        
        print("Created BIO annotations file")
    except Exception as e:
        print(f"Error creating spaCy binary: {str(e)}")
    
    # Create visualizations and metrics
    entity_counts = Counter()
    bio_tag_counts = Counter()
    token_counts = []
    entity_length_counts = []
    
    for res in valid_results:
        entity_counts.update(res.get('entity_types', {}))
        bio_tag_counts.update(res.get('bio_tags', []))
        if res.get('tokens'):
            token_counts.append(len(res.get('tokens')))
        for start, end, _ in res.get('entities', []):
            entity_length_counts.append(end - start)
    
    # Entity type distribution
    plt.figure(figsize=(10, 6))
    plt.bar(entity_counts.keys(), entity_counts.values())
    plt.title('Entity Type Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/'entity_distribution.png')
    
    # BIO tag distribution (excluding 'O' tags)
    bio_tag_counts_no_o = {k: v for k, v in bio_tag_counts.items() if k != 'O'}
    
    if bio_tag_counts_no_o:
        plt.figure(figsize=(12, 6))
        plt.bar(bio_tag_counts_no_o.keys(), bio_tag_counts_no_o.values())
        plt.title('BIO Tag Distribution (excluding O tags)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR/'bio_tag_distribution.png')
    
    # Entity length histogram
    if entity_length_counts:
        plt.figure(figsize=(10, 6))
        plt.hist(entity_length_counts, bins=30)
        plt.title('Entity Length Distribution (characters)')
        plt.xlabel('Number of Characters')
        plt.ylabel('Frequency')
        plt.savefig(OUTPUT_DIR/'entity_length_distribution.png')
    
    print("Created visualizations")
    print("Processing complete!")

if __name__ == "__main__":
    main() 