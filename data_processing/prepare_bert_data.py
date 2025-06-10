import json
import os
import random
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

def load_bio_files(data_dir: str) -> List[List[List[str]]]:
    """Load all BIO annotation files from the directory."""
    all_sentences = []
    for filename in os.listdir(data_dir):
        if filename.endswith('_BIO.json'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                sentences = json.load(f)
                all_sentences.extend(sentences)
    return all_sentences

def convert_to_bert_format(sentences: List[List[List[str]]]) -> List[Dict[str, List[str]]]:
    """Convert BIO annotations to BERT format."""
    bert_format = []
    for sentence in sentences:
        tokens = []
        labels = []
        for token, label in sentence:
            tokens.append(token)
            labels.append(label)
        
        bert_format.append({
            "tokens": tokens,
            "labels": labels
        })
    return bert_format

def split_dataset(data: List[Dict[str, List[str]]], 
                 train_ratio=0.8,
                 val_ratio=0.1,
                 test_ratio=0.1) -> Tuple[List, List, List]:
    """Split dataset into train, validation and test sets."""
    train_val, test = train_test_split(data, test_size=test_ratio, random_state=42)
    train, val = train_test_split(train_val, 
                                 test_size=val_ratio/(train_ratio + val_ratio), 
                                 random_state=42)
    return train, val, test

def save_to_file(data: List[Dict[str, List[str]]], filename: str):
    """Save the data in the format expected by HuggingFace."""
    # Save as a proper JSON file with all examples in a single array
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def main():
    # Input directory containing BIO annotation files
    input_dir = "../data_collection/annotated_data/articles_bio_annotations"
    
    # Output directory for BERT-formatted files
    output_dir = "bert_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process all files
    print("Loading BIO files...")
    sentences = load_bio_files(input_dir)
    
    print("Converting to BERT format...")
    bert_data = convert_to_bert_format(sentences)
    
    print("Splitting dataset...")
    train_data, val_data, test_data = split_dataset(bert_data)
    
    print("Saving files...")
    save_to_file(train_data, os.path.join(output_dir, "train.json"))
    save_to_file(val_data, os.path.join(output_dir, "val.json"))
    save_to_file(test_data, os.path.join(output_dir, "test.json"))
    
    print(f"✅ Done! Created {len(train_data)} training, {len(val_data)} validation, "
          f"and {len(test_data)} test examples.")

if __name__ == "__main__":
    main() 