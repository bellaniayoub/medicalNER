import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import AdamW
from tqdm import tqdm
import os
from typing import List, Dict
import numpy as np

class NERDataset(Dataset):
    def __init__(self, data_file: str, tokenizer: BertTokenizerFast, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self.load_data(data_file)
        
        # Get unique labels from the data
        self.label_set = set()
        for example in self.examples:
            self.label_set.update(example['labels'])
        self.label_set = sorted(list(self.label_set))
        self.label2id = {label: i for i, label in enumerate(self.label_set)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
    def load_data(self, data_file: str) -> List[Dict]:
        examples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize the words into subwords
        tokenized = self.tokenizer(
            example['tokens'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors='pt'
        )
        
        # Convert labels to ids and align with subwords
        labels = example['labels']
        word_ids = tokenized.word_ids()
        
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # special tokens
            else:
                label_ids.append(self.label2id[labels[word_id]])
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids)
        }

def train_model(model, train_dataloader, val_dataloader, device, 
                num_epochs=3, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            train_steps += 1
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': train_loss / train_steps})
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                val_steps += 1
        
        print(f"Average training loss: {train_loss / train_steps}")
        print(f"Average validation loss: {val_loss / val_steps}")

def main():
    # Parameters
    max_length = 128
    batch_size = 32
    num_epochs = 3
    learning_rate = 2e-5
    
    # Paths
    data_dir = "data_collection/bert_data"
    output_dir = "models/bert_ner"
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer and load datasets
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    
    print("Loading datasets...")
    train_dataset = NERDataset(
        os.path.join(data_dir, "train.json"),
        tokenizer,
        max_length
    )
    val_dataset = NERDataset(
        os.path.join(data_dir, "val.json"),
        tokenizer,
        max_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    
    # Initialize model
    model = BertForTokenClassification.from_pretrained(
        'bert-base-cased',
        num_labels=len(train_dataset.label2id),
        id2label=train_dataset.id2label,
        label2id=train_dataset.label2id
    )
    model.to(device)
    
    # Train the model
    print("Starting training...")
    train_model(
        model,
        train_dataloader,
        val_dataloader,
        device,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    
    # Save the model and tokenizer
    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings
    with open(os.path.join(output_dir, "label_mappings.json"), "w") as f:
        json.dump({
            "label2id": train_dataset.label2id,
            "id2label": train_dataset.id2label
        }, f, indent=2)
    
    print("✅ Training complete! Model saved to:", output_dir)

if __name__ == "__main__":
    main() 