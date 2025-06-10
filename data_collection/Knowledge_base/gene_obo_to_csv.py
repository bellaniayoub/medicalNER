import csv
import re
import os

def parse_obo_file(obo_file_path):
    """
    Parse the OBO file and extract relevant information.
    Returns a list of dictionaries containing term information.
    """
    terms = []
    current_term = {}
    
    with open(obo_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines and header information
            if not line or line.startswith('format-version:') or line.startswith('data-version:'):
                continue
                
            # Start of a new term
            if line == '[Term]':
                if current_term:
                    terms.append(current_term)
                current_term = {}
                continue
                
            # End of a term
            if line == '' and current_term:
                terms.append(current_term)
                current_term = {}
                continue
                
            # Parse term attributes
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle different types of attributes
                if key == 'id':
                    current_term['id'] = value
                elif key == 'name':
                    current_term['name'] = value
                elif key == 'namespace':
                    current_term['namespace'] = value
                elif key == 'def':
                    # Remove citation brackets from definition
                    def_text = re.sub(r'\[.*?\]', '', value)
                    current_term['definition'] = def_text.strip()
                elif key == 'is_a':
                    if 'is_a' not in current_term:
                        current_term['is_a'] = []
                    current_term['is_a'].append(value.split('!')[0].strip())
                elif key == 'synonym':
                    if 'synonyms' not in current_term:
                        current_term['synonyms'] = []
                    # Extract just the synonym text without the type
                    synonym_text = re.match(r'"(.*?)"', value)
                    if synonym_text:
                        current_term['synonyms'].append(synonym_text.group(1))
    
    # Add the last term if exists
    if current_term:
        terms.append(current_term)
    
    return terms

def write_to_csv(terms, csv_file_path):
    """
    Write the parsed terms to a CSV file.
    """
    # Define the fieldnames for the CSV
    fieldnames = ['id', 'name', 'namespace', 'definition', 'is_a', 'synonyms']
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for term in terms:
            # Convert lists to strings for CSV writing
            row = {}
            for field in fieldnames:
                if field in term:
                    if isinstance(term[field], list):
                        row[field] = '; '.join(term[field])
                    else:
                        row[field] = term[field]
                else:
                    row[field] = ''
            writer.writerow(row)

def main():
    # Define file paths
    obo_file_path = 'data_collection/Knowledge_base/data/gene_ontology.obo'
    csv_file_path = 'data_collection/Knowledge_base/data/gene_ontology.csv'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    # Parse the OBO file
    print(f"Parsing OBO file: {obo_file_path}")
    terms = parse_obo_file(obo_file_path)
    
    # Write to CSV
    print(f"Writing to CSV file: {csv_file_path}")
    write_to_csv(terms, csv_file_path)
    
    print(f"Conversion complete. {len(terms)} terms processed.")

if __name__ == "__main__":
    main() 