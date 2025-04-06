import requests
import pandas as pd
import os
import time

def download_hgnc_data():
    """
    Download gene nomenclature data from HGNC REST API and process it for NER.
    Returns a list of gene symbols and names suitable for distant supervision.
    """
    print("Downloading HGNC gene data...")
    
    # Make API request to get all approved genes
    try:
        response = requests.get(
            'https://rest.genenames.org/fetch/status/Approved',
            headers={'Accept': 'application/json'},
            timeout=30
        )
        response.raise_for_status()  # Raise exception for non-200 status codes
    except requests.exceptions.RequestException as e:
        print(f"Error downloading HGNC data: {e}")
        return []
    
    # Parse the JSON response
    data = response.json()
    
    # Convert to DataFrame for easier processing
    if 'response' in data and 'docs' in data['response']:
        genes_data = pd.DataFrame(data['response']['docs'])
        print(f"Downloaded data for {len(genes_data)} genes")
        
        # Save raw data
        genes_data.to_csv('hgnc_genes_raw.csv', index=False, encoding='utf-8')
        
        # Process the data for NER
        gene_names = []
        
        # Add approved symbols
        if 'symbol' in genes_data.columns:
            gene_names.extend(genes_data['symbol'].dropna().tolist())
        
        # Add approved names
        if 'name' in genes_data.columns:
            gene_names.extend(genes_data['name'].dropna().tolist())
        
        # Add previous symbols if available
        if 'prev_symbol' in genes_data.columns:
            prev_symbols = genes_data['prev_symbol'].dropna()
            for symbols in prev_symbols:
                if isinstance(symbols, list):
                    gene_names.extend(symbols)
                elif isinstance(symbols, str):
                    gene_names.extend([s.strip() for s in symbols.split(',')])
        
        # Add alias symbols if available
        if 'alias_symbol' in genes_data.columns:
            alias_symbols = genes_data['alias_symbol'].dropna()
            for aliases in alias_symbols:
                if isinstance(aliases, list):
                    gene_names.extend(aliases)
                elif isinstance(aliases, str):
                    gene_names.extend([a.strip() for a in aliases.split(',')])
        
        # Remove duplicates and empty strings
        gene_names = list(set(filter(None, gene_names)))
        
        # Save processed gene names - USE UTF-8 ENCODING HERE
        with open('gene_names_for_ner.txt', 'w', encoding='utf-8') as f:
            for gene in gene_names:
                f.write(f"{gene}\n")
        
        print(f"Processed {len(gene_names)} unique gene names and synonyms")
        return gene_names
    else:
        print("Unexpected API response format")
        return []

def download_alternative_hgnc():
    """
    Alternative method to download HGNC data using the text download.
    """
    print("Using alternative download method...")
    
    try:
        # Download the complete HGNC dataset
        url = "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt"
        df = pd.read_csv(url, sep='\t')
        
        # Save raw data
        df.to_csv('hgnc_complete_set.csv', index=False, encoding='utf-8')
        
        # Process the data for NER
        gene_names = []
        
        # Add approved symbols
        gene_names.extend(df['symbol'].dropna().tolist())
        
        # Add approved names
        gene_names.extend(df['name'].dropna().tolist())
        
        # Add previous symbols
        prev_symbols = df['prev_symbol'].dropna()
        for symbols in prev_symbols:
            if isinstance(symbols, str):
                gene_names.extend([s.strip() for s in symbols.split('|')])
        
        # Add alias symbols
        alias_symbols = df['alias_symbol'].dropna()
        for aliases in alias_symbols:
            if isinstance(aliases, str):
                gene_names.extend([a.strip() for a in aliases.split('|')])
        
        # Remove duplicates and empty strings
        gene_names = list(set(filter(None, gene_names)))
        
        # Save processed gene names - USE UTF-8 ENCODING HERE
        with open('gene_names_for_ner.txt', 'w', encoding='utf-8') as f:
            for gene in gene_names:
                f.write(f"{gene}\n")
        
        print(f"Processed {len(gene_names)} unique gene names and synonyms")
        return gene_names
    
    except Exception as e:
        print(f"Error in alternative download: {e}")
        return []

def main():
    # Try the REST API approach first
    gene_names = download_hgnc_data()
    
    # If the API approach fails or returns empty, try the alternative
    if not gene_names:
        gene_names = download_alternative_hgnc()
    
    # Print some examples
    if gene_names:
        print("\nExample gene names and symbols:")
        for gene in gene_names[:10]:
            print(f"- {gene}")
        
        print(f"\nTotal unique gene identifiers: {len(gene_names)}")
        print(f"Data saved to 'gene_names_for_ner.txt'")

if __name__ == "__main__":
    main()