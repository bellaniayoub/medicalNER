import requests
import os

# Create a directory for the data if it doesn't exist
os.makedirs('data_collection/Knowledge_base', exist_ok=True)

# Fetch the Gene Ontology data
response = requests.get('https://current.geneontology.org/ontology/go.obo')

# Check if the request was successful
if response.status_code == 200:
    # Save the data to a file in OBO format
    with open('data_collection/Knowledge_base/gene_ontology.obo', 'wb') as f:
        f.write(response.content)
    print("Gene Ontology data successfully saved to gene_ontology.obo")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")