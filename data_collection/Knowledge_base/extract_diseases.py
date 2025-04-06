import obonet
import networkx
import pandas as pd

def extract_diseases(obo_file):
    # Read the OBO file
    graph = obonet.read_obo(obo_file)
    
    # Initialize lists to store disease data
    disease_data = []
    
    # Iterate through all nodes in the graph
    for node_id, data in graph.nodes(data=True):
        # Check if it's a disease term (you might need to adjust this condition)
        if 'name' in data and ('disease' in data.get('name', '').lower() or 
                             'disorder' in data.get('name', '').lower()):
            disease_info = {
                'ID': node_id,
                'Name': data.get('name', ''),
                'Definition': data.get('def', ''),
                'Synonyms': '; '.join(data.get('synonym', [])),
                'Xrefs': '; '.join(data.get('xref', [])),
                'Is_Obsolete': data.get('is_obsolete', False)
            }
            disease_data.append(disease_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(disease_data)
    
    # Save to CSV
    df.to_csv('extracted_diseases.csv', index=False)
    print(f"Extracted {len(disease_data)} diseases to 'extracted_diseases.csv'")

if __name__ == "__main__":
    extract_diseases('HumanDOr21.obo') 