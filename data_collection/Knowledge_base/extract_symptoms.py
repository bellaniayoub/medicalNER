import obonet
import networkx as nx
import pandas as pd

def extract_symptoms(obo_file):
    # Read the OBO file
    graph = obonet.read_obo(obo_file)
    
    # Initialize lists to store symptom data
    symptom_data = []
    
    # Iterate through all nodes in the graph
    for node_id, data in graph.nodes(data=True):
        # Check if it's a symptom term
        if 'name' in data and ('symptom' in data.get('name', '').lower() or 
                             'sign' in data.get('name', '').lower() or
                             'clinical feature' in data.get('name', '').lower()):
            symptom_info = {
                'ID': node_id,
                'Name': data.get('name', ''),
                'Definition': data.get('def', ''),
                'Synonyms': '; '.join(data.get('synonym', [])),
                'Xrefs': '; '.join(data.get('xref', [])),
                'Is_Obsolete': data.get('is_obsolete', False),
                'Related_Diseases': []
            }
            
            # Find related diseases
            for source, target, edge_data in graph.edges(data=True):
                if source == node_id and 'type' in edge_data:
                    if edge_data['type'] == 'has_symptom' or edge_data['type'] == 'has_feature':
                        disease_name = graph.nodes[target].get('name', target)
                        symptom_info['Related_Diseases'].append(disease_name)
            
            symptom_info['Related_Diseases'] = '; '.join(symptom_info['Related_Diseases'])
            symptom_data.append(symptom_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(symptom_data)
    
    # Save to CSV
    df.to_csv('extracted_symptoms.csv', index=False)
    print(f"Extracted {len(symptom_data)} symptoms to 'extracted_symptoms.csv'")
    
    # Print some statistics
    print("\nSymptom Analysis Summary:")
    print(f"Total symptoms found: {len(symptom_data)}")
    print(f"Average number of synonyms per symptom: {df['Synonyms'].str.count(';').mean():.2f}")
    print(f"Number of symptoms with related diseases: {df[df['Related_Diseases'] != ''].shape[0]}")

if __name__ == "__main__":
    extract_symptoms('HumanDOr21.obo') 