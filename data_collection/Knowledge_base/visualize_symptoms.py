import obonet
import networkx as nx
import matplotlib.pyplot as plt

def visualize_symptoms(obo_file, max_nodes=30):
    # Read the OBO file
    graph = obonet.read_obo(obo_file)
    
    # Create a subgraph with symptom nodes and their related diseases
    symptom_nodes = []
    node_labels = {}
    edge_labels = {}
    
    for node_id, data in graph.nodes(data=True):
        if 'name' in data and ('symptom' in data.get('name', '').lower() or 
                             'sign' in data.get('name', '').lower() or
                             'clinical feature' in data.get('name', '').lower()):
            symptom_nodes.append(node_id)
            node_labels[node_id] = data.get('name', node_id)
            
            # Find related diseases
            for source, target, edge_data in graph.edges(data=True):
                if source == node_id and 'type' in edge_data:
                    if edge_data['type'] == 'has_symptom' or edge_data['type'] == 'has_feature':
                        disease_name = graph.nodes[target].get('name', target)
                        edge_labels[(node_id, target)] = edge_data['type']
                        symptom_nodes.append(target)
                        node_labels[target] = disease_name
            
            if len(symptom_nodes) >= max_nodes:
                break
    
    # Create a subgraph
    subgraph = graph.subgraph(symptom_nodes)
    
    # Set up the plot
    plt.figure(figsize=(20, 15))
    
    # Use spring layout
    pos = nx.spring_layout(subgraph, k=2, iterations=100)
    
    # Draw the graph
    nx.draw(subgraph, pos, 
            labels=node_labels,
            with_labels=True,
            node_color=['lightblue' if 'symptom' in node_labels[n].lower() or 
                       'sign' in node_labels[n].lower() or 
                       'clinical feature' in node_labels[n].lower() 
                       else 'lightgreen' for n in subgraph.nodes()],
            node_size=3000,
            font_size=10,
            font_weight='bold')
    
    # Add edge labels
    nx.draw_networkx_edge_labels(subgraph, pos, 
                                edge_labels=edge_labels,
                                font_size=8)
    
    # Add title
    plt.title('Symptom-Disease Relationship Graph', fontsize=16, pad=20)
    
    # Add legend
    plt.figtext(0.5, 0.01, 
                "Blue nodes: Symptoms/Signs\nGreen nodes: Related Diseases", 
                ha='center', fontsize=12)
    
    # Save the plot
    plt.savefig('symptom_relationships.png', dpi=300, bbox_inches='tight')
    print("Graph visualization saved as 'symptom_relationships.png'")

if __name__ == "__main__":
    visualize_symptoms('HumanDOr21.obo') 