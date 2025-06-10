import obonet
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def visualize_obo_graph(obo_file, max_nodes=50):
    # Read the OBO file
    graph = obonet.read_obo(obo_file)
    
    # Create a subgraph with only disease nodes
    disease_nodes = []
    node_labels = {}  # Dictionary to store node labels (names)
    
    for node_id, data in graph.nodes(data=True):
        if 'name' in data and ('disease' in data.get('name', '').lower() or 
                             'disorder' in data.get('name', '').lower()):
            disease_nodes.append(node_id)
            # Store the disease name as the label
            node_labels[node_id] = data.get('name', node_id)
            if len(disease_nodes) >= max_nodes:
                break
    
    # Create a subgraph with the selected nodes
    subgraph = graph.subgraph(disease_nodes)
    
    # Set up the plot with larger figure size
    plt.figure(figsize=(20, 15))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(subgraph, k=2, iterations=100)
    
    # Draw the graph with disease names as labels
    nx.draw(subgraph, pos, 
            labels=node_labels,
            with_labels=True, 
            node_color='lightblue', 
            node_size=3000,
            font_size=10,
            font_weight='bold')
    
    # Add edge labels (relationship types)
    edge_labels = nx.get_edge_attributes(subgraph, 'type')
    nx.draw_networkx_edge_labels(subgraph, pos, 
                                edge_labels=edge_labels, 
                                font_size=8)
    
    # Add title
    plt.title(f'Disease Ontology Graph (showing {len(disease_nodes)} diseases)', 
              fontsize=16, pad=20)
    
    # Save the plot with higher resolution
    plt.savefig('disease_ontology_graph.png', 
                dpi=300, 
                bbox_inches='tight')
    print("Graph visualization saved as 'disease_ontology_graph.png'")

if __name__ == "__main__":
    visualize_obo_graph('HumanDOr21.obo') 