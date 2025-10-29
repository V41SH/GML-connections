import os
import torch
import networkx as nx
import fasttext
import numpy as np

def load_graph_from_gml(gml_path):
    """
    Load graph from GML file and prepare PyTorch Geometric Data object.
    """
    print(f"Loading graph from: {gml_path}")
    G = nx.read_gml(gml_path)
    
    # Get nodes in consistent order
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Build edge index and edge types
    edges = []
    edge_types = []
    relations = set()
    
    for u, v, data in G.edges(data=True):
        edges.append([node_to_idx[u], node_to_idx[v]])
        rel = data.get('relation', 'unknown')
        relations.add(rel)
        edge_types.append(rel)
    
    # Create relation to index mapping
    relation_to_idx = {rel: idx for idx, rel in enumerate(sorted(relations))}
    
    # Convert to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor([relation_to_idx[rel] for rel in edge_types], dtype=torch.long)
    
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges, {len(relations)} relation types")
    
    return nodes, edge_index, edge_type, node_to_idx, relation_to_idx

def load_fasttext_embeddings(nodes, embedding_dim=300):
    """
    Load FastText embeddings for the given nodes (words).
    Downloads the model if not already present.
    """
    print("Loading FastText embeddings...")
    
    # Download FastText model (this will cache it locally)
    ft = fasttext.load_model('embedding_models/cc.en.300.bin')
    
    # Get embeddings for each node
    embeddings = []
    for node in nodes:
        # FastText can handle out-of-vocabulary words
        vec = ft.get_word_vector(str(node))
        embeddings.append(vec)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # Save embeddings to file
    out_path = os.path.join(os.path.dirname(__file__), '../graphs/fasttext_node_embeddings.npy')
    np.save(out_path, embeddings)
    print(f"Embeddings saved to: {out_path}")

    return torch.from_numpy(embeddings)