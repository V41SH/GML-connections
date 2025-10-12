import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data

def load_swow_en18(csv_path, strength_col='R123.Strength', min_strength=0.0):
    """
    Load SWOW-EN18 data as a PyTorch Geometric graph.
    
    Args:
        csv_path (str): path to SWOW-EN csv (e.g. SWOW-EN.R123.csv)
        strength_col (str): which column to use for edge weights
        min_strength (float): filter out weak edges below this threshold
        
    Returns:
        data (torch_geometric.data.Data): graph data
        G (networkx.DiGraph): networkx graph for reference
        word2idx (dict): mapping from words to node indices
        idx2word (dict): mapping from indices to words
    """
    # Load CSV
    df = pd.read_csv(csv_path, sep="\t")
    
    # Filter
    df = df[df["R123.Strength"] >= min_strength]
    
    # Build directed weighted graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['cue'], row['response'], weight=row[strength_col])
    
    # Map words to node indices
    word2idx = {word: i for i, word in enumerate(G.nodes())}
    idx2word = {i: word for word, i in word2idx.items()}
    
    # Build PyTorch Geometric edge index
    edges = [(word2idx[u], word2idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Edge weights
    edge_weight = torch.tensor(
        [G[u][v]['weight'] for u, v in G.edges()],
        dtype=torch.float
    )
    
    # Optional: simple node features (e.g., identity matrix or degree)
    x = torch.eye(len(G))  # placeholder: one-hot per node
    
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    
    return data, G, word2idx, idx2word


# Example usage
if __name__ == "__main__":
    csv_file = "SWOW-EN18/strength.SWOW-EN.R123.20180827.csv"
    data, G, word2idx, idx2word = load_swow_en18(csv_file, min_strength=0.05)
    print(data)
    print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
    print(f"Example words: {list(idx2word.values())[:10]}")