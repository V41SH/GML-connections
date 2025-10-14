import pandas as pd
import torch
from torch_geometric.data import Data

def load_swow_en18(csv_path, strength_col='R123.Strength', min_strength=0.0):
    """
    Load SWOW-EN18 data as a PyTorch Geometric graph quickly.
    """
    # Load only needed columns, using efficient dtypes
    df = pd.read_csv(csv_path, sep="\t", usecols=['cue', 'response', strength_col])
    df = df[df[strength_col] >= min_strength]

    # Map words to integer IDs (vectorized)
    all_words = pd.Index(df['cue']).append(pd.Index(df['response'])).unique()
    word2idx = pd.Series(range(len(all_words)), index=all_words)
    idx2word = dict(enumerate(all_words))

    # Build edge indices using vectorized mapping
    src = df['cue'].map(word2idx).to_numpy()
    dst = df['response'].map(word2idx).to_numpy()
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Edge weights
    edge_weight = torch.tensor(df[strength_col].to_numpy(), dtype=torch.float)

    # Optional: simple node features
    num_nodes = len(all_words)
    x = torch.eye(num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    # If you still want a NetworkX graph:
    # (But note: building this will be slower no matter what)
    # import networkx as nx
    # G = nx.from_pandas_edgelist(df, 'cue', 'response', edge_attr=strength_col, create_using=nx.DiGraph)

    return data, None, word2idx.to_dict(), idx2word


# Example usage
if __name__ == "__main__":
    csv_file = "SWOW-EN18/strength.SWOW-EN.R123.20180827.csv"
    data, G, word2idx, idx2word = load_swow_en18(csv_file, min_strength=0.05)
    print(data)
    print(f"Nodes: {len(word2idx)}, Edges: {data.edge_index.shape[1]}")
    print(f"Example words: {list(idx2word.values())[:10]}")
