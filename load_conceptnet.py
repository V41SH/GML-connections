import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Data
import networkx as nx
import numpy as np


def load_conceptnet_graph(csv_path: str):
    """
    Load a pre-filtered ConceptNet CSV (start_word, end_word, relation, weight)
    and build a PyTorch Geometric graph with edge weights and relation types.
    """
    print(f"Loading ConceptNet edges from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalize & clean
    df['start_word'] = df['start_word'].astype(str).str.lower()
    df['end_word'] = df['end_word'].astype(str).str.lower()
    df['relation'] = df['relation'].astype(str).str.lower()

    # Build node vocabulary
    all_words = pd.Index(df['start_word']).append(pd.Index(df['end_word'])).unique()
    word2idx = pd.Series(range(len(all_words)), index=all_words)
    idx2word = dict(enumerate(all_words))

    # Edge indices
    src = df['start_word'].map(word2idx).to_numpy()
    dst = df['end_word'].map(word2idx).to_numpy()
    edge_index = torch.from_numpy(
        np.vstack([src, dst]).astype('int64')
    )

    # Edge weights
    edge_weight = torch.tensor(df['weight'].to_numpy(), dtype=torch.float)

    # Encode relation types as integers
    relation_types = pd.Index(df['relation'].unique())
    rel2idx = pd.Series(range(len(relation_types)), index=relation_types)
    edge_type = torch.tensor(df['relation'].map(rel2idx).to_numpy(), dtype=torch.long)

    # Node features (none for now)
    x = None

    # Build PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_type=edge_type
    )

    # ---- NetworkX directed graph ----
    G = nx.from_pandas_edgelist(
        df,
        source='start_word',
        target='end_word',
        edge_attr=['weight', 'relation'],
        create_using=nx.DiGraph
    )

    print(f"Loaded graph with {len(all_words):,} nodes, {len(df):,} edges, {len(relation_types)} relation types")
    return data, G, word2idx.to_dict(), idx2word, rel2idx.to_dict()


if __name__ == "__main__":
    # Example usage:
    graph_path = "conceptnet/conceptnet_filtered_edges.csv"
    data, G, word2idx, idx2word, rel2idx = load_conceptnet_graph(graph_path)