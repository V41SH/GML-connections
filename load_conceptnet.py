import os
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from time import time
from typing import Set, List, Tuple
from wordninja import split as wordninja_split


PICKLE_NAME_DF = "graphs/conceptnet_filtered_edges_with_compounds.pkl"


def load_conceptnet_graph(csv_path: str) -> tuple[Data, nx.DiGraph, dict, dict, dict]:
    """
    Load a pre-filtered ConceptNet CSV (start_word, end_word, relation, weight)
    and build a PyTorch Geometric graph with edge weights and relation types.
    """
    if os.path.exists(PICKLE_NAME_DF):
        print(f"Loading preprocessed DataFrame from: {PICKLE_NAME_DF}")
        df: pd.DataFrame = pd.read_pickle(PICKLE_NAME_DF)
    else:
        print(f"Loading ConceptNet edges from: {csv_path}")
        df: pd.DataFrame = pd.read_csv(csv_path)

    # Normalize & clean
    df["start_word"] = df["start_word"].astype(str).str.lower()
    df["end_word"] = df["end_word"].astype(str).str.lower()
    df["relation"] = df["relation"].astype(str).str.lower()

    df.to_pickle(PICKLE_NAME_DF)
    print(f"DataFrame saved to: {PICKLE_NAME_DF}")

    # Build node vocabulary
    all_words = pd.Index(df["start_word"]).append(pd.Index(df["end_word"])).unique()
    word2idx = pd.Series(range(len(all_words)), index=all_words)
    idx2word = dict(enumerate(all_words))

    # Edge indices
    src = df["start_word"].map(word2idx).to_numpy()
    dst = df["end_word"].map(word2idx).to_numpy()
    edge_index = torch.from_numpy(np.vstack([src, dst]).astype("int64"))

    # Edge weights
    edge_weight = torch.tensor(df["weight"].to_numpy(), dtype=torch.float)

    # Encode relation types as integers
    relation_types = pd.Index(df["relation"].unique())
    rel2idx = pd.Series(range(len(relation_types)), index=relation_types)
    edge_type = torch.tensor(df["relation"].map(rel2idx).to_numpy(), dtype=torch.long)

    # Node features (none for now)
    x = None

    # Build PyTorch Geometric Data object
    data = Data(
        x=x, edge_index=edge_index, edge_weight=edge_weight, edge_type=edge_type
    )

    # ---- NetworkX directed graph ----
    graphs_dir = os.path.join(os.path.dirname(__file__), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    graph_save_path = os.path.join(graphs_dir, "conceptnet_graph.gml")

    if os.path.exists(graph_save_path):
        print(f"Loading existing NetworkX graph from: {graph_save_path}")
        G = nx.read_gml(graph_save_path)
    else:
        G = nx.from_pandas_edgelist(
            df,
            source="start_word",
            target="end_word",
            edge_attr=["weight", "relation"],
            create_using=nx.DiGraph,
        )
        # Save the NetworkX graph to 'graphs' directory if freshly created
        nx.write_gml(G, graph_save_path)
        print(f"Graph saved to: {graph_save_path}")

    print(
        f"Loaded graph with {len(all_words):,} nodes, {len(df):,} edges, {len(relation_types)} relation types"
    )
    return data, G, word2idx.to_dict(), idx2word, rel2idx.to_dict()


if __name__ == "__main__":
    t1 = time()
    # Example usage:
    graph_path = "conceptnet/conceptnet_filtered_edges.csv"
    data, G, word2idx, idx2word, rel2idx = load_conceptnet_graph(graph_path)
    t2 = time()
    print(f"Graph loading took {t2 - t1:.2f} seconds")
