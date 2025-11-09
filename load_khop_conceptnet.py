import os
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from time import time
from typing import Set, List, Dict, Tuple
from collections import deque
from load_connections import load_connections_game


def get_k_hop_neighbors(
    df: pd.DataFrame,
    seed_words: List[str],
    k: int = 10,
    bidirectional: bool = True
) -> Set[str]:
    """
    Extract all nodes within k hops of seed_words using BFS.
    
    Args:
        df: DataFrame with columns ['start_word', 'end_word', 'relation', 'weight']
        seed_words: List of seed words to start the BFS from
        k: Number of hops to traverse
        bidirectional: If True, treat edges as undirected for BFS
    
    Returns:
        Set of all words within k hops of seed_words
    """
    # Normalize seed words
    seed_words = [w.lower().strip() for w in seed_words]
    
    # Build adjacency list for fast lookups
    adj_list = {}
    for _, row in df.iterrows():
        start, end = row['start_word'], row['end_word']
        
        if start not in adj_list:
            adj_list[start] = set()
        adj_list[start].add(end)
        
        if bidirectional:
            if end not in adj_list:
                adj_list[end] = set()
            adj_list[end].add(start)
    
    # BFS to find k-hop neighbors
    visited = set()
    queue = deque()
    
    # Initialize with seed words at distance 0
    for word in seed_words:
        if word in adj_list:
            queue.append((word, 0))
            visited.add(word)
    
    k_hop_nodes = set(seed_words)
    
    while queue:
        current_word, dist = queue.popleft()
        
        if dist >= k:
            continue
        
        # Explore neighbors
        if current_word in adj_list:
            for neighbor in adj_list[current_word]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    k_hop_nodes.add(neighbor)
                    queue.append((neighbor, dist + 1))
    
    print(f"Found {len(k_hop_nodes):,} nodes within {k} hops of {len(seed_words)} seed words")
    return k_hop_nodes


def load_conceptnet_subgraph(
    csv_path: str,
    connections_csv_path: str,
    k_hops: int = 10,
    cache_dir: str = "graphs"
) -> Tuple[Data, nx.DiGraph, Dict, Dict, Dict]:
    """
    Load a subgraph of ConceptNet containing only k-hop neighbors of ALL words
    from the entire Connections database.
    
    Args:
        csv_path: Path to ConceptNet CSV file
        connections_csv_path: Path to Connections game CSV
        k_hops: Number of hops for neighborhood extraction
        cache_dir: Directory to save cached files
    
    Returns:
        data: PyTorch Geometric Data object
        G: NetworkX DiGraph
        word2idx: Dictionary mapping words to indices
        idx2word: Dictionary mapping indices to words
        rel2idx: Dictionary mapping relation types to indices
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename based on k_hops
    cache_pkl = os.path.join(cache_dir, f"conceptnet_subgraph_all_connections_k{k_hops}.pkl")
    cache_gml = os.path.join(cache_dir, f"conceptnet_subgraph_all_connections_k{k_hops}.gml")
    
    # Load ALL words from Connections database
    print(f"Loading all words from Connections database...")
    connections_df = pd.read_csv(connections_csv_path)
    connections_df = connections_df.dropna(subset=["Word"])
    seed_words = connections_df["Word"].astype(str).str.strip().str.lower().unique().tolist()
    print(f"Found {len(seed_words)} unique words across all Connections games")
    print(f"Sample words: {seed_words[:10]}")
    
    # Check if cached subgraph exists
    if os.path.exists(cache_pkl):
        print(f"Loading cached subgraph from: {cache_pkl}")
        df_subgraph = pd.read_pickle(cache_pkl)
    else:
        # Load full ConceptNet CSV
        print(f"Loading full ConceptNet from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Normalize & clean
        df["start_word"] = df["start_word"].astype(str).str.lower().str.strip()
        df["end_word"] = df["end_word"].astype(str).str.lower().str.strip()
        df["relation"] = df["relation"].astype(str).str.lower().str.strip()
        
        # Extract k-hop subgraph
        print(f"Extracting {k_hops}-hop neighborhood...")
        t1 = time()
        k_hop_nodes = get_k_hop_neighbors(df, seed_words, k=k_hops, bidirectional=True)
        t2 = time()
        print(f"Neighborhood extraction took {t2 - t1:.2f} seconds")
        
        # Filter dataframe to only include edges within the subgraph
        df_subgraph = df[
            df['start_word'].isin(k_hop_nodes) & df['end_word'].isin(k_hop_nodes)
        ].copy()
        
        # Save cached subgraph
        df_subgraph.to_pickle(cache_pkl)
        print(f"Cached subgraph saved to: {cache_pkl}")
    
    # Build node vocabulary
    all_words = pd.Index(df_subgraph["start_word"]).append(
        pd.Index(df_subgraph["end_word"])
    ).unique()
    word2idx = pd.Series(range(len(all_words)), index=all_words)
    idx2word = dict(enumerate(all_words))
    
    # Edge indices
    src = df_subgraph["start_word"].map(word2idx).to_numpy()
    dst = df_subgraph["end_word"].map(word2idx).to_numpy()
    edge_index = torch.from_numpy(np.vstack([src, dst]).astype("int64"))
    
    # Edge weights
    edge_weight = torch.tensor(df_subgraph["weight"].to_numpy(), dtype=torch.float)
    
    # Encode relation types as integers
    relation_types = pd.Index(df_subgraph["relation"].unique())
    rel2idx = pd.Series(range(len(relation_types)), index=relation_types)
    edge_type = torch.tensor(
        df_subgraph["relation"].map(rel2idx).to_numpy(), dtype=torch.long
    )
    
    # Node features (none for now)
    x = None
    
    # Build PyTorch Geometric Data object
    data = Data(
        x=x, edge_index=edge_index, edge_weight=edge_weight, edge_type=edge_type
    )
    
    # ---- NetworkX directed graph ----
    if os.path.exists(cache_gml):
        print(f"Loading cached NetworkX graph from: {cache_gml}")
        G = nx.read_gml(cache_gml)
    else:
        print("Building NetworkX graph...")
        G = nx.from_pandas_edgelist(
            df_subgraph,
            source="start_word",
            target="end_word",
            edge_attr=["weight", "relation"],
            create_using=nx.DiGraph,
        )
        nx.write_gml(G, cache_gml)
        print(f"NetworkX graph saved to: {cache_gml}")
    
    print(
        f"Loaded subgraph with {len(all_words):,} nodes, {len(df_subgraph):,} edges, "
        f"{len(relation_types)} relation types"
    )
    
    return data, G, word2idx.to_dict(), idx2word, rel2idx.to_dict()


if __name__ == "__main__":
    t1 = time()
    
    # Example usage
    conceptnet_path = "conceptnet/conceptnet_filtered_edges.csv"
    connections_path = "connections_data/Connections_Data.csv"
    
    data, G, word2idx, idx2word, rel2idx = load_conceptnet_subgraph(
        csv_path=conceptnet_path,
        connections_csv_path=connections_path,
        k_hops=1
    )
    
    t2 = time()
    print(f"\nTotal time: {t2 - t1:.2f} seconds")
    
    # Print some statistics
    print(f"\nGraph statistics:")
    print(f"  Number of nodes: {data.edge_index.max().item() + 1:,}")
    print(f"  Number of edges: {data.edge_index.shape[1]:,}")
    print(f"  Number of relation types: {len(rel2idx)}")
    print(f"\nRelation types: {list(rel2idx.keys())}")