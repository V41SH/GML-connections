import pandas as pd
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from itertools import combinations
from utils import phonetic_similarity, phonetic_code


def load_swow_en18(csv_path, strength_col="R123.Strength", min_strength=0.0):
    """
    Load SWOW-EN18 data as a PyTorch Geometric graph quickly.
    """
    # Load only needed columns, using efficient dtypes
    df = pd.read_csv(csv_path, sep="\t", usecols=["cue", "response", strength_col])
    df = df[df[strength_col] >= min_strength]

    # Map words to integer IDs (vectorized)
    all_words = pd.Index(df["cue"]).append(pd.Index(df["response"])).unique()

    # get embeddings for all words. costly, but whatever.
    # all_embeddings = all_words.map(ft.get_word_vector)

    word2idx = pd.Series(range(len(all_words)), index=all_words)
    idx2word = dict(enumerate(all_words))

    # Build edge indices using vectorized mapping
    src = df["cue"].map(word2idx).to_numpy()
    dst = df["response"].map(word2idx).to_numpy()
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    # Edge weights
    edge_weight = torch.tensor(df[strength_col].to_numpy(), dtype=torch.float)

    # Optional: simple node features
    num_nodes = len(all_words)
    x = torch.eye(num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    G = nx.from_pandas_edgelist(
        df, "cue", "response", edge_attr=strength_col, create_using=nx.DiGraph
    )

    return data, G, word2idx.to_dict(), idx2word


def add_phonetic_edges(words_list, word2idx, phonetic_threshold=0.7):
    """
    Add phonetic similarity edges between words.

    Args:
        words_list: List of words to compute phonetic similarities for
        word2idx: Word to index mapping
        phonetic_threshold: Minimum phonetic similarity to create an edge

    Returns:
        phonetic_edges: List of (src_idx, dst_idx, similarity) tuples
    """
    print(
        f"Computing phonetic edges for {len(words_list)} words with threshold {phonetic_threshold}..."
    )
    phonetic_edges = []

    # For computational efficiency, limit comparisons for very large vocabularies
    max_comparisons = 100000  # Adjust based on your computational constraints

    total_pairs = len(words_list) * (len(words_list) - 1) // 2
    if total_pairs > max_comparisons:
        print(
            f"Warning: {total_pairs} total pairs. Using sampling to limit computation."
        )
        # Sample a subset of words for phonetic comparison
        import random

        sample_size = int(np.sqrt(max_comparisons * 2))
        sampled_words = random.sample(words_list, min(sample_size, len(words_list)))
        words_to_compare = sampled_words
    else:
        words_to_compare = words_list

    print(f"Comparing {len(words_to_compare)} words for phonetic similarity...")

    for i, word1 in enumerate(words_to_compare):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(words_to_compare)} words...")

        # Validate word1 is a string
        if not isinstance(word1, str) or not word1.strip():
            continue

        for j, word2 in enumerate(words_to_compare[i + 1 :], i + 1):
            # Validate word2 is a string
            if not isinstance(word2, str) or not word2.strip():
                continue

            try:
                similarity = phonetic_similarity(word1, word2)
                if similarity >= phonetic_threshold:
                    idx1 = word2idx[word1]
                    idx2 = word2idx[word2]
                    phonetic_edges.append((idx1, idx2, similarity))
                    # Add reverse edge for undirected phonetic similarity
                    phonetic_edges.append((idx2, idx1, similarity))
            except Exception as e:
                print(
                    f"Warning: Error computing phonetic similarity for '{word1}' and '{word2}': {e}"
                )
                continue

    print(f"Found {len(phonetic_edges)} phonetic edges")
    return phonetic_edges


def compute_node_features_with_phonetics(words_list, word2idx, phonetic_edges):
    """
    Compute enhanced node features including phonetic information.

    Args:
        words_list: List of all words
        word2idx: Word to index mapping
        phonetic_edges: List of phonetic edges from add_phonetic_edges

    Returns:
        features_dict: Dictionary with phonetic features for each node
    """
    print("Computing phonetic node features...")

    # Initialize counters
    phonetic_edge_counts = {idx: 0 for idx in word2idx.values()}
    phonetic_codes = {}

    # Count phonetic edges per node
    for src_idx, dst_idx, similarity in phonetic_edges:
        phonetic_edge_counts[src_idx] += 1

    # Compute phonetic codes for each word
    for word, idx in word2idx.items():
        if isinstance(word, str) and word.strip():
            phonetic_codes[idx] = phonetic_code(word)
        else:
            phonetic_codes[idx] = ""  # Empty code for invalid words

    # Create phonetic code embeddings (simple one-hot encoding of metaphone characters)
    all_phonetic_chars = set()
    for code in phonetic_codes.values():
        all_phonetic_chars.update(code)

    char_to_idx = {char: i for i, char in enumerate(sorted(all_phonetic_chars))}
    phonetic_dim = len(char_to_idx)

    features_dict = {}
    for word, idx in word2idx.items():
        # Phonetic one-hot encoding
        phonetic_vec = np.zeros(phonetic_dim)
        for char in phonetic_codes[idx]:
            if char in char_to_idx:
                phonetic_vec[char_to_idx[char]] = 1

        # Safe word length computation
        word_len = len(word) if isinstance(word, str) else 0

        features_dict[idx] = {
            "phonetic_embedding": phonetic_vec,
            "phonetic_edge_count": phonetic_edge_counts[idx],
            "word_length": word_len,
            "phonetic_code_length": len(phonetic_codes[idx]),
        }

    return features_dict


def load_swow_with_phonetics(
    csv_path,
    strength_col="R123.Strength",
    min_strength=0.0,
    phonetic_threshold=0.7,
    include_original_features=True,
):
    """
    Extended loader that adds phonetic edges and enhanced node features to SWOW graph.

    Args:
        csv_path: Path to SWOW CSV file
        strength_col: Column name for edge weights
        min_strength: Minimum strength threshold for SWOW edges
        phonetic_threshold: Minimum phonetic similarity for phonetic edges
        include_original_features: Whether to include original identity matrix features

    Returns:
        data: PyTorch Geometric data with enhanced features and edge types
        G: NetworkX graph with both edge types
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        phonetic_stats: Statistics about phonetic edges
    """
    print("Loading SWOW data with phonetic enhancement...")

    # First load the original SWOW data
    data, G, word2idx, idx2word = load_swow_en18(csv_path, strength_col, min_strength)

    # Get list of all words, filtering out non-string entries
    words_list = [
        word for word in word2idx.keys() if isinstance(word, str) and word.strip()
    ]
    print(
        f"Filtered to {len(words_list)} valid string words from {len(word2idx)} total entries"
    )

    # Add phonetic edges
    phonetic_edges = add_phonetic_edges(words_list, word2idx, phonetic_threshold)

    # Compute enhanced node features
    phonetic_features = compute_node_features_with_phonetics(
        words_list, word2idx, phonetic_edges
    )

    # Prepare combined edge information
    num_swow_edges = data.edge_index.shape[1]
    num_phonetic_edges = len(phonetic_edges)

    # Combine edge indices
    if phonetic_edges:
        phonetic_edge_tensor = torch.tensor(
            [[edge[0], edge[1]] for edge in phonetic_edges], dtype=torch.long
        ).t()
        combined_edge_index = torch.cat([data.edge_index, phonetic_edge_tensor], dim=1)

        # Combine edge weights (normalize phonetic similarities to similar scale as SWOW)
        phonetic_weights = torch.tensor(
            [edge[2] for edge in phonetic_edges], dtype=torch.float
        )
        combined_edge_weight = torch.cat([data.edge_weight, phonetic_weights])

        # Create edge type indicators (0 = SWOW, 1 = phonetic)
        swow_edge_types = torch.zeros(num_swow_edges, dtype=torch.long)
        phonetic_edge_types = torch.ones(num_phonetic_edges, dtype=torch.long)
        edge_type = torch.cat([swow_edge_types, phonetic_edge_types])
    else:
        combined_edge_index = data.edge_index
        combined_edge_weight = data.edge_weight
        edge_type = torch.zeros(num_swow_edges, dtype=torch.long)

    # Create enhanced node features
    num_nodes = len(word2idx)
    if phonetic_features:
        # Get dimensions
        sample_features = next(iter(phonetic_features.values()))
        phonetic_dim = len(sample_features["phonetic_embedding"])

        # Build feature matrix
        phonetic_embeddings = np.zeros((num_nodes, phonetic_dim))
        phonetic_edge_counts = np.zeros((num_nodes, 1))
        word_lengths = np.zeros((num_nodes, 1))
        phonetic_code_lengths = np.zeros((num_nodes, 1))

        for idx in range(num_nodes):
            if idx in phonetic_features:
                features = phonetic_features[idx]
                phonetic_embeddings[idx] = features["phonetic_embedding"]
                phonetic_edge_counts[idx, 0] = features["phonetic_edge_count"]
                word_lengths[idx, 0] = features["word_length"]
                phonetic_code_lengths[idx, 0] = features["phonetic_code_length"]

        # Combine all features
        feature_components = []

        if include_original_features:
            feature_components.append(data.x.numpy())  # Original identity matrix

        feature_components.extend(
            [
                phonetic_embeddings,
                phonetic_edge_counts,
                word_lengths,
                phonetic_code_lengths,
            ]
        )

        combined_features = np.concatenate(feature_components, axis=1)
        x = torch.tensor(combined_features, dtype=torch.float)
    else:
        x = data.x

    # Create enhanced PyTorch Geometric data
    enhanced_data = Data(
        x=x,
        edge_index=combined_edge_index,
        edge_weight=combined_edge_weight,
        edge_type=edge_type,
    )

    # Update NetworkX graph with phonetic edges
    enhanced_G = G.copy()
    for src_idx, dst_idx, similarity in phonetic_edges:
        src_word = idx2word[src_idx]
        dst_word = idx2word[dst_idx]
        enhanced_G.add_edge(src_word, dst_word, weight=similarity, edge_type="phonetic")

    # Add edge type to original SWOW edges
    for u, v, data_dict in enhanced_G.edges(data=True):
        if "edge_type" not in data_dict:
            data_dict["edge_type"] = "association"

    # Compute statistics
    phonetic_stats = {
        "num_swow_edges": num_swow_edges,
        "num_phonetic_edges": num_phonetic_edges,
        "total_edges": combined_edge_index.shape[1],
        "phonetic_threshold": phonetic_threshold,
        "phonetic_feature_dim": phonetic_dim if phonetic_features else 0,
        "total_feature_dim": x.shape[1],
    }

    print("Enhanced graph statistics:")
    print(f"  Nodes: {num_nodes}")
    print(f"  SWOW edges: {num_swow_edges}")
    print(f"  Phonetic edges: {num_phonetic_edges}")
    print(f"  Total edges: {phonetic_stats['total_edges']}")
    print(f"  Node feature dimension: {phonetic_stats['total_feature_dim']}")

    return enhanced_data, enhanced_G, word2idx, idx2word, phonetic_stats


# Example usage
if __name__ == "__main__":
    csv_file = "SWOW-EN18/strength.SWOW-EN.R123.20180827.csv"

    # Test original loader
    print("=== Testing original loader ===")
    data, G, word2idx, idx2word = load_swow_en18(csv_file, min_strength=0.05)
    print(data)
    print(f"Nodes: {len(word2idx)}, Edges: {data.edge_index.shape[1]}")
    print(f"Example words: {list(idx2word.values())[:10]}")

    # Test enhanced loader with phonetics
    print("\n=== Testing enhanced loader with phonetics ===")
    enhanced_data, enhanced_G, word2idx, idx2word, stats = load_swow_with_phonetics(
        csv_file, min_strength=0.05, phonetic_threshold=0.8
    )
    print(enhanced_data)
    print(f"Enhanced statistics: {stats}")
