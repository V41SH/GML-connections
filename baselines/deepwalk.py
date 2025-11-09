import os
import sys
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_conceptnet import load_conceptnet_graph
from load_connections import load_connections_game
from scripts.utils import load_fasttext_embeddings

# Global variable to store node features (replaces embeddings_fasttext)
node_features = None


def train_node2vec(
    csv_path,
    min_strength=0.05,
    dimensions=64,
    walk_length=100,
    num_walks=10,
    window_size=5,
    workers=4,
    p=1,
    q=1,
    model_path="models/node2vec_conceptnet_model.pkl",
):
    """
    Train Node2Vec on ConceptNet data with model saving/loading.

    Args:
        csv_path: Path to ConceptNet CSV file
        min_strength: Minimum edge strength threshold (unused for ConceptNet)
        dimensions: Embedding dimension
        walk_length: Length of random walks
        num_walks: Number of walks per node
        window_size: Context window size
        workers: Number of parallel workers
        p: Return parameter (controls backtracking)
        q: In-out parameter (controls exploration)
        model_path: Path to save/load the Node2Vec model

    Returns:
        embedding: Node embeddings (numpy array)
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        model: Trained Node2Vec model
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Try to load existing model
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)
        return (
            saved_data["embedding"],
            saved_data["word2idx"],
            saved_data["idx2word"],
            saved_data["model"],
        )

    # Load data
    _, G_directed, word2idx, idx2word, rel2idx = load_conceptnet_graph(csv_path)
    print(f"Loaded: {len(G_directed.nodes())} nodes, {len(G_directed.edges())} edges")

    # Convert to undirected and relabel nodes
    G = nx.relabel_nodes(G_directed.to_undirected(), word2idx)
    print("Converted to undirected graph")

    # Train Node2Vec
    print(f"Training Node2Vec ({dimensions}D, p={p}, q={q})...")
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        p=p,
        q=q,
    )
    model = node2vec.fit(window=window_size, min_count=1, batch_words=4)

    # Extract embeddings
    embedding = np.array([model.wv[str(i)] for i in range(len(word2idx))])
    print(f"Embedding shape: {embedding.shape}")

    # Save model and related data
    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "embedding": embedding,
                "word2idx": word2idx,
                "idx2word": idx2word,
                "model": model,
            },
            f,
        )
    print("Model saved successfully!")

    return embedding, word2idx, idx2word, model


def plot_embeddings(
    embedding, words, word2idx, idx2word, title="Word Embeddings", groups=None
):
    """Plot word embeddings in 2D using PCA."""
    # Filter valid words
    print("words:")
    print(words)
    # TODO remove
    # normalise words to get more matches
    for i, word in enumerate(words):
        words[i] = word.lower()
    indices = [word2idx[w] for w in words if w in word2idx]
    valid_words = [w for w in words if w in word2idx]
    print("valid_words:")
    print(valid_words)

    if not valid_words:
        print("No valid words found!")
        return

    # Apply PCA
    pca_result = PCA(n_components=2).fit_transform(embedding[indices])

    # Plot
    plt.figure(figsize=(12, 8))

    # Use different colors for different groups if provided
    if groups:
        colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
        ]  # Different colors for each group
        for group_idx, group_words in enumerate(groups):
            group_indices = [
                i for i, word in enumerate(valid_words) if word in group_words
            ]
            if group_indices:
                plt.scatter(
                    pca_result[group_indices, 0],
                    pca_result[group_indices, 1],
                    alpha=0.6,
                    s=100,
                    color=colors[group_idx % len(colors)],
                    label=f"Group {group_idx + 1}",
                )
    else:
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=100)

    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            (pca_result[i, 0], pca_result[i, 1]),
            fontsize=10,
            alpha=0.8,
        )

    if groups:
        plt.legend()

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{title.replace(' ', '_').lower()}.png")
    plt.show()


def find_similar(embedding, word, word2idx, idx2word, top_k=10):
    """Find most similar words based on cosine similarity."""
    if word not in word2idx:
        print(f"Word '{word}' not in vocabulary")
        return []

    word_vec = embedding[word2idx[word]].reshape(1, -1)
    similarities = cosine_similarity(word_vec, embedding)[0]
    top_indices = np.argsort(similarities)[::-1][1 : top_k + 1]

    return [(idx2word[i], similarities[i]) for i in top_indices]


def test_with_connections():
    """Example usage with specific connections game"""
    global node_features

    # Configuration
    CSV_FILE = "conceptnet/conceptnet_filtered_edges.csv"
    MIN_STRENGTH = 0.05  # Not used for ConceptNet but kept for compatibility
    DIMENSIONS = 64

    # Train model
    embedding, word2idx, idx2word, model = train_node2vec(
        CSV_FILE,
        min_strength=MIN_STRENGTH,
        dimensions=DIMENSIONS,
        walk_length=100,
        num_walks=10,
        window_size=5,
        p=1,
        q=1,
    )

    # Load node features if not already loaded
    if node_features is None:
        print("Loading node features using FastText...")
        # Get all nodes from the word2idx
        nodes = list(word2idx.keys())
        node_features = load_fasttext_embeddings(nodes, embedding_dim=300)
        node_features = node_features.numpy()  # Convert to numpy for compatibility
        print(f"Loaded node features with shape: {node_features.shape}")

    # Load connections game data
    connections_data = load_connections_game(
        "connections_data/Connections_Data.csv", game_id=870
    )
    words = connections_data["all_words"]

    # Get embeddings for valid words
    valid_words = [w.lower() for w in words if w.lower() in word2idx]
    valid_indices = [word2idx[w] for w in valid_words]

    # For words not in vocabulary, find closest word based on FastText embeddings
    import fasttext

    ft = fasttext.load_model("embedding_models/cc.en.300.bin")

    for nonword in set(words) - set(valid_words):
        nonword_lower = nonword.lower()
        current_embedding = ft.get_word_vector(nonword_lower)

        # Find closest word in our vocabulary based on FastText similarity
        best_distance = float("inf")
        best_idx = None

        for word, idx in word2idx.items():
            word_embedding = node_features[idx]
            distance = np.linalg.norm(word_embedding - current_embedding)
            if distance < best_distance:
                best_distance = distance
                best_idx = idx

        if best_idx is not None:
            valid_indices.append(best_idx)
            print(f"Unseen word '{nonword}' replaced by '{idx2word[best_idx]}'")
        else:
            print(f"Could not find replacement for '{nonword}'")

    valid_embeddings = embedding[valid_indices]

    # Calculate pairwise similarities
    similarities = cosine_similarity(valid_embeddings)

    # Create groups based on mutual similarity
    def group_average_similarity(word_indices, similarities_matrix):
        """Calculate the average pairwise similarity within a group"""
        if len(word_indices) <= 1:
            return 0
        pair_similarities = []
        for i in range(len(word_indices)):
            for j in range(i + 1, len(word_indices)):
                pair_similarities.append(
                    similarities_matrix[word_indices[i]][word_indices[j]]
                )
        return np.mean(pair_similarities)

    def find_best_group(available_indices, similarities_matrix, size=4):
        """Find the group of given size with highest average mutual similarity"""
        best_group = []
        best_score = -1

        # Try each word as a starting point
        for start_idx in available_indices:
            # Get similarities to all other available words
            current_sims = similarities_matrix[start_idx]
            # Get indices of most similar available words
            candidate_indices = [i for i in available_indices if i != start_idx]
            candidate_indices.sort(key=lambda x: current_sims[x], reverse=True)

            # Try top N most similar words
            for subset in range(min(10, len(candidate_indices) - size + 2)):
                potential_group = [start_idx] + candidate_indices[
                    subset : subset + size - 1
                ]
                if len(potential_group) == size:
                    score = group_average_similarity(
                        potential_group, similarities_matrix
                    )
                    if score > best_score:
                        best_score = score
                        best_group = potential_group

        return best_group, best_score

    # Create groups based on mutual similarity
    groups = []
    available_indices = set(range(len(valid_words)))

    while len(available_indices) >= 4:
        best_group, score = find_best_group(available_indices, similarities)
        if not best_group or score < 0:
            break

        # Add group and remove used words
        groups.append([valid_words[i] for i in best_group])
        available_indices -= set(best_group)

    # Plot with the automatically generated groups
    plot_embeddings(
        embedding,
        words,
        word2idx,
        idx2word,
        title="Connections Game 870 ConceptNet (Auto-Grouped)",
        groups=groups,
    )

    # Print the groups for analysis
    print("\nAutomatically generated groups based on embedding similarity:")
    for i, group in enumerate(groups):
        print(f"Group {i + 1}: {', '.join(group)}")


def init():
    # Create necessary directories
    for directory in ["plots", "models"]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")


def main():
    # test saloni made
    # hand_made_test()
    # load and test with latest connections game
    test_with_connections()


if __name__ == "__main__":
    init()
    main()
