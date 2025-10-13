import sys
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_graphs import load_swow_en18
from load_connections import load_connections_game


def train_node2vec(csv_path, min_strength=0.05, dimensions=64, 
                   walk_length=100, num_walks=10, window_size=5, 
                   workers=4, p=1, q=1):
    """
    Train Node2Vec on SWOW data.
    
    Args:
        csv_path: Path to SWOW CSV file
        min_strength: Minimum edge strength threshold
        dimensions: Embedding dimension
        walk_length: Length of random walks
        num_walks: Number of walks per node
        window_size: Context window size
        workers: Number of parallel workers
        p: Return parameter (controls backtracking)
        q: In-out parameter (controls exploration)
        
    Returns:
        embedding: Node embeddings (numpy array)
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        model: Trained Node2Vec model
    """
    # Load data
    _, G_directed, word2idx, idx2word = load_swow_en18(csv_path, min_strength=min_strength)
    print(f"Loaded: {len(G_directed.nodes())} nodes, {len(G_directed.edges())} edges")
    
    # Convert to undirected and relabel nodes
    G = nx.relabel_nodes(G_directed.to_undirected(), word2idx)
    print("Converted to undirected graph")
    
    # Train Node2Vec
    print(f"Training Node2Vec ({dimensions}D, p={p}, q={q})...")
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                       num_walks=num_walks, workers=workers, p=p, q=q)
    model = node2vec.fit(window=window_size, min_count=1, batch_words=4)
    
    # Extract embeddings
    embedding = np.array([model.wv[str(i)] for i in range(len(word2idx))])
    print(f"Embedding shape: {embedding.shape}\n")
    
    return embedding, word2idx, idx2word, model


def plot_embeddings(embedding, words, word2idx, idx2word, title='Word Embeddings'):
    """Plot word embeddings in 2D using PCA."""
    # Filter valid words
    print("words:")
    print(words)
    # TODO remove
    # normalise words to get more matches
    for i,word in enumerate(words):
        words[i] =  word.lower()
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
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=100)
    
    for i, word in enumerate(valid_words):
        plt.annotate(word, (pca_result[i, 0], pca_result[i, 1]), 
                    fontsize=10, alpha=0.8)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def find_similar(embedding, word, word2idx, idx2word, top_k=10):
    """Find most similar words based on cosine similarity."""
    if word not in word2idx:
        print(f"Word '{word}' not in vocabulary")
        return []
    
    word_vec = embedding[word2idx[word]].reshape(1, -1)
    similarities = cosine_similarity(word_vec, embedding)[0]
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    return [(idx2word[i], similarities[i]) for i in top_indices]


def main():
    # test saloni made
    hand_made_test()
    # load and test with latest connections game
    #test_with_connections()
   


def test_with_connections():
    """Example usage with specific connections game"""
    # Configuration
    CSV_FILE = "SWOW-EN18/strength.SWOW-EN.R123.20180827.csv"
    MIN_STRENGTH = 0.05
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
        q=1
    )
    
    # TODO load connections game 870 and print embeddings of all words and then compute similarities 
    connections_data = load_connections_game("connections_data/Connections_Data.csv", game_id=870)
    words = connections_data["all_words"]
    plot_embeddings(embedding, words, word2idx, idx2word, title="Connections Game 870")

def hand_made_test():
    """Example usage."""
    # Configuration
    CSV_FILE = "SWOW-EN18/strength.SWOW-EN.R123.20180827.csv"
    MIN_STRENGTH = 0.05
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
        q=1
    )
    
    # Example 1: Plot semantic categories
    print("=== Plotting semantic categories ===")
    words = ['dog', 'cat', 'animal', 'pet', 'house', 'home', 
             'happy', 'sad', 'joy', 'anger']
    plot_embeddings(embedding, words, word2idx, idx2word, 
                   'Semantic Word Embeddings')
    
    # Example 2: Find similar words
    print("\n=== Similar words to 'happy' ===")
    for word, score in find_similar(embedding, 'happy', word2idx, idx2word):
        print(f"  {word}: {score:.3f}")
    
    # Example 3: Compare emotion words
    print("\n=== Similar words to 'sad' ===")
    for word, score in find_similar(embedding, 'sad', word2idx, idx2word):
        print(f"  {word}: {score:.3f}")


if __name__ == "__main__":
    main()