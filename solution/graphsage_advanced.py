"""
GraphSAGE training script with contrastive learning for SWOW word embeddings.

Copied from graphsage_baseline.py, but will be extended for use with
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm

# Try to import efficient neighbor sampling, fall back to simpler approach
try:
    from torch_geometric.loader import NeighborLoader

    HAS_NEIGHBOR_SAMPLING = True
except ImportError:
    HAS_NEIGHBOR_SAMPLING = False
    print(
        "Warning: Efficient neighbor sampling not available. Using full-batch training."
    )

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from load_graphs import load_swow_en18, load_swow_with_phonetics
from load_connections import load_connections_game
from get_embeddings import get_embeddings
from models.graphsage_baseline_model import (
    GraphSAGEContrastive,
    GraphAugmentor,
    nt_xent_loss,
    skip_gram_loss,
)


class GraphSAGETrainer:
    """Trainer for GraphSAGE with contrastive learning."""

    def __init__(
        self,
        model: GraphSAGEContrastive,
        data,
        word2idx: dict,
        idx2word: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 512,
        num_neighbors: list = [10, 5],
        augment_prob: float = 0.8,
        contrastive_weight: float = 0.7,
    ):
        """
        Args:
            model: GraphSAGE model
            data: PyTorch Geometric data object
            word2idx: Word to index mapping
            idx2word: Index to word mapping
            device: Device to run training on
            learning_rate: Learning rate
            weight_decay: Weight decay
            batch_size: Batch size for training
            num_neighbors: Number of neighbors to sample per layer
            augment_prob: Probability of applying augmentation
            contrastive_weight: Weight for contrastive loss vs skip-gram loss
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.device = device
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.augment_prob = augment_prob
        self.contrastive_weight = contrastive_weight

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Graph augmentor
        self.augmentor = GraphAugmentor()

        # Create data loaders if neighbor sampling is available
        if HAS_NEIGHBOR_SAMPLING:
            self.train_loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 for Windows compatibility
            )
            self.use_batching = True
        else:
            self.train_loader = None
            self.use_batching = False
            print("Using full-batch training mode")

    def generate_walk_samples(self, num_walks: int = 5, walk_length: int = 5) -> tuple:
        """Generate positive and negative samples from random walks."""
        # Simple random walk sampling
        pos_samples = []

        for _ in range(num_walks):
            # Start from random nodes
            start_nodes = torch.randint(0, self.data.num_nodes, (self.batch_size,))

            for start_node in start_nodes:
                current_node = start_node.item()

                # Random walk
                for _ in range(walk_length - 1):
                    # Get neighbors
                    neighbors = self.data.index[1][
                        self.data.edge_index[0] == current_node
                    ]
                    if len(neighbors) > 0:
                        # Choose neighbor based on edge weights if available
                        if (
                            hasattr(self.data, "edge_weight")
                            and self.data.edge_weight is not None
                        ):
                            edge_mask = self.data.edge_index[0] == current_node
                            weights = self.data.edge_weight[edge_mask]
                            weights = weights / weights.sum()
                            next_idx = torch.multinomial(weights, 1).item()
                            next_node = neighbors[next_idx].item()
                        else:
                            next_node = neighbors[
                                torch.randint(0, len(neighbors), (1,))
                            ].item()

                        pos_samples.append([current_node, next_node])
                        current_node = next_node
                    else:
                        break

        if not pos_samples:
            # Fallback: use some edges directly
            pos_samples = self.data.edge_index.t()[
                : min(1000, self.data.edge_index.size(1))
            ].tolist()

        # Generate negative samples
        neg_samples = []
        for _ in range(len(pos_samples)):
            src = torch.randint(0, self.data.num_nodes, (1,)).item()
            dst = torch.randint(0, self.data.num_nodes, (1,)).item()
            neg_samples.append([src, dst])

        return torch.tensor(pos_samples), torch.tensor(neg_samples)

    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        contrastive_loss_total = 0
        skipgram_loss_total = 0
        num_batches = 0

        # Generate walk samples for skip-gram loss
        pos_samples, neg_samples = self.generate_walk_samples()
        pos_samples = pos_samples.to(self.device)
        neg_samples = neg_samples.to(self.device)

        if self.use_batching and self.train_loader is not None:
            # Use neighbor sampling
            iterator = tqdm(self.train_loader, desc="Training")
        else:
            # Full-batch training - create single batch
            iterator = [self.data]
            print("Running full-batch training...")

        for batch in iterator:
            self.optimizer.zero_grad()

            # Get batch data
            x, edge_index, edge_weight = (
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_weight", None),
            )

            # Apply augmentations for contrastive learning
            if torch.rand(1).item() < self.augment_prob:
                # Create two augmented views
                aug_data1 = self.augmentor.augment_graph(batch)
                aug_data2 = self.augmentor.augment_graph(batch)

                # Get embeddings and projections
                _, proj1 = self.model(
                    aug_data1.x,
                    aug_data1.edge_index,
                    getattr(aug_data1, "edge_weight", None),
                )
                _, proj2 = self.model(
                    aug_data2.x,
                    aug_data2.edge_index,
                    getattr(aug_data2, "edge_weight", None),
                )

                # Contrastive loss
                contrastive_loss = nt_xent_loss(
                    proj1, proj2, temperature=self.model.temperature
                )
                contrastive_loss_total += contrastive_loss.item()
            else:
                contrastive_loss = torch.tensor(0.0, device=self.device)

            # Skip-gram loss
            embeddings, _ = self.model(x, edge_index, edge_weight)

            # Use subset of samples that fit in current batch
            if self.use_batching:
                batch_size = (
                    batch.batch_size if hasattr(batch, "batch_size") else x.size(0)
                )
                max_node_id = batch_size
            else:
                # Full-batch mode
                batch_size = x.size(0)
                max_node_id = self.data.num_nodes

            if len(pos_samples) > 0 and len(neg_samples) > 0:
                # Take samples that reference nodes in current batch
                valid_pos_mask = (pos_samples[:, 0] < max_node_id) & (
                    pos_samples[:, 1] < max_node_id
                )
                valid_neg_mask = (neg_samples[:, 0] < max_node_id) & (
                    neg_samples[:, 1] < max_node_id
                )

                if valid_pos_mask.sum() > 0 and valid_neg_mask.sum() > 0:
                    batch_pos_samples = pos_samples[valid_pos_mask][
                        : min(100, valid_pos_mask.sum())
                    ]
                    batch_neg_samples = neg_samples[valid_neg_mask][
                        : min(100, valid_neg_mask.sum())
                    ]

                    skipgram_loss = skip_gram_loss(
                        embeddings, batch_pos_samples, batch_neg_samples
                    )
                    skipgram_loss_total += skipgram_loss.item()
                else:
                    skipgram_loss = torch.tensor(0.0, device=self.device)
            else:
                skipgram_loss = torch.tensor(0.0, device=self.device)

            # Combined loss - ensure we always have at least one component with gradients
            if contrastive_loss.requires_grad and skipgram_loss.requires_grad:
                loss = (
                    self.contrastive_weight * contrastive_loss
                    + (1 - self.contrastive_weight) * skipgram_loss
                )
            elif contrastive_loss.requires_grad:
                loss = contrastive_loss
            elif skipgram_loss.requires_grad:
                loss = skipgram_loss
            else:
                # Skip this batch if no gradients
                continue

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {
            "total_loss": total_loss / max(num_batches, 1),
            "contrastive_loss": contrastive_loss_total / max(num_batches, 1),
            "skipgram_loss": skipgram_loss_total / max(num_batches, 1),
        }

    def get_all_embeddings(self) -> torch.Tensor:
        """Get embeddings for all nodes."""
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embeddings(
                self.data.x,
                self.data.edge_index,
                getattr(self.data, "edge_weight", None),
            )
        return embeddings.cpu().numpy()

    def find_similar_words(
        self, word: str, embeddings: np.ndarray, top_k: int = 10
    ) -> list:
        """Find most similar words based on cosine similarity."""
        if word not in self.word2idx:
            return []

        word_idx = self.word2idx[word]
        word_vec = embeddings[word_idx].reshape(1, -1)
        similarities = cosine_similarity(word_vec, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][1 : top_k + 1]

        return [(self.idx2word[i], similarities[i]) for i in top_indices]


def load_pretrained_embeddings(embeddings_path: str, word2idx: dict) -> torch.Tensor:
    """Load pretrained embeddings (e.g., from sentence transformers)."""
    with open(embeddings_path, "rb") as f:
        embeddings_dict = pickle.load(f)

    # Create embedding matrix
    embedding_dim = next(iter(embeddings_dict.values())).shape[0]
    embeddings = torch.randn(len(word2idx), embedding_dim)  # Random init

    for word, idx in word2idx.items():
        if idx in embeddings_dict:
            embeddings[idx] = torch.tensor(embeddings_dict[idx])

    return embeddings


def test_connections_game(trainer: GraphSAGETrainer, game_id: int = 870):
    """Test the model on a connections game."""
    # Load connections game data
    connections_data = load_connections_game(
        "connections_data/Connections_Data.csv", game_id=game_id
    )
    words = [w.lower() for w in connections_data["all_words"]]

    # Find valid words
    valid_words = [w for w in words if w in trainer.word2idx]

    if len(valid_words) < 4:
        print(f"Only {len(valid_words)} words found in vocabulary, skipping...")
        return

    print(f"\nTesting on Connections Game {game_id}")
    print(f"Valid words: {valid_words}")
    print("True groups:")
    for group_name, group_info in connections_data["groups"].items():
        group_words_lower = [w.lower() for w in group_info["words"]]
        valid_group_words = [w for w in group_words_lower if w in valid_words]
        print(f"  {group_name}: {valid_group_words}")


def main():
    """Main training function."""
    print("=== GraphSAGE Training for SWOW Word Embeddings ===")

    # Configuration
    CSV_FILE = "SWOW-EN18/strength.SWOW-EN.R123.20180827.csv"
    MIN_STRENGTH = 0.05
    PHONETIC_THRESHOLD = 0.8  # New parameter for phonetic similarity
    USE_PHONETIC_ENHANCEMENT = True  # Flag to enable/disable phonetic features
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    PROJECTION_DIM = 64
    NUM_EPOCHS = 100

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Load data with phonetic enhancement
    print("Loading SWOW data with phonetic enhancement...")
    if USE_PHONETIC_ENHANCEMENT:
        data, _, word2idx, idx2word, phonetic_stats = load_swow_with_phonetics(
            CSV_FILE,
            min_strength=MIN_STRENGTH,
            phonetic_threshold=PHONETIC_THRESHOLD,
            include_original_features=False,  # Don't include identity matrix
        )
        print(
            f"Loaded enhanced graph: {len(word2idx)} nodes, {data.edge_index.shape[1]} edges"
        )
        print(f"Phonetic statistics: {phonetic_stats}")
    else:
        # Fall back to original loader
        data, _, word2idx, idx2word = load_swow_en18(
            CSV_FILE, min_strength=MIN_STRENGTH
        )
        print(f"Loaded: {len(word2idx)} nodes, {data.edge_index.shape[1]} edges")

    # Load and combine pretrained embeddings with phonetic features
    print("Loading pretrained embeddings...")
    if not os.path.exists("models/embeddings.pickle"):
        get_embeddings()

    pretrained_embeddings = load_pretrained_embeddings(
        "models/embeddings.pickle", word2idx
    )

    if USE_PHONETIC_ENHANCEMENT:
        # Combine pretrained embeddings with phonetic features
        print("Combining pretrained embeddings with phonetic features...")
        combined_features = torch.cat([pretrained_embeddings, data.x], dim=1)
        data.x = combined_features
    else:
        data.x = pretrained_embeddings

    print(f"Final node features shape: {data.x.shape}")

    # Add edge type information to data for potential future use
    if hasattr(data, "edge_type"):
        print(
            f"Edge types: {torch.bincount(data.edge_type)}"
        )  # Count of each edge type

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GraphSAGEContrastive(
        vocab_size=len(word2idx),
        input_dim=data.x.shape[1],
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
        num_layers=2,
        aggregator="mean",
        dropout=0.3,
        temperature=0.1,
    )

    # Initialize trainer
    trainer = GraphSAGETrainer(
        model=model,
        data=data,
        word2idx=word2idx,
        idx2word=idx2word,
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=512,
        num_neighbors=[10, 5],
        contrastive_weight=0.7,
    )

    # Training loop
    print("\nStarting training...")
    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        metrics = trainer.train_epoch()

        print(
            f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
            f"Loss: {metrics['total_loss']:.4f} | "
            f"Contrastive: {metrics['contrastive_loss']:.4f} | "
            f"Skip-gram: {metrics['skipgram_loss']:.4f}"
        )

        # Save best model
        if metrics["total_loss"] < best_loss:
            best_loss = metrics["total_loss"]
            model_data = {
                "model_state_dict": model.state_dict(),
                "word2idx": word2idx,
                "idx2word": idx2word,
                "config": {
                    "vocab_size": len(word2idx),
                    "input_dim": data.x.shape[1],
                    "hidden_dim": HIDDEN_DIM,
                    "embedding_dim": EMBEDDING_DIM,
                    "projection_dim": PROJECTION_DIM,
                },
                "use_phonetic_enhancement": USE_PHONETIC_ENHANCEMENT,
                "phonetic_threshold": PHONETIC_THRESHOLD
                if USE_PHONETIC_ENHANCEMENT
                else None,
            }

            # Add phonetic statistics if available
            if USE_PHONETIC_ENHANCEMENT and "phonetic_stats" in locals():
                model_data["phonetic_stats"] = phonetic_stats

            torch.save(model_data, "models/graphsage_best.pth")

    # Test on connections game
    print("\n=== Testing on Connections Game ===")
    test_connections_game(trainer, game_id=870)

    # Get final embeddings and test similarity
    print("\n=== Testing Word Similarities ===")
    embeddings = trainer.get_all_embeddings()

    test_words = ["happy", "sad", "dog", "cat", "phone", "tone", "cat", "bat"]
    for word in test_words:
        if word in word2idx:
            similar = trainer.find_similar_words(word, embeddings, top_k=5)
            print(f"Similar to '{word}': {[(w, f'{s:.3f}') for w, s in similar]}")

    # If using phonetic enhancement, test some phonetically similar words
    if USE_PHONETIC_ENHANCEMENT:
        print("\n=== Testing Phonetic Similarity Impact ===")
        from utils import phonetic_similarity

        phonetic_test_pairs = [
            ("cat", "bat"),
            ("phone", "tone"),
            ("right", "write"),
            ("see", "sea"),
            ("flower", "flour"),
        ]

        for word1, word2 in phonetic_test_pairs:
            if word1 in word2idx and word2 in word2idx:
                phon_sim = phonetic_similarity(word1, word2)

                # Get embeddings and compute cosine similarity
                idx1, idx2 = word2idx[word1], word2idx[word2]
                emb1 = embeddings[idx1].reshape(1, -1)
                emb2 = embeddings[idx2].reshape(1, -1)
                cos_sim = cosine_similarity(emb1, emb2)[0, 0]

                print(
                    f"'{word1}' <-> '{word2}': phonetic={phon_sim:.3f}, learned={cos_sim:.3f}"
                )

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
