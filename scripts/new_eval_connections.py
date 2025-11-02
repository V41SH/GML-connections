import os
import pickle
import numpy as np
import torch
from pathlib import Path
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity, safe_sparse_dot, check_pairwise_arrays
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
from load_connections import load_connections_game
import utils
import dine

# ---
# We include the EmbeddingEvaluator class from eval_connections.py
# to make this script self-contained and runnable.
# ---

class EmbeddingEvaluator:
    """
    Evaluates .npy graph embeddings on the Connections game task.
    Requires 'node_to_idx.pkl' in the same directory as the embedding file.
    """

    def __init__(self, embedding_path: str):
        self.embedding_path = embedding_path
        self.embeddings = None
        self.word2idx = None
        self.idx2word = None
        self._load_embeddings()

    def _load_embeddings(self):
        """Load embeddings (.npy) and word-to-index mapping."""
        base_dir = os.path.dirname(self.embedding_path)
        node_to_idx_path = os.path.join(base_dir, "node_to_idx.pkl")

        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_path}")
        if not os.path.exists(node_to_idx_path):
            raise FileNotFoundError(f"Mapping file not found: {node_to_idx_path}")

        print(f"Loading embeddings from {self.embedding_path}")
        self.embeddings = np.load(self.embedding_path)
        with open(node_to_idx_path, "rb") as f:
            self.word2idx = pickle.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        print(f"Loaded {len(self.word2idx)} words, embedding dim = {self.embeddings.shape[1]}")

    def get_word_embedding(self, word: str):
        """Return embedding for a given word if available."""
        try:
            word = word.lower()
        except:
            print(f"get_word_embedding failed with word `{word}`")
            raise Exception
        if word in self.word2idx:
            return self.embeddings[self.word2idx[word]]
        return None

    def group_average_similarity(self, word_indices: List[int], sim_matrix: np.ndarray) -> float:
        """Average pairwise cosine similarity within a group."""
        if len(word_indices) <= 1:
            return 0.0
        return np.mean([sim_matrix[i, j] for i, j in combinations(word_indices, 2)])

    def find_best_group(self, available: set, sim_matrix: np.ndarray, size: int = 4) -> List[int]:
        """Find the most mutually similar group of given size."""
        best_group, best_score = [], -1
        for i in available:
            sims = sim_matrix[i]
            candidates = sorted([j for j in available if j != i],
                                key=lambda x: sims[x], reverse=True)
            if len(candidates) < size - 1:
                continue
            group = [i] + candidates[:size - 1]
            score = self.group_average_similarity(group, sim_matrix)
            if score > best_score:
                best_score, best_group = score, group
        return best_group

    def predict_groups(self, words: List[str], group_size: int = 4, num_groups: int = 4) -> List[List[str]]:
        """Predict groups of related words."""
        valid_words, word_embs = [], []
        for w in words:
            emb = self.get_word_embedding(w)
            if emb is not None:
                valid_words.append(w.lower())
                word_embs.append(emb)

        if len(valid_words) < group_size * num_groups:
            print(f"Warning: only {len(valid_words)} valid words found.")
            if len(valid_words) == 0:
                return []  # Return empty list if no valid words are found

        word_embs = np.array(word_embs)
        # sim = cosine_similarity(np.array(word_embs))
        X, Y = check_pairwise_arrays(word_embs, None)
        sim = safe_sparse_dot(X, Y.T)
        available = set(range(len(valid_words)))
        groups = []

        for _ in range(num_groups):
            if len(available) < group_size:
                break
            group = self.find_best_group(available, sim, group_size)
            if not group:
                break
            groups.append([valid_words[i] for i in group])
            available -= set(group)
        return groups

    def _calculate_metrics(self, true_groups: List[List[str]], pred_groups: List[List[str]]) -> Dict:
        """Compute accuracy, exact matches, and pairwise F1."""
        true_groups = [[w.lower() for w in g] for g in true_groups]
        pred_groups = [[w.lower() for w in g] for g in pred_groups]

        # Exact matches
        exact_matches = sum(set(pg) in [set(tg) for tg in true_groups] for pg in pred_groups)

        # Group accuracy
        if not pred_groups:  # If no predictions were made
            group_accs = [0.0 for _ in true_groups]  # Zero accuracy for all groups
        else:
            group_accs = [
                max(len(set(tg) & set(pg)) / len(tg) for pg in pred_groups)
                for tg in true_groups
            ]

        # Pairwise F1
        def pairs(groups):
            return {tuple(sorted(p)) for g in groups for p in combinations(g, 2)}

        true_pairs = pairs(true_groups)
        pred_pairs = pairs(pred_groups) if pred_groups else set()
        tp = len(true_pairs & pred_pairs)
        fp = len(pred_pairs - true_pairs)
        fn = len(true_pairs - pred_pairs)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        return {
            "exact_matches": exact_matches,
            "avg_group_accuracy": np.mean(group_accs),
            "pairwise_f1": f1,
        }

    def evaluate_game(self, game_id: int, csv_path: str) -> Dict:
        """Evaluate model on one Connections game."""
        try:
            data = load_connections_game(csv_path, game_id)
        except Exception as e:
            print(f"⚠️ Skipping game {game_id}: {e}")
            return None

        true_groups = [g["words"] for g in data["groups"].values()]
        pred_groups = self.predict_groups(data["all_words"])
        return self._calculate_metrics(true_groups, pred_groups)

    def evaluate_multiple_games(self, game_ids: List[int], csv_path: str) -> Dict:
        """Evaluate over multiple games and average results."""
        all_metrics = []
        best_game = {"id": None, "score": -1, "metrics": None}

        for gid in tqdm(game_ids, desc="Evaluating games", unit="game"):
            metrics = self.evaluate_game(gid, csv_path)
            if metrics:
                all_metrics.append(metrics)
                # Calculate average score for this game
                avg_score = np.mean([
                    metrics["exact_matches"],
                    metrics["avg_group_accuracy"],
                    metrics["pairwise_f1"]
                ])
                if avg_score > best_game["score"]:
                    best_game = {"id": gid, "score": avg_score, "metrics": metrics}

        if not all_metrics:
            print("No valid games evaluated.")
            return {}

        agg = {
            "avg_exact_matches": np.mean([m["exact_matches"] for m in all_metrics]),
            "avg_group_accuracy": np.mean([m["avg_group_accuracy"] for m in all_metrics]),
            "avg_pairwise_f1": np.mean([m["pairwise_f1"] for m in all_metrics]),
        }

        print("\nAggregated results (averaged over valid games):")
        for k, v in agg.items():
            print(f"  {k}: {v:.3f}")
        print(f"  Games evaluated: {len(all_metrics)} / {len(game_ids)}")
        
        print("\nBest performing game:")
        print(f"  Game ID: {best_game['id']}")
        print(f"  Average score: {best_game['score']:.3f}")
        print("  Individual metrics:")
        for k, v in best_game['metrics'].items():
            print(f"    {k}: {v:.3f}")
        
        agg["best_game"] = best_game
        return agg

# ---
# New Analysis Functions
# ---

def get_subgraph_edge_set(edge_index: torch.Tensor) -> Set[Tuple[int, int]]:
    """
    Converts a [2, N] edge_index tensor into a set of sorted (u, v) tuples
    for efficient lookup.
    """
    if edge_index.shape[1] == 0:
        return set()
    
    # Transpose to [N, 2], sort pairs to handle (a,b) vs (b,a), convert to tuples
    edges = edge_index.t().cpu().numpy()
    sorted_edges = np.sort(edges, axis=1)
    return set(map(tuple, sorted_edges))

def analyze_best_game(
    best_game_id: int, 
    evaluator: EmbeddingEvaluator, 
    all_subgraphs_by_dim: Dict[int, torch.Tensor], 
    csv_path: str
):
    """
    Provides a detailed analysis of the best game, explaining predicted groups
    using the DINE dimension subgraphs.
    """
    print(f"\n--- Analyzing Best Game (ID: {best_game_id}) ---")
    
    # 1. Get Game Data (True Groups and Predicted Groups)
    try:
        data = load_connections_game(csv_path, best_game_id)
    except Exception as e:
        print(f"Error loading game {best_game_id}: {e}")
        return

    true_groups = [g["words"] for g in data["groups"].values()]
    true_groups_lower = [[w.lower() for w in g] for g in true_groups]
    
    pred_groups = evaluator.predict_groups(data["all_words"])
    
    print("\n--- TRUE GROUPS ---")
    for i, group in enumerate(true_groups_lower):
        print(f"  {i+1}. {group}")
        
    print("\n--- PREDICTED GROUPS & DIMENSIONAL EXPLANATION ---")
    if not pred_groups:
        print("No groups were predicted for this game.")
        return

    # 2. Pre-process subgraphs for fast lookup
    print("Pre-processing subgraphs...")
    dim_edge_sets = {
        dim: get_subgraph_edge_set(edge_index) 
        for dim, edge_index in all_subgraphs_by_dim.items()
    }
    
    # 3. Analyze each predicted group
    for i, group in enumerate(pred_groups):
        print(f"\n  Predicted Group {i+1}: {group}")
        
        # Get word indices
        group_indices = [evaluator.word2idx.get(w) for w in group]
        group_indices = [idx for idx in group_indices if idx is not None]
        
        if len(group_indices) < 2:
            print("    -> Not enough known words in this group to analyze.")
            continue
            
        # Get all internal pairs (e.g., 6 pairs for a 4-word group)
        group_pairs = set(
            tuple(sorted(pair)) for pair in combinations(group_indices, 2)
        )
        
        # Grade each dimension
        dim_scores = {}
        for dim, edge_set in dim_edge_sets.items():
            if not edge_set:
                dim_scores[dim] = 0
                continue
            
            # Count how many of the group's pairs are in this dim's subgraph
            count = len(group_pairs.intersection(edge_set))
            dim_scores[dim] = count

        # Find the best dimension
        best_dim = max(dim_scores, key=dim_scores.get)
        best_score = dim_scores[best_dim]
        
        print(f"    -> Best Explanation: Dimension {best_dim}")
        print(f"    -> Internal pairs found in subgraph: {best_score} / {len(group_pairs)}")
        
        # Show the specific pairs found
        if best_score > 0:
            found_pairs = group_pairs.intersection(dim_edge_sets[best_dim])
            for u_idx, v_idx in found_pairs:
                u_word = evaluator.idx2word.get(u_idx, "???")
                v_word = evaluator.idx2word.get(v_idx, "???")
                print(f"        - Found Link: ({u_word} <-> {v_word})")

def main():
    """
    Main workflow:
    1. Find best game by running evaluation.
    2. Load DINE dimension subgraphs.
    3. Run detailed analysis on the best game.
    """
    
    # --- Configuration ---
    base_dir = "graphs"
    csv_path = "connections_data/Connections_Data.csv"
    
    subgraphs_path = os.path.join(base_dir, "dimension_subgraphs.pkl")
    embedding_path = os.path.join(base_dir, "compgcn_node_embeddings.npy")
    
    game_ids = list(range(1, 872)) 
    # game_ids = [870, 871] # Use a small list for quick testing
    
    # 1. Load Dimension Subgraphs
    print(f"Loading dimension subgraphs from {subgraphs_path}")
    try:
        with open(subgraphs_path, "rb") as f:
            all_subgraphs_by_dim = pickle.load(f)
        print(f"Loaded {len(all_subgraphs_by_dim)} dimension subgraphs.")
    except FileNotFoundError:
        print(f"Error: Could not find subgraph file at {subgraphs_path}")
        print("Please run eval_dimensions.py first.")
        return
    except Exception as e:
        print(f"Error loading subgraphs: {e}")
        return

    # 2. Run Evaluation to Find Best Game
    print("\n--- Running Evaluation to Find Best Game ---")
    evaluator = EmbeddingEvaluator(embedding_path)
    agg_results = evaluator.evaluate_multiple_games(game_ids, csv_path)
    
    if "best_game" not in agg_results:
        print("Evaluation failed to find a best game.")
        return
        
    best_game_id = agg_results['best_game']['id']
    
    # 3. Run Detailed Analysis
    analyze_best_game(
        best_game_id, 
        evaluator, 
        all_subgraphs_by_dim, 
        csv_path
    )
    
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()
