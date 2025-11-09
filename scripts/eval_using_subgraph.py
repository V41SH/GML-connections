import os
import pickle
import numpy as np
import torch
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
from load_connections import load_connections_game
import utils
import dine

SOURCE_FILE = "dimension_subgraphs_connections_only_dim512.pkl"


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


# ---
# New Evaluator Class
# ---


class EmbeddingEvaluator:
    """
    Evaluates .npy graph embeddings on the Connections game task
    using ONLY DINE subgraph explanations.

    Requires:
    - 'compgcn_node_embeddings.npy' (embedding file)
    - 'node_to_idx.pkl' (mapping file)
    - 'dimension_subgraphs.pkl' (DINE explanation file)
    """

    def __init__(self, embedding_path: str, subgraphs_path: str):
        self.embedding_path = embedding_path
        self.subgraphs_path = subgraphs_path

        self.embeddings = None
        self.word2idx = None
        self.idx2word = None
        self.dim_edge_sets = {}
        self.dim_best_count = {}

        self._load_embeddings()
        self._load_subgraphs()

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

        print(
            f"Loaded {len(self.word2idx)} words, embedding dim = {self.embeddings.shape[1]}"
        )

    def _load_subgraphs(self):
        """Load and pre-process the DINE dimension subgraphs."""
        if not os.path.exists(self.subgraphs_path):
            raise FileNotFoundError(
                f"Subgraph file not found: {self.subgraphs_path}. Run eval_dimensions.py first."
            )

        print(f"Loading dimension subgraphs from {self.subgraphs_path}")
        with open(self.subgraphs_path, "rb") as f:
            all_subgraphs_by_dim = pickle.load(f)

        print("Pre-processing subgraphs for fast lookup...")
        self.dim_edge_sets = {
            dim: get_subgraph_edge_set(edge_index)
            for dim, edge_index in tqdm(
                all_subgraphs_by_dim.items(), desc="Processing dims"
            )
        }
        
        # Initialize the counter for dimension usage
        self.dim_best_count = {dim: 0 for dim in self.dim_edge_sets.keys()}
        
        print(f"Loaded and processed {len(self.dim_edge_sets)} dimension subgraphs.")

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

    def predict_groups_dine(
        self, words: List[str], group_size: int = 4, num_groups: int = 4
    ) -> Tuple[List[List[str]], List[Dict]]:
        """
        Predicts groups by finding the 4-word combination best explained
        by a single DINE dimension subgraph.
        """

        # 1. Get all valid word indices from the game board
        valid_words_indices = []
        word_map = {}  # map from game board index to original word
        for i, w in enumerate(words):
            w_lower = w.lower()
            idx = self.word2idx.get(w_lower)
            if idx is not None:
                valid_words_indices.append(idx)
                word_map[idx] = w_lower

        if len(valid_words_indices) < num_groups * group_size:
            # print(f"Warning: Only {len(valid_words_indices)} valid words found on board.")
            if len(valid_words_indices) < group_size:
                return [], []

        # 2. Generate all possible 4-word groups (e.g., 1820 combinations for 16 words)
        scored_groups = []

        for group_indices in combinations(valid_words_indices, group_size):
            # 3. For each group, find its best dimension
            group_pairs = set(
                tuple(sorted(pair)) for pair in combinations(group_indices, 2)
            )
            total_pairs = len(group_pairs)  # Should be 6

            best_dim_score = -1
            best_dim = -1

            for dim, edge_set in self.dim_edge_sets.items():
                if not edge_set:
                    continue

                count = len(group_pairs.intersection(edge_set))

                if count > best_dim_score:
                    best_dim_score = count
                    best_dim = dim

            # Score is (pairs_found / total_pairs).
            score = (
                best_dim_score + 1e-5
            ) / total_pairs  # Add epsilon to prioritize 1-pair over 0-pair
            scored_groups.append((score, group_indices, best_dim))
            
            # --- NEW: Increment the counter for the best dimension ---
            if best_dim != -1:
                self.dim_best_count[best_dim] += 1
            # --- END NEW ---

        # 4. Sort all possible groups by their best score
        scored_groups.sort(key=lambda x: x[0], reverse=True)

        # 5. Greedily select the top 4 non-overlapping groups
        final_groups_info = []
        final_groups_list = []
        used_indices = set()

        for score, group_indices, best_dim in scored_groups:
            if not set(group_indices).intersection(used_indices):
                group_words = [word_map[idx] for idx in group_indices]

                final_groups_list.append(group_words)
                final_groups_info.append(
                    {
                        "group": group_words,
                        "dim": best_dim,
                        "score": score,
                        "pairs_found": int(
                            round(score * total_pairs)
                        ),  # round to remove epsilon
                    }
                )

                used_indices.update(group_indices)
                if len(final_groups_info) == num_groups:
                    break

        return final_groups_list, final_groups_info

    def _calculate_metrics(
        self, true_groups: List[List[str]], pred_groups: List[List[str]]
    ) -> Dict:
        """Compute accuracy, exact matches, and pairwise F1."""
        true_groups = [[w.lower() for w in g] for g in true_groups]
        pred_groups = [[w.lower() for w in g] for g in pred_groups]

        # Exact matches
        exact_matches = sum(
            set(pg) in [set(tg) for tg in true_groups] for pg in pred_groups
        )

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

    def evaluate_game(self, game_id: int, csv_path: str) -> Tuple[Dict, List[Dict]]:
        """Evaluate model on one Connections game using DINE subgraphs."""
        try:
            data = load_connections_game(csv_path, game_id)
        except Exception as e:
            # print(f"⚠️ Skipping game {game_id}: {e}")
            return None, None, None

        true_groups = [g["words"] for g in data["groups"].values()]

        # Use the DINE prediction method
        pred_groups_list, pred_groups_info = self.predict_groups_dine(
            data["all_words"]
        )

        metrics = self._calculate_metrics(true_groups, pred_groups_list)

        return metrics, pred_groups_info, true_groups

    def evaluate_multiple_games(self, game_ids: List[int], csv_path: str) -> Dict:
        """Evaluate over multiple games and average results."""
        all_metrics = []
        all_game_results = [] # --- NEW: Store all game results ---
        top_n = 2

        for gid in tqdm(game_ids, desc="Evaluating games", unit="game"):
            metrics, info, true_groups = self.evaluate_game(gid, csv_path)

            if metrics:
                all_metrics.append(metrics)
                # Calculate average score for this game
                avg_score = np.mean(
                    [
                        metrics["exact_matches"],
                        metrics["avg_group_accuracy"],
                        metrics["pairwise_f1"],
                    ]
                )
                
                # --- NEW: Store results for all games ---
                all_game_results.append({
                    "id": gid, 
                    "score": avg_score, 
                    "metrics": metrics,
                    "info": info,
                    "true_groups": true_groups
                })
                # --- END NEW ---

        if not all_metrics:
            print("No valid games evaluated.")
            return {}

        agg = {
            "avg_exact_matches": np.mean([m["exact_matches"] for m in all_metrics]),
            "avg_group_accuracy": np.mean(
                [m["avg_group_accuracy"] for m in all_metrics]
            ),
            "avg_pairwise_f1": np.mean([m["pairwise_f1"] for m in all_metrics]),
        }

        print("\nAggregated results (averaged over valid games):")
        for k, v in agg.items():
            print(f"  {k}: {v:.3f}")
        print(f"  Games evaluated: {len(all_metrics)} / {len(game_ids)}")

        # --- NEW: Sort and print top N games ---
        all_game_results.sort(key=lambda x: x["score"], reverse=True)
        top_n_games = all_game_results[:top_n]

        print(f"\n--- Top {top_n} Performing Games ---")
        for i, game in enumerate(top_n_games):
            print(f"\n--- Rank {i+1} Game (ID: {game['id']}) ---")
            print(f"  Average score: {game['score']:.3f}")
            print("  Individual metrics:")
            for k, v in game["metrics"].items():
                print(f"    {k}: {v:.3f}")

            print("  Ground truth groups:")
            for group in game["true_groups"]:
                print(f"    - {group}")

            print("  Predicted Groups & Explanations:")
            if game["info"]:
                for info in game["info"]:
                    print(f"    - Group: {info['group']}")
                    print(
                        f"      -> Explained by: Dim {info['dim']} (Pairs: {info['pairs_found']}/6)"
                    )
            else:
                print("    - No groups predicted for this game.")
        
        agg["top_games"] = top_n_games
        # --- END NEW ---
        
        # --- NEW: Print dimension usage stats ---
        agg["dimension_usage_stats"] = self.dim_best_count

        print("\n--- Dimension Usage Statistics ---")
        print("(How many times a dimension was 'best' for a 4-word group)")
        
        # Sort by count, descending
        sorted_dims = sorted(
            self.dim_best_count.items(), key=lambda item: item[1], reverse=True
        )
        
        for dim, count in sorted_dims[:20]: # Print top 20
            if count > 0: # Only print dimensions that were actually used
                print(f"  Dim {dim}: {count} times")
        
        non_zero_dims = sum(1 for c in self.dim_best_count.values() if c > 0)
        if non_zero_dims > 20:
            print(f"  ... and {non_zero_dims - 20} other dimensions")
        # --- END NEW ---

        return agg


def main():
    """
    Main workflow:
    1. Load embeddings and DINE subgraphs.
    2. Run evaluation using only the subgraph method.
    3. Print results.
    """

    # --- Configuration ---
    base_dir = "graphs"
    csv_path = "connections_data/Connections_Data.csv"

    # subgraphs_path = os.path.join(base_dir, "dimension_subgraphs.pkl")
    subgraphs_path = os.path.join(
        # base_dir, "dimension_subgraphs_connections_only_dim64.pkl"
        base_dir, SOURCE_FILE
    )
    embedding_path = os.path.join(base_dir, "compgcn_node_embeddings_dim64.npy")

    game_ids = list(range(1, 872))
    # game_ids = [870, 871] # Use a small list for quick testing

    # 1. Initialize Evaluator
    # This will load both embeddings and subgraphs
    try:
        evaluator = EmbeddingEvaluator(embedding_path, subgraphs_path)
    except FileNotFoundError as e:
        print(f"\nError initializing evaluator: {e}")
        print("Please ensure all required files exist.")
        return

    # 2. Run Evaluation
    print(f"\n--- Running DINE Subgraph Evaluation on {len(game_ids)} games ---")
    agg_results = evaluator.evaluate_multiple_games(game_ids, csv_path)

    if "top_games" not in agg_results or not agg_results["top_games"]:
        print("Evaluation failed to find any valid games.")
        return

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()