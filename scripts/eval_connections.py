import os
import pickle
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity, safe_sparse_dot, check_pairwise_arrays
from typing import Dict, List
from tqdm import tqdm
from load_connections import load_connections_game
# import scripts.utils
import utils
import dine
import torch

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


def main_evaluation():
    """
    Runs the Connections game evaluation.
    """
    print("--- STARTING EVALUATION ---")
    # Note: Assumes your compgcn output is in 'graphs/compgcn_node_embeddings.npy'
    # If it's somewhere else, change this path.
    evaluator = EmbeddingEvaluator("graphs/compgcn_node_embeddings.npy")
    game_ids = list(range(1, 872)) 
    # game_ids = [870, 871]
    evaluator.evaluate_multiple_games(game_ids, "connections_data/Connections_Data.csv")
    print("--- EVALUATION COMPLETE ---")
    return evaluator # Return for use in interpretation

def main_interpretation(evaluator: EmbeddingEvaluator, graph_path: str, dim_to_check: int = 0):
    """
    Runs the DINE subgraph interpretation.
    
    This is where you use get_explanation_subgraph.
    """
    print(f"\n--- STARTING INTERPRETATION FOR DIMENSION {dim_to_check} ---")
    
    # 1. We need the original graph structure (edge_index)
    # This data is NOT in the EmbeddingEvaluator, so we must load it.
    print(f"Loading original graph from {graph_path}...")
    try:
        # We only need the edge_index and relation_to_idx from this
        _, edge_index, _, _, relation_to_idx = utils.load_graph_from_gml(graph_path)
    except FileNotFoundError:
        print(f"Error: Could not find graph GML file at {graph_path}")
        print("Skipping interpretation.")
        return
    except Exception as e:
        print(f"Error loading GML: {e}")
        print("Skipping interpretation.")
        return

    # 2. We need the final embeddings (as a torch tensor)
    # The evaluator already loaded them as numpy, so we just convert them.
    embeddings_tensor = torch.from_numpy(evaluator.embeddings)
    
    if dim_to_check >= embeddings_tensor.shape[1]:
        print(f"Error: Dimension {dim_to_check} is out of bounds.")
        print(f"Embeddings only have {embeddings_tensor.shape[1]} dimensions (0 to {embeddings_tensor.shape[1]-1}).")
        return

    # 3. Call the DINE function
    print(f"Extracting explanation subgraph for dimension {dim_to_check}...")
    subgraph_edge_index = dine.get_explanation_subgraph(
        embeddings_tensor, 
        edge_index, 
        dimension_idx=dim_to_check
    )
    
    print(f"\nDimension {dim_to_check} 'explains' {subgraph_edge_index.shape[1]} edges.")
    
    # 4. Make the results human-readable
    # We can use the idx2word map from the evaluator
    print("Top 20 edges from this subgraph:")
    edges_to_show = min(20, subgraph_edge_index.shape[1])
    if edges_to_show == 0:
        print("(No edges found for this dimension)")
        return
        
    for i in range(edges_to_show):
        u_idx = subgraph_edge_index[0, i].item()
        v_idx = subgraph_edge_index[1, i].item()
        
        u_word = evaluator.idx2word.get(u_idx, f"idx_{u_idx}")
        v_word = evaluator.idx2word.get(v_idx, f"idx_{v_idx}")
        
        print(f"  {u_word} <---> {v_word}")
        
    print("--- INTERPRETATION COMPLETE ---")


if __name__ == "__main__":
    # Path to the original graph GML file
    gml_path = "graphs/conceptnet_graph.gml"
    
    # Run the evaluation first
    eval_obj = main_evaluation()
    # eval_obj = main()
    
    # Then, run the interpretation using the results from the evaluation
    # if eval_obj:
    #     # Check a few dimensions
    #     main_interpretation(eval_obj, gml_path, dim_to_check=0)
    #     main_interpretation(eval_obj, gml_path, dim_to_check=1)
    #     main_interpretation(eval_obj, gml_path, dim_to_check=5)

