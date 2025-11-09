import os
import pickle
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional
from tqdm import tqdm
from load_connections import load_connections_game
import fasttext
import fasttext.util


class Node2VecEvaluator:
    """
    Evaluates Node2Vec model (.pkl) on the Connections game task.
    Works with gensim Word2Vec models saved as pickle files.
    Supports replacing missing words using FastText embeddings.
    """

    def __init__(self, model_path: str, use_fasttext_fallback: bool = True, 
                 fasttext_model_path: str = 'cc.en.300.bin',
                 embeddings_pickle_path: str = 'embeddings.pickle'):
        self.model_path = model_path
        self.model = None
        self.use_fasttext_fallback = use_fasttext_fallback
        self.ft = None
        self.embeddings_fasttext = None
        self.idx2word = None
        
        self._load_model()
        
        if use_fasttext_fallback:
            self._load_fasttext(fasttext_model_path, embeddings_pickle_path)

    def _load_model(self):
        """Load Node2Vec model from pickle file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading Node2Vec model from {self.model_path}")
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(model_data, tuple):
            # Format: (embedding, word2idx, idx2word, ...)
            print("Detected tuple format pickle")
            if len(model_data) >= 3:
                embedding, word2idx, idx2word = model_data[0], model_data[1], model_data[2]
                self.model = None  # No gensim model in this format
                self.wv = None
                self.idx2word = idx2word
                self.word2idx = word2idx
                self.embedding_matrix = embedding
                print(f"Loaded {len(word2idx)} nodes, embedding dim = {embedding.shape[1]}")
                return
        
        #  NEW: Handle custom dictionary format
        if isinstance(model_data, dict):
            print("Detected dictionary format pickle")
            # --- THIS IS AN EXAMPLE - ADJUST KEYS BASED ON YOUR INSPECTION ---
            if 'embeddings' in model_data and 'word2idx' in model_data and 'idx2word' in model_data:
                self.embedding_matrix = model_data['embeddings']
                self.word2idx = model_data['word2idx']
                self.idx2word = model_data['idx2word']
                self.model = None
                self.wv = None
                print(f"Loaded {len(self.word2idx)} nodes from dict.")
                return
            elif 'model' in model_data:
                 print("Found 'model' key in dict, loading it.")
                 model_data = model_data['model'] # Treat the inner model as the data
            else:
                raise ValueError("Pickle is a dictionary, but required keys are missing.")
        

        # Standard gensim model
        self.model = model_data
        
        # Check if it's a gensim Word2Vec model
        if hasattr(self.model, 'wv'):
            # Gensim 4.x style - uses KeyedVectors
            self.wv = self.model.wv
            vocab_size = len(self.wv)
            emb_dim = self.wv.vector_size
        elif hasattr(self.model, 'key_to_index'):
            # Already a KeyedVectors object
            self.wv = self.model
            vocab_size = len(self.wv)
            emb_dim = self.wv.vector_size
        elif hasattr(self.model, 'vocab') or hasattr(self.model, 'index2word'):
            # --- THIS IS THE ROBUST FIX ---
            # Older gensim or custom format. 
            # We explicitly assign self.wv to the model itself,
            # assuming it behaves like KeyedVectors (which it does).
            # BUT, if it has a .wv attribute, use that instead.
            print("Detected older gensim model format.")
            self.wv = self.model
        else:
            # --- ADD A FINAL CHECK ---
            raise TypeError(
                "Loaded pickle file is not a recognized format. "
                "It's not a (embedding, word2idx, idx2word) tuple "
                "and does not appear to be a gensim model (missing .wv or .key_to_index)."
            )

        # --- MOVE WV-DEPENDENT ATTRIBUTES OUTSIDE THE IF/ELSE ---
        vocab_size = len(self.wv)
        emb_dim = self.wv.vector_size
        
        # Build idx2word mapping for fallback
        if hasattr(self.wv, 'index_to_key'):
            self.idx2word = {i: word for i, word in enumerate(self.wv.index_to_key)}
        elif hasattr(self.wv, 'index2word'):
            self.idx2word = {i: word for i, word in enumerate(self.wv.index2word)}
        else:
            print("Warning: Could not build idx2word mapping. Fallback may fail.")

        print(f"Loaded model with {vocab_size} nodes, embedding dim = {emb_dim}")

    def _load_fasttext(self, model_path: str, embeddings_pickle_path: str):
        """Load FastText model and pre-computed embeddings for fallback."""
        # Download if needed
        if not os.path.exists(model_path):
            print("FastText model not found. Downloading (this may take a few minutes)...")
            try:
                fasttext.util.download_model('en', if_exists='ignore')
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading FastText model: {e}")
                print("Continuing without FastText fallback...")
                self.use_fasttext_fallback = False
                return
        
        try:
            import sys
            print(f"Loading FastText model from {model_path}")
            print("⏳ This will take about 60 seconds for the 6GB model file...")
            print("(If it seems stuck, just wait - it's loading)")
            sys.stdout.flush()  # Force output to display
            
            self.ft = fasttext.load_model(model_path)
            
            print("✓ FastText model loaded successfully!")
            sys.stdout.flush()
        except Exception as e:
            print(f"✗ Error loading FastText model: {e}")
            print("Continuing without FastText fallback...")
            self.use_fasttext_fallback = False
            return
        
        # Load pre-computed embeddings
        if os.path.exists(embeddings_pickle_path):
            print(f"Loading pre-computed embeddings from {embeddings_pickle_path}")
            sys.stdout.flush()
            with open(embeddings_pickle_path, 'rb') as handle:
                self.embeddings_fasttext = pickle.load(handle)
            print(f"✓ Loaded {len(self.embeddings_fasttext)} pre-computed embeddings")
        else:
            print(f"⚠️  {embeddings_pickle_path} not found.")
            if self.idx2word and len(self.idx2word) > 0:
                print("Computing embeddings for vocabulary...")
                self._compute_fasttext_embeddings()
            else:
                print("Cannot compute embeddings without vocabulary. Disabling FastText fallback.")
                self.use_fasttext_fallback = False
    
    def _compute_fasttext_embeddings(self):
        """Compute FastText embeddings for all words in vocabulary."""
        if self.idx2word is None:
            print("Warning: Cannot compute embeddings without idx2word mapping")
            self.use_fasttext_fallback = False
            return
        
        print(f"Computing FastText embeddings for {len(self.idx2word)} vocabulary words...")
        print("This will be saved to embeddings.pickle for future use...")
        self.embeddings_fasttext = {}
        for idx, word in tqdm(self.idx2word.items(), desc="Computing embeddings"):
            try:
                self.embeddings_fasttext[idx] = self.ft.get_word_vector(word)
            except Exception as e:
                print(f"Warning: Could not get embedding for '{word}': {e}")
        
        # Save for future use
        try:
            with open('embeddings.pickle', 'wb') as handle:
                pickle.dump(self.embeddings_fasttext, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {len(self.embeddings_fasttext)} embeddings to embeddings.pickle")
        except Exception as e:
            print(f"Warning: Could not save embeddings: {e}")

    def _find_closest_word_idx(self, word: str) -> Optional[int]:
        """Find closest word in vocabulary using FastText embeddings."""
        if not self.use_fasttext_fallback or self.ft is None or self.embeddings_fasttext is None:
            return None
        
        current_embedding = self.ft.get_word_vector(word)
        
        # Find closest embedding
        myitem = next(iter(self.embeddings_fasttext.items()))
        closest_embedding = myitem[1]
        closest_embedding_index = myitem[0]
        closest_distance = np.linalg.norm(closest_embedding - current_embedding)
        
        for idx, ft_embedding in self.embeddings_fasttext.items():
            dist = np.linalg.norm(ft_embedding - current_embedding)
            if dist < closest_distance:
                closest_distance = dist
                closest_embedding = ft_embedding
                closest_embedding_index = idx
        
        return closest_embedding_index

    def get_word_embedding(self, word: str, verbose: bool = False):
        """Return embedding for a given word if available, with FastText fallback."""
        word = word.lower()
        
        # Handle tuple format (direct embedding matrix)
        if hasattr(self, 'embedding_matrix') and hasattr(self, 'word2idx'):
            if word in self.word2idx:
                idx = self.word2idx[word]
                return self.embedding_matrix[idx], word
            # Try fallback for tuple format
            if self.use_fasttext_fallback:
                closest_idx = self._find_closest_word_idx(word)
                if closest_idx is not None:
                    closest_word = self.idx2word[closest_idx]
                    if verbose:
                        print(f"Unseen word '{word}' replaced by '{closest_word}'")
                    return self.embedding_matrix[closest_idx], closest_word
            return None, None
        
       # Handle gensim format
        try:
            # Use get_vector() as it's the standard.
            # This works for Word2Vec (In-Vocab) and FastText (In-Vocab & OOV).
            if hasattr(self.wv, 'get_vector'):
                return self.wv.get_vector(word), word
            # Fallback for dict-like models
            elif hasattr(self.wv, '__getitem__'):
                return self.wv[word], word
                
        except KeyError:
            # This block is *only* reached for OOV words in
            # non-FastText models (like Word2Vec).
            
            if self.use_fasttext_fallback:
                # This now assumes self.idx2word and self._find_closest_word_idx
                # are correctly populated/working with the gensim vocab.
                closest_idx = self._find_closest_word_idx(word)
                
                if closest_idx is not None and self.idx2word is not None:
                    closest_word = self.idx2word[closest_idx]
                    if verbose:
                        print(f"Unseen word '{word}' replaced by '{closest_word}'")
                    
                    try:
                        # Try to get the vector for the replacement word
                        if hasattr(self.wv, 'get_vector'):
                            return self.wv.get_vector(closest_word), closest_word
                        elif hasattr(self.wv, '__getitem__'):
                            return self.wv[closest_word], closest_word
                    except (KeyError, Exception) as e:
                        # Be explicit about the failure, don't pass silently
                        if verbose:
                            print(f"Fallback word '{closest_word}' also not found: {e}")
                        return None, None
                        
        except Exception as e:
            # Catch any other unexpected errors
            if verbose:
                print(f"An unexpected error occurred for word '{word}': {e}")
            return None, None
        
        # All attempts failed
        return None, None

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

    def predict_groups(self, words: List[str], group_size: int = 4, num_groups: int = 4, 
                      verbose: bool = False) -> List[List[str]]:
        """Predict groups of related words."""
        valid_words, word_embs, replaced_words = [], [], {}
        
        for w in words:
            emb, actual_word = self.get_word_embedding(w, verbose=verbose)
            if emb is not None:
                valid_words.append(w.lower())
                word_embs.append(emb)
                if actual_word != w.lower():
                    replaced_words[w.lower()] = actual_word

        if len(valid_words) < group_size * num_groups:
            if verbose:
                print(f"Warning: only {len(valid_words)} valid words found.")
            if len(valid_words) == 0:
                return []  # Return empty list if no valid words are found
        
        if verbose and replaced_words:
            print(f"Replaced {len(replaced_words)} words using FastText fallback")

        word_embs = np.array(word_embs)
        sim = cosine_similarity(np.array(word_embs))
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


def main():
    # Example usage - adjust path to your Node2Vec model
    evaluator = Node2VecEvaluator(
        model_path="models/swow_node2vec_64d.pkl",
        use_fasttext_fallback=True,  # Enable FastText fallback for missing words
        fasttext_model_path='embedding_models/cc.en.300.bin',
        embeddings_pickle_path='embeddings.pickle'
    )
    
    # Evaluate on all games or a subset
    game_ids = list(range(1, 872))  # All games
    # game_ids = [870, 871]  # Specific games for testing
    
    evaluator.evaluate_multiple_games(game_ids, "connections_data/Connections_Data.csv")


if __name__ == "__main__":
    main()