import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from itertools import chain

from load_connections import load_connections_game
from utils import load_graph_from_gml
from dine import get_explanation_subgraph

# --- Configuration ---
base_dir = "graphs"
csv_path = "connections_data/Connections_Data.csv"
game_ids = list(range(1, 872))
gml_path = os.path.join(base_dir, "conceptnet_graph.gml")
embedding_path = os.path.join(base_dir, "compgcn_node_embeddings_dim128.npy")
output_path = os.path.join(base_dir, "dimension_subgraphs_connections_only_dim128.pkl")
   


# def get_all_connections_words(csv_path: str, game_ids: List[int]) -> Set[str]:
def get_all_connections_words(csv_path, game_ids):
    """
    Loops through all Connections games and returns a single
    set of all unique words used.
    """
    all_words = set()
    print(f"Loading all words from {len(game_ids)} games...")
    for gid in tqdm(game_ids, desc="Loading game words"):
        try:
            data = load_connections_game(csv_path, gid)
            all_words.update(w.lower() for w in data["all_words"])
        except Exception as e:
            # Skip games that fail to load
            continue
    print(f"Found {len(all_words)} unique words in the Connections dataset.")
    return all_words

def main():
    """
    Generates and saves DINE explanation subgraphs, but only for the
    subset of the graph relevant to the Connections dataset.
    """
    
    
    
    
    print(f"--- Starting Filtered Subgraph Extraction ---")

    # 1. Get all unique words from the Connections dataset
    connections_words_set = get_all_connections_words(csv_path, game_ids)

    # 2. Load Original Graph Data
    print(f"Loading original graph from {gml_path}...")
    try:
        _, full_edge_index, _, node_to_idx, _ = load_graph_from_gml(gml_path)
        print(f"Loaded full graph with {full_edge_index.shape[1]} total edges.")
    except FileNotFoundError:
        print(f"Error: Could not find graph GML file at {gml_path}")
        return
    except Exception as e:
        print(f"Error loading GML: {e}")
        return

    # 3. Convert Connections words to their node indices
    connections_indices_set = set()
    for word in connections_words_set:
        idx = node_to_idx.get(word)
        if idx is not None:
            connections_indices_set.add(idx)
    print(f"Mapped {len(connections_words_set)} words to {len(connections_indices_set)} unique node indices.")

    # 4. Filter the full edge_index
    print("Filtering full edge index to only Connections-related edges...")
    
    # Convert to numpy for fast np.isin
    u = full_edge_index[0].cpu().numpy()
    v = full_edge_index[1].cpu().numpy()
    
    # Create a list/array for np.isin's 'test_elements' argument
    indices_array = np.array(list(connections_indices_set))

    # Find edges where *both* source and target nodes are in our set
    mask_u = np.isin(u, indices_array)
    mask_v = np.isin(v, indices_array)
    final_mask = mask_u & mask_v
    
    # Apply the mask
    filtered_edge_index = full_edge_index[:, torch.from_numpy(final_mask)]
    print(f"Filtered graph has {filtered_edge_index.shape[1]} edges (down from {full_edge_index.shape[1]}).")
    
    # Clean up memory
    del u, v, mask_u, mask_v, final_mask, indices_array, full_edge_index

    # 5. Load Final Embeddings
    print(f"Loading embeddings from {embedding_path}...")
    try:
        embeddings_numpy = np.load(embedding_path)
        embeddings_tensor = torch.from_numpy(embeddings_numpy)
        num_dimensions = embeddings_tensor.shape[1]
        print(f"Loaded embeddings with shape: {embeddings_tensor.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find embedding file at {embedding_path}")
        return
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    # 6. Loop, Calculate, and Store Subgraphs (on the *filtered* graph)
    all_subgraphs = {}
    print(f"Extracting subgraphs for all {num_dimensions} dimensions...")
    
    for dim_idx in tqdm(range(num_dimensions), desc="Processing Dimensions"):
        # Call the DINE function, but ONLY on our small, filtered edge list
        subgraph_edge_index = get_explanation_subgraph(
            embeddings_tensor, 
            filtered_edge_index, # <-- The key optimization!
            dimension_idx=dim_idx
        )
        all_subgraphs[dim_idx] = subgraph_edge_index.cpu()

    # 7. Save the new, smaller results file
    print(f"\nSaving filtered subgraphs to {output_path}...")
    try:
        with open(output_path, "wb") as f:
            pickle.dump(all_subgraphs, f)
        print("Successfully saved filtered dimension subgraphs.")
    except Exception as e:
        print(f"Error saving subgraphs: {e}")
        
    print("--- Process Complete ---")

if __name__ == "__main__":
    main()