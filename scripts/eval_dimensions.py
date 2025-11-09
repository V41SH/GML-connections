import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

from utils import load_graph_from_gml
from dine import get_explanation_subgraph


def main():
    """
    Evaluates all dimensions of an embedding and saves their
    corresponding explanation subgraphs (edge indices) to a file.
    """

    # --- Configuration ---
    base_dir = "graphs"
    gml_path = os.path.join(base_dir, "conceptnet_graph.gml")
    embedding_path = os.path.join(base_dir, "compgcn_node_embeddings.npy")
    output_path = os.path.join(base_dir, "dimension_subgraphs.pkl")

    print("--- Starting Subgraph Extraction ---")

    # 1. Load Original Graph Data
    print(f"Loading original graph from {gml_path}...")
    try:
        _, edge_index, _, node_to_idx, _ = load_graph_from_gml(gml_path)
        print(f"Loaded graph with {edge_index.shape[1]} total edges.")
    except FileNotFoundError:
        print(f"Error: Could not find graph GML file at {gml_path}")
        return
    except Exception as e:
        print(f"Error loading GML: {e}")
        return

    # 2. Load Final Embeddings
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

    # 3. Loop, Calculate, and Store Subgraphs
    all_subgraphs = {}
    print(f"Extracting subgraphs for all {num_dimensions} dimensions...")

    for dim_idx in tqdm(range(num_dimensions), desc="Processing Dimensions"):
        # Call the DINE function to get the subgraph for this dimension
        subgraph_edge_index = get_explanation_subgraph(
            embeddings_tensor, edge_index, dimension_idx=dim_idx
        )

        # Store the resulting edge_index.
        # .cpu() is good practice in case you ever run this on a GPU.
        all_subgraphs[dim_idx] = subgraph_edge_index.cpu()

    # 4. Save the results
    print(f"\nSaving all subgraphs to {output_path}...")
    try:
        with open(output_path, "wb") as f:
            pickle.dump(all_subgraphs, f)
        print("Successfully saved dimension subgraphs.")
    except Exception as e:
        print(f"Error saving subgraphs: {e}")

    print("--- Process Complete ---")


if __name__ == "__main__":
    main()
