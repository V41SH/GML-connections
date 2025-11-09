import pickle
import os

# --- CONFIGURE THIS ---
# Point this to the file you just used for evaluation
subgraphs_path = "graphs/dimension_subgraphs_connections_only_dim128.pkl"
# ---

print(f"Inspecting subgraph file: {subgraphs_path}")

try:
    with open(subgraphs_path, "rb") as f:
        subgraphs = pickle.load(f)
except FileNotFoundError:
    print(f"Error: File not found. Make sure this path is correct: {subgraphs_path}")
    exit()

if not subgraphs:
    print("Error: Subgraphs file is empty.")
    exit()

non_empty_dims = 0
total_edges = 0

for dim, edge_index in subgraphs.items():
    num_edges = edge_index.shape[1]
    total_edges += num_edges
    
    if num_edges > 0:
        print(f"Dimension {dim}: {num_edges} edges")
        non_empty_dims += 1

print("\n--- Summary ---")
print(f"Total dimensions: {len(subgraphs)}")
print(f"Total non-empty dimensions: {non_empty_dims}")
print(f"Total edges explained: {total_edges}")

if non_empty_dims <= 1:
    print("\nDIAGNOSIS: MODEL COLLAPSE CONFIRMED.")
    print("Almost all dimensions are empty. The model has failed to disentangle.")
else:
    print("\nDIAGNOSIS: Model seems to have learned in multiple dimensions.")