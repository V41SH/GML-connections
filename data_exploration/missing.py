import os
import pandas as pd
import sys
from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_graphs import load_swow_en18
from load_connections import get_all_connections_words

def compare_connections_to_swow(connections_csv, swow_csv, min_strength=0.05, output_path="analysis/connections_swow_overlap.csv"):
    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load datasets
    print("Loading Connections data...")
    connections_words = get_all_connections_words(connections_csv)

    print("Loading SWOW data...")
    _, _, word2idx, idx2word = load_swow_en18(swow_csv, min_strength=min_strength)
    swow_words = set(word2idx.keys())
    print(f"✅ Loaded {len(swow_words)} SWOW words")

    # Compare
    overlap = connections_words & swow_words
    missing = connections_words - swow_words

    # Stats
    coverage = len(overlap) / len(connections_words) * 100
    print(f"\n=== Comparison Summary ===")
    print(f"Connections words: {len(connections_words)}")
    print(f"Overlap with SWOW: {len(overlap)} ({coverage:.2f}%)")
    print(f"Missing words: {len(missing)} ({len(missing)/len(connections_words)*100:.2f}%)")
    
    missing_path =  "data_exploration/_missing.csv"
    pd.DataFrame(sorted(missing), columns=["missing_word"]).to_csv(missing_path, index=False)

    return overlap, missing


# --- Step 3: Run the comparison ---
if __name__ == "__main__":
    CONNECTIONS_CSV = "connections_data/Connections_Data.csv"
    SWOW_CSV = "SWOW-EN18/strength.SWOW-EN.R123.20180827.csv"

    overlap, missing = compare_connections_to_swow(
        connections_csv=CONNECTIONS_CSV,
        swow_csv=SWOW_CSV,
        min_strength=0.05,
        output_path="analysis/connections_swow_overlap.csv"
    )