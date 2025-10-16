#!/usr/bin/env python3
"""
Test script for phonetic enhancement functionality.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_graphs import load_swow_en18, load_swow_with_phonetics
from utils import phonetic_similarity, phonetic_code


def test_phonetic_functions():
    """Test basic phonetic functionality."""
    print("=== Testing Phonetic Functions ===")

    test_pairs = [
        ("cat", "bat"),
        ("phone", "tone"),
        ("right", "write"),
        ("happy", "sad"),
        ("dog", "log"),
        ("see", "sea"),
    ]

    for word1, word2 in test_pairs:
        similarity = phonetic_similarity(word1, word2)
        code1 = phonetic_code(word1)
        code2 = phonetic_code(word2)
        print(f"'{word1}' ({code1}) <-> '{word2}' ({code2}): {similarity:.3f}")


def test_graph_loading():
    """Test the enhanced graph loading functionality."""
    print("\n=== Testing Graph Loading ===")

    csv_file = "SWOW-EN18/strength.SWOW-EN.R123.20180827.csv"

    try:
        # Test original loader
        print("Testing original loader...")
        data_orig, G_orig, word2idx_orig, idx2word_orig = load_swow_en18(
            csv_file, min_strength=0.1
        )
        print(
            f"Original: {len(word2idx_orig)} nodes, {data_orig.edge_index.shape[1]} edges"
        )
        print(f"Original features shape: {data_orig.x.shape}")

        # Test enhanced loader with high threshold to limit computation
        print("\nTesting enhanced loader...")
        data_enh, G_enh, word2idx_enh, idx2word_enh, stats = load_swow_with_phonetics(
            csv_file,
            min_strength=0.1,
            phonetic_threshold=0.4,  # Very high threshold
        )
        print(
            f"Enhanced: {len(word2idx_enh)} nodes, {data_enh.edge_index.shape[1]} edges"
        )
        print(f"Enhanced features shape: {data_enh.x.shape}")
        print(f"Statistics: {stats}")

        # Test edge types
        if hasattr(data_enh, "edge_type"):
            import torch

            edge_type_counts = torch.bincount(data_enh.edge_type)
            print(f"Edge type distribution: {edge_type_counts}")

        return True

    except FileNotFoundError:
        print(f"Warning: {csv_file} not found. Skipping graph loading test.")
        return False
    except Exception as e:
        print(f"Error during graph loading test: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing phonetic enhancement implementation...\n")

    test_phonetic_functions()

    success = test_graph_loading()

    if success:
        print("\n✓ All tests completed successfully!")
    else:
        print("\n⚠ Some tests were skipped due to missing data files.")


if __name__ == "__main__":
    main()
