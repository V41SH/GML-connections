# GraphSAGE Enhancement: Adding Phonetic Similarity

## On phonetic shit

I put the phonetic weight to 0.4 to get a reasonable number of phonetic edges. It is kind of arbitrary but I wanted <=0.5, so that words like cat and bat are considered similar (they rhyme, so they should be similar). If you set it too high (like 0.9), you get very few edges (only identical words). If you set it too low (like 0.1), you get way too many edges and it takes forever to compute.

When running `uv run test_phonetic_enhancement.py`, you should see output like this: (takes 2 mins to compute)

```sh
Testing phonetic enhancement implementation...

=== Testing Phonetic Functions ===
'cat' (KT) <-> 'bat' (BT): 0.500
'phone' (FN) <-> 'tone' (TN): 0.500
'right' (RT) <-> 'write' (RT): 1.000
'happy' (HP) <-> 'sad' (ST): 0.000
'dog' (TK) <-> 'log' (LK): 0.500
'see' (S) <-> 'sea' (S): 1.000

=== Testing Graph Loading ===
Testing original loader...
Original: 7563 nodes, 8832 edges
Original features shape: torch.Size([7563, 7563])

Testing enhanced loader...
Loading SWOW data with phonetic enhancement...
Filtered to 7562 valid string words from 7563 total entries
Computing phonetic edges for 7562 words with threshold 0.4...
Warning: 28588141 total pairs. Using sampling to limit computation.
Comparing 447 words for phonetic similarity...
  Processed 0/447 words...
Found 15494 phonetic edges
Computing phonetic node features...
Enhanced graph statistics:
  Nodes: 7563
  SWOW edges: 8832
  Phonetic edges: 15494
  Total edges: 24326
  Node feature dimension: 7588
Enhanced: 7563 nodes, 24326 edges
Enhanced features shape: torch.Size([7563, 7588])
Statistics: {'num_swow_edges': 8832, 'num_phonetic_edges': 15494, 'total_edges': 24326, 'phonetic_threshold': 0.4, 'phonetic_feature_dim': 22, 'total_feature_dim': 7588}
Edge type distribution: tensor([ 8832, 15494])

✓ All tests completed successfully!
```

## Objective

Extend the current GraphSAGE implementation on SWOW network to incorporate phonetic similarity between words as additional graph structure. Currently only uses R1.strength as edge weights - goal is to add phonetic edges and leverage this information in the GNN model.

## Option A: Enhanced Node Features (Recommended)

- **Graph Construction**: Create hybrid graph with both association edges (SWOW) and phonetic edges (similarity > threshold)
- **Edge Types**: Add `edge_type` attribute to distinguish association vs phonetic edges
- **Node Features**: Augment existing node features with:
  - Phonetic embedding vectors
  - Phonetic edge count per node
  - Association edge count per node
  - Phonetic/association connectivity ratio
- **Model**: Use existing GraphSAGE architecture - learns phonetic patterns implicitly through enhanced node features
- **Complexity**: Low - leverages existing GraphSAGE strengths

## Option B: Edge-Aware GraphSAGE

- **Graph Construction**: Same hybrid graph as Option A
- **Model Architecture**: Implement edge-type-aware message passing
  - Separate aggregation functions per edge type
  - Optional edge-type-specific attention mechanisms
  - Edge type embedding layers
- **Message Passing**: Explicitly handle different edge types during neighbor aggregation
- **Complexity**: High - requires custom GraphSAGE implementation

## Expansion Effort (A → B)

- **Reusable (~70%)**: Graph construction, data loading, training loops, phonetic computation
- **New Implementation (~30%)**: Edge-aware message passing, separate aggregators, modified forward pass
- **Estimated Time**: 4-7 days (2-3 implementation, 1-2 testing, 1-2 tuning)
- **Risk**: Low-Medium - well-documented approach with existing infrastructure

**Strategy**: Start with Option A for quick validation, expand to Option B if performance gains justify complexity.
