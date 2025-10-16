# GraphSAGE Enhancement: Adding Phonetic Similarity

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
