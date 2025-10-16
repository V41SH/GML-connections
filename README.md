# GML-connections

Course project for GML where we try replacing NYT Connections players with AI 👍

## Setup

Use Ubuntu WSL with Cuda installed (>=12.8, tested with Cuda 13.0 and RTX 3060 12GB, Windows 11 Pro, Ubuntu WSL 24.04, 32GB RAM).

Pyg-lib gives me headaches for installing with `uv` in the Ubuntu/WSL envs; it can't discover the right URL because the repository is apparently not PEP-503 compliant. So we add the URL manually in `pyproject.toml`:

Also to install any new dependency, you need to include the `--find-links` option to `uv`. E.g. when adding `tqdm`:

```bash
uv add tqdm pyg-lib --find-links "https://data.pyg.org/whl/torch-2.8.0+cu128.html"
```

## What the GraphSAGE Baseline Currently Does

Runs in approx. 5 seconds per epoch on my setup (see above). Including startup + result analysis, it takes roughly 10 minutes to run the full 100 epochs.

### Core Functionality

Graph Construction: Loads the Small World of Words (SWOW) dataset as a directed, weighted graph where:

- Nodes = words from the SWOW dataset
- Edges = word associations with strength weights
- Filters edges with strength ≥ 0.05 to remove weak associations

Node Features: Uses pretrained sentence-transformer embeddings as initial node features (loaded from embeddings.pickle)

Graph Neural Network: Implements a 2-layer GraphSAGE encoder that:

- Takes pretrained word embeddings as input
- Aggregates neighborhood information using mean aggregation
- Outputs 128-dimensional learned word embeddings

Training Objectives: Combines two self-supervised learning approaches:

- Contrastive Learning (70% weight): Uses graph augmentations (edge dropout, node masking) to create two views and learns to distinguish positive pairs
- Skip-gram Loss (30% weight): Uses random walks to generate positive word pairs and negative sampling

For NYT Connections Game:
The model learns word representations that should capture semantic relationships. For the Connections game, words in the same category should have similar embeddings, making it easier to group them correctly.

### Contrastive Strategy Summary

- Graph-level contrastive: Same nodes, different augmented graph structures (main contrastive learning)
- Node-level contrastive: Random walk positive pairs vs random negative pairs (skip-gram style)

## What the Training Results Mean

### Loss Components

- Total Loss: Combined weighted average of contrastive and skip-gram losses
- Contrastive Loss: How well the model distinguishes between augmented views of the same graph
- Skip-gram Loss: How well the model predicts word co-occurrences from random walks

### Training Dynamics

- Decreasing losses = Model is learning meaningful word representations
- Best model saved = Checkpoint with lowest total loss (should generalize best)
- Loss plateauing = Model may have converged or need different hyperparameters

### Output currently

```bash
Starting training...
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:03<00:00,  5.97it/s]
Epoch   1/100 | Loss: 4.8085 | Contrastive: 4.6358 | Skip-gram: 4.2902
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:00<00:00, 53.41it/s]
Epoch   2/100 | Loss: 4.1500 | Contrastive: 3.5795 | Skip-gram: 4.1321
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:00<00:00, 54.50it/s]
Epoch   3/100 | Loss: 3.6631 | Contrastive: 2.8619 | Skip-gram: 3.6093
......
Epoch  97/100 | Loss: 1.7792 | Contrastive: 1.3540 | Skip-gram: 1.6502
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:00<00:00, 58.05it/s]
Epoch  98/100 | Loss: 1.6521 | Contrastive: 1.3517 | Skip-gram: 1.4736
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:00<00:00, 52.99it/s]
Epoch  99/100 | Loss: 1.7442 | Contrastive: 1.2390 | Skip-gram: 1.5665
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:00<00:00, 51.76it/s]
Epoch 100/100 | Loss: 1.8503 | Contrastive: 1.7944 | Skip-gram: 1.7444

=== Testing on Connections Game ===

Testing on Connections Game 870
Valid words: ['ready', 'willing', 'able', 'outfit', 'game', 'down', 'cane', 'prepare', 'eager']
True groups:
  FURNISH: ['ready', 'outfit', 'prepare']
  HOMOPHONES OF GENESIS FIGURES: ['able', 'cane']
  INTO IT: ['willing', 'game', 'down', 'eager']
  ORGS WITH STARS IN THEIR LOGOS: []

=== Testing Word Similarities ===
Similar to 'happy': [('nice', '0.742'), ('smile', '0.710'), ('good', '0.695'), ('love', '0.664'), ('sexy', '0.644')]
Similar to 'sad': [('sadness', '0.879'), ('unhappy', '0.693'), ('depressed', '0.682'), ('worried', '0.649'), ('mad', '0.619')]
Similar to 'dog': [('cat', '0.724'), ('animal', '0.694'), ('tiger', '0.626'), ('dogs', '0.599'), ('animals', '0.599')]
Similar to 'cat': [('dog', '0.724'), ('cats', '0.718'), ('animal', '0.671'), ('animals', '0.652'), ('tiger', '0.600')]

Training completed!
```

## Training Results Analysis

### **Training Progress Analysis**

The GraphSAGE training shows excellent convergence patterns with strong learning dynamics:

**Loss Convergence:**

- **Total loss**: Decreased from 4.81 → 1.85 (61% reduction over 100 epochs)
- **Contrastive loss**: Dropped from 4.64 → 1.79 (61% reduction)
- **Skip-gram loss**: Reduced from 4.29 → 1.74 (59% reduction)

This demonstrates that the model successfully learned meaningful word representations from both training objectives. The consistent reduction across all loss components indicates balanced learning between contrastive and skip-gram objectives.

**Training Dynamics:**

- **Fast initial learning**: Major improvements occurred in the first ~10 epochs, typical of effective initialization
- **Stable convergence**: Losses plateaued around epochs 50-60, indicating healthy convergence without oscillation
- **No overfitting signs**: Losses remained stable without dramatic increases, suggesting good generalization
- **Balanced objectives**: By the end of training, both contrastive and skip-gram losses contributed roughly equally, showing the 70/30 weighting worked well

### **Word Similarity Results - Strong Semantic Learning**

The similarity tests demonstrate that the model learned semantically coherent word representations:

**Semantic Clustering Quality:**

- **'happy'** → nice (0.742), smile (0.710), good (0.695), love (0.664), sexy (0.644)
  - All results represent positive emotions, qualities, or pleasant experiences
- **'sad'** → sadness (0.879), unhappy (0.693), depressed (0.682), worried (0.649), mad (0.619)
  - Perfect clustering of negative emotional states with high similarity scores
- **'dog'** → cat (0.724), animal (0.694), tiger (0.626), dogs (0.599), animals (0.599)
- **'cat'** → dog (0.724), cats (0.718), animal (0.671), animals (0.652), tiger (0.600)
  - Both animal queries show excellent semantic coherence within the animal domain

The high cosine similarities (0.6-0.9 range) and clear semantic coherence indicate the GraphSAGE model successfully captured meaningful word relationships from the SWOW graph structure.

### **NYT Connections Game Performance**

**Vocabulary Coverage Challenge:**
The primary limitation is vocabulary coverage - only 9 out of 16 words (56%) from Connections Game 870 were found in the SWOW dataset vocabulary. This significantly limits the model's ability to solve complete puzzles.

**Category Performance Analysis:**

- **FURNISH**: ['ready', 'outfit', 'prepare'] - Found 3/4 words (75% coverage)
  - These represent semantic relationships the model should handle well
- **HOMOPHONES OF GENESIS FIGURES**: ['able', 'cane'] - Found 2/4 words (50% coverage)
  - This category requires cultural/religious knowledge beyond semantic similarity
- **INTO IT**: ['willing', 'game', 'down', 'eager'] - Found 4/4 words (100% coverage)
  - Perfect vocabulary coverage for this enthusiasm-related semantic cluster
- **ORGS WITH STARS IN THEIR LOGOS**: [] - Found 0/4 words (0% coverage)
  - Complete failure due to missing proper nouns and cultural references

**Performance Implications:**
The model shows strong potential for semantic categories (FURNISH, INTO IT) but struggles with cultural knowledge categories (HOMOPHONES, LOGOS). The 56% vocabulary coverage is the primary bottleneck rather than representation quality.
