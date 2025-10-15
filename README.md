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

## What the Training Results Mean

### Loss Components

- Total Loss: Combined weighted average of contrastive and skip-gram losses
- Contrastive Loss: How well the model distinguishes between augmented views of the same graph
- Skip-gram Loss: How well the model predicts word co-occurrences from random walks

### Training Dynamics

- Decreasing losses = Model is learning meaningful word representations
- Best model saved = Checkpoint with lowest total loss (should generalize best)
- Loss plateauing = Model may have converged or need different hyperparameters
