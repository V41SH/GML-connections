# GML-connections

Course project for GML where we try solving NYT Connections using Graph ML techniques to generate embeddings.

1. Simple Baseline - baselines/deepwalk.py
2. Advanced Baseline - scripts/compgcn.py
3. Our Contribution - scripts/dine.py
4. Evaluation using cosine similarity - eval.py
5. Evaluation using subgraph - scripts/eval_using_subgraph.py
6. Evaluation for DeepWalk (SWoW dataset) - eval_deepwalk.py

Package manager: uv

Use: uv sync

## Known issues

Building on WSL with Python 3.12.x, fasttext (max supported py version is 3.6) may struggle building. Make sure to run `sudo apt install python3.12-dev` to install the necessary headers for building fasttext.

