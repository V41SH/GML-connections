# GML-connections

Course project for GML where we try replacing NYT Connections players with AI 👍

## Setup

Pyg-lib gives me headaches for installing with `uv` in the Ubuntu/WSL envs; it can't discover the right URL because the repository is apparently not PEP-503 compliant. So we add the URL manually in `pyproject.toml`:

Also to install any new dependency, you need to include the `--find-links` option to `uv`. E.g. when adding `tqdm`:

```bash
uv add tqdm pyg-lib --find-links "https://data.pyg.org/whl/torch-2.8.0+cu128.html"
```
