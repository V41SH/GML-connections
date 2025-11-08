import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import fasttext
import numpy as np
from torch_geometric.utils import negative_sampling
import dine

def load_graph_from_gml(gml_path):
    """
    Load graph from GML file and prepare PyTorch Geometric Data object.
    """
    print(f"Loading graph from: {gml_path}")
    G = nx.read_gml(gml_path)
    
    # Get nodes in consistent order
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Build edge index and edge types
    edges = []
    edge_types = []
    relations = set()
    
    for u, v, data in G.edges(data=True):
        edges.append([node_to_idx[u], node_to_idx[v]])
        rel = data.get('relation', 'unknown')
        relations.add(rel)
        edge_types.append(rel)
    
    # Create relation to index mapping
    relation_to_idx = {rel: idx for idx, rel in enumerate(sorted(relations))}
    
    # Convert to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor([relation_to_idx[rel] for rel in edge_types], dtype=torch.long)
    
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges, {len(relations)} relation types")
    
    return nodes, edge_index, edge_type, node_to_idx, relation_to_idx

def load_fasttext_embeddings(nodes, embedding_dim=300):
    """
    Load FastText embeddings for the given nodes (words).
    Downloads the model if not already present.
    """
    print("Loading FastText embeddings...")
    
    # Download FastText model (this will cache it locally)
    ft = fasttext.load_model('embedding_models/cc.en.300.bin')
    
    # Get embeddings for each node
    embeddings = []
    for node in nodes:
        # FastText can handle out-of-vocabulary words
        vec = ft.get_word_vector(str(node))
        embeddings.append(vec)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # Save embeddings to file
    out_path = os.path.join(os.path.dirname(__file__), '../graphs/fasttext_node_embeddings.npy')
    np.save(out_path, embeddings)
    print(f"Embeddings saved to: {out_path}")

    return torch.from_numpy(embeddings)


class LinkPredictor(nn.Module):
    """
    Simple decoder for link prediction.
    Takes node embeddings and predicts edge existence.
    """
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        # Two-layer MLP decoder
        self.lin1 = nn.Linear(2 * in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
    
    def forward(self, z, edge_index):
        """
        Args:
            z: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge indices [2, num_edges]
        Returns:
            Edge scores [num_edges]
        """
        # Get embeddings for source and target nodes
        src_emb = z[edge_index[0]]  # [num_edges, embedding_dim]
        dst_emb = z[edge_index[1]]  # [num_edges, embedding_dim]
        
        # Concatenate source and target embeddings
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)  # [num_edges, 2*embedding_dim]
        
        # MLP decoder
        x = self.lin1(edge_emb)
        x = F.relu(x)
        x = self.lin2(x)
        
        return x.squeeze(-1)  # [num_edges]


def compute_link_prediction_loss(embeddings, pos_edge_index, neg_edge_index, link_predictor):
    """
    Compute link prediction loss using positive and negative samples.
    
    Args:
        embeddings: Node embeddings from CompGCN
        pos_edge_index: Positive (real) edges
        neg_edge_index: Negative (fake) edges
        link_predictor: Decoder model / predesigned function
    
    Returns:
        loss: Binary cross-entropy loss
    """
    # Predict scores for positive edges
    pos_scores = link_predictor(embeddings, pos_edge_index)
    
    # Predict scores for negative edges
    neg_scores = link_predictor(embeddings, neg_edge_index)
    
    # Combine scores and labels
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([
        torch.ones(pos_scores.size(0), device=pos_scores.device),
        torch.zeros(neg_scores.size(0), device=neg_scores.device)
    ], dim=0)
    
    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    
    return loss


def compute_reconstruction_loss(embeddings, original_features):
    """
    Simple reconstruction loss (MSE between embeddings and original features).
    This is a placeholder - not recommended for production use.
    
    Args:
        embeddings: Learned node embeddings
        original_features: Original node features
    
    Returns:
        loss: MSE loss
    """
    # Truncate original features to match embedding dimension
    target = original_features[:, :embeddings.size(1)]
    loss = F.mse_loss(embeddings, target)
    return loss



def compute_contrastive_loss(pos_scores, neg_scores, margin=1.0):
    """
    Computes the margin-based ranking loss.
    
    Args:
        pos_scores: Scores for positive edges [num_pos_edges]
        neg_scores: Scores for negative edges [num_neg_edges]
        margin: The desired margin between positive and negative scores
    
    Returns:
        loss: Scalar loss value
    """
    # We want pos_scores > neg_scores + margin
    # Loss = max(0, margin - pos_scores + neg_scores)
    
    # Assuming num_pos_edges == num_neg_edges, as in your training loop
    loss = F.relu(margin - pos_scores + neg_scores)
    return loss.mean()


def train_compgcn(data, model, link_predictor, optimizer, device,
                  loss_fn="link_prediction",
                  ortloss_coeff=1.0,
                  sizeloss_coeff=1.0,
                  margin=1.0  # <-- Add margin as a new parameter
    ):
    """
    Training step for CompGCN with configurable loss function.
    
    Args:
        data: PyTorch Geometric Data object
        model: CompGCN model
        link_predictor: LinkPredictor model or scoring function
        optimizer: Optimizer
        device: Device (cpu or cuda)
        loss_fn: Loss function to use.
        ortloss_coeff: Coefficient for DINE orthogonality loss
        sizeloss_coeff: Coefficient for DINE size loss
        margin: Margin for contrastive loss
    
    Returns:
        loss: Scalar loss value
    """
    model.train()
    if isinstance(link_predictor, LinkPredictor):
        link_predictor.train()
    
    optimizer.zero_grad()

    # Forward pass through CompGCN
    embeddings = model(data.x, data.edge_index, data.edge_type)
    
    # Shared logic for link-based losses
    if loss_fn in ["link_prediction", "dine", "contrastive"]:
        if loss_fn == "dine":
            embeddings = F.relu(embeddings)  # DINE uses ReLU
        
        # Generate negative samples
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.x.size(0),
            num_neg_samples=data.edge_index.size(1),  # Same number as positive edges
        )
        
        # Get scores from the predictor
        pos_scores = link_predictor(embeddings, data.edge_index)
        neg_scores = link_predictor(embeddings, neg_edge_index)

        # --- Compute loss based on selected function ---
        if loss_fn == "link_prediction":
            scores = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([
                torch.ones(pos_scores.size(0), device=pos_scores.device),
                torch.zeros(neg_scores.size(0), device=neg_scores.device)
            ], dim=0)
            loss = F.binary_cross_entropy_with_logits(scores, labels)

        elif loss_fn == "dine":
            scores = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([
                torch.ones(pos_scores.size(0), device=pos_scores.device),
                torch.zeros(neg_scores.size(0), device=neg_scores.device)
            ], dim=0)
            # DINE is BCE loss + regularization
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss += ortloss_coeff * dine.compute_orthogonality_loss(embeddings)
            loss += sizeloss_coeff * dine.compute_size_loss(embeddings)

        elif loss_fn == "contrastive":
            # Use our new contrastive loss function
            loss = compute_contrastive_loss(pos_scores, neg_scores, margin=margin)
    
    elif loss_fn == "reconstruction":
        # Compute reconstruction loss
        loss = compute_reconstruction_loss(embeddings, data.x)
    
    else:
        raise ValueError(f"Unknown loss fn: {loss_fn}. Choose 'link_prediction', 'dine', 'contrastive', or 'reconstruction'.")

    loss.backward()
    optimizer.step()

    return loss.item()
