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
        embeddings: Node embeddings from CompGCN [num_nodes, embedding_dim]
        pos_edge_index: Positive (real) edges [2, num_pos_edges]
        neg_edge_index: Negative (fake) edges [2, num_neg_edges]
        link_predictor: Decoder model or scoring function
    
    Returns:
        loss: Binary cross-entropy loss (scalar)
    """
    pos_scores = link_predictor(embeddings, pos_edge_index)
    neg_scores = link_predictor(embeddings, neg_edge_index)
    
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([
        torch.ones(pos_scores.size(0), device=pos_scores.device),
        torch.zeros(neg_scores.size(0), device=neg_scores.device)
    ], dim=0)
    
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_dine_loss(embeddings, pos_edge_index, neg_edge_index, link_predictor, 
                     orth_coeff=1.0, size_coeff=1.0):
    """
    Compute DINE loss: BCE + orthogonality regularization + size regularization.
    
    Args:
        embeddings: Node embeddings from CompGCN [num_nodes, embedding_dim]
        pos_edge_index: Positive edges [2, num_pos_edges]
        neg_edge_index: Negative edges [2, num_neg_edges]
        link_predictor: Decoder model or scoring function
        orth_coeff: Weight for orthogonality loss
        size_coeff: Weight for size loss
    
    Returns:
        loss: Combined DINE loss (scalar)
    """
    # Apply ReLU to embeddings as required by DINE
    embeddings = F.relu(embeddings)
    
    # Base loss (BCE)
    base_loss = compute_link_prediction_loss(embeddings, pos_edge_index, neg_edge_index, link_predictor)
    
    # Regularization losses
    orth_loss = dine.compute_orthogonality_loss(embeddings)
    size_loss = dine.compute_size_loss(embeddings)
    
    return base_loss + orth_coeff * orth_loss + size_coeff * size_loss


def compute_reconstruction_loss(embeddings, original_features):
    """
    Compute reconstruction loss (MSE between embeddings and original features).
    
    Args:
        embeddings: Learned node embeddings [num_nodes, embedding_dim]
        original_features: Original node features [num_nodes, feature_dim]
    
    Returns:
        loss: MSE loss (scalar)
    """
    target = original_features[:, :embeddings.size(1)]
    return F.mse_loss(embeddings, target)


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

def compute_dine_contrastive_loss( embeddings, pos_edge_index, neg_edge_index, link_predictor, dine_weight=0.5, contrastive_weight=0.5, orth_coeff=1.0, size_coeff=1.0, margin=1.0):
    dine_loss = compute_dine_loss(
        embeddings, pos_edge_index, neg_edge_index, link_predictor,
        orth_coeff=orth_coeff, size_coeff=size_coeff
    )

    pos_scores = link_predictor(embeddings, pos_edge_index)
    neg_scores = link_predictor(embeddings, neg_edge_index)
    contrastive_loss = compute_contrastive_loss(pos_scores, neg_scores, margin=margin)

    total_loss = dine_weight * dine_loss + contrastive_weight * contrastive_loss
    return total_loss

def train_compgcn(data, model, link_predictor, optimizer, device,
                  loss_fn="link_prediction",
                  ortloss_coeff=1.0,
                  sizeloss_coeff=1.0,
                  margin=1.0):
    """
    Training step for CompGCN with configurable loss function.
    
    Args:
        data: PyTorch Geometric Data object containing:
            - x: Node features
            - edge_index: Edge indices
            - edge_type: Edge type labels
        model: CompGCN model
        link_predictor: LinkPredictor model or scoring function
        optimizer: PyTorch optimizer
        device: Device (cpu or cuda)
        loss_fn: One of ["link_prediction", "dine", "contrastive", "reconstruction"]
        ortloss_coeff: Weight for DINE orthogonality loss
        sizeloss_coeff: Weight for DINE size loss
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
    
    # Handle link-based losses (link prediction, DINE, contrastive)
    if loss_fn in ["link_prediction", "dine", "contrastive", "dine_contrastive"]:
        # Generate negative samples
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.x.size(0),
            num_neg_samples=data.edge_index.size(1)
        )
        
        if loss_fn == "link_prediction":
            loss = compute_link_prediction_loss(embeddings, data.edge_index, neg_edge_index, link_predictor)
        
        elif loss_fn == "dine":
            loss = compute_dine_loss(
                embeddings, data.edge_index, neg_edge_index, link_predictor,
                orth_coeff=ortloss_coeff, size_coeff=sizeloss_coeff
            )
        
        elif loss_fn == "contrastive":
            pos_scores = link_predictor(embeddings, data.edge_index)
            neg_scores = link_predictor(embeddings, neg_edge_index)
            loss = compute_contrastive_loss(pos_scores, neg_scores, margin=margin)

        elif loss_fn == "dine_contrastive":
            loss = compute_dine_contrastive_loss(
                embeddings, data.edge_index, neg_edge_index, link_predictor,
                dine_weight=0.7, contrastive_weight=0.3,
                orth_coeff=ortloss_coeff, size_coeff=sizeloss_coeff, margin=margin
            )
    
    # Handle reconstruction loss
    elif loss_fn == "reconstruction":
        loss = compute_reconstruction_loss(embeddings, data.x)
    
    else:
        raise ValueError(
            f"Unknown loss function: {loss_fn}. "
            "Choose from: link_prediction, dine, contrastive, dine_contrastive"
        )

    loss.backward()
    optimizer.step()

    return loss.item()
