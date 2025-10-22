"""
GraphSAGE model implementation for SWOW word embeddings with contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from typing import Optional, Tuple


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder with multiple aggregator types."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        aggregator: str = "mean",
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Final embedding dimension
            num_layers: Number of GraphSAGE layers
            aggregator: Aggregator type ('mean', 'max', 'lstm', 'pool')
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggregator))
        else:
            # Single layer case
            self.convs[0] = SAGEConv(input_dim, output_dim, aggr=aggregator)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through GraphSAGE layers."""

        # Note: SAGEConv doesn't support edge_weight parameter directly
        # We'll ignore edge_weight for now, but could implement weighted sampling later

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)

            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer without activation
        x = self.convs[-1](x, edge_index)

        return x


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GraphSAGEContrastive(nn.Module):
    """GraphSAGE with contrastive learning for word embeddings."""

    def __init__(
        self,
        vocab_size: int,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        projection_dim: int = 64,
        num_layers: int = 2,
        aggregator: str = "mean",
        dropout: float = 0.3,
        temperature: float = 0.1,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            input_dim: Input feature dimension (e.g., fastText embedding size)
            hidden_dim: Hidden dimension for GraphSAGE
            embedding_dim: Final embedding dimension
            projection_dim: Projection head output dimension for contrastive learning
            num_layers: Number of GraphSAGE layers
            aggregator: Aggregator type
            dropout: Dropout rate
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.temperature = temperature

        # GraphSAGE encoder
        self.encoder = GraphSAGEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            aggregator=aggregator,
            dropout=dropout,
        )

        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=projection_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both embeddings and projections.

        Returns:
            embeddings: Node embeddings from GraphSAGE encoder
            projections: Projected embeddings for contrastive learning
        """
        embeddings = self.encoder(x, edge_index, edge_weight)
        projections = self.projection_head(embeddings)

        return embeddings, projections

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get node embeddings without projections."""
        return self.encoder(x, edge_index, edge_weight)


class GraphAugmentor:
    """Graph augmentation for contrastive learning."""

    def __init__(
        self,
        edge_drop_rate: float = 0.2,
        node_feature_mask_rate: float = 0.1,
        edge_weight_jitter: float = 0.1,
    ):
        self.edge_drop_rate = edge_drop_rate
        self.node_feature_mask_rate = node_feature_mask_rate
        self.edge_weight_jitter = edge_weight_jitter

    def augment_graph(self, data: Data) -> Data:
        """Apply random augmentations to graph data."""
        augmented_data = data.clone()

        # Edge dropout (prefer dropping weak edges)
        if self.edge_drop_rate > 0:
            augmented_data = self._edge_dropout(augmented_data)

        # Node feature masking
        if self.node_feature_mask_rate > 0:
            augmented_data = self._node_feature_masking(augmented_data)

        # Edge weight jittering
        if self.edge_weight_jitter > 0 and hasattr(augmented_data, "edge_weight"):
            augmented_data = self._edge_weight_jitter(augmented_data)

        return augmented_data

    def _edge_dropout(self, data: Data) -> Data:
        """Randomly drop edges, preferring weak edges."""
        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            # Random dropout if no edge weights
            num_edges = data.edge_index.size(1)
            keep_mask = torch.rand(num_edges) > self.edge_drop_rate
        else:
            # Probability inversely proportional to edge weight
            edge_weights_norm = data.edge_weight / data.edge_weight.max()
            drop_probs = self.edge_drop_rate * (1 - edge_weights_norm)
            keep_mask = torch.rand_like(drop_probs) > drop_probs

        data.edge_index = data.edge_index[:, keep_mask]
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            data.edge_weight = data.edge_weight[keep_mask]

        return data

    def _node_feature_masking(self, data: Data) -> Data:
        """Randomly mask node features."""
        mask = torch.rand_like(data.x) > self.node_feature_mask_rate
        data.x = data.x * mask.float()
        return data

    def _edge_weight_jitter(self, data: Data) -> Data:
        """Add small noise to edge weights."""
        noise = torch.randn_like(data.edge_weight) * self.edge_weight_jitter
        data.edge_weight = torch.clamp(data.edge_weight + noise, min=0.0)
        return data


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) loss for contrastive learning.

    Args:
        z1, z2: Projected embeddings from two augmented views [batch_size, projection_dim]
        temperature: Temperature scaling parameter

    Returns:
        Contrastive loss
    """
    batch_size = z1.size(0)

    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate representations
    representations = torch.cat([z1, z2], dim=0)  # [2*batch_size, projection_dim]

    # Compute similarity matrix
    sim_matrix = torch.mm(representations, representations.t()) / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    sim_matrix.masked_fill_(mask, -float("inf"))

    # Positive pairs: (0,N), (1,N+1), ..., (N-1, 2N-1), (N,0), (N+1,1), ..., (2N-1, N-1)
    pos_indices = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([pos_indices + batch_size, pos_indices])

    # Compute cross entropy loss
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def skip_gram_loss(
    embeddings: torch.Tensor, pos_samples: torch.Tensor, neg_samples: torch.Tensor
) -> torch.Tensor:
    """
    Skip-gram style loss with negative sampling.

    Args:
        embeddings: Node embeddings [num_nodes, embed_dim]
        pos_samples: Positive samples [num_pos, 2] (source, target pairs)
        neg_samples: Negative samples [num_neg, 2] (source, negative pairs)

    Returns:
        Skip-gram loss
    """
    # Positive samples
    pos_src, pos_dst = pos_samples[:, 0], pos_samples[:, 1]
    pos_embeddings_src = embeddings[pos_src]
    pos_embeddings_dst = embeddings[pos_dst]
    pos_scores = torch.sum(pos_embeddings_src * pos_embeddings_dst, dim=1)
    pos_loss = -F.logsigmoid(pos_scores).mean()

    # Negative samples
    neg_src, neg_dst = neg_samples[:, 0], neg_samples[:, 1]
    neg_embeddings_src = embeddings[neg_src]
    neg_embeddings_dst = embeddings[neg_dst]
    neg_scores = torch.sum(neg_embeddings_src * neg_embeddings_dst, dim=1)
    neg_loss = -F.logsigmoid(-neg_scores).mean()

    return pos_loss + neg_loss
