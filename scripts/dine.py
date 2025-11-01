import torch
import torch.nn as nn

def compute_orthogonality_loss(h):
    """From https://github.com/simonepiaggesi/dine/blob/main/model.py"""

    partitions = (h * h.sum(axis=0)).T
    O = partitions.matmul(partitions.T)
    I = torch.eye(O.shape[0], device=O.device)
    loss = nn.functional.mse_loss(O/O.norm(), I/I.norm(), reduction="mean")

    return loss

def compute_size_loss(h, EPS=1e-15):
    """From https://github.com/simonepiaggesi/dine/blob/main/model.py"""

    mask = h.T
    axs = torch.arange(mask.dim())        
    mask_size = torch.sum(mask, axis=tuple(axs[1:]))
    mask_norm = mask_size / torch.sum(mask_size, axis=0)
    mask_ent = torch.sum(- mask_norm * torch.log(mask_norm + EPS), axis=0)
    max_ent = torch.log(torch.tensor(mask.shape[0], dtype=torch.float32, device=mask.device))
   
    return max_ent - torch.mean(mask_ent)

def embedding_product(embeddings, edge_index):
    """Replaces link predictor model!"""
    # Retrieve embeddings for source and target nodes
    u = embeddings[edge_index[0]]
    v = embeddings[edge_index[1]]    

    # scores = torch.dot(u,v)
    scores = torch.sum(u * v, dim=1)
    return scores



def get_explanation_subgraph(embeddings, edge_index, dimension_idx, EPS=1e-15):
    """
    Implements Algorithm 1 from the DINE paper.
    """
    num_nodes, embedding_dim = embeddings.shape
    
    # if embedding_dim <= 1:
    #     print("Warning: Cannot compute marginal utility with embedding_dim <= 1.")
    #     # Return an empty subgraph
    #     return torch.tensor([[], []], dtype=edge_index.dtype, device=edge_index.device)

    # mu_d(u,v) = Delta_D(u,v) - Delta_{D\{d}}(u,v)
    
    full_dot_product = embedding_product(embeddings, edge_index)
    delta_D = full_dot_product / embedding_dim
    
    u_emb = embeddings[edge_index[0]]
    v_emb = embeddings[edge_index[1]]

    # Delta_{D\{d}}(u,v) = (1 / (D-1)) * ( (u . v) - u_d * v_d )
    # u_d * v_d for all edges
    dim_d_product = u_emb[:, dimension_idx] * v_emb[:, dimension_idx] # Shape: [num_edges]
    dot_product_minus_d = full_dot_product - dim_d_product   
    delta_D_minus_d = dot_product_minus_d / (embedding_dim - 1)
    mu_d = delta_D - delta_D_minus_d # Shape: [num_edges]
    
    # Filter edges where mu_d > 0
    positive_mask = mu_d > 0
    explanation_edge_index = edge_index[:, positive_mask]
    
    return explanation_edge_index