import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from utils import load_graph_from_gml, load_fasttext_embeddings


class CompGCNConv(MessagePassing):
    """
    CompGCN layer that learns compositions of node and relation embeddings.
    """
    def __init__(self, in_channels, out_channels, num_relations, composition='sub'):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.composition = composition
        
        # Separate weight matrices for each direction
        self.W_O = nn.Linear(in_channels, out_channels, bias=False)  # Self-loop
        self.W_I = nn.Linear(in_channels, out_channels, bias=False)  # Incoming edges
        self.W_R = nn.Linear(in_channels, out_channels, bias=False)  # Relation transformation
        
        # Relation embeddings
        self.rel_emb = nn.Parameter(torch.Tensor(num_relations, in_channels))
        
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.xavier_uniform_(self.W_I.weight)
        nn.init.xavier_uniform_(self.W_R.weight)
        nn.init.xavier_uniform_(self.rel_emb)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_type):
        # Add self-loops
        num_nodes = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Extend edge_type with self-loop relation (use relation 0 or a special one)
        self_loop_type = torch.zeros(num_nodes, dtype=torch.long, device=edge_type.device)
        edge_type = torch.cat([edge_type, self_loop_type], dim=0)
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_type=edge_type)
        
        return out + self.bias
    
    def message(self, x_j, edge_type):
        # Get relation embeddings for each edge
        rel = self.rel_emb[edge_type]
        
        # Composition function
        if self.composition == 'sub':
            # Subtraction: h_j - r
            composed = x_j - rel
        elif self.composition == 'mult':
            # Multiplication: h_j * r
            composed = x_j * rel
        elif self.composition == 'corr':
            # Circular correlation (simplified)
            composed = x_j * rel
        else:
            composed = x_j
        
        return self.W_I(composed)
    
    def update(self, aggr_out, x):
        # Combine aggregated messages with self-loop
        return aggr_out + self.W_O(x)


class CompGCN(nn.Module):
    """
    CompGCN model for learning node embeddings with relation composition.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, 
                 num_layers=2, dropout=0.3, composition='sub'):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # CompGCN layers
        self.convs = nn.ModuleList()
        self.convs.append(CompGCNConv(in_channels, hidden_channels, num_relations, composition))
        
        for _ in range(num_layers - 2):
            self.convs.append(CompGCNConv(hidden_channels, hidden_channels, num_relations, composition))
        
        if num_layers > 1:
            self.convs.append(CompGCNConv(hidden_channels, out_channels, num_relations, composition))
    
    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def train_compgcn(data, model, optimizer, device):
    """
    Training step for CompGCN (unsupervised).
    Uses a simple reconstruction loss or can be adapted for specific tasks.
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    embeddings = model(data.x, data.edge_index, data.edge_type)
    
    # Simple unsupervised loss: encourage smooth embeddings
    # You can replace this with task-specific loss (link prediction, classification, etc.)
    loss = F.mse_loss(embeddings, data.x[:, :embeddings.size(1)])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    # Paths
    script_dir = os.path.dirname(__file__)
    gml_path = os.path.join(script_dir, '../graphs/conceptnet_graph.gml')
    output_dir = os.path.join(script_dir, '../graphs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load graph
    nodes, edge_index, edge_type, node_to_idx, relation_to_idx = load_graph_from_gml(gml_path)
    
    # Load FastText embeddings (use saved file if available)
    fasttext_emb_path = os.path.join(output_dir, 'fasttext_node_embeddings.npy')
    if os.path.exists(fasttext_emb_path):
        print(f"Loading saved FastText embeddings from: {fasttext_emb_path}")
        node_features = torch.from_numpy(np.load(fasttext_emb_path))
    else:
        node_features = load_fasttext_embeddings(nodes, embedding_dim=300)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_type=edge_type
    ).to(device)
    
    # Model hyperparameters
    in_channels = 300  # FastText embedding dimension
    hidden_channels = 128
    out_channels = 64
    num_relations = len(relation_to_idx)
    num_layers = 2
    dropout = 0.3
    
    # Initialize model
    model = CompGCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_relations=num_relations,
        num_layers=num_layers,
        dropout=dropout,
        composition='sub'
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Training loop
    print("\nTraining CompGCN...")
    num_epochs = 100
    
    for epoch in range(num_epochs):
        loss = train_compgcn(data, model, optimizer, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Generate final embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, data.edge_type)
        embeddings = embeddings.cpu().numpy()
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, 'compgcn_node_embeddings.npy')
    np.save(embeddings_path, embeddings)
    print(f"\nNode embeddings saved to: {embeddings_path}")
    
    # Save node to index mapping
    import pickle
    mapping_path = os.path.join(output_dir, 'node_to_idx.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump(node_to_idx, f)
    print(f"Node mapping saved to: {mapping_path}")
    
    # Save relation to index mapping
    rel_mapping_path = os.path.join(output_dir, 'relation_to_idx.pkl')
    with open(rel_mapping_path, 'wb') as f:
        pickle.dump(relation_to_idx, f)
    print(f"Relation mapping saved to: {rel_mapping_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
