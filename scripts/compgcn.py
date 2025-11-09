import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from utils import (
    load_graph_from_gml,
    load_fasttext_embeddings,
    LinkPredictor,
    train_compgcn,
)
from tqdm import tqdm
from time import time
import pickle

import dine


class CompGCNConv(MessagePassing):
    """
    CompGCN layer that learns compositions of node and relation embeddings.
    """

    def __init__(self, in_channels, out_channels, num_relations, composition="sub"):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.composition = composition

        # Separate weight matrices for each direction
        self.W_O = nn.Linear(in_channels, out_channels, bias=False)  # Self-loop
        self.W_I = nn.Linear(in_channels, out_channels, bias=False)  # Incoming edges
        self.W_R = nn.Linear(
            in_channels, out_channels, bias=False
        )  # Relation transformation

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
        self_loop_type = torch.zeros(
            num_nodes, dtype=torch.long, device=edge_type.device
        )
        edge_type = torch.cat([edge_type, self_loop_type], dim=0)

        # Message passing
        out = self.propagate(edge_index, x=x, edge_type=edge_type)

        return out + self.bias

    def message(self, x_j, edge_type):
        # Get relation embeddings for each edge
        rel = self.rel_emb[edge_type]

        # Composition function
        if self.composition == "sub":
            # Subtraction: h_j - r
            composed = x_j - rel
        elif self.composition == "mult":
            # Multiplication: h_j * r
            composed = x_j * rel
        elif self.composition == "corr":
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

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_relations,
        num_layers=2,
        dropout=0.3,
        composition="sub",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # CompGCN layers
        self.convs = nn.ModuleList()
        self.convs.append(
            CompGCNConv(in_channels, hidden_channels, num_relations, composition)
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                CompGCNConv(
                    hidden_channels, hidden_channels, num_relations, composition
                )
            )

        if num_layers > 1:
            self.convs.append(
                CompGCNConv(hidden_channels, out_channels, num_relations, composition)
            )

        print(
            f"Successfully initialized CompGCN with {num_layers} layers and '{composition}' composition."
        )

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def main():
    t1 = time()
    # Paths
    script_dir = os.path.dirname(__file__)
    gml_path = os.path.join(script_dir, "../graphs/conceptnet_graph.gml")
    output_dir = os.path.join(script_dir, "../graphs")
    os.makedirs(output_dir, exist_ok=True)

    # Device

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gabriel's stupid prompt
    if torch.cuda.is_available():
        print("!!!!")
        answer = input(
            "USER PROMPT: WOULD YOU LIKE TO USE CUDA?[type anything but 'no']> "
        )
        if answer == "no":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load graph
    nodes, edge_index, edge_type, node_to_idx, relation_to_idx = load_graph_from_gml(
        gml_path
    )
    t2 = time()

    print(f"Graph loaded in {t2 - t1:.2f} seconds with {len(nodes):,} nodes.")

    # Load FastText embeddings (use saved file if available)
    fasttext_emb_path = os.path.join(output_dir, "fasttext_node_embeddings.npy")
    if os.path.exists(fasttext_emb_path):
        print(f"Loading saved FastText embeddings from: {fasttext_emb_path}")
        try:
            loaded_emb = np.load(fasttext_emb_path)
            # Verify shape matches current graph
            if loaded_emb.shape[0] == len(nodes):
                node_features = torch.from_numpy(loaded_emb)
                print(f"Successfully loaded embeddings with shape: {loaded_emb.shape}")
            else:
                print(
                    f"Shape mismatch: saved={loaded_emb.shape[0]} nodes, current={len(nodes)} nodes"
                )
                print("Regenerating embeddings...")
                node_features = load_fasttext_embeddings(nodes, embedding_dim=300)
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            print("Regenerating embeddings...")
            node_features = load_fasttext_embeddings(nodes, embedding_dim=300)
    else:
        node_features = load_fasttext_embeddings(nodes, embedding_dim=300)

    t3 = time()
    print(
        f"Node features prepared in {t3 - t2:.2f} seconds. Total elapsed time: {t3 - t1:.2f} seconds."
    )
    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_type=edge_type).to(device)

    # Model hyperparameters
    in_channels = 300  # FastText embedding dimension
    hidden_channels = 128
    out_channels = 64
    num_relations = len(relation_to_idx)
    num_layers = 2
    dropout = 0.3

    # Training configuration
    # loss_function = "link_prediction"  # Options: 'link_prediction', 'reconstruction', 'dine', 'contrastive'
    loss_function = "dine"
    num_epochs = 100
    learning_rate = 0.01
    weight_decay = 5e-4
    margin = 1.0

    # Initialize CompGCN model
    model = CompGCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_relations=num_relations,
        num_layers=num_layers,
        dropout=dropout,
        composition="sub",
    ).to(device)

    # Initialize LinkPredictor (decoder) if using link prediction loss
    link_predictor = None
    if loss_function == "link_prediction" or loss_function == "contrastive":
        link_predictor = LinkPredictor(in_channels=out_channels, hidden_channels=64).to(
            device
        )
        print(f"Initialized LinkPredictor for {loss_function} loss.")
    elif loss_function == "dine" or loss_function == "dine_contrastive":
        link_predictor = dine.embedding_product
        print("Using 'dine.embedding_product' as link predictor.")

    # Optimizer - optimize both model and link predictor parameters
    params = list(model.parameters())
    # if link_predictor is not None:
    if isinstance(link_predictor, LinkPredictor):
        params += list(link_predictor.parameters())

    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    t4 = time()
    print(
        f"Model initialized in {t4 - t3:.2f} seconds. Total elapsed time: {t4 - t1:.2f} seconds."
    )

    # Training loop
    print(f"\nTraining CompGCN with '{loss_function}' loss...")

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        loss = train_compgcn(
            data,
            model,
            link_predictor,
            optimizer,
            device,
            loss_fn=loss_function,
            margin=margin,
            ortloss_coeff=5.0
        )
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    t5 = time()
    print(
        f"Training completed in {t5 - t4:.2f} seconds. Total elapsed time: {t5 - t1:.2f} seconds."
    )
    # Generate final embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, data.edge_type)
        embeddings = embeddings.cpu().numpy()

    t6 = time()
    print(
        f"Embeddings generated in {t6 - t5:.2f} seconds. Total elapsed time: {t6 - t1:.2f} seconds."
    )
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "compgcn_node_embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"\nNode embeddings saved to: {embeddings_path}")

    mapping_path = os.path.join(output_dir, "node_to_idx.pkl")
    with open(mapping_path, "wb") as f:
        pickle.dump(node_to_idx, f)
    print(f"Node mapping saved to: {mapping_path}")

    # Save relation to index mapping
    rel_mapping_path = os.path.join(output_dir, "relation_to_idx.pkl")
    with open(rel_mapping_path, "wb") as f:
        pickle.dump(relation_to_idx, f)
    print(f"Relation mapping saved to: {rel_mapping_path}")

    t7 = time()
    print(f"\nTraining complete! Total elapsed time: {t7 - t1:.2f} seconds.")


if __name__ == "__main__":
    main()
