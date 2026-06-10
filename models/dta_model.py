"""Top-level DTA model and custom batching for nested PyG graph objects."""

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.utils import to_dense_batch

from .encoder import AtomGNN, PocketGraphEncoder


class DTABatch:
    """Container for drug graphs, pocket graphs, globals, and labels."""

    __slots__ = (
        "hetero",
        "pocket_graph",
        "fingerprint",
        "esm_global",
        "y",
        "smiles",
        "key",
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to(self, device: torch.device) -> "DTABatch":
        for attr in self.__slots__:
            value = getattr(self, attr, None)
            if isinstance(value, (Tensor, Data, HeteroData, Batch)):
                setattr(self, attr, value.to(device))
        return self

    def __repr__(self) -> str:
        batch_size = self.y.size(0) if self.y is not None else "?"
        return f"DTABatch(B={batch_size})"


def dta_collate_fn(data_list: List[Any]) -> DTABatch:
    """Batch nested heterogeneous drug graphs and pocket residue graphs."""

    return DTABatch(
        hetero=Batch.from_data_list([data.hetero for data in data_list]),
        pocket_graph=Batch.from_data_list([data.pocket_graph for data in data_list]),
        fingerprint=torch.cat([data.fingerprint for data in data_list], dim=0),
        esm_global=torch.cat([data.esm_global for data in data_list], dim=0),
        y=torch.cat([data.y for data in data_list], dim=0),
        smiles=[getattr(data, "smiles", "") for data in data_list],
        key=[getattr(data, "key", "") for data in data_list],
    )


class MLPDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        binary: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, binary)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


class Interaction(nn.Module):
    """Single-direction hierarchical cross-attention.

    Atom nodes and motif nodes independently query protein residue nodes.
    Protein residues act as memory and are not used as queries.
    """

    def __init__(self, emb_dim: int = 128, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")

        self.atom_to_prot = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.motif_to_prot = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.atom_norm = nn.LayerNorm(emb_dim)
        self.motif_norm = nn.LayerNorm(emb_dim)

    @staticmethod
    def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
        return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)

    def forward(
        self,
        atom_feat: Tensor,
        atom_mask: Tensor,
        motif_feat: Tensor,
        motif_mask: Tensor,
        prot_feat: Tensor,
        prot_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        prot_key_padding = ~prot_mask

        atom_attn, _ = self.atom_to_prot(
            query=atom_feat,
            key=prot_feat,
            value=prot_feat,
            key_padding_mask=prot_key_padding,
            need_weights=False,
        )
        atom_ctx = self.atom_norm(atom_feat + atom_attn)

        motif_attn, _ = self.motif_to_prot(
            query=motif_feat,
            key=prot_feat,
            value=prot_feat,
            key_padding_mask=prot_key_padding,
            need_weights=False,
        )
        motif_ctx = self.motif_norm(motif_feat + motif_attn)

        return (
            self.masked_mean(atom_ctx, atom_mask),
            self.masked_mean(motif_ctx, motif_mask),
        )


class DTAPocketCrossHead(nn.Module):
    """Hierarchical atom/motif-to-protein interaction head."""

    def __init__(self, emb_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fp_proj = nn.Sequential(
            nn.Linear(3239, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, emb_dim),
        )
        self.esm_global_proj = nn.Sequential(
            nn.Linear(480, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )
        self.interaction = Interaction(emb_dim, num_heads=4, dropout=dropout)
        self.mlp = MLPDecoder(emb_dim * 4, 1024, 256, 1, dropout=dropout)

    def forward(
        self,
        atom_feat: Tensor,
        atom_batch: Tensor,
        motif_feat: Tensor,
        motif_batch: Tensor,
        prot_feat: Tensor,
        prot_batch: Tensor,
        fingerprint: Tensor,
        esm_global: Tensor,
    ) -> Tensor:
        atom_d, atom_mask = to_dense_batch(atom_feat, atom_batch)
        motif_d, motif_mask = to_dense_batch(motif_feat, motif_batch)
        prot_d, prot_mask = to_dense_batch(prot_feat, prot_batch)

        atom_ctx, motif_ctx = self.interaction(
            atom_d,
            atom_mask,
            motif_d,
            motif_mask,
            prot_d,
            prot_mask,
        )
        d_readout = self.fp_proj(fingerprint)
        p_readout = self.esm_global_proj(esm_global)

        return self.mlp(torch.cat([atom_ctx, motif_ctx, d_readout, p_readout], dim=-1))


class DTAModel(nn.Module):
    """DTA model: atom/motif GNN + pocket EGNN -> hierarchical interaction."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_tasks: int = 1,
        task: str = "regression",
        dropout: float = 0.2,
        atom_in_dim: int = 37,
        atom_edge_dim: int = 13,
        motif_in_dim: int = 50,
        motif_edge_dim: int = 37,
        pocket_in_dim: int = 608,
        pocket_num_layers: int = 2,
        atom_num_layers: int = 2,
        motif_num_layers: int = 2,
    ):
        super().__init__()
        self.task = task
        self.atom_encoder = AtomGNN(
            in_dim=atom_in_dim,
            hidden_dim=hidden_dim,
            edge_dim=atom_edge_dim,
            num_layers=atom_num_layers,
            node_key="atom",
            edge_key=("atom", "bond", "atom"),
            dropout=dropout,
        )
        self.motif_encoder = AtomGNN(
            in_dim=motif_in_dim,
            hidden_dim=hidden_dim,
            edge_dim=motif_edge_dim,
            num_layers=motif_num_layers,
            node_key="motif",
            edge_key=("motif", "connects", "motif"),
            dropout=dropout,
        )
        self.pocket_encoder = PocketGraphEncoder(
            pocket_in_dim=pocket_in_dim,
            hidden_dim=hidden_dim,
            num_layers=pocket_num_layers,
            dropout=dropout,
        )
        self.fusion_head = DTAPocketCrossHead(
            emb_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, data: DTABatch) -> Tensor:
        atom_nodes, atom_batch = self.atom_encoder(data.hetero)
        motif_nodes, motif_batch = self.motif_encoder(data.hetero)
        prot_nodes, prot_batch = self.pocket_encoder(
            data.pocket_graph.x[:, 41:],
            data.pocket_graph.edge_index,
            data.pocket_graph.edge_attr,
            data.pocket_graph.coords,
            data.pocket_graph.batch,
        )
        return self.fusion_head(
            atom_feat=atom_nodes,
            atom_batch=atom_batch,
            motif_feat=motif_nodes,
            motif_batch=motif_batch,
            prot_feat=prot_nodes,
            prot_batch=prot_batch,
            fingerprint=data.fingerprint,
            esm_global=data.esm_global,
        )

    def count_parameters(self) -> dict:
        def count(module):
            return sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )

        print(self.atom_encoder)
        print(self.motif_encoder)
        print(self.pocket_encoder)
        print(self.fusion_head)

        return {
            "total": count(self),
        }
