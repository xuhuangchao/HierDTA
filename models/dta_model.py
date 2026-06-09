"""Top-level DTA model and custom batching for nested PyG graph objects."""

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from .encoder import AtomGNN, PocketGraphEncoder


class DTABatch:
    """Container for drug graphs, pocket graphs, globals, and labels."""

    __slots__ = (
        "hetero",
        "pocket_graph",
        "fingerprint",
        "esm_global",
        "chemberta_tokens",  # [B, max_tokens, 384] padded
        "chemberta_mask",    # [B, max_tokens] bool
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
        chemberta_tokens=torch.stack([data.chemberta_tokens for data in data_list]),
        chemberta_mask=torch.stack([data.chemberta_mask for data in data_list]),
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



# ── 3. 双向交互模块 ──
class BiDirectionalInteraction(nn.Module):
    """Bidirectional cross-attention: drug ⇄ protein.

    Two outputs per interaction: drug_context (drug aware of protein)
    and prot_context (protein aware of drug).  Separate Q/K projections
    for richer expressivity.
    """
    def __init__(self, emb_dim=128):
        super().__init__()
        self.scale = emb_dim ** -0.5
        self.shared_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, drug_feat, d_mask, prot_feat, p_mask):
        Q = self.shared_proj(drug_feat)
        K = self.shared_proj(prot_feat)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = attn.masked_fill(~(d_mask.unsqueeze(2) & p_mask.unsqueeze(1)), -1e9)

        # Drug-aware protein: each residue attends to drug atoms
        drug_context = torch.bmm(F.softmax(attn, dim=2), prot_feat)
        drug_interacted = (drug_context * d_mask.unsqueeze(-1)).sum(1) \
                          / d_mask.sum(1).clamp_min(1).unsqueeze(-1)

        # Protein-aware drug: each atom attends to protein residues
        prot_context = torch.bmm(F.softmax(attn.transpose(1, 2), dim=2), drug_feat)
        prot_interacted = (prot_context * p_mask.unsqueeze(-1)).sum(1) \
                          / p_mask.sum(1).clamp_min(1).unsqueeze(-1)

        return drug_interacted, prot_interacted



# ── 5. 双路径预测头 ──
class DTAPocketCrossHead(nn.Module):
    """Atom ⇄ EGNN bidirectional interaction + fp + esm → MLP."""
    def __init__(self, emb_dim=128, dropout=0.2):
        super().__init__()
        self.struct_interaction = BiDirectionalInteraction(emb_dim)
        self.fp_proj = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, emb_dim))
        self.esm_global_proj = nn.Sequential(
            nn.Linear(480, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, emb_dim))
        self.mlp = MLPDecoder(emb_dim * 4, 1024, 256, 1, dropout=dropout)

    def forward(self, atom_feat, atom_batch, egnn_prot, egnn_batch,
                fingerprint, esm_global):
        atom_d, a_mask = to_dense_batch(atom_feat, atom_batch)
        egnn_d, egnn_mask = to_dense_batch(egnn_prot, egnn_batch)

        d_struct, p_struct = self.struct_interaction(atom_d, a_mask, egnn_d, egnn_mask)

        return self.mlp(torch.cat([
            d_struct, p_struct,
            self.fp_proj(fingerprint), self.esm_global_proj(esm_global),
        ], dim=-1))


class DTAModel(nn.Module):
    """DTA model: atom GNN + ChemBERTa + protein EGNN → 2-way vote fusion."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_tasks: int = 1,
        task: str = "regression",
        dropout: float = 0.2,
        atom_in_dim: int = 37,
        motif_in_dim: int = 50,
        pocket_feat_dim: int = 608,
        pocket_num_layers: int = 2,
        atom_num_layers: int = 2,
        phys_feat_dim: int = 41,
        esm_res_dim: int = 480,
        motif_num_layers: int = 2,
    ):
        super().__init__()
        self.task = task
        self.drug_encoder = AtomGNN(
            in_dim=atom_in_dim, edge_dim=13,
            num_layers=atom_num_layers,
            node_key="atom", edge_key=("atom", "bond", "atom"),
            dropout=dropout,
        )
        self.pocket_encoder = PocketGraphEncoder(
            pocket_in_dim=pocket_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=pocket_num_layers,
            dropout=dropout,
        )
        self.fusion_head = DTAPocketCrossHead(
            emb_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, data: DTABatch) -> Tensor:
        atom_nodes, atom_batch = self.drug_encoder(data.hetero)
        egnn_prot, egnn_batch = self.pocket_encoder(    
            data.pocket_graph.x[:, 41:],
            data.pocket_graph.edge_index,
            data.pocket_graph.edge_attr,
            data.pocket_graph.coords,
            data.pocket_graph.batch,
        )
        return self.fusion_head(
            atom_feat=atom_nodes, atom_batch=atom_batch,
            egnn_prot=egnn_prot, egnn_batch=egnn_batch,
            fingerprint=data.fingerprint,
            esm_global=data.esm_global,
        )

    def count_parameters(self) -> dict:
        def count(module):
            return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        print(self.drug_encoder)
        # print(self.motif_encoder)
        print(self.pocket_encoder)
        print(self.fusion_head)
                
        return {
            "total": count(self),
        }
