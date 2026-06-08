"""Top-level DTA model and custom batching for nested PyG graph objects."""

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import global_max_pool
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



# ── 3. Dual-Interaction 预测头 ──
class DTAPocketCrossHead(nn.Module):
    """Atom + Motif → Protein interaction with learned per-residue fusion.

    Atom and motif votes are concatenated (not averaged) and fused by a
    1×1 convolution — each pocket residue gets its own atom/motif blend.
    This preserves both signals rather than diluting motif through averaging
    with overlapping atom copies.
    """
    def __init__(self, emb_dim=128, fp_dim=1024, esm_dim=480, dropout=0.2):
        super().__init__()
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, emb_dim),
        )
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )
        self.atom2prot = nn.Linear(emb_dim, emb_dim)
        self.motif2prot = nn.Linear(emb_dim, emb_dim)
        self.score_fusion = nn.Conv1d(2, 1, kernel_size=1)
        self.mlp = MLPDecoder(emb_dim * 3, 1024, 256, 1, dropout=dropout)

    def _attention(self, layer, query, key, key_mask):
        q = torch.relu(layer(query))
        k = torch.relu(layer(key))
        scores = torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask.unsqueeze(1), -1e9)
        return scores

    def forward(self, atom_feat, atom_batch, motif_feat, motif_batch,
                prot_feat, prot_batch, fingerprint, esm_global):
        atom_d, a_mask = to_dense_batch(atom_feat, atom_batch)
        motif_d, m_mask = to_dense_batch(motif_feat, motif_batch)
        prot_d, p_mask = to_dense_batch(prot_feat, prot_batch)

        a_scores = self._attention(self.atom2prot, atom_d, prot_d, p_mask)
        m_scores = self._attention(self.motif2prot, motif_d, prot_d, p_mask)

        a_pool = (a_scores * a_mask.unsqueeze(-1)).sum(dim=1) \
                 / a_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        m_pool = (m_scores * m_mask.unsqueeze(-1)).sum(dim=1) \
                 / m_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)

        # Per-residue fusion: stack [atom, motif] scores → 1×1 conv → 1 channel
        combined = torch.stack([a_pool, m_pool], dim=1)     # [B, 2, Np]
        combined = self.score_fusion(combined).squeeze(1)   # [B, Np]
        attn = F.softmax(combined, dim=-1)

        weighted_prot = torch.bmm(attn.unsqueeze(1), prot_d).squeeze(1)
        interaction = weighted_prot + global_max_pool(prot_feat, prot_batch)

        return self.mlp(torch.cat([
            interaction,
            self.fp_proj(fingerprint),
            self.esm_proj(esm_global),
        ], dim=-1))


class DTAModel(nn.Module):
    """DTA model: atom GNN + motif GNN + protein EGNN → learnable fusion."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_tasks: int = 1,
        task: str = "regression",
        dropout: float = 0.2,
        atom_in_dim: int = 37,
        motif_in_dim: int = 50,
        pocket_feat_dim: int = 41,
        pocket_num_layers: int = 2,
    ):
        super().__init__()
        self.task = task
        self.drug_encoder = AtomGNN(
            in_dim=atom_in_dim, edge_dim=13,
            node_key="atom", edge_key=("atom", "bond", "atom"),
            dropout=dropout,
        )
        self.motif_encoder = AtomGNN(
            in_dim=motif_in_dim, edge_dim=37,
            node_key="motif", edge_key=("motif", "connects", "motif"),
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
        motif_nodes, motif_batch = self.motif_encoder(data.hetero)
        prot_nodes, prot_batch = self.pocket_encoder(
            data.pocket_graph.x[:, 41:], # Exclude one-hot amino acid type 
            data.pocket_graph.edge_index,
            data.pocket_graph.edge_attr,
            data.pocket_graph.coords,
            data.pocket_graph.batch,
        )
        return self.fusion_head(
            atom_feat=atom_nodes, atom_batch=atom_batch,
            motif_feat=motif_nodes, motif_batch=motif_batch,
            prot_feat=prot_nodes, prot_batch=prot_batch,
            fingerprint=data.fingerprint,
            esm_global=data.esm_global,
        )

    def count_parameters(self) -> dict:
        def count(module):
            return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        print(self.drug_encoder)
        print(self.motif_encoder)
        print(self.pocket_encoder)
        print(self.fusion_head)
                
        return {
            "total": count(self),
        }
