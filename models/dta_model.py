"""Top-level DTA model and custom batching for nested PyG graph objects."""

from typing import Any, List

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData

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


class GatedDrugFusion(nn.Module):
    """Per-sample, per-dimension gated fusion of atom-level and motif-level drug
    graph representations.

    Given atom_vec [B, D] and motif_vec [B, D], a gate network produces a
    sigmoid mask g ∈ [0,1]^D that soft-selects between the two sources per
    dimension.  The fused output retains dimension D so that downstream
    FusionHead requires no structural change.
    """

    def __init__(self, atom_dim: int = 1024, motif_dim: int = 1024, fused_dim: int = 1024):
        super().__init__()
        self.atom_proj = nn.Linear(atom_dim, fused_dim)
        self.motif_proj = nn.Linear(motif_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(atom_dim + motif_dim, fused_dim),
            nn.Sigmoid(),
        )

    def forward(self, atom_vec: Tensor, motif_vec: Tensor) -> Tensor:
        atom_h = self.atom_proj(atom_vec)     # [B, fused_dim]
        motif_h = self.motif_proj(motif_vec)  # [B, fused_dim]
        g = self.gate(torch.cat([atom_vec, motif_vec], dim=-1))  # [B, fused_dim]
        return g * atom_h + (1.0 - g) * motif_h


class FinalFCLayers(nn.Module):
    def __init__(
        self,
        in_dim: int,
        binary: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, binary),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class FusionHead(nn.Module):
    """Pooled atom/protein/global feature fusion head."""

    def __init__(
        self,
        drug_graph_dim: int = 1024,
        pocket_graph_dim: int = 1024,
        fp_dim: int = 2048,
        fp_out_dim: int = 256,
        esm_in_dim: int = 1280,
        esm_out_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_dim, fp_out_dim),
            nn.BatchNorm1d(fp_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.esm_global_proj = nn.Sequential(
            nn.Linear(esm_in_dim, esm_out_dim),
            nn.BatchNorm1d(esm_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fusion_dim = drug_graph_dim + pocket_graph_dim + fp_out_dim + esm_out_dim
        self.mlp = FinalFCLayers(fusion_dim, 1, dropout=0.5)

    def forward(
        self,
        atom_graph: Tensor,
        prot_graph: Tensor,
        fingerprint: Tensor,
        esm_global: Tensor,
    ) -> Tensor:
        d_readout = self.fp_proj(fingerprint)
        p_readout = self.esm_global_proj(esm_global)

        return self.mlp(torch.cat([atom_graph, prot_graph, d_readout, p_readout], dim=-1))


class DTAModel(nn.Module):
    """DTA model: pooled drug GNN + pooled protein GIN fusion."""

    def __init__(
        self,
        drug_graph_type: str = "atom",
        atom_in_dim: int = 37,
        atom_edge_dim: int = 13,
        motif_in_dim: int = 50,
        motif_edge_dim: int = 37,
        pocket_in_dim: int = 41,
        pocket_num_layers: int = 1,
        pocket_hidden_dim: int = 256,
        pocket_graph_out_dim: int = 1024,
        atom_num_layers: int = 1,
        motif_num_layers: int = 1,
        drug_graph_out_dim: int = 1024,
        fp_dim: int = 2048,
        fp_out_dim: int = 256,
        esm_in_dim: int = 1280,
        esm_out_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        if drug_graph_type not in ("atom", "motif", "dual"):
            raise ValueError("drug_graph_type must be 'atom', 'motif', or 'dual'")

        self.drug_graph_type = drug_graph_type

        # ── drug encoder(s) ──────────────────────────────────────────
        if drug_graph_type in ("atom", "dual"):
            self.atom_encoder = AtomGNN(
                in_dim=atom_in_dim,
                edge_dim=atom_edge_dim,
                num_layers=atom_num_layers,
                graph_out_dim=drug_graph_out_dim,
                node_key="atom",
                edge_key=("atom", "bond", "atom"),
            )
        else:
            self.atom_encoder = None

        if drug_graph_type in ("motif", "dual"):
            self.motif_encoder = AtomGNN(
                in_dim=motif_in_dim,
                edge_dim=motif_edge_dim,
                num_layers=motif_num_layers,
                graph_out_dim=drug_graph_out_dim,
                node_key="motif",
                edge_key=("motif", "connects", "motif"),
            )
        else:
            self.motif_encoder = None

        if drug_graph_type == "dual":
            self.drug_fusion = GatedDrugFusion(
                atom_dim=drug_graph_out_dim,
                motif_dim=drug_graph_out_dim,
                fused_dim=drug_graph_out_dim,
            )
        else:
            self.drug_fusion = None

        # ── protein encoder ──────────────────────────────────────────
        self.pocket_encoder = PocketGraphEncoder(
            pocket_in_dim=pocket_in_dim,
            hidden_dim=pocket_hidden_dim,
            num_layers=pocket_num_layers,
            graph_out_dim=pocket_graph_out_dim,
        )

        # ── fusion head (dimensions unchanged for backward compat) ───
        self.fusion_head = FusionHead(
            drug_graph_dim=drug_graph_out_dim,
            pocket_graph_dim=pocket_graph_out_dim,
            fp_dim=fp_dim,
            fp_out_dim=fp_out_dim,
            esm_in_dim=esm_in_dim,
            esm_out_dim=esm_out_dim,
            dropout=dropout,
        )

    def forward(self, data: DTABatch) -> Tensor:
        if self.drug_graph_type == "dual":
            atom_graph = self.atom_encoder(data.hetero)
            motif_graph = self.motif_encoder(data.hetero)
            drug_graph = self.drug_fusion(atom_graph, motif_graph)
        elif self.drug_graph_type == "motif":
            drug_graph = self.motif_encoder(data.hetero)
        else:
            drug_graph = self.atom_encoder(data.hetero)

        prot_graph = self.pocket_encoder(
            data.pocket_graph.x,
            data.pocket_graph.edge_index,
            data.pocket_graph.batch,
        )
        return self.fusion_head(
            atom_graph=drug_graph,
            prot_graph=prot_graph,
            fingerprint=data.fingerprint,
            esm_global=data.esm_global,
        )

    def count_parameters(self) -> dict:
        def count(module):
            if module is None:
                return 0
            return sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )

        print("── atom_encoder ──")
        if self.atom_encoder is not None:
            print(self.atom_encoder)
        print("── motif_encoder ──")
        if self.motif_encoder is not None:
            print(self.motif_encoder)
        print("── drug_fusion ──")
        if self.drug_fusion is not None:
            print(self.drug_fusion)
        print("── pocket_encoder ──")
        print(self.pocket_encoder)
        print("── fusion_head ──")
        print(self.fusion_head)

        return {
            "total": count(self),
        }
