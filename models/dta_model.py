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
    """DTA model: pooled atom GNN + pooled protein GIN fusion."""

    def __init__(
        self,
        atom_in_dim: int = 37,
        atom_edge_dim: int = 13,
        pocket_in_dim: int = 41,
        pocket_num_layers: int = 1,
        pocket_hidden_dim: int = 256,
        pocket_graph_out_dim: int = 1024,
        atom_num_layers: int = 1,
        drug_graph_out_dim: int = 1024,
        fp_dim: int = 2048,
        fp_out_dim: int = 256,
        esm_in_dim: int = 1280,
        esm_out_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.atom_encoder = AtomGNN(
            in_dim=atom_in_dim,
            edge_dim=atom_edge_dim,
            num_layers=atom_num_layers,
            graph_out_dim=drug_graph_out_dim,
            node_key="atom",
            edge_key=("atom", "bond", "atom"),
        )
        self.pocket_encoder = PocketGraphEncoder(
            pocket_in_dim=pocket_in_dim,
            hidden_dim=pocket_hidden_dim,
            num_layers=pocket_num_layers,
            graph_out_dim=pocket_graph_out_dim,
        )
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
        atom_graph = self.atom_encoder(data.hetero)
        prot_graph = self.pocket_encoder(
            data.pocket_graph.x,
            data.pocket_graph.edge_index,
            data.pocket_graph.batch,
        )
        return self.fusion_head(
            atom_graph=atom_graph,
            prot_graph=prot_graph,
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
        print(self.pocket_encoder)
        print(self.fusion_head)

        return {
            "total": count(self),
        }
