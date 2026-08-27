"""Top-level DTA model and custom batching for nested PyG graph objects."""

from typing import Any, List, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData

from .cross_attention import MultiScaleAttention
from .encoder import AtomGNN, ProteinGraphEncoder
from .interaction import BottomUpAtomMotifFusion


VALID_INTERACTION_TYPES = (
    "atom",
    "motif",
    "global",
    "atom_motif",
    "atom_global",
    "motif_global",
    "atom_motif_global",
)


def _interaction_components(interaction_type: str):
    if interaction_type not in VALID_INTERACTION_TYPES:
        raise ValueError(
            "interaction_type must be one of "
            f"{list(VALID_INTERACTION_TYPES)}"
        )
    return frozenset(interaction_type.split("_"))


class DTABatch:
    """Container for drug graphs, protein graphs, globals, and labels."""

    __slots__ = (
        "hetero",
        "protein_graph",
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
    """Batch nested heterogeneous drug graphs and protein residue graphs."""

    return DTABatch(
        hetero=Batch.from_data_list([data.hetero for data in data_list]),
        protein_graph=Batch.from_data_list([data.protein_graph for data in data_list]),
        fingerprint=torch.cat([data.fingerprint for data in data_list], dim=0),
        esm_global=torch.cat([data.esm_global for data in data_list], dim=0),
        y=torch.cat([data.y for data in data_list], dim=0),
        smiles=[getattr(data, "smiles", "") for data in data_list],
        key=[getattr(data, "key", "") for data in data_list],
    )


# Experimental graph-level fusion retained for a future ablation, but inactive
# in the current cross-attention training path.
#
# class GatedFusionLayer(nn.Module):
#     def __init__(self, v_dim, q_dim, output_dim=128, dropout_rate=0.2):
#         super(GatedFusionLayer, self).__init__()
#         self.v_transform = nn.Linear(v_dim, output_dim)
#         self.q_transform = nn.Linear(q_dim, output_dim)
#         self.gate_transform = nn.Linear(output_dim * 2, output_dim)
#         self.activation = nn.Tanh()
#         self.output_dim = output_dim
#
#     def get_output_shape(self):
#         return self.output_dim
#
#     def forward(self, v, q, return_gate=False):
#         v_proj = self.activation(self.v_transform(v))
#         q_proj = self.activation(self.q_transform(q))
#         concat_proj = torch.cat([v_proj, q_proj], dim=1)
#         gate = torch.sigmoid(self.gate_transform(concat_proj))
#         gated_output = gate * v_proj + (1 - gate) * q_proj
#         if return_gate:
#             return gated_output, gate
#         return gated_output


class FinalFCLayers(nn.Module):
    def __init__(
        self,
        in_dim: int,
        binary: int = 1,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        if in_dim < 1 or binary < 1:
            raise ValueError("in_dim and binary must be positive")
        if any(hidden_dim < 1 for hidden_dim in hidden_dims):
            raise ValueError("all hidden dimensions must be positive")

        layers = []
        current_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, binary))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class FusionHead(nn.Module):
    """Fuse only the representations selected by ``interaction_type``."""

    def __init__(
        self,
        interaction_dim: int = 256,
        fp_dim: int = 2048,
        fp_out_dim: int = 256,
        esm_in_dim: int = 1280,
        esm_out_dim: int = 256,
        global_out_dim: int = 256,
        dropout: float = 0.3,
        interaction_type: str = "atom_motif_global",
    ):
        super().__init__()
        self.interaction_type = interaction_type
        self.components = _interaction_components(interaction_type)

        self.fp_proj = nn.Sequential(
            nn.Linear(fp_dim, fp_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.esm_global_proj = nn.Sequential(
            nn.Linear(esm_in_dim, esm_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.global_encoder = FinalFCLayers(
            in_dim=fp_out_dim + esm_out_dim,
            binary=global_out_dim,
            hidden_dims=(512,),
            dropout=dropout,
        )

        fusion_dim = 0
        if "atom" in self.components:
            fusion_dim += interaction_dim
        if "motif" in self.components:
            fusion_dim += interaction_dim
        if "global" in self.components:
            fusion_dim += global_out_dim

        # first_hidden_dim = min(512, fusion_dim)
        # second_hidden_dim = max(128, first_hidden_dim // 2)
        self.mlp = FinalFCLayers(
            in_dim=fusion_dim,
            binary=1,
            hidden_dims=(1024, 256), # 768->1024->256
            dropout=dropout,
        )

    def forward(
        self,
        atom_repr: Tensor,
        motif_repr: Tensor,
        fingerprint: Tensor,
        esm_global: Tensor,
    ) -> Tensor:
        fp_repr = self.fp_proj(fingerprint)
        esm_repr = self.esm_global_proj(esm_global)
        global_repr = self.global_encoder(
            torch.cat([fp_repr, esm_repr], dim=-1)
        )

        fusion_inputs = []
        if "atom" in self.components:
            fusion_inputs.append(atom_repr)
        if "motif" in self.components:
            fusion_inputs.append(motif_repr)
        if "global" in self.components:
            fusion_inputs.append(global_repr)

        return self.mlp(torch.cat(fusion_inputs, dim=-1))


class DTAModel(nn.Module):
    """Hierarchical DTA model with drug-residue cross-attention pooling."""

    def __init__(
        self,
        interaction_type: str = "atom_motif_global",
        atom_in_dim: int = 37,
        atom_edge_dim: int = 13,
        motif_in_dim: int = 50,
        motif_edge_dim: int = 37,
        protein_in_dim: int = 41,
        protein_num_layers: int = 1,
        protein_hidden_dim: int = 256,
        atom_num_layers: int = 1,
        motif_num_layers: int = 1,
        drug_gnn_type: str = "gat",
        graph_pool_type: str = "mean_add_max",
        embed_dim: int = 256,
        num_heads: int = 8,
        fp_dim: int = 2048,
        fp_out_dim: int = 256,
        esm_in_dim: int = 1280,
        esm_out_dim: int = 256,
        global_out_dim: int = 256,
        dropout: float = 0.3,
        use_agg: bool = True,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.interaction_type = interaction_type
        self.use_agg = use_agg

        # ── drug encoder(s) ──────────────────────────────────────────
        self.atom_encoder = AtomGNN(
            in_dim=atom_in_dim,
            edge_dim=atom_edge_dim,
            num_layers=atom_num_layers,
            graph_pool_type=graph_pool_type,
            build_graph_head=False,
            gnn_type=drug_gnn_type,
            node_key="atom",
            edge_key=("atom", "bond", "atom"),
        )

        self.motif_encoder = AtomGNN(
            in_dim=motif_in_dim,
            edge_dim=motif_edge_dim,
            num_layers=motif_num_layers,
            graph_pool_type=graph_pool_type,
            build_graph_head=False,
            gnn_type=drug_gnn_type,
            node_key="motif",
            edge_key=("motif", "connects", "motif"),
        )
        self.atom_motif_fusion = None
        if self.use_agg:
            self.atom_motif_fusion = BottomUpAtomMotifFusion(
                atom_dim=self.atom_encoder.node_out_dim,
                motif_dim=motif_in_dim,
            )

        # ── protein encoder ──────────────────────────────────────────
        self.protein_encoder = ProteinGraphEncoder(
            protein_in_dim=protein_in_dim,
            hidden_dim=protein_hidden_dim,
            num_layers=protein_num_layers,
            graph_pool_type=graph_pool_type,
            protein_edge_dim=10,
            build_graph_head=False,
        )
        self.cross_attention = MultiScaleAttention(
            atom_dim=self.atom_encoder.node_out_dim,
            motif_dim=self.motif_encoder.node_out_dim,
            protein_dim=protein_hidden_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        # ── Separate atom/motif interactions + 256-D global pair head ─────
        self.fusion_head = FusionHead(
            interaction_dim=embed_dim,
            fp_dim=fp_dim,
            fp_out_dim=fp_out_dim,
            esm_in_dim=esm_in_dim,
            esm_out_dim=esm_out_dim,
            global_out_dim=global_out_dim,
            dropout=dropout,
            interaction_type=interaction_type,
        )

    def forward(self, data: DTABatch, return_attention: bool = False):
        atom_h = self.atom_encoder.encode_nodes(data.hetero)
        motif_x = data.hetero["motif"].x
        if self.atom_motif_fusion is not None:
            motif_x = self.atom_motif_fusion(
                atom_h=atom_h,
                motif_x=motif_x,
                membership_edge_index=data.hetero[
                    "atom", "in", "motif"
                ].edge_index,
            )
        motif_h = self.motif_encoder.encode_nodes(
            data.hetero,
            x_override=motif_x,
        )

        protein_h = self.protein_encoder.encode_nodes(
            data.protein_graph.x,
            data.protein_graph.edge_index,
            data.protein_graph.edge_attr,
        )
        atom_repr, motif_repr, attention_info = self.cross_attention(
            atom_h=atom_h,
            atom_batch=data.hetero["atom"].batch,
            motif_h=motif_h,
            motif_batch=data.hetero["motif"].batch,
            protein_h=protein_h,
            protein_batch=data.protein_graph.batch,
            return_attention=return_attention,
        )

        prediction = self.fusion_head(
            atom_repr=atom_repr,
            motif_repr=motif_repr,
            fingerprint=data.fingerprint,
            esm_global=data.esm_global,
        )
        if return_attention:
            return prediction, attention_info
        return prediction

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
        print("── atom_motif_fusion ──")
        if self.atom_motif_fusion is None:
            print("Disabled (use_agg=False)")
        else:
            print(self.atom_motif_fusion)
        print("── protein_encoder ──")
        print(self.protein_encoder)
        print("── cross_attention ──")
        print(self.cross_attention)
        print("── fusion_head ──")
        print(self.fusion_head)

        return {
            "total": count(self),
        }

    def load_state_dict(self, state_dict, strict=True):
        """Load current or legacy checkpoints with renamed protein encoder keys."""
        legacy_prefix = "pocket_encoder."
        if any(key.startswith(legacy_prefix) for key in state_dict):
            state_dict = state_dict.copy()
            for key in list(state_dict):
                if key.startswith(legacy_prefix):
                    new_key = "protein_encoder." + key[len(legacy_prefix):]
                    state_dict[new_key] = state_dict.pop(key)
        return super().load_state_dict(state_dict, strict=strict)
