"""Top-level hierarchical drug-target affinity model."""

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from cross_attention import AttentionBranch
from encoder import DrugGraphEncoder, ProteinGraphEncoder
from torch_scatter import scatter_mean, scatter_sum


class LocalAugmentation(nn.Module):
    """Local multi-head attention from HimGNN.

    The Query attends to two message sources (fine and coarse) via multi-head
    self-attention, producing an updated representation.  Both Query and the
    two message vectors live in the same hidden space ``hid_dim``.
    """

    def __init__(self, hid_dim, heads):
        super(LocalAugmentation, self).__init__()
        if hid_dim < 1 or heads < 1 or hid_dim % heads != 0:
            raise ValueError("hid_dim must be positive and divisible by heads")
        self.linear_layers = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim, bias=False) for _ in range(3)]
        )
        self.W_o = nn.Linear(hid_dim, hid_dim)
        self.heads = heads
        self.d_k = hid_dim // heads

    def forward(self, fine_messages, coarse_messages, query_features):
        batch_size = fine_messages.shape[0]
        hid_dim = fine_messages.shape[-1]
        Q = query_features
        K = []
        K.append(fine_messages.unsqueeze(1))
        K.append(coarse_messages.unsqueeze(1))
        K = torch.cat(K, dim=1)
        V = K
        Q = Q.view(batch_size, -1, 1, hid_dim).transpose(1, 2)
        K = K.view(batch_size, -1, 1, hid_dim).transpose(1, 2)
        V = K
        Q, K, V = [
            l(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (Q, K, V))
        ]
        message_interaction = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        att_score = F.softmax(message_interaction, dim=-1)
        out = (
            torch.matmul(att_score, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, hid_dim)
        )
        out = self.W_o(out)
        return out.squeeze(1)


class MotifToAtomFusion(nn.Module):
    """Inject motif-level functional-group context into atom embeddings.

    Uses LocalAugmentation: each atom (as Query) attends to two sources:
      * fine_messages   — projected motif semantic of its parent functional group
      * coarse_messages — the atom's own state after atom-level MP.
    """

    def __init__(self, atom_dim=256, motif_dim=256, num_heads=4):
        super().__init__()
        self.la = LocalAugmentation(hid_dim=atom_dim, heads=num_heads)

    def forward(
        self,
        atom_h,
        motif_h,
        membership_edge_index,
        atom_query=None,
    ):
        atom_idx, motif_idx = membership_edge_index

        # fine_messages: each atom receives its parent motif's projected context.
        fine_messages = scatter_mean(motif_h[motif_idx], atom_idx, dim=0, dim_size=atom_h.size(0))
        # fine_messages = scatter_sum(motif_h[motif_idx], atom_idx, dim=0, dim_size=atom_h.size(0))

        # coarse_messages: the atom's own MP state
        coarse_messages = atom_h

        # Dynamic Query: the atom state entering the current GNN layer.
        query = atom_h if atom_query is None else atom_query

        # LA: atom decides how much of its parent motif context to absorb
        atom_messages = self.la(fine_messages, coarse_messages, query)
        return atom_messages


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
    """Fuse atom-residue interaction and global drug-protein representations."""

    def __init__(
        self,
        interaction_dim: int = 256,
        fp_dim: int = 2048,
        fp_out_dim: int = 256,
        esm_in_dim: int = 1280,
        esm_out_dim: int = 256,
        global_out_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
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

        fusion_dim = interaction_dim + global_out_dim

        self.mlp = FinalFCLayers(
            in_dim=fusion_dim,
            binary=1,
            hidden_dims=(1024, 256),
            dropout=dropout,
        )

    def forward(
        self,
        atom_repr: Tensor,
        fingerprint: Tensor,
        esm_global: Tensor,
    ) -> Tensor:
        fp_repr = self.fp_proj(fingerprint)
        esm_repr = self.esm_global_proj(esm_global)
        global_repr = self.global_encoder(
            torch.cat([fp_repr, esm_repr], dim=-1)
        )
        return self.mlp(
            torch.cat([atom_repr, global_repr], dim=-1)
        )


class DTAModel(nn.Module):
    """Hierarchical DTA model.

    Architecture (``use_agg=True``):
        atom GNN ──┐
                   ├→ Motif→Atom LA fusion (per-layer) ──→ atom-residue attn
        motif GNN ─┘                                              │
                                                                  ↓
                                                           FusionHead
                                                                  ↑
        global (ECFP4 + ESM-2) ───────────────────────────────────┘
    """

    def __init__(
        self,
        atom_in_dim: int = 37,
        atom_edge_dim: int = 13,
        motif_in_dim: int = 50,
        motif_edge_dim: int = 37,
        protein_in_dim: int = 41,
        protein_num_layers: int = 2,
        drug_num_layers: int = 2,
        motif_num_layers: int = 1,
        drug_gnn_type: str = "gat",
        hidden_dim: int = 256,
        num_heads: int = 8,
        fp_dim: int = 2048,
        fp_out_dim: int = 256,
        esm_in_dim: int = 1280,
        esm_out_dim: int = 256,
        global_out_dim: int = 256,
        dropout: float = 0.2,
        use_agg: bool = True,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be positive and divisible by num_heads"
            )
        if drug_num_layers < 1:
            raise ValueError("drug_num_layers must be positive")
        self.use_agg = use_agg and drug_num_layers > 1
        if self.use_agg and hidden_dim % 4 != 0:
            raise ValueError(
                "hidden_dim must be divisible by 4 when LocalAugmentation "
                "is enabled"
            )
        self.drug_num_layers = drug_num_layers

        # ── drug encoder(s) ──────────────────────────────────────────
        self.atom_proj = nn.Sequential(
            nn.Linear(atom_in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.motif_proj = None
        self.atom_encoder = DrugGraphEncoder(
            in_dim=hidden_dim,
            edge_dim=atom_edge_dim,
            num_layers=drug_num_layers,
            heads=2,
            gnn_type=drug_gnn_type,
            node_key="atom",
            edge_key=("atom", "bond", "atom"),
        )
        self.motif_encoder = None
        self.agg = None
        if self.use_agg:
            self.motif_proj = nn.Sequential(
                nn.Linear(motif_in_dim, hidden_dim),
                nn.ReLU(),
            )
            self.motif_encoder = DrugGraphEncoder(
                in_dim=hidden_dim,
                edge_dim=motif_edge_dim,
                num_layers=motif_num_layers,
                heads=2,
                gnn_type=drug_gnn_type,
                node_key="motif",
                edge_key=("motif", "connects", "motif"),
            )
            self.agg = MotifToAtomFusion(
                atom_dim=self.atom_encoder.node_out_dim,
                motif_dim=self.motif_encoder.node_out_dim,
                num_heads=4,
            )

        # ── protein encoder ──────────────────────────────────────────
        self.protein_encoder = ProteinGraphEncoder(
            protein_in_dim=protein_in_dim,
            hidden_dim=hidden_dim,
            num_layers=protein_num_layers,
            protein_edge_dim=10,
        )
        self.atom_attention = AttentionBranch(
            drug_dim=self.atom_encoder.node_out_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── fusion head ──────────────────────────────────────────────
        self.fusion_head = FusionHead(
            interaction_dim=hidden_dim,
            fp_dim=fp_dim,
            fp_out_dim=fp_out_dim,
            esm_in_dim=esm_in_dim,
            esm_out_dim=esm_out_dim,
            global_out_dim=global_out_dim,
            dropout=dropout,
        )

    def _encode_drug_nodes(self, hetero):
        atom_h = self.atom_proj(hetero["atom"].x)
        # Motif features are only needed when the motif branch is active.
        motif_h = self.motif_proj(hetero["motif"].x) if self.motif_encoder is not None else None
        
        if self.motif_encoder is not None:
            motif_h = self.motif_encoder.encode_nodes(hetero, motif_h)
        
        for layer_idx in range(self.drug_num_layers):
            # Capture the atom state entering this layer as Query anchor.
            atom_query = atom_h
            atom_h = self.atom_encoder.encode_layer(
                hetero, atom_h, layer_idx
            )
            # Top-down fusion: motif semantics -> atoms
            if self.agg is not None and layer_idx < self.drug_num_layers - 1:
                atom_h = atom_h + self.agg(atom_h, motif_h, hetero["atom", "in", "motif"].edge_index, atom_query) # residual connection

        return atom_h

    def forward(self, data, return_attention: bool = False):
        atom_h = self._encode_drug_nodes(data.hetero)

        attention_info = {} if return_attention else None
        protein_h = self.protein_encoder.encode_nodes(
            data.protein_graph.x,
            data.protein_graph.edge_index,
            data.protein_graph.edge_attr,
        )
        atom_repr, atom_info = self.atom_attention(
            atom_h,
            data.hetero["atom"].batch,
            protein_h,
            data.protein_graph.batch,
            return_attention,
        )
        if return_attention:
            attention_info.update({
                "atom_weights": atom_info["weights"],
                "atom_mask": atom_info["drug_mask"],
                "atom_protein_mask": atom_info["protein_mask"],
            })

        prediction = self.fusion_head(
            atom_repr=atom_repr,
            fingerprint=data.fingerprint,
            esm_global=data.esm_global,
        )
        if return_attention:
            return prediction, attention_info
        return prediction

    def count_parameters(self) -> dict:
        print(self)
        total = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {"total": total}
