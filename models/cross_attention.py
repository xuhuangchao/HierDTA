"""Independent atom–residue and motif–residue interaction branches."""

import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import to_dense_batch


class AttentionBranch(nn.Module):
    """Condition one drug-token scale on protein residue tokens."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        # Optional token-wise FFN retained for ablation reference.
        # self.ffn = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim * 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.Dropout(dropout),
        # )
        # self.ffn_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        count = mask_f.sum(dim=1).clamp_min(1.0)
        return (x * mask_f).sum(dim=1) / count

    def forward(
        self,
        drug_h: Tensor,
        drug_batch: Tensor,
        protein_h: Tensor,
        protein_batch: Tensor,
        return_attention: bool = False,
    ):
        drug_tokens, drug_mask = to_dense_batch(drug_h, drug_batch)
        protein_tokens, protein_mask = to_dense_batch(
            protein_h, protein_batch
        )

        attended, attention_weights = self.cross_attention(
            query=drug_tokens,
            key=protein_tokens,
            value=protein_tokens,
            key_padding_mask=~protein_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        valid_drug = drug_mask.unsqueeze(-1)
        attended = self.attention_norm(drug_tokens + attended)
        attended = attended.masked_fill(~valid_drug, 0.0)
        # attended = self.ffn_norm(attended + self.ffn(attended))
        # attended = attended.masked_fill(~valid_drug, 0.0)

        pooled = self._masked_mean(attended, drug_mask)
        attention_info = None
        if return_attention:
            attention_info = {
                "weights": attention_weights,
                "drug_mask": drug_mask,
                "protein_mask": protein_mask,
            }
        return pooled, attention_info
