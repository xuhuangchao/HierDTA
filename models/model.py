import torch
import torch.nn as nn
import torch.nn.functional as F
from .drug_model import *
from .protein_model import *
from torch_geometric.utils import to_dense_batch

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CoAttentionModule(nn.Module):
    """Bidirectional cross-attention with independent Q/K/V projections per direction."""
    def __init__(self, emb_dim, fp_dim=1024, esm_dim=480, dropout=0.2,
                 use_fingerprint=True, use_p_global=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_fingerprint = use_fingerprint
        self.use_p_global = use_p_global

        # D->P direction: drug as Query, protein as Key / Value
        self.drug_q_proj = nn.Linear(emb_dim, emb_dim)
        self.prot_k_proj = nn.Linear(emb_dim, emb_dim)
        self.prot_v_proj = nn.Linear(emb_dim, emb_dim)

        # P->D direction: protein as Query, drug as Key / Value
        self.prot_q_proj = nn.Linear(emb_dim, emb_dim)
        self.drug_k_proj = nn.Linear(emb_dim, emb_dim)
        self.drug_v_proj = nn.Linear(emb_dim, emb_dim)

        # Auxiliary priors
        if self.use_fingerprint:
            self.fp_projection = nn.Sequential(
                nn.Linear(fp_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, emb_dim)
            )
        if self.use_p_global:
            self.esm_projection = nn.Sequential(
                nn.Linear(esm_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, emb_dim)
            )

        # Bilinear interaction + MLP decoder
        drug_in = emb_dim + (emb_dim if use_fingerprint else 0)
        prot_in = emb_dim + (emb_dim if use_p_global else 0)
        self.bilinear = nn.Bilinear(drug_in, prot_in, emb_dim)
        self.predictor = MLPDecoder(emb_dim, 256, 1, dropout=dropout)

    def _masked_softmax(self, scores, mask, dim):
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        return F.softmax(scores, dim=dim)

    def forward(self, h_node_d, batch_d, h_node_p, batch_p, fingerprint, esm_feat):
        # Dense batch packing
        drug_rep, drug_mask = to_dense_batch(h_node_d, batch_d)
        prot_rep, prot_mask = to_dense_batch(h_node_p, batch_p)

        # Q/K/V projections (no activation — standard Transformer-style linear projections)
        drug_q = self.drug_q_proj(drug_rep)
        prot_k = self.prot_k_proj(prot_rep)
        prot_v = self.prot_v_proj(prot_rep)

        prot_q = self.prot_q_proj(prot_rep)
        drug_k = self.drug_k_proj(drug_rep)
        drug_v = self.drug_v_proj(drug_rep)

        # D->P cross-attention: drug queries read protein keys/values
        scores_d2p = torch.bmm(drug_q, prot_k.transpose(1, 2)) / (self.emb_dim ** 0.5)
        attn_d2p = self._masked_softmax(scores_d2p, prot_mask, dim=2)
        drug_cross = torch.bmm(attn_d2p, prot_v)           # (B, N_drug, D) 

        # P->D cross-attention: protein queries read drug keys/values
        scores_p2d = torch.bmm(prot_q, drug_k.transpose(1, 2)) / (self.emb_dim ** 0.5)
        attn_p2d = self._masked_softmax(scores_p2d, drug_mask, dim=2)
        prot_cross = torch.bmm(attn_p2d, drug_v)           # (B, N_prot, D) 

        # Masked mean pooling over cross-attention outputs
        drug_mask_f = drug_mask.float().unsqueeze(-1)
        prot_mask_f = prot_mask.float().unsqueeze(-1)

        # Standard semantics: drug_cross 是 drug 节点的输出 → 池化后为 drug 侧表征
        drug_repr = (drug_cross * drug_mask_f).sum(dim=1) / drug_mask_f.sum(dim=1).clamp(min=1e-9)
        # Standard semantics: prot_cross 是 protein 节点的输出 → 池化后为 protein 侧表征
        prot_repr = (prot_cross * prot_mask_f).sum(dim=1) / prot_mask_f.sum(dim=1).clamp(min=1e-9)

        # Auxiliary priors concatenated to their own side
        drug_final = drug_repr
        if self.use_fingerprint:
            drug_final = torch.cat([drug_final, self.fp_projection(fingerprint)], dim=-1)
        prot_final = prot_repr 
        if self.use_p_global:
            prot_final = torch.cat([prot_final, self.esm_projection(esm_feat)], dim=-1)

        fused = self.bilinear(drug_final, prot_final)
        logits = self.predictor(fused)

        output_dict = {
            'attn_d2p': attn_d2p,
            'attn_p2d': attn_p2d,
            'scores_d2p': scores_d2p,
            'scores_p2d': scores_p2d,
        }
        return logits, output_dict


class HierDTA(nn.Module):
    def __init__(self,
                 num_features_xd=266,
                 drug_hidden=64,
                 drug_out=128,
                 n_layers_drug=2,
                 dropout=0.2,
                 num_features_xt=608,
                 protein_hidden=128,
                 protein_out=128,
                 n_layers_protein=4,
                 use_surface=True,
                 emb_dim=128,
                 fp_dim=1024,
                 esm_dim=480,
                 use_fingerprint=True,
                 use_p_global=True):
        super().__init__()

        assert drug_out == emb_dim, f"Drug out_channels ({drug_out}) must equal emb_dim ({emb_dim})"
        assert protein_out == emb_dim, f"Protein output_dim ({protein_out}) must equal emb_dim ({emb_dim})"

        self.drug_net = DrugModel(
            in_channels=num_features_xd,
            hidden_channels=drug_hidden,
            out_channels=drug_out,
            num_layers=n_layers_drug,
            dropout=dropout
        )
        print(self.drug_net)

        self.protein_net = ProteinEGNN(
            num_features_xt=num_features_xt,
            hidden_nf=protein_hidden,
            output_dim=protein_out,
            n_layers=n_layers_protein,
            use_surface=use_surface
        )
        # print(self.protein_net)

        self.interaction = CoAttentionModule(
            emb_dim=emb_dim,
            fp_dim=fp_dim,
            esm_dim=esm_dim,
            dropout=dropout,
            use_fingerprint=use_fingerprint,
            use_p_global=use_p_global,
        )
        print(self.interaction)

    def forward(self, data):
        h_node_d, batch_d, fingerprint = self.drug_net(data.drug_graph)
        h_node_p, batch_p, esm_feat = self.protein_net(data.protein_graph)

        affinity, output_dict = self.interaction(
            h_node_d, batch_d, h_node_p, batch_p, fingerprint, esm_feat
        )

        if self.training:
            return affinity
        else:
            return affinity, output_dict
