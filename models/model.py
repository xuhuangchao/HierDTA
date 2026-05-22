import torch
import torch.nn as nn
import torch.nn.functional as F
from .drug_model import *
from .protein_model import *
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool, global_mean_pool

class MLPFusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout_rate=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2

        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, concatenated_features):
        # concatenated_features = torch.cat(feature_streams, dim=1)
        fused_representation = self.fusion_mlp(concatenated_features)
        return fused_representation

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1, dropout=0.2):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        #self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class DualInteractionModule(nn.Module):
    def __init__(self, emb_dim, fp_dim=1024, esm_dim=480, dropout=0.2,
                 use_fingerprint=True, use_p_global=True,
                 use_a2p_attn=True, use_m2p_attn=True, use_protein_max_pool=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_fingerprint = use_fingerprint
        self.use_p_global = use_p_global
        self.use_a2p_attn = use_a2p_attn
        self.use_m2p_attn = use_m2p_attn
        self.use_protein_max_pool = use_protein_max_pool

        if self.use_fingerprint:
            self.fp_projection = nn.Sequential(nn.Linear(fp_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, emb_dim))
        if self.use_p_global:
            self.esm_projection = nn.Sequential(nn.Linear(esm_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, emb_dim))

        if self.use_a2p_attn:
            self.atom2protein_layer = nn.Linear(emb_dim, emb_dim)
        if self.use_m2p_attn:
            self.motif2protein_layer = nn.Linear(emb_dim, emb_dim)

        # Predictor input dimension depends on active branches
        predictor_in_dim = emb_dim
        if self.use_fingerprint:
            predictor_in_dim += emb_dim
        if self.use_p_global:
            predictor_in_dim += emb_dim

        self.predictor = MLPDecoder(predictor_in_dim, 1024, 256, 1, dropout=dropout)

    def _calculate_attention(self, attention_layer, query_rep, key_rep, key_mask=None):
        query_h = torch.relu(attention_layer(query_rep))
        key_h = torch.relu(attention_layer(key_rep))
        d_k = query_h.size(-1)
        scale = d_k ** -0.5
        scores = torch.bmm(query_h, key_h.transpose(1, 2)) * scale
        if key_mask is not None:
            mask_expanded = key_mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)  # [B, N_max_q, N_max_k]
        return scores

    def _masked_mean_pooling(self, scores, query_mask):
        mask_expanded = query_mask.unsqueeze(-1).expand_as(scores)
        sum_scores = (scores * mask_expanded).sum(dim=1)
        num_nodes = torch.clamp(query_mask.sum(dim=1).unsqueeze(-1), min=1e-9)
        mean_scores = sum_scores / num_nodes
        return mean_scores

    def calculate_context_rep(self, query_feat, batch_d, node_mask, protein_rep, protein_mask, attention_layer, is_supernode=False):
        if is_supernode:
            query_rep_dense = query_feat.unsqueeze(1)
            attention_scores = self._calculate_attention(attention_layer, query_rep_dense, protein_rep, key_mask=protein_mask)
            attention_weights = F.softmax(attention_scores.squeeze(1), dim=1)
        else:
            # 从 protein_rep 获取完整的批次大小
            full_batch_size = protein_rep.size(0)
            device = protein_rep.device

            if node_mask is not None:
                batch_d_masked = batch_d[node_mask]
            else:
                batch_d_masked = batch_d

            if query_feat.size(0) == 0:
                # 如果 query_feat 完全为空，我们需要创建正确形状的零张量以避免后续错误
                N_max_k = protein_rep.size(1)
                aggregated_scores = torch.zeros(full_batch_size, N_max_k, device=device)
            else:
                query_rep_dense_active, dense_mask_active = to_dense_batch(query_feat, batch_d_masked)
                N_max_q = query_rep_dense_active.size(1) # 任何一个图里 motif 的最大数量
                D = query_rep_dense_active.size(2)      # 特征维度
                query_rep_dense = torch.zeros(full_batch_size, N_max_q, D, device=device)
                dense_mask = torch.zeros(full_batch_size, N_max_q, dtype=torch.bool, device=device)
                active_batch_indices = torch.unique(batch_d_masked)

                query_rep_dense[active_batch_indices] = query_rep_dense_active[active_batch_indices] # 解决motif节点缺失的情况
                dense_mask[active_batch_indices] = dense_mask_active[active_batch_indices]
                attention_scores = self._calculate_attention(attention_layer, query_rep_dense, protein_rep, key_mask=protein_mask)
                aggregated_scores = self._masked_mean_pooling(attention_scores, dense_mask)

            attention_weights = F.softmax(aggregated_scores, dim=1)

        # context_rep = torch.bmm(attention_weights.unsqueeze(1), protein_rep).squeeze(1)
        return attention_scores, attention_weights

    def forward(self, atom_feat, motif_feat, batch_d, node_types, fingerprint, h_node_p, batch_p, esm_feat):
        protein_rep, protein_mask = to_dense_batch(h_node_p, batch_p)

        # Projections
        if self.use_fingerprint:
            fingerprint = self.fp_projection(fingerprint)
        if self.use_p_global:
            p_global_embedded = self.esm_projection(esm_feat)

        # Cross-attention branches
        attn_list = []
        if self.use_a2p_attn:
            a2p_scores, a2p_attn = self.calculate_context_rep(
                atom_feat, batch_d, node_types==0, protein_rep, protein_mask, self.atom2protein_layer, False
            )
            attn_list.append(a2p_attn)
        if self.use_m2p_attn:
            m2p_scores, m2p_attn = self.calculate_context_rep(
                motif_feat, batch_d, node_types==1, protein_rep, protein_mask, self.motif2protein_layer, False
            )
            attn_list.append(m2p_attn)

        if len(attn_list) > 0:
            attention_weights = torch.stack(attn_list, dim=0).mean(dim=0)
            multi_level_reps = torch.bmm(attention_weights.unsqueeze(1), protein_rep).squeeze(1)
        else:
            # Fallback: mean-pooled protein representation when no cross-attention
            multi_level_reps = global_mean_pool(h_node_p, batch_p)

        if self.use_protein_max_pool:
            protein_max_pooled = global_max_pool(h_node_p, batch_p)
            multi_level_reps = multi_level_reps + protein_max_pooled

        # Build final representation dynamically based on active branches
        reps = [multi_level_reps]
        if self.use_fingerprint:
            reps.append(fingerprint)
        if self.use_p_global:
            reps.append(p_global_embedded)

        final_representation = torch.cat(reps, dim=1)
        logits = self.predictor(final_representation)

        # Collect attention outputs for inspection
        output_dict = {}
        if self.use_a2p_attn:
            output_dict['a2p_attn'] = a2p_attn
            output_dict['a2p_scores'] = a2p_scores
        if self.use_m2p_attn:
            output_dict['m2p_attn'] = m2p_attn
            output_dict['m2p_scores'] = m2p_scores
        return logits, output_dict

class HierDTA(nn.Module):
    def __init__(self,
                 num_features_xd=133,
                 drug_hidden=64,
                 drug_out=128,
                 n_layers_drug=2,
                 heads=2,
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
                 use_p_global=True,
                 use_a2p_attn=True,
                 use_m2p_attn=True,
                 use_protein_max_pool=True):
        super().__init__()

        assert drug_out == emb_dim, f"Drug out_channels ({drug_out}) must equal emb_dim ({emb_dim})"
        assert protein_out == emb_dim, f"Protein output_dim ({protein_out}) must equal emb_dim ({emb_dim})"

        # Drug representation network
        self.drug_net = DrugMotifGAT(
            in_channels=num_features_xd,
            hidden_channels=drug_hidden,
            out_channels=drug_out,
            num_layers=n_layers_drug,
            heads=heads,
            dropout=dropout
        )

        self.protein_net = ProteinEGNN(
            num_features_xt=num_features_xt,
            hidden_nf=protein_hidden,
            output_dim=protein_out,
            n_layers=n_layers_protein,
            use_surface=use_surface
        )

        self.interaction = DualInteractionModule(
            emb_dim=emb_dim,
            fp_dim=fp_dim,
            esm_dim=esm_dim,
            dropout=dropout,
            use_fingerprint=use_fingerprint,
            use_p_global=use_p_global,
            use_a2p_attn=use_a2p_attn,
            use_m2p_attn=use_m2p_attn,
            use_protein_max_pool=use_protein_max_pool
        )


    def forward(self, data):
        # Get drug and protein representations
        atom_feat, motif_feat, batch_d, node_types, fingerprint = self.drug_net(data.drug_graph)
        h_node_p, batch_p, esm_feat = self.protein_net(data.protein_graph)

        affinity, output_dict = self.interaction(atom_feat, motif_feat, batch_d, node_types, fingerprint, h_node_p, batch_p, esm_feat)

        if self.training:
            return affinity
        else:
            return affinity, output_dict
