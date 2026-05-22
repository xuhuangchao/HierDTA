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
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        #self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)
        self.dropout = nn.Dropout(0.2)  

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class DualInteractionModule(nn.Module):
    def __init__(self, emb_dim, fp_dim=1024, esm_dim=480, dropout=0.2, 
        # 三个细粒度消融
        use_a2p_attn=True,
        use_m2p_attn=True,
        use_protein_max_pool=True,
        ):
        super().__init__()
        self.emb_dim = emb_dim
        
        # ablation flags
        self.use_a2p_attn = use_a2p_attn
        self.use_m2p_attn = use_m2p_attn
        self.use_protein_max_pool = use_protein_max_pool

        self.fp_projection = nn.Sequential(nn.Linear(fp_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, emb_dim))
        self.esm_projection = nn.Sequential(nn.Linear(esm_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, emb_dim))
        
        self.motif2protein_layer = nn.Linear(emb_dim, emb_dim)
        self.atom2protein_layer = nn.Linear(emb_dim, emb_dim)
        
        self.predictor = MLPDecoder(emb_dim*3, 1024, 256, 1)
    
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
                aggregated_scores = self._masked_mean_pooling(attention_scores, dense_mask)  # TODO:max_pooling/mean_pooling
            
            attention_weights = F.softmax(aggregated_scores, dim=1)
        
        # context_rep = torch.bmm(attention_weights.unsqueeze(1), protein_rep).squeeze(1)
        return attention_scores, attention_weights 
    
    def forward(self, atom_feat, motif_feat, batch_d, node_types, fingerprint, h_node_p, batch_p, esm_feat):
        protein_rep, protein_mask = to_dense_batch(h_node_p, batch_p)
        fingerprint_emb = self.fp_projection(fingerprint)  # [B, emb_dim]
        p_global_embedded = self.esm_projection(esm_feat)  # [B, emb_dim]
        
        batch_size, n_prot, d = protein_rep.size()
        device = protein_rep.device

        # 初始化 attention_weights 为全零
        combined_weights = torch.zeros(batch_size, n_prot, device=device)
        weight_count = torch.zeros(batch_size, 1, device=device)  # 用于做平均
        output_dict = {}
        
        # 药物视角下的蛋白质特征  
        if self.use_a2p_attn:      
            a2p_scores, a2p_attn = self.calculate_context_rep(atom_feat, batch_d, node_types==0, protein_rep, protein_mask, self.atom2protein_layer, False)
            combined_weights += a2p_attn
            weight_count += 1.0
            output_dict["a2p_attn"] = a2p_attn
            output_dict["a2p_scores"] = a2p_scores
        if self.use_m2p_attn:
            m2p_scores, m2p_attn = self.calculate_context_rep(motif_feat, batch_d, node_types==1, protein_rep, protein_mask, self.motif2protein_layer, False)
            combined_weights += m2p_attn
            weight_count += 1.0
            output_dict["m2p_attn"] = m2p_attn
            output_dict["m2p_scores"] = m2p_scores
        
        attention_weights = combined_weights / weight_count  # [B, N]
        multi_level_reps = torch.bmm(attention_weights.unsqueeze(1), protein_rep).squeeze(1)
        
        if self.use_protein_max_pool:
            protein_max_pooled = global_max_pool(h_node_p, batch_p)  # [B, D]
            multi_level_reps = multi_level_reps + protein_max_pooled
        
        final_representation = torch.cat([multi_level_reps, fingerprint_emb, p_global_embedded], dim=1)
        logits = self.predictor(final_representation)
        
        return logits, output_dict

class HierDTA(nn.Module):
    def __init__(self, num_features_xd=47, num_features_xt=608, dropout=0.2, use_a2p_attn=True, use_m2p_attn=True, use_protein_max_pool=True): 
        super().__init__()
        
        # Drug representation network
        self.drug_net = DrugMotifGAT(
            in_channels=num_features_xd,
            hidden_channels=64,
            out_channels=128,
            num_layers=2,
            heads=2,
            dropout=dropout
        )

        self.protein_net = ProteinEGNN(num_features_xt=num_features_xt, hidden_nf=128, output_dim=128, n_layers=4)
        
        # Interaction module 带三种消融
        self.interaction = DualInteractionModule(emb_dim=128, dropout=dropout, use_a2p_attn=use_a2p_attn, use_m2p_attn=use_m2p_attn, use_protein_max_pool=use_protein_max_pool)  # 统一为128才能这样写        

    def forward(self, data):
        # Get drug and protein representations
        atom_feat, motif_feat, batch_d, node_types, fingerprint = self.drug_net(data.drug_graph)
        h_node_p, batch_p, esm_feat = self.protein_net(data.protein_graph)
        
        affinity, output_dict = self.interaction(atom_feat, motif_feat, batch_d, node_types, fingerprint, h_node_p, batch_p, esm_feat)
        
        if self.training:
            return affinity
        else:
            return affinity, output_dict