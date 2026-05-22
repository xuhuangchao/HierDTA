import torch
import torch.nn as nn
from torch_geometric.data import Batch
from .egnn_clean import EGNN

class ProteinEGNN(nn.Module):
    """基于 EGNN 的蛋白质口袋图网络"""
    def __init__(self, num_features_xt=608, hidden_nf=128, output_dim=128, n_layers=4, attention=False, normalize=True, tanh=True):  
        super(ProteinEGNN, self).__init__()
        
        self.egnn = EGNN(
            in_node_nf=num_features_xt,
            hidden_nf=hidden_nf,
            out_node_nf=output_dim,
            in_edge_nf=0,
            n_layers=n_layers,
            attention=attention,
            normalize=normalize,
            tanh=tanh,
            residual=True
        )
        print("num_features_xt:", num_features_xt)

    def forward(self, protein_graph):
        if isinstance(protein_graph, list):
            protein_graph = Batch.from_data_list(protein_graph)
            
        h, edge_index, x, batch = (
            protein_graph.x,
            protein_graph.edge_index,
            protein_graph.pos,
            protein_graph.batch
        )
        h = h[:, 41:]  # esm_feat和surface特征
        h_out, x_out = self.egnn(h=h, x=x, edges=edge_index, edge_attr=None)
        
        # ESM特征处理
        target_features = protein_graph.target_features
        if target_features.dim() > 2:
            target_features = target_features.squeeze(1)
        
        return h_out, batch, target_features
    