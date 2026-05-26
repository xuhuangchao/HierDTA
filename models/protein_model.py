import torch
import torch.nn as nn
from torch_geometric.data import Batch
from .egnn_clean import EGNN

class ProteinEGNN(nn.Module):
    """基于 EGNN 的蛋白质口袋图网络。
    """
    def __init__(self, num_features_xt=None, hidden_nf=128, output_dim=128, n_layers=4, use_surface=True, 
                 residual=True, attention=False, normalize=True, tanh=True):
        super(ProteinEGNN, self).__init__()
        self.use_surface = use_surface

        actual_in_dim = 480 + 128 if use_surface else 480
        self.egnn = EGNN(
            in_node_nf=actual_in_dim,
            hidden_nf=hidden_nf,
            out_node_nf=output_dim,
            in_edge_nf=0,
            n_layers=n_layers,
            attention=attention,
            normalize=normalize,
            tanh=tanh,
            residual=residual
        )
        print(f"ProteinEGNN input dim: {actual_in_dim}")

    def forward(self, protein_graph):
        if isinstance(protein_graph, list):
            protein_graph = Batch.from_data_list(protein_graph)

        h, edge_index, x, batch = (
            protein_graph.x,
            protein_graph.edge_index,
            protein_graph.pos,
            protein_graph.batch
        )

        if self.use_surface:
            h = h[:, 41:]  
        else:
            h = h[:, 41:521]  # 只保留ESM特征，去掉表面特征
        h_out, x_out = self.egnn(h=h, x=x, edges=edge_index, edge_attr=None)

        # ESM特征处理
        target_features = protein_graph.target_features
        if target_features.dim() > 2:
            target_features = target_features.squeeze(1)

        return h_out, batch, target_features
    