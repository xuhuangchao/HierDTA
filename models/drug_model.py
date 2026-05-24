import torch
import torch.nn as nn
from torch_geometric.data import Batch
from .attentivefp import AttentiveFP

EDGE_DIM = 14  # chemprop bond features dim


class DrugModel(nn.Module):
    """
    PyG 内置 AttentiveFP 药物编码器。
    输入原子图，直接输出分子级别表示 [batch_size, out_channels]。
    """
    def __init__(self, in_channels=266, hidden_channels=128, out_channels=128,
                 num_layers=2, edge_dim=EDGE_DIM, dropout=0.2,
                 heads=None, num_timesteps=2):
        super().__init__()
       
        self.attentive_fp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )

    def forward(self, drug_graph):
        if isinstance(drug_graph, list):
            drug_graph = Batch.from_data_list(drug_graph)

        x = drug_graph.x.float()
        edge_index = drug_graph.edge_index
        edge_attr = drug_graph.edge_attr.float()
        batch = drug_graph.batch
        fingerprint = drug_graph.fingerprint
        if fingerprint.dim() > 2:
            fingerprint = fingerprint.squeeze(1)

        atom_feat = self.attentive_fp(x, edge_index, edge_attr, batch)
        return atom_feat, batch, fingerprint
