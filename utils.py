import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch

class TestbedDataset3D(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None, pre_transform=None,
                 smile_graph=None, pocket_graph=None, fingerprint=None,
                 esm_feats=None): # Removed 'surfaces' parameter

        self.dataset = dataset
        self.xd = xd
        self.xt = xt
        self.y = y
        self.smile_graph = smile_graph
        self.pocket_graph = pocket_graph # This will now be the point cloud data
        self.fingerprint = fingerprint
        self.esm_feats = esm_feats

        super(TestbedDataset3D, self).__init__(root, transform, pre_transform)

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        assert (self.xd is not None and self.xt is not None and self.y is not None), "The three lists must be the same length!"
        data_list = []
        data_len = len(self.xd)

        for i in range(data_len):
            print('Converting data pair to graph: {}/{}'.format(i + 1, data_len))
            smiles = self.xd[i]
            key = self.xt[i]
            labels = self.y[i]

            # Process drug data (3D molecular graph)
            c_size, features, edge_index, bond_features, coordinates = self.smile_graph[smiles]
            drug_graph = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                edge_attr=torch.FloatTensor(bond_features),
                coordinates=torch.FloatTensor(coordinates)
            )
            drug_graph.c_size = torch.LongTensor([c_size])
            drug_graph.smiles = smiles
            drug_graph.fingerprint = torch.FloatTensor([self.fingerprint[smiles]])
            
            # Process protein data
            pocket_data = self.pocket_graph[key]
            
            num_residues = torch.FloatTensor(pocket_data[0])
            node_features = torch.Tensor(pocket_data[1])
            edge_index_protein = torch.LongTensor(pocket_data[2])
            residue_coords = torch.FloatTensor(pocket_data[3])
            
            # print(node_features, residue_coords)

            protein_graph = DATA.Data(
                x=node_features,
                edge_index=edge_index_protein,
                pos=residue_coords
            )

            protein_graph.key = key
            target_features = self.esm_feats[key]
            protein_graph.target_features = torch.FloatTensor([target_features])
            
            # 创建包含两个图的复合数据对象
            data = DATA.Data(
                drug_graph=drug_graph,
                protein_graph=protein_graph,
                y=torch.FloatTensor([labels]),
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
class TestbedDatasetHMol(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None, pre_transform=None,
                 smile_graph=None, pocket_graph=None, fingerprint=None, esm_feats=None): 

        self.dataset = dataset
        self.xd = xd
        self.xt = xt
        self.y = y
        self.smile_graph = smile_graph
        self.pocket_graph = pocket_graph # This will now be the point cloud data
        self.fingerprint = fingerprint
        self.esm_feats = esm_feats

        super(TestbedDatasetHMol, self).__init__(root, transform, pre_transform)

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        assert (self.xd is not None and self.xt is not None and self.y is not None), "The three lists must be the same length!"
        data_list = []
        data_len = len(self.xd)

        for i in range(data_len):
            print('Converting data pair to graph: {}/{}'.format(i + 1, data_len))
            smiles = self.xd[i]
            key = self.xt[i]
            labels = self.y[i]

            # Process drug data (3D molecular graph)
            smile_data = self.smile_graph[smiles]
       
            drug_graph = DATA.Data(
                x=torch.LongTensor(smile_data["x"]), 
                edge_index=torch.LongTensor(smile_data["edge_index"]),
                edge_attr=torch.LongTensor(smile_data["edge_attr"]),  
            )
                            
            drug_graph.c_size = torch.LongTensor([smile_data["num_part"]])
            drug_graph.smiles = smiles
            drug_graph.fingerprint = torch.FloatTensor([self.fingerprint[smiles]]) 
            
            # Process protein data
            pocket_data = self.pocket_graph[key]
            
            # num_residues = torch.LongTensor(pocket_data[0])
            node_features = torch.Tensor(pocket_data[1])
            edge_index_protein = torch.LongTensor(pocket_data[2])
            residue_coords = torch.FloatTensor(pocket_data[3])
            
            # print(node_features, residue_coords)

            protein_graph = DATA.Data(
                x=node_features,
                edge_index=edge_index_protein,
                pos=residue_coords
            )

            protein_graph.key = key
            target_features = self.esm_feats[key]
            protein_graph.target_features = torch.FloatTensor([target_features])
            
            # 创建包含两个图的复合数据对象
            data = DATA.Data(
                drug_graph=drug_graph,
                protein_graph=protein_graph,
                y=torch.FloatTensor([labels]),
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
        
# 测试指标
def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def ci(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)
    return CI

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / (float(y_obs_sq * y_pred_sq) + 0.00000001)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / (float(sum(y_pred * y_pred)) + 0.00000001)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / (float(down) + 0.00000001))

def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def rmse_gpu(y, f):
    """Root Mean Squared Error - GPU version"""
    return torch.sqrt(((y - f) ** 2).mean(dim=0))

def mse_gpu(y, f):
    """Mean Squared Error - GPU version"""
    return ((y - f) ** 2).mean(dim=0)

def pearson_gpu(y, f):
    """Pearson correlation coefficient - GPU version"""
    y_mean = torch.mean(y)
    f_mean = torch.mean(f)
    
    y_centered = y - y_mean
    f_centered = f - f_mean
    
    numerator = torch.sum(y_centered * f_centered)
    denominator = torch.sqrt(torch.sum(y_centered ** 2) * torch.sum(f_centered ** 2))
    
    return numerator / (denominator + 1e-8)

# def spearman_gpu(y, f):
#     """Spearman correlation coefficient - GPU version"""
#     # 需要将数据移回CPU以使用scipy的实现
#     y_cpu = y.cpu().numpy()
#     f_cpu = f.cpu().numpy()
    
#     return stats.spearmanr(y_cpu, f_cpu)[0]

def ci_gpu(gt, pred):
    """Concordance index - GPU version"""
    gt_reshaped1 = gt.view(1, -1)
    gt_reshaped2 = gt.view(-1, 1)
    
    pred_reshaped1 = pred.view(1, -1)
    pred_reshaped2 = pred.view(-1, 1)
    
    gt_mask = gt_reshaped1 > gt_reshaped2
    diff = pred_reshaped1 - pred_reshaped2
    
    h_one = (diff > 0).float()
    h_half = (diff == 0).float() * 0.5
    
    return torch.sum(gt_mask.float() * h_one + gt_mask.float() * h_half) / torch.sum(gt_mask.float())

def r_squared_error_gpu(y_obs, y_pred):
    """R-squared error - GPU version"""
    y_obs_mean = torch.mean(y_obs)
    y_pred_mean = torch.mean(y_pred)
    
    covariance = torch.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = covariance * covariance
    
    y_obs_sq = torch.sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = torch.sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    
    return mult / (y_obs_sq * y_pred_sq + 1e-8)

def get_k_gpu(y_obs, y_pred):
    """Get scaling factor k - GPU version"""
    return torch.sum(y_obs * y_pred) / (torch.sum(y_pred * y_pred) + 1e-8)

def squared_error_zero_gpu(y_obs, y_pred):
    """Squared error at origin - GPU version"""
    k = get_k_gpu(y_obs, y_pred)
    
    y_obs_mean = torch.mean(y_obs)
    
    upp = torch.sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = torch.sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    
    return 1 - (upp / (down + 1e-8))

def get_rm2_gpu(ys_orig, ys_line):
    """Get RM2 metric - GPU version"""
    r2 = r_squared_error_gpu(ys_orig, ys_line)
    r02 = squared_error_zero_gpu(ys_orig, ys_line)
    
    diff = (r2 * r2) - (r02 * r02)
    abs_diff = torch.abs(diff)
    return r2 * (1 - torch.sqrt(abs_diff))