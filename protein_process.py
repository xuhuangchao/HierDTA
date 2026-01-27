import os
import re
import numpy as np
import pandas as pd
import json
import pickle
from collections import OrderedDict
import time
from tqdm import tqdm
import traceback
import time
import esm
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from itertools import product, groupby, permutations
from scipy.spatial import distance_matrix
import torch
from Bio.PDB import PDBParser
from Bio import SeqIO
import matplotlib.pyplot as plt
from rdkit import Chem
import glob

# 导入必要的常量和函数
METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA', "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", 
         "CE", "PR", "ND", "GD", "TB", "DY", "ER", "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def obtain_resname(res):
    """获取残基名称，处理特殊情况如金属离子"""
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    if resname in METAL:
        return "M"
    else:
        return resname

def obtain_self_dist(res):
    """计算残基内部原子间距离"""
    try:
        xx = res.atoms
        from MDAnalysis.analysis import distances
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1,
                distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except Exception as e:
        print(f"Error calculating self-distances for residue {res.resname}{res.resid}: {e}")
        return [0, 0, 0, 0, 0]

def obtain_dihediral_angles(res):
    """计算残基二面角"""
    try:
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except Exception as e:
        print(f"Error calculating dihedral angles for residue {res.resname}{res.resid}: {e}")
        return [0, 0, 0, 0]
    
def obtain_resname(res):
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    if resname in METAL:
        return "M"
    else:
        return resname

def calc_dist(res1, res2):

    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array


def obtain_edge(u, cutoff=10.0):
    edgeids = []
    dismin = []
    dismax = []
    for res1, res2 in permutations(u.residues, 2):
        dist = calc_dist(res1, res2)
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)
            dismax.append(dist.max() * 0.1)
    return edgeids, np.array([dismin, dismax]).T


# def obtain_ca_pos(res):
#     if obtain_resname(res) == "M":
#         return res.atoms.positions[0]
#     else:
#         try:
#             pos = res.atoms.select_atoms("name CA").positions[0]
#             return pos
#         except:  ##some residues loss the CA atoms
#             return res.atoms.positions.mean(axis=0)

# def check_connect(u, i, j):
#     if abs(i - j) != 1:
#         return 0
#     else:
#         if i > j:
#             i = j
#         nb1 = len(u.residues[i].get_connections("bonds"))
#         nb2 = len(u.residues[i + 1].get_connections("bonds"))
#         nb3 = len(u.residues[i:i + 2].get_connections("bonds"))
#         if nb1 + nb2 == nb3 + 1:
#             return 1
#         else:
#             return 0

def calc_res_features(res):
    """计算残基特征：类型 + 距离 + 二面角"""
    return np.array(one_of_k_encoding_unk(obtain_resname(res),
                                        ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                                         'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                                         'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                         'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +  # 32 残基类型
                obtain_self_dist(res) +  # 5 内部距离
                obtain_dihediral_angles(res)  # 4 二面角
                )
 

def get_aa_code(three_letter_code):
    """将三字母氨基酸代码转换为单字母代码"""
    aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        'MSE': 'M'  # 硒代蛋氨酸通常用M表示
    }
    return aa_dict.get(three_letter_code, 'X')  # 如果未知，返回X
   

# max_length = 2048确保显存不会溢出
def get_esm_embeddings(sequence, model=None, alphabet=None, max_length=2048):
    """使用ESM2获取蛋白质序列的嵌入表示
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 截断序列
    sequence = sequence[:max_length]
    
    # 准备数据
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
    batch_tokens = batch_tokens.to(device)
    
    # 计算嵌入
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12], return_contacts=True)
    
    # 提取嵌入并移除特殊标记
    token_embeddings = results["representations"][12][0, 1:len(sequence)+1].cpu().numpy()
    
    # 清理GPU内存
    del batch_tokens
    del results
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return token_embeddings


# def target_to_graph(protein_pdb, cutoff_distance=10.0, model=None, alphabet=None):
#     try:
#         # 使用RDKit和MDAnalysis加载蛋白质
#         prot = Chem.MolFromPDBFile(protein_pdb, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
#         Chem.AssignStereochemistryFrom3D(prot)
#         u = mda.Universe(prot)
        
#         # 获取残基数量
#         num_residues = len(u.residues)
        
#         sequence = ''.join([get_aa_code(res.resname) for res in u.residues])
#         print("mda sequence:", sequence)
#         esm_feats = get_esm_embeddings(sequence, model, alphabet)
        
#         # 计算残基物理化学特征
#         res_feats = np.array([calc_res_features(res) for res in u.residues])
#         print("res esm shape:",esm_feats.shape)
#         print("res physical shape:", res_feats.shape)
        
#         # 合并节点特征
#         node_features = np.concatenate((res_feats, esm_feats), axis=1)
                
#         # 构建边的连接关系
#         edgeids, distm = obtain_edge(u, cutoff_distance)
#         if len(edgeids) == 0:
#             print("警告: 未找到任何边连接，请检查截断距离或蛋白质结构")
#             return 0, torch.zeros((num_residues, node_features.shape[1])), torch.zeros((2, 0), dtype=torch.long)
            
#         src_list, dst_list = zip(*edgeids)
        
#         # 构建最终返回的边索引
#         edge_index = np.array([src_list, dst_list])
            
#         return num_residues, node_features, edge_index
        
#     except Exception as e:
#         print(f"处理蛋白质时出错: {e}")
#         import traceback
#         traceback.print_exc()
#         return None
    

def get_pocket_file(pocket_dir, key, suffix=".pdb"):
    # 确保目录存在
    if not os.path.exists(pocket_dir):
        print(f"警告: 目录 {pocket_dir} 不存在")
        return None
    
    processed_key = re.sub(r'[.\-() ]', '', key.lower())
    # print(processed_key)
    all_pdb_files = glob.glob(os.path.join(pocket_dir, f"{processed_key}*{suffix}"))
    
    if all_pdb_files:
        return all_pdb_files[0]  # 返回第一个匹配的文件
        
    return None
        

def preprocess_and_save_proteins(dataset_name):
    print(f"开始预处理 {dataset_name} 数据集")
    
    # 创建保存目录
    base_dir = f'data/{dataset_name}'
    seq_dir = f'{base_dir}/sequence'
    graph_dir = f'{base_dir}/pocket_graph'
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    
    # 准备序列到键的映射
    fpath = f'data/{dataset_name}/'
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    
    # 收集所有蛋白质键和序列
    all_keys = list(proteins.keys())
    all_prots = list(proteins.values())
    
    print(f"总共有 {len(all_keys)} 个唯一蛋白质键和 {len(all_prots)} 个唯一蛋白质序列")
    # 加载ESM模型（只需一次）
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t12_35M_UR50D")
    model.eval()
    
    # 1. 处理蛋白质序列特征
    print("提取蛋白质序列特征...")
    for i, (key, seq) in enumerate(tqdm(zip(all_keys, all_prots), desc="处理蛋白质序列", total=len(all_keys))):
        npy_path = f'{seq_dir}/{key}.npy'
        if not os.path.exists(npy_path):
            try:
                print(f"处理序列 {key}...长度为{len(seq)}")
                sequence_embedding = get_esm_embeddings(seq, model, alphabet)
                # 对特征进行均值池化，得到整个序列的表示
                pooled_embedding = np.mean(sequence_embedding, axis=0)
                np.save(npy_path, pooled_embedding)
            except Exception as e:
                print(f"处理序列 {key} 时出错: {e}")
                traceback.print_exc()
        else:
            print("序列文件已存在")
            
            
    # 2. 处理蛋白质口袋图
    # print("构建所有蛋白质口袋图...")
    # pocket_dir = f'data/{dataset_name}/pocket1_{dataset_name}'
    # if not os.path.exists(pocket_dir):
    #     raise FileNotFoundError(f"口袋目录不存在: {pocket_dir}")
    
    # # 构建口袋图并保存为单独的文件
    # for key in tqdm(all_keys, desc="构建蛋白质口袋图"):
    #     npz_path = f'{graph_dir}/{key}.npz'
        
    #     # 获取口袋文件路径
    #     pocket_file = get_pocket_file(pocket_dir, key)
    #     print(f"find {pocket_file}")
    #     if not pocket_file:
    #         print(f"警告: 未找到键 '{key}' 的口袋文件")
    #         continue
                
    #     if not os.path.exists(npz_path):
    #         try:
    #             # 构建靶点图
    #             target_graph = target_to_graph(pocket_file, cutoff_distance=10.0, model=model, alphabet=alphabet)
                
    #             if target_graph is None:
    #                 print(f"未能处理键 '{key}' 的口袋文件，跳过")
    #                 continue
                
    #             # 使用np.savez保存多个数组到一个文件
    #             np.savez(npz_path, 
    #                     num_nodes=target_graph[0], 
    #                     node_features=target_graph[1], 
    #                     edge_index=target_graph[2])
                
    #             # 单次处理后主动清理内存
    #             del target_graph
    #             if torch.cuda.is_available():
    #                 torch.cuda.empty_cache()
    #         except Exception as e:
    #             print(f"处理靶点时出错 (key={key}): {e}")
    #             traceback.print_exc()

    print("预处理完成!")


# preprocess_and_save_proteins('davis')
preprocess_and_save_proteins('2hyy')