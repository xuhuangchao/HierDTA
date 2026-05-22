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
import argparse

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
def get_esm_embeddings(sequence, model=None, alphabet=None, max_length=2048, device=None):
    """使用ESM2获取蛋白质序列的嵌入表示
    """
    if device is None:
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
        

def preprocess_and_save_proteins(dataset_name, device=None):
    print(f"开始预处理 {dataset_name} 数据集")
    
    # 创建保存目录（整合到 preprocessed 下）
    base_dir = f'data/{dataset_name}'
    seq_dir = f'{base_dir}/preprocessed/sequence'
    os.makedirs(seq_dir, exist_ok=True)
    
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
                sequence_embedding = get_esm_embeddings(seq, model, alphabet, device=device)
                # 对特征进行均值池化，得到整个序列的表示
                pooled_embedding = np.mean(sequence_embedding, axis=0)
                np.save(npy_path, pooled_embedding)
            except Exception as e:
                print(f"处理序列 {key} 时出错: {e}")
                traceback.print_exc()
        else:
            print("序列文件已存在")
            

    print("预处理完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess protein sequence features')
    parser.add_argument('--dataset', type=str, default='2hyy', help='Dataset name to process')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index to use for ESM inference')
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    preprocess_and_save_proteins(args.dataset, device=device)