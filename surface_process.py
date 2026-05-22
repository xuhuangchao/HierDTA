import os
import subprocess
import glob
import json
import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from openbabel import openbabel
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from scipy.spatial import KDTree
from easydict import EasyDict
from rdkit import Chem
import MDAnalysis as mda
import yaml
import re
import argparse

# 导入dMaSIF相关模块
from dmasif_encoder.protein_surface_encoder import dMaSIF
from dmasif_encoder.geometry_processing import atoms_to_points_normals

class ProteinSurfacePreprocessor:
    def __init__(self, dataset_name, k=5):
        self.dataset_name = dataset_name
        self.fpath = f'data/{dataset_name}/'
        self.pocket_dir = f'{self.fpath}pocket1_{dataset_name}'
        self.k = k

        # 设置随机种子
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # 加载配置和模型
        self.device = "cpu"
        with open("config.yml", 'r') as f:
            config = EasyDict(yaml.safe_load(f))
        self.net = dMaSIF(config.model.dmasif).to(self.device)
        self.net.eval()

        # 元素到数字的映射
        self.ele2num = {'H':0, 'LI':1, 'C':2, 'N':3, 'O':4, 'NA':5, 'MG':6, 'P':7, 'S':8,
                       'K':9, 'CA':10, 'MN':11, 'FE':12, 'CO':13, 'NI':14, 'CU':15, 'ZN':16,
                       'SE':17, 'SR':18, 'CD':19, 'CS':20, 'HG':21}

        # 加载蛋白质数据
        self.proteins = json.load(open(self.fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
        self.all_keys = list(self.proteins.keys())

        # 创建输出目录（整合到 preprocessed 下）
        self.preprocessed_dir = f'{self.fpath}preprocessed'
        self.surface_dir = f'{self.preprocessed_dir}/surface_points'
        self.map_dir = f'{self.preprocessed_dir}/residue_surface/k{k}'
        os.makedirs(self.surface_dir, exist_ok=True)
        os.makedirs(self.map_dir, exist_ok=True)


    def get_pocket_file(self, key, suffix=".pdb"):
        """根据蛋白质键名找到对应的PDB口袋文件"""
        if not os.path.exists(self.pocket_dir):
            print(f"警告: 目录 {self.pocket_dir} 不存在")
            return None
        
        processed_key = re.sub(r'[.\-() ]', '', key.lower())
        all_pdb_files = glob.glob(os.path.join(self.pocket_dir, f"{processed_key}*{suffix}"))
        
        return all_pdb_files[0] if all_pdb_files else None

    def load_structure_np(self, fname):
        """加载蛋白原子信息"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", fname)
        atoms = structure.get_atoms()

        coords = []
        types = []
        for atom in atoms:
            coords.append(atom.get_coord())
            element = atom.element.upper()
            if element in self.ele2num:
                types.append(self.ele2num[element])
            else:
                print(f"警告: 未知元素 {element}, 跳过")
                continue

        if not coords:
            raise ValueError("未找到有效的原子坐标")

        coords = np.stack(coords)
        types_array = np.zeros((len(types), 22))
        for i, t in enumerate(types):
            types_array[i, t] = 1.0

        return coords, types_array

    def load_protein_atoms(self, protein_path):
        """加载蛋白质原子数据"""
        try:
            atom_coords, atom_types = self.load_structure_np(protein_path)
        except Exception as e:
            print(f"加载蛋白质原子数据失败: {e}")
            return None

        protein_data = Data(
            atom_coords=torch.tensor(atom_coords, dtype=torch.float32),
            atom_types=torch.tensor(atom_types, dtype=torch.float32),
        )
        return protein_data

    def extract_single(self, P_batch, number):
        """提取单个批次的表面信息"""
        P = {}  
        suface_batch = P_batch["batch"] == number
        P["batch"] = P_batch["batch"][suface_batch]
        P["xyz"] = P_batch["xyz"][suface_batch]
        P["normals"] = P_batch["normals"][suface_batch]
        return P

    def select_pocket(self, P_batch, ligand_center):
        """选择口袋区域的表面点"""
        surface_list = []
        batch_list = []
        normal_list = []
        protein_batch_size = P_batch["batch_atoms"][-1].item() + 1
        
        for i in range(protein_batch_size):
            P = self.extract_single(P_batch, i)
            distances = torch.norm(P["xyz"] - ligand_center[i].squeeze(), dim=1)
            sorted_indices = torch.argsort(distances)
            point_nums = 512
            closest_protein_indices = sorted_indices[:point_nums]

            surface_list.append(P["xyz"][closest_protein_indices])
            normal_list.append(P["normals"][closest_protein_indices])
            batch_list.append(P["batch"][:closest_protein_indices.shape[0]])

        p_xyz = torch.cat(surface_list, dim=0)
        p_batch = torch.cat(batch_list, dim=0)
        p_normals = torch.cat(normal_list, dim=0)

        return p_xyz, p_normals, p_batch

    def process_surface(self, protein_single, ligand_center):
        """处理蛋白质表面"""
        P = {}
        P["atoms"] = protein_single.atom_coords
        P["atomtypes"] = protein_single.atom_types
        P["atom_xyz"] = protein_single.atom_coords
        N = P["atoms"].shape[0]
        P["batch_atoms"] = torch.zeros(N, dtype=torch.long)

        P["xyz"], P["normals"], P["batch"] = atoms_to_points_normals(
            P["atoms"], P["batch_atoms"], atomtypes=P["atomtypes"],
            num_atoms=22, resolution=1.0, sup_sampling=20, distance=1.05,
        )
        P["xyz"], P["normals"], P["batch"] = self.select_pocket(P, ligand_center)
        return P

    def extract_residue_coords(self, pdb_file):
        prot = Chem.MolFromPDBFile(pdb_file, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
        Chem.AssignStereochemistryFrom3D(prot)
        u = mda.Universe(prot)
        protein_residues = u.select_atoms("protein").residues
        
        residue_coords = []
        for res in protein_residues:
            # 尝试寻找Alpha碳 (CA)
            ca_atoms = res.atoms.select_atoms("name CA")
            if len(ca_atoms) > 0:
                # 如果找到CA，使用其坐标
                coord = ca_atoms.positions[0]
            else:
                # 如果没有CA (可能是配体或其他非氨基酸残基)，计算该残基所有原子的几何中心
                coord = res.atoms.center_of_geometry()
            residue_coords.append(coord)
        
        residue_coords = np.array(residue_coords)
        print(residue_coords.shape)
        return residue_coords

    def map_surface_to_residues(self, surface_pt_file, residue_coords):
        """将表面特征映射到残基"""
        surface_data = torch.load(surface_pt_file, map_location="cpu")
        surface_xyz = surface_data["xyz"].numpy()         # [512, 3]
        surface_feat = surface_data["embedding"].numpy()  # [512, 128]

        # 构建 KDTree 查找最近 surface 点
        surface_tree = KDTree(surface_xyz)
        distances, indices = surface_tree.query(residue_coords, k=self.k)   # (N_res, k)

        # 收集最近 surface 的特征
        selected_feat = surface_feat[indices]  # (N_residues, k, 128) 

        # 求平均特征作为 residue 的补充特征
        avg_feat = selected_feat.mean(axis=1) # (N_residues, 128) 

        return {
            "residue_to_surface_indices": torch.LongTensor(indices),
            "residue_to_surface_distances": torch.FloatTensor(distances),
            "residue_feat_from_surface": torch.FloatTensor(avg_feat),
        }

    def run_pipeline(self):
        """运行完整的预处理流程: 表面特征提取 -> 残基-表面映射"""
        print(f"开始预处理 {self.dataset_name} 数据集")
        print(f"总共有 {len(self.all_keys)} 个唯一蛋白质")
        
        for key in tqdm(self.all_keys, desc="蛋白质表面预处理流水线"):
            try:
                # 检查最终输出是否已存在
                map_file = f"{self.map_dir}/{key}.pt"
                surface_file = f"{self.surface_dir}/{key}.pt"
                
                # Step 1: OpenBabel + Reduce 预处理--跳过
                pocket_file = self.get_pocket_file(key)
                if not pocket_file:
                    print(f"警告: 未找到键 '{key}' 的口袋文件")
                    continue
                
                # Step 1: 表面特征提取
                if not os.path.exists(surface_file):
                    protein_single = self.load_protein_atoms(pocket_file)
                    if protein_single is None:
                        print(f"警告: 无法加载蛋白质原子数据 '{key}'")
                        continue

                    protein_single = protein_single.to(self.device)
                    # 使用几何中心作为伪 ligand_center
                    ligand_center = protein_single.atom_coords.mean(dim=0).unsqueeze(0).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        P = self.process_surface(protein_single, ligand_center)
                        surface_emb = self.net(P)["embedding"]

                    # 保存表面特征
                    torch.save({
                        "embedding": surface_emb.cpu(),
                        "xyz": P["xyz"].cpu(),
                        "normals": P["normals"].cpu(),
                    }, surface_file)
                    
                    print(f"Step 1 完成 - 表面特征提取: {surface_file}")
                else:
                    print(f"Step 1 跳过 - 表面特征已存在: {surface_file}")
                
                # Step 2: 残基-表面映射
                if not os.path.exists(map_file):
                    residue_coords = self.extract_residue_coords(pocket_file)
                    
                    if os.path.exists(surface_file):
                        result = self.map_surface_to_residues(surface_file, residue_coords)
                        torch.save(result, map_file)
                        print(f"Step 2 完成 - 残基-表面映射: {map_file}")
                    else:
                        print(f"警告: 表面特征文件不存在，跳过残基映射: {surface_file}")
                else:
                    print(f"Step 2 跳过 - 残基映射已存在: {map_file}")
                
                print(f"蛋白质 {key} 处理完成")
                
            except Exception as e:
                print(f"处理蛋白质 {key} 时出错: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess protein surface features')
    parser.add_argument('--dataset', type=str, default='2hyy', help='Dataset name to process')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest surface points for residue mapping')
    args = parser.parse_args()
    preprocessor = ProteinSurfacePreprocessor(args.dataset, k=args.k)
    preprocessor.run_pipeline()