import os
import glob
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from easydict import EasyDict
import yaml
import re
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入dMaSIF相关模块
from dmasif_encoder.protein_surface_encoder import dMaSIF
from dmasif_encoder.geometry_processing import atoms_to_points_normals


class ProteinSurfacePreprocessor:
    def __init__(self, dataset_name, data_root=PROJECT_ROOT / "data"):
        self.dataset_name = dataset_name
        self.dataset_root = Path(data_root) / dataset_name
        self.fpath = str(self.dataset_root) + os.sep
        self.pocket_dir = str(self.dataset_root / f'pocket1_{dataset_name}')

        # 设置随机种子
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # 加载配置和模型
        self.device = "cpu"
        with open(PROJECT_ROOT / "config.yml", 'r') as f:
            config = EasyDict(yaml.safe_load(f))
        self.net = dMaSIF(config.model.dmasif).to(self.device)
        self.net.eval()

        # 元素到数字的映射
        self.ele2num = {'H':0, 'LI':1, 'C':2, 'N':3, 'O':4, 'NA':5, 'MG':6, 'P':7, 'S':8,
                       'K':9, 'CA':10, 'MN':11, 'FE':12, 'CO':13, 'NI':14, 'CU':15, 'ZN':16,
                       'SE':17, 'SR':18, 'CD':19, 'CS':20, 'HG':21}

        # 加载蛋白质数据
        prot_path = self.dataset_root / f"{dataset_name}_prots.csv"
        if not prot_path.exists():
            raise FileNotFoundError(f"Protein CSV not found: {prot_path}")
        prot_df = pd.read_csv(prot_path)
        required_columns = {"target_key", "target_sequence"}
        missing_columns = required_columns - set(prot_df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {prot_path}: {sorted(missing_columns)}. "
                f"Available columns: {list(prot_df.columns)}"
            )
        self.proteins = OrderedDict(
            (str(row.target_key), str(row.target_sequence))
            for row in prot_df[["target_key", "target_sequence"]].drop_duplicates("target_key").itertuples(index=False)
        )
        self.all_keys = list(self.proteins.keys())

        # Surface points are stored directly under the dataset directory.
        self.surface_dir = str(self.dataset_root / 'surface_points')
        os.makedirs(self.surface_dir, exist_ok=True)


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
            element = atom.element.upper()
            if element in self.ele2num:
                coords.append(atom.get_coord())
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

    def run_pipeline(self):
        """提取每个口袋最多 512 个表面点的特征。"""
        print(f"开始预处理 {self.dataset_name} 数据集")
        print(f"总共有 {len(self.all_keys)} 个唯一蛋白质")
        
        for key in tqdm(self.all_keys, desc="蛋白质表面预处理流水线"):
            try:
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

                    if surface_emb.dim() != 2 or surface_emb.size(1) != 128:
                        raise ValueError(
                            f"表面特征形状应为 [N, 128]，实际为 {tuple(surface_emb.shape)}"
                        )

                    # xyz 和 normals 保留用于独立的表面分析；训练仅使用 embedding。
                    torch.save({
                        "embedding": surface_emb.cpu(),
                        "xyz": P["xyz"].cpu(),
                        "normals": P["normals"].cpu(),
                    }, surface_file)
                    
                    print(f"Step 1 完成 - 表面特征提取: {surface_file}")
                else:
                    print(f"Step 1 跳过 - 表面特征已存在: {surface_file}")

                print(f"蛋白质 {key} 处理完成")
                
            except Exception as e:
                print(f"处理蛋白质 {key} 时出错: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess protein surface features')
    parser.add_argument('--dataset', type=str, default='davis', help='Dataset name to process')
    parser.add_argument('--data_root', type=str, default=str(PROJECT_ROOT / 'data'), help='Root containing {dataset}/{dataset}_prots.csv')
    args = parser.parse_args()
    preprocessor = ProteinSurfacePreprocessor(args.dataset, data_root=args.data_root)
    preprocessor.run_pipeline()
