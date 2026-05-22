import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """运行命令并捕获输出"""
    print(f"\n{'='*60}")
    print(f"[Step] {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[Error] Command failed with code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)
    print(f"[Done] {description}")
    return result


def check_split_exists(dataset, seed, strategy):
    """检查指定数据集和策略的划分文件是否已存在"""
    split_dir = Path(f"split_data/seed_{seed}/{strategy}")
    files = [
        split_dir / f"{dataset}_train.csv",
        split_dir / f"{dataset}_val.csv",
        split_dir / f"{dataset}_test.csv",
    ]
    return all(f.exists() for f in files)


def check_protein_sequence_exists(dataset):
    """检查蛋白质序列特征是否存在（至少有一个 .npy 文件）"""
    seq_dir = Path(f"data/{dataset}/preprocessed/sequence")
    if not seq_dir.exists():
        return False
    return len(list(seq_dir.glob("*.npy"))) > 0


def check_surface_exists(dataset, k):
    """检查指定 k 的表面特征是否已处理"""
    map_dir = Path(f"data/{dataset}/preprocessed/residue_surface/k{k}")
    if not map_dir.exists():
        return False
    return len(list(map_dir.glob("*.pt"))) > 0


def check_cache_exists(dataset, k):
    """检查全局 cache 是否已存在"""
    cache_dir = Path("data/cache")
    cache_files = [
        cache_dir / f"{dataset}_smile_graph.pt",
        cache_dir / f"{dataset}_fingerprint.pt",
        cache_dir / f"{dataset}_protein_graphs_k{k}.pt",
        cache_dir / f"{dataset}_esm_feats.pt",
    ]
    return all(f.exists() for f in cache_files)


def main():
    parser = argparse.ArgumentParser(
        description="HierDTA 训练前完整预处理流水线（划分 -> 蛋白序列 -> 表面特征 -> 全局缓存）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["davis", "kiba"],
        help="要处理的数据集名称",
    )
    parser.add_argument(
        "--k_list",
        type=str,
        default="3,5,8",
        help="要测试的表面映射 k 值列表，用逗号分隔（默认: 3,5,8）",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="数据划分的随机种子列表，用逗号分隔（默认: 0,1,2,3,4）",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="random",
        help="数据划分策略，逗号分隔（默认: random; 可选: random,cold_drug,cold_target,all_cold）",
    )
    parser.add_argument(
        "--skip_split",
        action="store_true",
        help="跳过数据划分步骤（如果划分文件已存在）",
    )
    parser.add_argument(
        "--skip_protein",
        action="store_true",
        help="跳过蛋白质序列特征提取（如果已存在）",
    )
    parser.add_argument(
        "--skip_surface",
        action="store_true",
        help="跳过表面特征提取（如果已存在）",
    )
    parser.add_argument(
        "--force_cache",
        action="store_true",
        help="强制重建全局 cache（即使 cache 文件已存在）",
    )
    parser.add_argument(
        "--force_surface",
        action="store_true",
        help="强制重建表面特征（即使已存在）",
    )
    parser.add_argument(
        "--gpu_idx",
        type=int,
        default=0,
        help="GPU index to use for ESM inference (default: 0)",
    )

    args = parser.parse_args()

    dataset = args.dataset
    k_values = [int(k.strip()) for k in args.k_list.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]

    print(f"\n{'#'*60}")
    print(f"# HierDTA Preprocessing Pipeline")
    print(f"# Dataset: {dataset}")
    print(f"# GPU index: {args.gpu_idx}")
    print(f"# Surface k values: {k_values}")
    print(f"# Seeds: {seeds}")
    print(f"# Strategies: {strategies}")
    print(f"{'#'*60}\n")

    # ============================================================
    # Step 1: 数据划分 (split_all_dataset.py)
    # ============================================================
    if not args.skip_split:
        all_exist = all(
            check_split_exists(dataset, seed, strategy)
            for seed in seeds
            for strategy in strategies
        )
        if all_exist:
            print(f"[Skip] Split files for {dataset} (seeds={seeds}, strategies={strategies}) already exist.")
        else:
            run_command(
                [sys.executable, "split_all_dataset.py"],
                "Data splitting for all datasets, seeds and strategies",
            )
    else:
        print("[Skip] Data splitting (--skip_split specified).")

    # ============================================================
    # Step 2: 蛋白质序列特征 (protein_process.py)
    # ============================================================
    if not args.skip_protein:
        if check_protein_sequence_exists(dataset) and not args.force_surface:
            print(f"[Skip] Protein sequence features for {dataset} already exist.")
        else:
            run_command(
                [sys.executable, "protein_process.py", "--dataset", dataset, "--gpu_idx", str(args.gpu_idx)],
                f"Protein sequence feature extraction ({dataset})",
            )
    else:
        print("[Skip] Protein sequence feature extraction (--skip_protein specified).")

    # ============================================================
    # Step 3 & 4: 表面特征 + 全局缓存（按 k 值循环）
    # ============================================================
    for k in k_values:
        print(f"\n{'#'*60}")
        print(f"# Processing surface mapping with k={k}")
        print(f"{'#'*60}")

        # Step 3: 表面特征 (surface_process.py)
        if not args.skip_surface:
            if check_surface_exists(dataset, k) and not args.force_surface:
                print(f"[Skip] Surface features for {dataset} (k={k}) already exist.")
            else:
                run_command(
                    [sys.executable, "surface_process.py", "--dataset", dataset, "--k", str(k)],
                    f"Surface feature extraction and mapping ({dataset}, k={k})",
                )
        else:
            print("[Skip] Surface feature extraction (--skip_surface specified).")

        # Step 4: 构建全局 cache (create_data_motif.py)
        if check_cache_exists(dataset, k) and not args.force_cache:
            print(f"[Skip] Global cache for {dataset} (k={k}) already exists.")
        else:
            cache_cmd = [
                sys.executable,
                "create_data_motif.py",
                "--dataset",
                dataset,
                "--surface_k",
                str(k),
                "--gpu_idx",
                str(args.gpu_idx),
            ]
            if args.force_cache:
                cache_cmd.append("--force")
            run_command(
                cache_cmd,
                f"Build global feature cache ({dataset}, k={k})",
            )

    print(f"\n{'#'*60}")
    print("# Preprocessing pipeline completed successfully!")
    print("# You can now run training with:")
    print(f"#   python training_warmup.py --dataset_idx {'1' if dataset == 'kiba' else '0'} --surface_k <k> ...")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
