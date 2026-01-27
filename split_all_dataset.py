import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path


def create_fold_setting_cold(df, fold_seed, frac, entities):
    """
    根据给定实体进行冷启动分割数据集
    
    参数:
    df: 数据框
    fold_seed: 随机种子
    frac: 训练、验证、测试集的比例 [train_frac, val_frac, test_frac]
    entities: 用于冷启动的列名(如'Drug'或'Target'或['Drug', 'Target'])
    """
    if isinstance(entities, str):
        entities = [entities]
    train_frac, val_frac, test_frac = frac
    test_entity_instances = [df[e].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values for e in entities]

    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]                  

    if len(test) == 0:
        raise ValueError("No test samples found. Try another seed, increasing the test frac or a less stringent splitting strategy.")
    
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [train_val[e].drop_duplicates().sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed).values for e in entities]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError("No validation samples found. Try another seed, increasing the test frac or a less stringent splitting strategy.")

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def get_task_dir(task_name: str):
    """获取任务数据路径"""
    task_paths = {
        'davis': './data/davis',
        'kiba': './data/kiba',
    }
    return Path(task_paths[task_name.lower()]).resolve()


def split_data(data_dir, output_base_dir, task_name, split_strategy, seed=0, header=0, index_col=0, sep=","):
    """
    分割数据并保存为CSV文件
    
    参数:
    data_dir: 数据目录
    output_base_dir: 输出基础目录
    task_name: 任务名称
    split_strategy: 分割策略字典
    seed: 随机种子
    """
    strategy_name = split_strategy["name"]
    use_cold_split = split_strategy["use_cold_split"]
    cold_entity = split_strategy["cold_entity"]
    
    # 创建种子对应的输出目录
    seed_dir = os.path.join(output_base_dir, f"seed_{seed}")
    
    # 创建策略对应的输出目录
    output_dir = os.path.join(seed_dir, strategy_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    csv_kwargs = {
        "header": header,
        "index_col": index_col,
        "sep": sep,
    }
    
    data_path = Path(data_dir) / "process.csv"
    print(f"Reading data from {data_path}")
    df = pd.read_csv(data_path, **csv_kwargs)
    
    # 分割数据
    if use_cold_split:
        if isinstance(cold_entity, list):
            entity_str = "Drug_Target" if cold_entity == ["Drug", "Target"] else "_".join(cold_entity)
            print(f"Using cold-start splitting strategy for {entity_str}")
        else:
            print(f"Using {cold_entity} cold-start splitting strategy")
            
        df_train, df_val, df_test = create_fold_setting_cold(
            df, fold_seed=seed, frac=[0.8, 0.1, 0.1], entities=cold_entity
        )
    else:
        print("Using random splitting strategy")
        df_train, temp = train_test_split(df, test_size=0.2, random_state=seed)
        df_val, df_test = train_test_split(temp, test_size=0.5, random_state=seed)
    
    # 打印分割结果统计
    print(f"Split statistics:")
    print(f"  Train: {len(df_train)} samples ({len(df_train)/len(df)*100:.2f}%)")
    print(f"  Validation: {len(df_val)} samples ({len(df_val)/len(df)*100:.2f}%)")
    print(f"  Test: {len(df_test)} samples ({len(df_test)/len(df)*100:.2f}%)")
    
    # 保存分割后的CSV
    train_path = os.path.join(output_dir, f"{task_name}_train.csv")
    val_path = os.path.join(output_dir, f"{task_name}_val.csv")
    test_path = os.path.join(output_dir, f"{task_name}_test.csv")
    
    df_train.to_csv(train_path)
    df_val.to_csv(val_path)
    df_test.to_csv(test_path)
    
    print(f"Files saved to:")
    print(f"  Train: {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Test: {test_path}")
    print("-" * 80)
    
    return df_train, df_val, df_test


def main():
    # 多个种子
    seeds = [0, 1, 2, 3, 4]
    
    # 输出基础目录
    output_base_dir = "./split_data"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 数据集
    datasets = ["davis", "kiba"]
    
    # 分割策略
    split_strategies = [
        {"name": "random", "use_cold_split": False, "cold_entity": None},
        {"name": "cold_drug", "use_cold_split": True, "cold_entity": "Drug"},
        {"name": "cold_target", "use_cold_split": True, "cold_entity": "Target"},
        {"name": "all_cold", "use_cold_split": True, "cold_entity": ["Drug", "Target"]}
    ]
    
     # 循环处理每个种子
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"Processing with seed: {seed}")
        print(f"{'#'*60}")
        
        # 循环处理每个数据集和分割策略
        for dataset in datasets:
            print(f"\n{'='*40}")
            print(f"Processing dataset: {dataset}")
            print(f"{'='*40}")
            
            data_dir = get_task_dir(dataset)
            
            for strategy in split_strategies:
                print(f"\n{'-'*40}")
                print(f"Applying {strategy['name']} splitting strategy")
                print(f"{'-'*40}")
                
                split_data(
                    data_dir=data_dir,
                    output_base_dir=output_base_dir,
                    task_name=dataset,
                    split_strategy=strategy,
                    seed=seed
                )


if __name__ == "__main__":
    main()