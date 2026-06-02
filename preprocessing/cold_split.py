import pandas as pd
import argparse
from pathlib import Path

# referenced from https://tdc.readthedocs.io/en/main/index.html
def create_fold(df, fold_seed, frac):
    """create random split

    Args:
        df (pd.DataFrame): dataset dataframe
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(
        frac=val_frac / (1 - test_frac), replace=False, random_state=1
    )
    train = train_val[~train_val.index.isin(val.index)]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def create_fold_setting_cold(df, fold_seed, frac, entities):
    """create cold-split where given one or multiple columns, it first splits based on
    entities in the columns and then maps all associated data points to the partition

    Args:
            df (pd.DataFrame): dataset dataframe
            fold_seed (int): the random seed
            frac (list): a list of train/valid/test fractions
            entities (Union[str, List[str]]): either a single "cold" entity or a list of
                    "cold" entities on which the split is done

    Returns:
            dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    # For each entity, sample the instances belonging to the test datasets
    test_entity_instances = [
        df[e]
        .drop_duplicates()
        .sample(frac=test_frac, replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]

    # Select samples where all entities are in the test set
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            "No test samples found. Try another seed, increasing the test frac or a "
            "less stringent splitting strategy."
        )

    # Proceed with validation data
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e]
        .drop_duplicates()
        .sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            "No validation samples found. Try another seed, increasing the test frac "
            "or a less stringent splitting strategy."
        )

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

dataset = 'davis' # davis, kiba, metz, bindingDB

SEED = 41
DEFAULT_SEEDS = [41, 42, 43, 32, 33]
FRAC = [0.8, 0.1, 0.1]


def save_fold(fold, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in fold.items():
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)


def print_fold_info(dataset_name, seed, fold_name, fold):
    print(
        f"{dataset_name} seed-{seed} {fold_name} done!  the shape of train, valid, test are: ",
        fold["train"].shape,
        fold["valid"].shape,
        fold["test"].shape,
    )


def split_dataset(df, dataset_root, dataset_name, seed):
    seed_root = dataset_root / f"seed_{seed}"

    warm_fold = create_fold(df, seed, FRAC)
    save_fold(warm_fold, seed_root / "warm")
    print_fold_info(dataset_name, seed, "warm_fold", warm_fold)

    cold_target_fold = create_fold_setting_cold(df, seed, FRAC, ['target_key'])
    save_fold(cold_target_fold, seed_root / "unseen_prot")
    print_fold_info(dataset_name, seed, "cold_target_fold", cold_target_fold)

    cold_drug_fold = create_fold_setting_cold(df, seed, FRAC, ['compound_iso_smiles'])
    save_fold(cold_drug_fold, seed_root / "unseen_drug")
    print_fold_info(dataset_name, seed, "cold_drug_fold", cold_drug_fold)

    cold_target_drug_fold = create_fold_setting_cold(df, seed, FRAC, ['target_key', 'compound_iso_smiles'])
    save_fold(cold_target_drug_fold, seed_root / "unseen_pair")
    print_fold_info(dataset_name, seed, "cold_target_drug_fold", cold_target_drug_fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=dataset, choices=["davis", "kiba", "metz", "bindingDB"])
    parser.add_argument("--SEED", type=int, default=None, help="Generate splits for one seed. Kept for backward compatibility.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Generate splits for multiple seeds.")

    args = parser.parse_args()
    if args.seeds is not None:
        seeds = args.seeds
    elif args.SEED is not None:
        seeds = [args.SEED]
    else:
        seeds = DEFAULT_SEEDS

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / args.dataset
    data_path = dataset_root / "data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # read data
    df = pd.read_csv(data_path)
    required_columns = {"target_key", "compound_iso_smiles"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {data_path}: {sorted(missing_columns)}")

    for seed in seeds:
        split_dataset(df, dataset_root, args.dataset, seed)
