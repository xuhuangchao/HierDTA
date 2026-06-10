# -*- coding: utf-8 -*-
"""Generate ESM-C embeddings for local protein sequences (conda activate esmc)."""

import pickle
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import transformers
from transformers import AutoModel, AutoTokenizer

print("Transformers version:", transformers.__version__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "esm_pt" / "ESMC-600M"

# Load model (use relative path + offline env to bypass huggingface_hub issues)
import os as _os
_os.environ["TRANSFORMERS_OFFLINE"] = "1"
_os.environ["HF_HUB_OFFLINE"] = "1"
_model_dir = _os.path.relpath(str(MODEL_PATH.resolve()), _os.getcwd())
model = AutoModel.from_pretrained(_model_dir, local_files_only=True).eval()

tokenizer = AutoTokenizer.from_pretrained(_model_dir, local_files_only=True)

MAX_SEQ_LENGTH = 1200

def get_esmc_pretrain(df_dir, db_name):
    df = pd.read_csv(df_dir).drop_duplicates("target_key")

    emb_dict, emb_mat_dict, length_target = {}, {}, {}

    for prot_id, seq in tqdm(zip(df["target_key"], df["target_sequence"]), total=len(df)):
        seq = seq[:MAX_SEQ_LENGTH]

        inputs = tokenizer(seq, return_tensors="pt")

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            
        reps = outputs.last_hidden_state[0]
        reps = reps.cpu().numpy()

        emb_mat_dict[prot_id] = reps
        emb_dict[prot_id] = reps.mean(axis=0)
        length_target[prot_id] = len(seq)

        print(f"Embedding shape: {reps.shape} for {prot_id}")

    with open(DATA_ROOT / db_name / f"{db_name}_esmc_pretrain.pkl", "wb") as f:
        pickle.dump(
            {
                "dataset": db_name,
                "vec_dict": emb_dict,
                "mat_dict": emb_mat_dict,
                "length_dict": length_target,
            },
            f,
        )

    print(f"Saved {db_name} ESM-C features.")


if __name__ == "__main__":
    db_names = ["davis", "kiba"]
    for dataset in db_names:
        df_dir = DATA_ROOT / dataset / f"{dataset}_processed_300_add_range.csv"
        print(f"Compute {df_dir} protein pretrain features by local ESM-C.")
        get_esmc_pretrain(df_dir, dataset)
