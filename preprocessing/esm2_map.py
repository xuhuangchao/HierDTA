# -*- coding: utf-8 -*-
"""Generate ESM2 contact maps for local protein sequences."""

import pickle
from pathlib import Path

# fair-esm==2.0.0
import esm
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "esm2_pt" / "esm2_t36_3B_UR50D.pt"

model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(MODEL_PATH))
batch_converter = alphabet.get_batch_converter()
model = model.to("cuda").eval()


def get_esm_contact_map(model, df_dir, db_name, max_length=1200):
    df = pd.read_csv(df_dir).drop_duplicates("target_key")
    prot_data = []
    for prot_id, seq in zip(df["target_key"], df["target_sequence"]):
        truncated_seq = seq[:max_length]
        prot_data.append((prot_id, truncated_seq))

    target_graph = {}
    length_target = {}

    for prot_id, seq in tqdm(prot_data):
        _, _, batch_tokens = batch_converter([(prot_id, seq)])

        with torch.inference_mode():
            results = model(batch_tokens.to("cuda"), return_contacts=True)

        contact_map = results["contacts"][0].cpu().numpy()
        target_graph[prot_id] = contact_map
        length_target[prot_id] = len(seq)
        print("  ESM2 contact map shape:", contact_map.shape, "for target", prot_id, "with sequence length", len(seq))
        
        # cuda memory cleanup
        del results
        del batch_tokens
        torch.cuda.empty_cache()

    dump_data = {
        "dataset": db_name,
        "contact_map": target_graph,
        "length_dict": length_target,
    }

    output_path = DATA_ROOT / db_name / f"{db_name}_esm2_contact_map.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(dump_data, f)

    print(f"Saved contact maps for {db_name}.")


if __name__ == "__main__":
    db_names = ["davis", "kiba"]
    for dataset in db_names:
        df_dir = DATA_ROOT / dataset / f"{dataset}_processed_300_add_range.csv"
        print(f"Compute {df_dir} protein contact maps by local ESM2.")
        get_esm_contact_map(model, df_dir, dataset)
