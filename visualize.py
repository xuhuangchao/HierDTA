"""Visualize HierDTA LocalAugmentation and atom--protein cross-attention.

The script does not modify the model implementation.  It registers a forward
hook on ``model.agg.la`` and reconstructs the two-source LocalAugmentation
softmax from the module inputs and trained projection weights.

For every automatically selected active drug--target pair, exactly four files
are written into a sample-specific directory:

    cross_attention_gat.png
    local_aug_comparison.png
    atom_scores.csv
    metadata.json

Example
-------
python visualize.py \
    --dataset davis --strategy warm --seed 41 --split test \
    --checkpoint_dir checkpoint_seed41 --num_samples 5 \
    --output_dir outputs/visualization_davis_warm_seed41
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import colors as mpl_colors
from matplotlib import colormaps
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from dta_model import DTAModel
from utils import TestbedDatasetHMol, dta_collate_fn


BACKBONES = ("gat", "gcn", "gin")
DEFAULT_CHECKPOINTS = {
    "gat": "ckpt_motif2atom_mlayer1_best.pt",
    "gcn": "ckpt_motif2atom_mlayer1_gcn_best.pt",
    "gin": "ckpt_motif2atom_mlayer1_gin_best.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create LocalAugmentation and GAT CrossAttention atom maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--strategy", default="warm")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--split_csv",
        default="data/davis/seed_41/warm/test.csv",
        help="Pair CSV used for automatic active-pair selection.",
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--cache_dir", default="data/cache")
    parser.add_argument("--checkpoint_dir", default="checkpoint_seed41")
    parser.add_argument("--gat_checkpoint", default=None)
    parser.add_argument("--gcn_checkpoint", default=None)
    parser.add_argument("--gin_checkpoint", default=None)
    parser.add_argument("--output_dir", default="outputs/attention_visualization")
    parser.add_argument("--num_samples", type=int, default=5, choices=range(3, 6))
    parser.add_argument(
        "--active_threshold",
        type=float,
        default=7.0,
        help="Minimum affinity for an active DAVIS pair (higher is stronger).",
    )
    parser.add_argument("--min_atoms", type=int, default=8)
    parser.add_argument("--max_atoms", type=int, default=60)
    parser.add_argument(
        "--candidate_pool",
        type=int,
        default=80,
        help="Top active structural candidates scored by the GAT model.",
    )
    parser.add_argument(
        "--selection",
        choices=("auto", "index"),
        default="auto",
        help="Auto-select active pairs or visualize rows supplied by --indices.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Row indices within the selected split; used with --selection index.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--drug_layer", type=int, default=3)
    parser.add_argument("--motif_layer", type=int, default=1)
    parser.add_argument("--protein_layer", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def choose_device(args: argparse.Namespace) -> torch.device:
    if args.device != "auto":
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu_idx}")
    return torch.device("cpu")


def torch_load(path: Path, device: object, weights_only: bool):
    """Load on both recent and older PyTorch releases."""
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_checkpoint_paths(args: argparse.Namespace) -> Dict[str, Path]:
    checkpoint_dir = Path(args.checkpoint_dir)
    explicit = {
        "gat": args.gat_checkpoint,
        "gcn": args.gcn_checkpoint,
        "gin": args.gin_checkpoint,
    }
    paths = {}
    for backbone in BACKBONES:
        path = Path(explicit[backbone]) if explicit[backbone] else (
            checkpoint_dir / DEFAULT_CHECKPOINTS[backbone]
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing {backbone.upper()} checkpoint: {path}")
        paths[backbone] = path
    return paths


def resolve_split_path(args: argparse.Namespace) -> Path:
    if args.split_csv:
        path = Path(args.split_csv)
    else:
        path = (
            Path(args.data_root)
            / args.dataset
            / f"seed_{args.seed}"
            / args.strategy
            / f"{args.split}.csv"
        )
    if path.is_file():
        return path

    fallback = Path(args.data_root) / args.dataset / "data.csv"
    if fallback.is_file():
        warnings.warn(
            f"Split CSV was not found at {path}. Falling back to {fallback}. "
            "The selected samples are therefore not guaranteed to belong to "
            f"the {args.strategy}/{args.split} split."
        )
        return fallback
    raise FileNotFoundError(f"Neither split CSV {path} nor fallback {fallback} exists")


def read_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"compound_iso_smiles", "target_key", "affinity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["source_row_index"] = np.arange(len(frame), dtype=int)
    frame["affinity"] = pd.to_numeric(frame["affinity"], errors="coerce")
    frame = frame.dropna(subset=["compound_iso_smiles", "target_key", "affinity"])
    return frame


def load_feature_caches(args: argparse.Namespace):
    cache_dir = Path(args.cache_dir)
    drug_path = cache_dir / f"{args.dataset}_drug_features.pt"
    protein_path = cache_dir / f"{args.dataset}_protein_graphs.pt"
    if not drug_path.is_file() or not protein_path.is_file():
        raise FileNotFoundError(
            f"Expected feature caches {drug_path} and {protein_path}"
        )
    drugs = torch_load(drug_path, "cpu", weights_only=False)
    proteins = torch_load(protein_path, "cpu", weights_only=False)
    return drugs, proteins


def make_dataset(
    rows: pd.DataFrame,
    args: argparse.Namespace,
    drug_features,
    protein_features,
) -> TestbedDatasetHMol:
    return TestbedDatasetHMol(
        xd=rows["compound_iso_smiles"].astype(str).tolist(),
        xt=rows["target_key"].astype(str).tolist(),
        y=rows["affinity"].astype(float).tolist(),
        dataset_name=args.dataset,
        cache_dir=args.cache_dir,
        drug_features=drug_features,
        protein_features=protein_features,
    )


def move_batch(batch, device: torch.device):
    # DTABatch.to mutates the object.  Do not rely on its return value because
    # some repository revisions do not explicitly return ``self``.
    batch.to(device)
    return batch


def model_for(backbone: str, args: argparse.Namespace, checkpoint: Path, device):
    model = DTAModel(
        drug_num_layers=args.drug_layer,
        motif_num_layers=args.motif_layer,
        protein_num_layers=args.protein_layer,
        drug_gnn_type=backbone,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_agg=True,
    ).to(device)
    state = torch_load(checkpoint, device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class LocalAugmentationCapture:
    """Forward hook that exactly reconstructs LocalAugmentation attention."""

    def __init__(self):
        self.motif_scores: List[np.ndarray] = []

    def __call__(self, module, inputs, output):
        fine_messages, coarse_messages, query_features = inputs[:3]
        n_atoms, hidden_dim = fine_messages.shape
        keys = torch.cat(
            [fine_messages.unsqueeze(1), coarse_messages.unsqueeze(1)], dim=1
        )
        query = query_features.view(n_atoms, -1, 1, hidden_dim).transpose(1, 2)
        keys = keys.view(n_atoms, -1, 1, hidden_dim).transpose(1, 2)

        query = (
            module.linear_layers[0](query)
            .view(n_atoms, -1, module.heads, module.d_k)
            .transpose(1, 2)
        )
        keys = (
            module.linear_layers[1](keys)
            .view(n_atoms, -1, module.heads, module.d_k)
            .transpose(1, 2)
        )
        logits = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(
            module.d_k
        )
        attention = F.softmax(logits, dim=-1)
        # [atom, head, query=1, source=0(motif)] -> [atom]
        motif = attention[:, :, 0, 0].mean(dim=1)
        self.motif_scores.append(motif.detach().float().cpu().numpy())


def run_model_with_capture(model, batch, return_cross_attention: bool):
    capture = LocalAugmentationCapture()
    if model.agg is None:
        raise RuntimeError("The checkpoint model has no LocalAugmentation module")
    handle = model.agg.la.register_forward_hook(capture)
    try:
        with torch.no_grad():
            result = model(batch, return_attention=return_cross_attention)
    finally:
        handle.remove()

    if return_cross_attention:
        prediction, info = result
    else:
        prediction, info = result, None
    return float(prediction.reshape(-1)[0].detach().cpu()), capture.motif_scores, info


def score_gat_candidates(
    rows: pd.DataFrame,
    args: argparse.Namespace,
    drug_features,
    protein_features,
    gat_model,
    device,
) -> pd.DataFrame:
    """Predict a small high-affinity pool to avoid selecting failed examples."""
    dataset = make_dataset(rows, args, drug_features, protein_features)
    predictions = []
    for sample in dataset:
        batch = move_batch(dta_collate_fn([sample]), device)
        with torch.no_grad():
            prediction = gat_model(batch)
        predictions.append(float(prediction.reshape(-1)[0].detach().cpu()))
    scored = rows.copy()
    scored["gat_prediction"] = predictions
    scored["gat_absolute_error"] = np.abs(
        scored["gat_prediction"] - scored["affinity"]
    )
    return scored


def structural_candidates(
    frame: pd.DataFrame, args: argparse.Namespace, drug_features
) -> pd.DataFrame:
    # DAVIS mutation constructs are conventionally written as e.g. ABL1(T315I).
    # Keep wild-type/non-parenthesized target names for clearer biological cases.
    target_names = frame["target_key"].astype(str)
    non_mutation = frame[~target_names.str.contains(r"[()]", regex=True)].copy()
    if non_mutation.empty:
        raise RuntimeError(
            "No non-mutant target remained after excluding target names with parentheses"
        )

    candidates = non_mutation[
        non_mutation["affinity"] >= args.active_threshold
    ].copy()
    if candidates.empty:
        warnings.warn(
            f"No non-mutant pairs met affinity >= {args.active_threshold}; "
            "using the highest-affinity non-mutant pairs instead."
        )
        candidates = non_mutation.copy()

    atom_counts = []
    valid = []
    for row in candidates.itertuples(index=False):
        smiles = str(row.compound_iso_smiles)
        mol = Chem.MolFromSmiles(smiles)
        ok = mol is not None and smiles in drug_features
        count = mol.GetNumAtoms() if mol is not None else -1
        ok = ok and args.min_atoms <= count <= args.max_atoms
        valid.append(ok)
        atom_counts.append(count)
    candidates["num_atoms"] = atom_counts
    candidates = candidates[np.asarray(valid, dtype=bool)]
    if candidates.empty:
        raise RuntimeError("No active candidates passed the molecule-size filters")

    candidates = candidates.sort_values("affinity", ascending=False)
    # Remove exact repeated pairs before the relatively expensive GAT scoring.
    candidates = candidates.drop_duplicates(
        subset=["compound_iso_smiles", "target_key"], keep="first"
    )
    return candidates.head(args.candidate_pool).copy()


def diverse_selection(scored: pd.DataFrame, count: int) -> pd.DataFrame:
    """Prefer accurately predicted active pairs with distinct drugs/targets."""
    affinity_rank = scored["affinity"].rank(ascending=False, method="min")
    error_rank = scored["gat_absolute_error"].rank(ascending=True, method="min")
    prediction_rank = scored["gat_prediction"].rank(ascending=False, method="min")
    scored = scored.copy()
    scored["selection_score"] = (
        0.45 * affinity_rank + 0.40 * error_rank + 0.15 * prediction_rank
    )
    scored = scored.sort_values("selection_score", ascending=True)

    chosen = []
    used_drugs, used_targets = set(), set()
    # First pass: maximize both drug and target diversity.
    for idx, row in scored.iterrows():
        drug, target = row["compound_iso_smiles"], row["target_key"]
        if drug not in used_drugs and target not in used_targets:
            chosen.append(idx)
            used_drugs.add(drug)
            used_targets.add(target)
        if len(chosen) == count:
            break
    # Second pass: fill remaining slots while avoiding exact pair duplication.
    if len(chosen) < count:
        for idx in scored.index:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) == count:
                break
    if len(chosen) < count:
        raise RuntimeError(f"Only {len(chosen)} suitable pairs were available")
    return scored.loc[chosen].reset_index(drop=True)


def select_rows(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    drug_features,
    protein_features,
    gat_model,
    device,
) -> pd.DataFrame:
    if args.selection == "index":
        if not args.indices:
            raise ValueError("--selection index requires --indices")
        if not 3 <= len(args.indices) <= 5:
            raise ValueError("Provide 3 to 5 values with --indices")
        by_source_index = frame.set_index("source_row_index", drop=False)
        missing = [idx for idx in args.indices if idx not in by_source_index.index]
        if missing:
            raise IndexError(f"Row indices not found in the split: {missing}")
        return by_source_index.loc[args.indices].reset_index(drop=True)

    candidates = structural_candidates(frame, args, drug_features)
    scored = score_gat_candidates(
        candidates,
        args,
        drug_features,
        protein_features,
        gat_model,
        device,
    )
    return diverse_selection(scored, args.num_samples)


def atom_memberships(membership_edge_index: torch.Tensor, n_atoms: int):
    memberships: List[List[int]] = [[] for _ in range(n_atoms)]
    edge_index = membership_edge_index.detach().cpu().numpy()
    for atom_idx, motif_idx in zip(edge_index[0], edge_index[1]):
        atom_idx, motif_idx = int(atom_idx), int(motif_idx)
        if motif_idx not in memberships[atom_idx]:
            memberships[atom_idx].append(motif_idx)
    for values in memberships:
        values.sort()
    return memberships


def categorical_motif_colors(memberships: Sequence[Sequence[int]]):
    cmap = colormaps.get_cmap("tab20")
    primary = [values[0] if values else -1 for values in memberships]
    unique = sorted({value for value in primary if value >= 0})
    color_by_motif = {
        motif: tuple(float(x) for x in cmap(i % cmap.N)[:3])
        for i, motif in enumerate(unique)
    }
    return primary, color_by_motif


def draw_molecule_image(
    mol: Chem.Mol,
    atom_colors: Dict[int, Tuple[float, float, float]],
    atom_radii: Optional[Dict[int, float]] = None,
    bond_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
    width: int = 620,
    height: int = 430,
) -> Image.Image:
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    options.clearBackground = True
    options.setBackgroundColour((1.0, 1.0, 1.0))
    options.padding = 0.08
    atom_indices = sorted(atom_colors)
    bond_colors = bond_colors or {}
    drawer.DrawMolecule(
        mol,
        highlightAtoms=atom_indices,
        highlightBonds=sorted(bond_colors),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        highlightAtomRadii=atom_radii or {idx: 0.38 for idx in atom_indices},
    )
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    # Cairo PNGs can retain transparent pixels on some RDKit versions.  Paste
    # onto white explicitly so PIL/Matplotlib never render them as black.
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    white.alpha_composite(image)
    return white.convert("RGB")


def motif_partition_image(mol: Chem.Mol, memberships) -> Image.Image:
    # Motifs may overlap (notably a ring atom can also belong to a functional
    # group).  DrawMoleculeWithHighlights accepts multiple colors per atom/bond,
    # so no arbitrary "primary motif" assignment is needed and rings remain
    # visually intact.
    motif_ids = sorted({motif for values in memberships for motif in values})
    cmap = colormaps.get_cmap("tab10")
    color_by_motif = {
        motif: tuple(float(x) for x in cmap(i % cmap.N)[:3])
        for i, motif in enumerate(motif_ids)
    }
    atom_highlights = {
        atom_idx: [color_by_motif[motif] for motif in motif_list]
        for atom_idx, motif_list in enumerate(memberships)
        if motif_list
    }
    bond_highlights = {}
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        shared = sorted(set(memberships[begin]).intersection(memberships[end]))
        if shared:
            bond_highlights[bond.GetIdx()] = [
                color_by_motif[motif] for motif in shared
            ]

    drawer = rdMolDraw2D.MolDraw2DCairo(760, 1060)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    options.clearBackground = True
    options.setBackgroundColour((1.0, 1.0, 1.0))
    options.fillHighlights = False
    options.padding = 0.08
    drawer.DrawMoleculeWithHighlights(
        mol,
        "",
        atom_highlights,
        bond_highlights,
        {idx: 0.32 for idx in atom_highlights},
        {idx: 2 for idx in bond_highlights},
    )
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    white.alpha_composite(image)
    return white.convert("RGB")


def continuous_atom_image(
    mol: Chem.Mol,
    values: Sequence[float],
    cmap_name: str,
    norm: mpl_colors.Normalize,
) -> Image.Image:
    cmap = colormaps.get_cmap(cmap_name)
    atom_colors = {
        idx: tuple(float(x) for x in cmap(norm(float(value)))[:3])
        for idx, value in enumerate(values)
    }
    return draw_molecule_image(mol, atom_colors)


def cross_attention_scores(info, n_atoms: int, n_residues: int):
    weights = info["atom_weights"].detach().float().cpu()
    # Batch size is one: [1, heads, padded_atoms, padded_residues].
    weights = weights[0, :, :n_atoms, :n_residues]
    mean_matrix = weights.mean(dim=0).numpy()
    atom_max = mean_matrix.max(axis=1)
    top_residue = mean_matrix.argmax(axis=1)
    residue_mean = mean_matrix.mean(axis=0)
    return mean_matrix, atom_max, top_residue, residue_mean


def save_cross_attention_figure(
    path: Path,
    mol: Chem.Mol,
    atom_scores: np.ndarray,
    target_key: str,
    affinity: float,
    prediction: float,
    dpi: int,
):
    vmax = float(max(atom_scores.max(), np.finfo(float).eps))
    norm = mpl_colors.Normalize(vmin=0.0, vmax=vmax)
    image = continuous_atom_image(mol, atom_scores, "magma", norm)
    fig, ax = plt.subplots(figsize=(8.8, 6.2), facecolor="white")
    ax.set_facecolor("white")
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(
        f"GAT CrossAttention ligand map | target={target_key}\n"
        f"affinity={affinity:.3f}, prediction={prediction:.3f}\n"
        "atom score = max residue attention after head averaging",
        fontsize=11,
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="magma")
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax, fraction=0.045, pad=0.025, label="Attention score")
    fig.tight_layout()
    fig.savefig(
        path, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False
    )
    plt.close(fig)


def save_local_aug_figure(
    path: Path,
    mol: Chem.Mol,
    memberships,
    local_scores: Dict[str, Sequence[np.ndarray]],
    dpi: int,
):
    norm = mpl_colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    # Draw the motif partition once in an axis spanning all three rows.  The
    # remaining two columns compare the two augmentation stages by backbone.
    # A dedicated final column keeps the colorbar away from molecule panels.
    fig = plt.figure(figsize=(19.5, 14.5), facecolor="white")
    grid = fig.add_gridspec(
        3,
        4,
        width_ratios=(1.12, 1.0, 1.0, 0.045),
        left=0.045,
        right=0.94,
        bottom=0.045,
        top=0.925,
        wspace=0.035,
        hspace=0.10,
    )
    motif_ax = fig.add_subplot(grid[:, 0])
    motif_ax.set_facecolor("white")
    motif_ax.imshow(motif_partition_image(mol, memberships))
    motif_ax.axis("off")
    motif_ax.set_title("Motif membership (overlaps allowed)", fontsize=15, pad=14)

    axes = np.empty((3, 2), dtype=object)
    for row in range(3):
        for col in range(2):
            axes[row, col] = fig.add_subplot(grid[row, col + 1])
            axes[row, col].set_facecolor("white")
    colorbar_ax = fig.add_subplot(grid[:, 3])
    colorbar_ax.set_facecolor("white")

    panel_labels = {
        "gat": "(a) GAT",
        "gcn": "(b) GCN",
        "gin": "(c) GIN",
    }
    for row, backbone in enumerate(BACKBONES):
        if len(local_scores[backbone]) < 2:
            raise RuntimeError(
                f"Expected two LocalAugmentation calls for {backbone.upper()}, "
                f"but captured {len(local_scores[backbone])}. Check --drug-layer."
            )
        images = [
            continuous_atom_image(mol, local_scores[backbone][0], "coolwarm", norm),
            continuous_atom_image(mol, local_scores[backbone][1], "coolwarm", norm),
        ]
        for col, image in enumerate(images):
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
        axes[row, 0].text(
            0.015,
            0.985,
            panel_labels[backbone],
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=17,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.5},
        )

    axes[0, 0].set_title("Layer 1 motif attention", fontsize=15, pad=14)
    axes[0, 1].set_title("Layer 2 motif attention", fontsize=15, pad=14)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        cax=colorbar_ax,
    )
    colorbar.set_label(
        "Motif-source attention (blue=self, red=motif)", fontsize=12, labelpad=12
    )
    fig.suptitle(
        "LocalAugmentation motif-to-atom attention comparison",
        fontsize=16,
        y=0.985,
    )
    fig.savefig(
        path, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False
    )
    plt.close(fig)


def safe_name(value: str, max_length: int = 40) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (cleaned or "unknown")[:max_length]


def sample_directory(root: Path, rank: int, target: str, smiles: str) -> Path:
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:8]
    path = root / f"sample_{rank:02d}_{safe_name(target)}_{digest}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sequence_for_target(args: argparse.Namespace, target_key: str) -> Optional[str]:
    path = Path(args.data_root) / args.dataset / f"{args.dataset}_prots.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path, usecols=lambda c: c in {"target_key", "target_sequence"})
    matches = frame[frame["target_key"].astype(str) == str(target_key)]
    if matches.empty or "target_sequence" not in matches:
        return None
    return str(matches.iloc[0]["target_sequence"])


def native_float_list(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values]


def write_sample_outputs(
    output_dir: Path,
    sample,
    row: pd.Series,
    rank: int,
    args: argparse.Namespace,
    checkpoint_paths: Dict[str, Path],
    models: Dict[str, DTAModel],
    device: torch.device,
    split_path: Path,
):
    smiles = str(row["compound_iso_smiles"])
    target_key = str(row["target_key"])
    affinity = float(row["affinity"])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse selected SMILES: {smiles}")

    n_atoms = int(sample.hetero["atom"].x.size(0))
    n_residues = int(sample.protein_graph.x.size(0))
    if mol.GetNumAtoms() != n_atoms:
        raise ValueError(
            f"RDKit/graph atom mismatch for {smiles}: "
            f"RDKit={mol.GetNumAtoms()}, graph={n_atoms}"
        )
    memberships = atom_memberships(
        sample.hetero["atom", "in", "motif"].edge_index, n_atoms
    )
    batch = move_batch(dta_collate_fn([sample]), device)

    predictions: Dict[str, float] = {}
    local_scores: Dict[str, Sequence[np.ndarray]] = {}
    gat_info = None
    for backbone in BACKBONES:
        prediction, traces, info = run_model_with_capture(
            models[backbone], batch, return_cross_attention=(backbone == "gat")
        )
        predictions[backbone] = prediction
        local_scores[backbone] = traces
        if backbone == "gat":
            gat_info = info

    if gat_info is None:
        raise RuntimeError("GAT CrossAttention information was not returned")
    cross_matrix, cross_atom, top_residue, residue_mean = cross_attention_scores(
        gat_info, n_atoms, n_residues
    )

    out = sample_directory(output_dir, rank, target_key, smiles)
    save_cross_attention_figure(
        out / "cross_attention_gat.png",
        mol,
        cross_atom,
        target_key,
        affinity,
        predictions["gat"],
        args.dpi,
    )
    save_local_aug_figure(
        out / "local_aug_comparison.png",
        mol,
        memberships,
        local_scores,
        args.dpi,
    )

    sequence = sequence_for_target(args, target_key)
    sequence_valid = sequence is not None and len(sequence) == n_residues
    atom_rows = []
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        residue_idx = int(top_residue[atom_idx])
        residue_position = residue_idx + 1 if sequence_valid else None
        residue_letter = (
            sequence[residue_idx]
            if sequence_valid
            else ""
        )
        atom_rows.append(
            {
                "atom_index": atom_idx,
                "atom_symbol": atom.GetSymbol(),
                "motif_ids": ";".join(map(str, memberships[atom_idx])),
                "gat_local_layer1": float(local_scores["gat"][0][atom_idx]),
                "gat_local_layer2": float(local_scores["gat"][1][atom_idx]),
                "gcn_local_layer1": float(local_scores["gcn"][0][atom_idx]),
                "gcn_local_layer2": float(local_scores["gcn"][1][atom_idx]),
                "gin_local_layer1": float(local_scores["gin"][0][atom_idx]),
                "gin_local_layer2": float(local_scores["gin"][1][atom_idx]),
                "gat_cross_max_attention": float(cross_atom[atom_idx]),
                "gat_cross_top_residue_graph_index": residue_idx,
                "gat_cross_top_residue_position_1based": residue_position,
                "gat_cross_top_residue_letter": residue_letter,
            }
        )
    pd.DataFrame(atom_rows).to_csv(out / "atom_scores.csv", index=False)

    top_residue_indices = np.argsort(residue_mean)[::-1][: min(20, n_residues)]
    top_residues = []
    for idx in top_residue_indices:
        idx = int(idx)
        top_residues.append(
            {
                "graph_index": idx,
                "position_1based": idx + 1 if sequence_valid else None,
                "residue": sequence[idx] if sequence_valid else None,
                "mean_atom_attention": float(residue_mean[idx]),
            }
        )

    metadata = {
        "dataset": args.dataset,
        "strategy": args.strategy,
        "seed": args.seed,
        "split": args.split,
        "split_csv": str(split_path.resolve()),
        "source_row_index": int(row["source_row_index"]),
        "selection_rank": rank,
        "selection_mode": args.selection,
        "active_threshold": args.active_threshold,
        "smiles": smiles,
        "target_key": target_key,
        "affinity": affinity,
        "predictions": predictions,
        "absolute_errors": {
            key: abs(value - affinity) for key, value in predictions.items()
        },
        "num_atoms": n_atoms,
        "num_motifs": int(sample.hetero["motif"].x.size(0)),
        "num_protein_graph_nodes": n_residues,
        "target_sequence_length": len(sequence) if sequence else None,
        "residue_numbering_valid": sequence_valid,
        "checkpoints": {key: str(value.resolve()) for key, value in checkpoint_paths.items()},
        "model_config": {
            "drug_layer": args.drug_layer,
            "motif_layer": args.motif_layer,
            "protein_layer": args.protein_layer,
            "hidden_dim": args.hidden_dim,
            "cross_attention_num_heads": args.num_heads,
            "dropout": args.dropout,
            "use_agg": True,
        },
        "score_definitions": {
            "local_motif_attention": (
                "Mean over LocalAugmentation heads of the softmax weight assigned "
                "to the motif semantic source; the other source is the atom coarse message."
            ),
            "gat_cross_max_attention": (
                "Maximum over protein residues after averaging GAT-model "
                "CrossAttention weights across heads."
            ),
        },
        "local_attention_ranges": {
            backbone: {
                f"layer_{layer_idx + 1}": [float(values.min()), float(values.max())]
                for layer_idx, values in enumerate(local_scores[backbone][:2])
            }
            for backbone in BACKBONES
        },
        "gat_cross_attention_range": [
            float(cross_atom.min()),
            float(cross_atom.max()),
        ],
        "gat_top_residues_by_mean_atom_attention": top_residues,
        "motif_memberships_by_atom": {
            str(idx): values for idx, values in enumerate(memberships)
        },
        "notes": [
            "CrossAttention values indicate model attention, not experimentally validated contacts.",
            "LocalAugmentation color scales are fixed to [0, 1] and centered at 0.5 for all backbones.",
            "Residue positions are trustworthy only when residue_numbering_valid is true.",
        ],
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"[{rank}] wrote {out}")


def main():
    args = parse_args()
    device = choose_device(args)
    checkpoint_paths = resolve_checkpoint_paths(args)
    split_path = resolve_split_path(args)
    frame = read_pairs(split_path)
    drug_features, protein_features = load_feature_caches(args)

    print(f"Device: {device}")
    print(f"Candidate source: {split_path}")
    print("Loading GAT model for active-pair selection...")
    gat_model = model_for("gat", args, checkpoint_paths["gat"], device)
    selected = select_rows(
        frame,
        args,
        drug_features,
        protein_features,
        gat_model,
        device,
    )
    print("Selected active pairs:")
    display_columns = [
        "source_row_index",
        "target_key",
        "affinity",
        "compound_iso_smiles",
    ]
    optional = ["gat_prediction", "gat_absolute_error"]
    print(selected[display_columns + [c for c in optional if c in selected]].to_string(index=False))

    models = {"gat": gat_model}
    for backbone in ("gcn", "gin"):
        models[backbone] = model_for(
            backbone, args, checkpoint_paths[backbone], device
        )

    selected_dataset = make_dataset(
        selected, args, drug_features, protein_features
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, (row_tuple, sample) in enumerate(
        zip(selected.itertuples(index=False), selected_dataset), start=1
    ):
        row = pd.Series(row_tuple._asdict())
        write_sample_outputs(
            output_dir,
            sample,
            row,
            rank,
            args,
            checkpoint_paths,
            models,
            device,
            split_path,
        )
    print(f"Done. Generated {len(selected)} sample directories under {output_dir}")


if __name__ == "__main__":
    main()
