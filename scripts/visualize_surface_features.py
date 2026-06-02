"""Visualize cached protein surface points and learned surface embeddings."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_residue_coords(pdb_path):
    """Read one CA coordinate per residue, falling back to its atom centroid."""
    residues = {}
    with open(pdb_path, encoding="utf-8") as pdb_file:
        for line in pdb_file:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            residue_id = (line[21].strip(), line[22:26].strip(), line[26].strip())
            coord = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
            residues.setdefault(residue_id, []).append((atom_name, coord))

    coords = []
    for atoms in residues.values():
        ca_coords = [coord for atom_name, coord in atoms if atom_name == "CA"]
        coords.append(ca_coords[0] if ca_coords else np.mean([coord for _, coord in atoms], axis=0))
    return np.asarray(coords, dtype=np.float32)


def pca_scores(features, num_components=3):
    """Project embeddings with PCA using only NumPy."""
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ right_vectors[:num_components].T
    explained = singular_values**2
    explained_ratio = explained[:num_components] / explained.sum()
    return scores, explained_ratio


def style_3d_axis(axis, title):
    axis.set_title(title)
    axis.set_xlabel("x (A)")
    axis.set_ylabel("y (A)")
    axis.set_zlabel("z (A)")
    axis.set_box_aspect((1, 1, 1))


def scatter_surface(axis, xyz, color, title, cmap="viridis"):
    points = axis.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=color,
        cmap=cmap,
        s=18,
        alpha=0.9,
        linewidths=0,
    )
    style_3d_axis(axis, title)
    return points


def visualize(args):
    surface = torch.load(args.surface_points, map_location="cpu", weights_only=False)
    residue_map = torch.load(args.residue_surface, map_location="cpu", weights_only=False)

    xyz = surface["xyz"].float().numpy()
    embeddings = surface["embedding"].float().numpy()
    normals = surface["normals"].float().numpy()
    indices = residue_map["residue_to_surface_indices"].long().numpy()
    distances = residue_map["residue_to_surface_distances"].float().numpy()
    residue_coords = load_residue_coords(args.pdb)

    if residue_coords.shape[0] != indices.shape[0]:
        raise ValueError(
            f"PDB residue count ({residue_coords.shape[0]}) does not match mapping count "
            f"({indices.shape[0]})."
        )

    scores, explained_ratio = pca_scores(embeddings)
    embedding_norm = np.linalg.norm(embeddings, axis=1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(18, 14), constrained_layout=True)
    axis = figure.add_subplot(2, 2, 1, projection="3d")
    axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=16, alpha=0.75, color="#2878b5")
    step = max(len(xyz) // 100, 1)
    axis.quiver(
        xyz[::step, 0],
        xyz[::step, 1],
        xyz[::step, 2],
        normals[::step, 0],
        normals[::step, 1],
        normals[::step, 2],
        length=1.2,
        normalize=True,
        color="#d73027",
        alpha=0.65,
    )
    style_3d_axis(axis, "AAK1 pocket surface points and normals")

    axis = figure.add_subplot(2, 2, 2, projection="3d")
    points = scatter_surface(axis, xyz, embedding_norm, "Surface embedding L2 norm")
    figure.colorbar(points, ax=axis, shrink=0.65, label="Embedding norm")

    axis = figure.add_subplot(2, 2, 3, projection="3d")
    title = f"Surface embedding PCA-1 ({explained_ratio[0] * 100:.1f}% variance)"
    points = scatter_surface(axis, xyz, scores[:, 0], title, cmap="coolwarm")
    figure.colorbar(points, ax=axis, shrink=0.65, label="PCA-1 score")

    axis = figure.add_subplot(2, 2, 4, projection="3d")
    axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=10, alpha=0.25, color="#808080")
    axis.scatter(
        residue_coords[:, 0],
        residue_coords[:, 1],
        residue_coords[:, 2],
        s=42,
        color="#111111",
        label="Residue CA / centroid",
    )
    for residue_index, surface_indices in enumerate(indices):
        for surface_index in surface_indices:
            line = np.stack([residue_coords[residue_index], xyz[surface_index]])
            axis.plot(line[:, 0], line[:, 1], line[:, 2], color="#f28e2b", alpha=0.65, linewidth=0.8)
    style_3d_axis(axis, f"Residue-to-surface mapping (k={residue_map['k']})")
    axis.legend(loc="upper right")

    overview_path = output_dir / "aak1_surface_overview.png"
    figure.savefig(overview_path, dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].hist(embedding_norm, bins=32, color="#2878b5", alpha=0.9)
    axes[0].set_title("Surface embedding norm distribution")
    axes[0].set_xlabel("L2 norm")
    axes[0].set_ylabel("Surface point count")

    axes[1].scatter(scores[:, 0], scores[:, 1], c=scores[:, 2], cmap="viridis", s=18, alpha=0.85)
    axes[1].set_title("Surface embedding PCA projection")
    axes[1].set_xlabel(f"PCA-1 ({explained_ratio[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PCA-2 ({explained_ratio[1] * 100:.1f}%)")

    axes[2].hist(distances.reshape(-1), bins=24, color="#f28e2b", alpha=0.9)
    axes[2].set_title(f"Residue-to-surface distances (k={residue_map['k']})")
    axes[2].set_xlabel("Distance (A)")
    axes[2].set_ylabel("Mapped pair count")

    distribution_path = output_dir / "aak1_surface_distributions.png"
    figure.savefig(distribution_path, dpi=220)
    plt.close(figure)

    print(f"Surface points: {xyz.shape}")
    print(f"Surface embeddings: {embeddings.shape}")
    print(f"Residue mappings: {indices.shape}")
    print(f"PCA explained variance: {explained_ratio}")
    print(f"Saved: {overview_path}")
    print(f"Saved: {distribution_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface_points", default="data/davis/surface_points/AAK1.pt")
    parser.add_argument("--residue_surface", default="data/davis/preprocessed/residue_surface/k3/AAK1.pt")
    parser.add_argument("--pdb", default="data/davis/pocket1_davis/aak1.pdb")
    parser.add_argument("--output_dir", default="outputs/aak1_surface_visualization")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
