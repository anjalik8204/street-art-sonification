#!/usr/bin/env python3
"""PCA sonification analysis: patch grid -> 7 features -> PCA latent space.

Author: Anjali Kolapalli
Method after Dalziel & Lieck, "Sonification of High-Dimensional Data Using
Harmonic Overtone Amplitudes" (ICAD 2025), adapted for street-art murals.

Each patch of the artwork is reduced to seven numbers -- mean R, G, B, the
standard deviation of R, G, B, and a Laplacian edge-density score. Those
features are standardized and projected onto their first principal components.
Each component then drives the amplitude of one harmonic in an overtone series
over a fixed fundamental, so visually unusual patches sound unusual.

The browser performs the synthesis; this module only produces the latent space.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import laplace
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B", "edge_density"]
N_COMPONENTS = 5
BASE_FREQ = 220.0
TARGET_PATCHES = 200
MIN_PATCH_SIZE = 32
MAX_PATCH_SIZE = 160
MIN_PATCHES_FOR_PCA = 12

# Amplitude mappings from a principal component to a harmonic amplitude.
# The notebook experiments measured each one's dynamic range over 195 patches:
# 10^|PC| put 185 of 195 patches below -40 dB (effectively silent), so it is
# kept only for reference. 2^|PC| is the default: ~20 dB of range, nothing
# inaudible.
AMPLITUDE_MAPPINGS = [
    {"id": "pow2", "label": "2^|PC|", "default": True},
    {"id": "linear", "label": "|PC|", "default": False},
    {"id": "offset", "label": "1 + |PC|", "default": False},
    {"id": "clipped", "label": "clip at 2, then 10^", "default": False},
    {"id": "pow10", "label": "10^|PC| (paper Eq. 2)", "default": False},
]

# Optional emphasis applied to the standardized features before PCA. With
# "edge3" the first component correlates +0.99 with edge density and only +0.05
# with brightness, which makes PC1 directly interpretable as "how busy is this
# patch". "none" reproduces the unweighted baseline.
FEATURE_WEIGHTS: dict[str, list[float]] = {
    "none": [1, 1, 1, 1, 1, 1, 1],
    "edge3": [1, 1, 1, 1, 1, 1, 3],
    "edge3_std2": [1, 1, 1, 2, 2, 2, 3],
    "edge3_std2_mean05": [0.5, 0.5, 0.5, 2, 2, 2, 3],
}


class PCAAnalysisError(RuntimeError):
    """The artwork could not be reduced to a usable latent space."""


def choose_patch_size(width: int, height: int, target: int = TARGET_PATCHES) -> int:
    """Pick a patch size that yields roughly `target` patches for any image.

    The notebook used a fixed 64px patch on an 832x960 mural, giving 195
    patches. Uploads are normalized to an 1800px long edge, so a fixed 64px
    would produce ~590 patches on a landscape photo -- three times the data and
    three times the scan duration. Deriving the size from image area keeps the
    patch count, and therefore the musical pacing, stable across uploads.

    On the notebook's own mural this returns exactly 64, so the original
    experiment is reproduced rather than altered.
    """
    if target < 1:
        raise PCAAnalysisError("Target patch count must be at least 1.")
    raw = math.sqrt(width * height / target)
    snapped = int(round(raw / 8.0) * 8)
    return max(MIN_PATCH_SIZE, min(MAX_PATCH_SIZE, snapped))


def patch_features(patch: np.ndarray) -> np.ndarray:
    """Seven numbers describing one patch: mean RGB, std RGB, edge density."""
    rgb = patch.reshape(-1, 3).astype(np.float64)
    mean_rgb = rgb.mean(axis=0)
    std_rgb = rgb.std(axis=0)
    gray = patch.mean(axis=2)
    edge_density = float(np.abs(laplace(gray)).mean())
    return np.concatenate([mean_rgb, std_rgb, [edge_density]])


def _content_bounds(
    content_box: Sequence[float] | None, width: int, height: int
) -> tuple[int, int, int, int]:
    """Clamp the real-photograph rectangle to the prepared image."""
    if not content_box or len(content_box) != 4:
        return 0, 0, width, height
    x1, y1, x2, y2 = (int(round(float(value))) for value in content_box)
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(x1, min(width, x2))
    y2 = max(y1, min(height, y2))
    if x2 - x1 < MIN_PATCH_SIZE or y2 - y1 < MIN_PATCH_SIZE:
        return 0, 0, width, height
    return x1, y1, x2, y2


def build_patch_grid(
    image: np.ndarray,
    patch_size: int,
    content_box: Sequence[float] | None = None,
) -> tuple[np.ndarray, list[dict[str, int]], dict[str, Any]]:
    """Cut the artwork into patches and extract features from each one.

    Only the real photograph is sampled. `image_preprocessing.prepare_uploaded_image`
    pads extreme panoramas with a flat averaged colour; those pixels have zero
    variance and near-zero edge density, so including them would plant a cluster
    of identical extreme outliers in the feature matrix and distort every
    component. The padding rectangle is recorded during preprocessing, so the
    grid is simply laid out inside it.
    """
    height, width = image.shape[:2]
    left, top, right, bottom = _content_bounds(content_box, width, height)
    usable_width = right - left
    usable_height = bottom - top

    n_cols = usable_width // patch_size
    n_rows = usable_height // patch_size
    if n_cols < 1 or n_rows < 1:
        raise PCAAnalysisError(
            f"The artwork is too small for {patch_size}px patches "
            f"({usable_width}x{usable_height} usable pixels)."
        )

    # Centre the grid in the usable area so the trimmed remainder is split
    # evenly rather than always falling off the right and bottom edges.
    origin_x = left + (usable_width - n_cols * patch_size) // 2
    origin_y = top + (usable_height - n_rows * patch_size) // 2

    rows: list[np.ndarray] = []
    positions: list[dict[str, int]] = []
    for row in range(n_rows):
        for col in range(n_cols):
            y = origin_y + row * patch_size
            x = origin_x + col * patch_size
            patch = image[y:y + patch_size, x:x + patch_size]
            rows.append(patch_features(patch))
            positions.append({"row": row, "col": col, "x": int(x), "y": int(y)})

    geometry = {
        "patch_size": patch_size,
        "cols": n_cols,
        "rows": n_rows,
        "origin": [int(origin_x), int(origin_y)],
        "content_box": [int(left), int(top), int(right), int(bottom)],
        "image_size": {"width": int(width), "height": int(height)},
    }
    return np.array(rows), positions, geometry


def run_pca(
    features: np.ndarray, weights_id: str = "none", n_components: int = N_COMPONENTS
) -> dict[str, Any]:
    """Standardize, optionally emphasise features, then project onto components."""
    if features.shape[0] < MIN_PATCHES_FOR_PCA:
        raise PCAAnalysisError(
            f"Only {features.shape[0]} patches available; "
            f"at least {MIN_PATCHES_FOR_PCA} are needed for a stable fit."
        )
    weights = np.array(FEATURE_WEIGHTS.get(weights_id, FEATURE_WEIGHTS["none"]), dtype=np.float64)

    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    # A patch grid can contain a constant feature (a perfectly flat artwork
    # channel). StandardScaler leaves those columns at zero rather than
    # dividing by zero, so they contribute nothing instead of exploding.
    weighted = standardized * weights

    components = min(n_components, features.shape[1], features.shape[0])
    pca = PCA(n_components=components)
    latent = pca.fit_transform(weighted)
    return {
        "latent": latent,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
        "feature_mean": scaler.mean_,
        "feature_scale": scaler.scale_,
        "weights": weights,
        "n_components": components,
    }


def analyze(
    image_path: Path,
    content_box: Sequence[float] | None = None,
    target_patches: int = TARGET_PATCHES,
    weights_id: str = "none",
    patch_size: int | None = None,
) -> dict[str, Any]:
    """Produce the complete PCA block for one artwork."""
    with Image.open(image_path) as opened:
        image = np.array(opened.convert("RGB"))

    height, width = image.shape[:2]
    if patch_size is None:
        bounds = _content_bounds(content_box, width, height)
        patch_size = choose_patch_size(bounds[2] - bounds[0], bounds[3] - bounds[1], target_patches)

    features, positions, geometry = build_patch_grid(image, patch_size, content_box)
    result = run_pca(features, weights_id)
    latent = result["latent"]

    # Normalizing by the largest absolute component keeps the browser's
    # amplitude curves identical across artworks, so one mural is not silent
    # next to another. The raw values are preserved for inspection.
    pc_abs_max = float(np.abs(latent).max()) or 1.0

    patches = []
    for index, position in enumerate(positions):
        patches.append({
            **position,
            "pc": [round(float(value), 4) for value in latent[index]],
        })

    correlations = {}
    if features.shape[0] >= MIN_PATCHES_FOR_PCA:
        brightness = features[:, :3].mean(axis=1)
        edge = features[:, 6]
        for axis in range(min(2, latent.shape[1])):
            column = latent[:, axis]
            correlations[f"PC{axis + 1}"] = {
                "edge_density": round(float(np.corrcoef(column, edge)[0, 1]), 3),
                "brightness": round(float(np.corrcoef(column, brightness)[0, 1]), 3),
            }

    return {
        "method": "patch-grid-standardized-pca",
        "reference": "Dalziel & Lieck, ICAD 2025 -- harmonic overtone amplitude sonification",
        "author": "Anjali Kolapalli",
        "geometry": geometry,
        "n_patches": len(patches),
        "feature_names": FEATURE_NAMES,
        "weights_id": weights_id,
        "weights": [float(value) for value in result["weights"]],
        "n_components": result["n_components"],
        "explained_variance_ratio": [
            round(float(value), 5) for value in result["explained_variance_ratio"]
        ],
        "total_explained": round(float(result["explained_variance_ratio"].sum()), 5),
        "components": [
            [round(float(value), 5) for value in row] for row in result["components"]
        ],
        "feature_mean": [round(float(value), 5) for value in result["feature_mean"]],
        "feature_scale": [round(float(value), 5) for value in result["feature_scale"]],
        "correlations": correlations,
        "pc_abs_max": round(pc_abs_max, 4),
        "audio": {
            "base_frequency": BASE_FREQ,
            "mappings": AMPLITUDE_MAPPINGS,
            "default_mapping": "pow2",
        },
        "patches": patches,
    }


def write_csv(result: dict[str, Any], features: np.ndarray, path: Path) -> None:
    """Optional parity output matching the notebook's inspection CSV."""
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["row", "col", *FEATURE_NAMES]
        header += [f"PC{i + 1}" for i in range(result["n_components"])]
        writer.writerow(header)
        for index, patch in enumerate(result["patches"]):
            writer.writerow(
                [patch["row"], patch["col"], *features[index].tolist(), *patch["pc"]]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Write the PCA block as JSON here.")
    parser.add_argument("--manifest", type=Path, help="Merge the PCA block into this regions.json.")
    parser.add_argument("--csv", type=Path, help="Also write the notebook-style inspection CSV.")
    parser.add_argument("--target-patches", type=int, default=TARGET_PATCHES)
    parser.add_argument("--patch-size", type=int, default=None, help="Override the derived size.")
    parser.add_argument("--weights", default="none", choices=sorted(FEATURE_WEIGHTS))
    parser.add_argument(
        "--content-box", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2")
    )
    args = parser.parse_args()

    result = analyze(
        args.image,
        content_box=args.content_box,
        target_patches=args.target_patches,
        weights_id=args.weights,
        patch_size=args.patch_size,
    )

    geometry = result["geometry"]
    print(
        f"Grid: {geometry['cols']} x {geometry['rows']} = {result['n_patches']} patches "
        f"at {geometry['patch_size']}px"
    )
    print("Explained variance ratio per component:")
    for index, value in enumerate(result["explained_variance_ratio"], start=1):
        print(f"  PC{index}: {value:.1%}")
    print(f"Total explained: {result['total_explained']:.1%}")
    for name, values in result["correlations"].items():
        print(f"  {name} corr: edge {values['edge_density']:+.2f}, "
              f"brightness {values['brightness']:+.2f}")

    if args.csv:
        features, _, _ = build_patch_grid(
            np.array(Image.open(args.image).convert("RGB")),
            geometry["patch_size"],
            args.content_box,
        )
        write_csv(result, features, args.csv)
        print(f"Saved {args.csv}")

    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved {args.output}")

    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest["pca"] = result
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Merged PCA block into {args.manifest}")


if __name__ == "__main__":
    main()
