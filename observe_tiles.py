#!/usr/bin/env python
"""observe_tiles.py — peek at the patch embeddings a running job is producing.

Reads cells straight from a site's patches.zarr (no GPU, read-only, safe to run while training),
reduces each cell's (gh, gw, C) patch grid to a 3-channel PCA-RGB image, and saves a montage PNG.
Lets you eyeball whether the embeddings carry spatial structure (good) vs look like noise (bad)
without waiting for the site to finish.

Usage:
    python observe_tiles.py                         # newest site under EMB_ROOT, 12 cells
    python observe_tiles.py <site_id_or_zarr_dir>   # explicit site (id under EMB_ROOT, or a path)
    python observe_tiles.py <site> 16               # how many cells to show
"""
import glob
import os
import sys

import numpy as np
import zarr

EMB_ROOT = os.environ.get("DINO_EMB_ROOT", "/mnt/ai/DeepThought/dino_embeddings")


def _find_zarr(arg):
    """Resolve a patches.zarr path from a CLI arg, or auto-pick the most recently written site."""
    if arg:
        for cand in (arg, os.path.join(arg, "patches.zarr"),
                     os.path.join(EMB_ROOT, arg, "patches.zarr"), os.path.join(EMB_ROOT, arg)):
            if os.path.isdir(cand) and os.path.basename(cand.rstrip("/")) == "patches.zarr":
                return cand
        sys.exit(f"no patches.zarr found for {arg!r}")
    zarrs = glob.glob(os.path.join(EMB_ROOT, "*", "patches.zarr"))
    if not zarrs:
        sys.exit(f"no */patches.zarr under {EMB_ROOT}")
    return max(zarrs, key=os.path.getmtime)


def _pca_rgb(grids, sample=20000, seed=0):
    """Fit a shared PCA(3) across all cells' patches, map to RGB, per-channel 2-98% stretch."""
    from sklearn.decomposition import PCA
    gh, gw, c = grids[0].shape
    flat = np.concatenate([g.reshape(-1, c) for g in grids], 0).astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.shape[0], min(sample, flat.shape[0]), replace=False)
    pca = PCA(n_components=3, random_state=seed).fit(flat[idx])
    proj = pca.transform(flat)
    lo, hi = np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)
    proj = np.clip((proj - lo) / (hi - lo + 1e-9), 0, 1)
    return [proj[i * gh * gw:(i + 1) * gh * gw].reshape(gh, gw, 3) for i in range(len(grids))]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    zpath = _find_zarr(arg)
    site = os.path.basename(os.path.dirname(zpath))
    z = zarr.open_array(zpath, mode="r")                 # (n_cells, gh, gw, C) fp16, FULL shape pre-allocated
    # The array is pre-allocated to n_cells; unwritten cells read back as fill_value (0). Only show
    # cells actually written = those with a chunk on disk under patches.zarr/c/<i>/... (newest first).
    cdir = os.path.join(zpath, "c")
    written = sorted(int(d) for d in os.listdir(cdir) if d.isdigit()) if os.path.isdir(cdir) else []
    if not written:
        sys.exit(f"no cells written yet in {zpath}")
    idxs = written[-n:]                                   # the n most-recently-written cells
    print(f"site: {site}\npatches.zarr {z.shape} {z.dtype} — {len(written)}/{z.shape[0]} cells written, "
          f"showing {idxs[0]}..{idxs[-1]}")

    grids = []
    for i in idxs:
        g = np.asarray(z[i]).astype(np.float32)           # (gh, gw, C)
        grids.append(g)
        print(f"  cell_{i:04d}: {g.shape} NaN={np.isnan(g).any()} min={g.min():.2f} max={g.max():.2f}")
    imgs = _pca_rgb(grids)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = min(4, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, img, i in zip(np.atleast_1d(axes).ravel(), imgs, idxs):
        ax.imshow(img)
        ax.set_title(f"{i:04d}", fontsize=8)
    fig.suptitle(f"{site} — PCA-RGB of {len(imgs)} patch grids", fontsize=10)
    fig.tight_layout()
    out = os.path.join("outputs", f"observe_{site}.png")
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
