#!/usr/bin/env python
"""compare_models.py — compare two DINO backbones on the SAME site (outputs are model-scoped).

Outputs live at <base>/<model>/<site_id>/patches.zarr. Both ViT-L and 7B use patch-16 + the same
upsample, so they share the cell grid (128x128 at high-res) and cell index i is the SAME ground
cell — only the embedding width differs (1024 vs 4096). For a site embedded by >=2 models this:
  (1) renders the SAME cells as each model's PCA-RGB, side by side (visual structure), and
  (2) reports each model's EFFECTIVE dimensionality — #dims carrying 90% of variance after L2-norm
      — a label-free proxy for how much of its width the model actually uses (is 4096 real or
      redundant vs 1024?), plus the explained-variance curves.

Pass a site_key ("Project/Site") instead of a site_id to also show the SOURCE RGB on the left
(rebuilds the grid like observe_tiles, so it works mid-run before cells.parquet exists).

Usage:
    python compare_models.py <site_id> [n_cells] [model_a model_b ...]      # PCA columns only
    python compare_models.py "<Project>/<Site>" [n_cells] [models...]       # source RGB | per-model PCA
    # default models: dinov3_vitl16 dinov3_vit7b16 ; default n_cells: 4
"""
import os
import sys

import numpy as np
import zarr

import observe_tiles as ot                      # reuse grid rebuild + RGB window read

BASE = os.environ.get("DINO_EMB_ROOT", "/mnt/ai/DeepThought/dino_embeddings")
DEFAULT_MODELS = ["dinov3_vitl16", "dinov3_vit7b16"]


def _written(zp):
    cdir = os.path.join(zp, "c")
    return sorted(int(d) for d in os.listdir(cdir) if d.isdigit()) if os.path.isdir(cdir) else []


def _pca_rgb(z, idxs, sample=20000, seed=0):
    """Per-model shared PCA(3) over the shown cells -> list of (gh,gw,3) in [0,1], 2-98% stretch."""
    from sklearn.decomposition import PCA
    grids = [np.asarray(z[i]).astype(np.float32) for i in idxs]
    gh, gw, c = grids[0].shape
    flat = np.concatenate([g.reshape(-1, c) for g in grids], 0)
    rng = np.random.default_rng(seed)
    pca = PCA(3, random_state=seed).fit(flat[rng.choice(len(flat), min(sample, len(flat)), replace=False)])
    proj = pca.transform(flat)
    lo, hi = np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)
    proj = np.clip((proj - lo) / (hi - lo + 1e-9), 0, 1)
    return [proj[k * gh * gw:(k + 1) * gh * gw].reshape(gh, gw, 3) for k in range(len(grids))]


def _eff_dim(z, written, n_cells=4, sample=15000, var=0.90, max_k=400, seed=0):
    """(embed_dim, #dims for `var` cumulative variance, cumulative-variance curve). L2-normed.

    Truncated randomized SVD (top max_k axes only) — 90% variance is reached well before that for
    DINO, and a full 4096-axis SVD is needlessly slow. If the curve never crosses `var`, returns
    max_k as a floor (rare). CPU-only, no GPU."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    pick = rng.choice(written, min(n_cells, len(written)), replace=False)
    X = np.concatenate([np.asarray(z[i]).reshape(-1, z.shape[-1]).astype(np.float32) for i in pick])
    X = X[rng.choice(len(X), min(sample, len(X)), replace=False)]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    k = min(max_k, X.shape[1] - 1, X.shape[0] - 1)
    p = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(X)
    cum = np.cumsum(p.explained_variance_ratio_)                  # sums to <1 (truncated)
    eff = int(np.searchsorted(cum, var) + 1) if cum[-1] >= var else k
    return z.shape[-1], eff, cum


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    arg1 = sys.argv[1]
    n, models = 4, []
    for a in sys.argv[2:]:
        (models.append(a) if not a.isdigit() else (n := int(a)))
    models = models or DEFAULT_MODELS

    rgb_mode = "/" in arg1                        # site_key -> also show the source RGB column
    if rgb_mode:
        site_id, grid_w, rgb_path = ot._grid_and_rgb(arg1)
    else:
        site_id = arg1

    avail = {}                                          # model -> (zarr, written-set, written-list)
    for m in models:
        zp = os.path.join(BASE, m, site_id, "patches.zarr")
        if not os.path.isdir(zp):
            print(f"[skip] {m}: no patches.zarr for {site_id}"); continue
        w = _written(zp)
        if not w:
            print(f"[skip] {m}: no cells written yet"); continue
        avail[m] = (zarr.open_array(zp, mode="r"), set(w), w)
    if not avail:
        sys.exit(f"no model has {site_id} under {BASE}")

    # cells written in ALL available models (same index == same ground cell)
    common = sorted(set.intersection(*[s for _, s, _ in avail.values()]))
    if not common:
        sys.exit("models have no written cell in common yet — let both run a bit further")
    idxs = common[-n:]
    mods = list(avail)
    print(f"site: {site_id}\nmodels: {mods}  |  common written cells: {len(common)}  |  showing {idxs}\n")

    # ---- quantitative: effective dimensionality ----
    print(f"{'model':24s} {'dim':>5s} {'eff-dim @90% var':>17s} {'ratio used':>11s}")
    cums = {}
    for m, (z, _, w) in avail.items():
        dim, k, cum = _eff_dim(z, w); cums[m] = cum
        print(f"{m:24s} {dim:5d} {k:17d} {k/dim:11.3%}")

    # ---- visual: rows = cells, cols = models ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = {m: _pca_rgb(avail[m][0], idxs) for m in mods}
    src = {i: ot._read_rgb(rgb_path, grid_w.geometry.iloc[i]) for i in idxs} if rgb_mode else {}
    cols = (["__rgb__"] + mods) if rgb_mode else mods       # optional source-RGB column on the left
    fig, axes = plt.subplots(len(idxs), len(cols), figsize=(len(cols) * 3, len(idxs) * 3), squeeze=False)
    for r, i in enumerate(idxs):
        for c, col in enumerate(cols):
            if col == "__rgb__":
                axes[r, c].imshow(src[i])
                title = "source RGB"
            else:
                axes[r, c].imshow(rgb[col][r])
                title = f"{col}\n{avail[col][0].shape[-1]}-d"
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(title, fontsize=9)
        axes[r, 0].text(-0.08, 0.5, f"cell {i}", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle(f"{site_id} — same cells: {'source RGB | ' if rgb_mode else ''}PCA-RGB per model", fontsize=11)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out1 = os.path.join("outputs", f"compare_{site_id}_cells.png")
    fig.savefig(out1, dpi=120, bbox_inches="tight")

    # ---- explained-variance curves ----
    fig2, ax = plt.subplots(figsize=(7, 5))
    for m in mods:
        ax.plot(np.arange(1, len(cums[m]) + 1), cums[m], label=f"{m} ({avail[m][0].shape[-1]}-d)")
    ax.axhline(0.90, ls="--", c="grey", lw=0.8); ax.set_xscale("log")
    ax.set_xlabel("# PCA components (log)"); ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"{site_id} — effective dimensionality"); ax.legend(); ax.grid(alpha=0.3)
    out2 = os.path.join("outputs", f"compare_{site_id}_variance.png")
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    print(f"\n-> {out1}\n-> {out2}")


if __name__ == "__main__":
    main()
