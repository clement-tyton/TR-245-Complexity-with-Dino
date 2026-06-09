"""Visual QA for blobs and their embeddings — all the plotting in one place.

Every function reads from a blob working directory (blob.tif / boxes.fgb / embedding.tif)
and writes a PNG, returning its path. They are pure QA (no recompute of embeddings).

  pca_rgb_preview      embedding -> PCA(3)->RGB thumbnail
  plot_blob_boxes      one blob: RGB + box outlines | per-pixel overlap heatmap
  plot_site_boxes_grid all blobs of a site: thumbnails with boxes overlaid
  plot_blob_vs_embedding  RGB | PCA-RGB of the embedding, side by side (quality check)
  cluster_blob         RGB | PCA-RGB | KMeans label map (unsupervised classification proxy)
"""
from __future__ import annotations

import os

import numpy as np
import rasterio
import geopandas as gpd

EDGE = "#ff1f5b"   # shared high-contrast box-outline colour


def _pca_rgb(emb, max_fit=200_000):
    """(C,H,W) embedding -> (H,W,3) in [0,1] from its top-3 PCA axes (robust 2-98% stretch).

    Uses the covariance/eigh ('Gram') route on a random subsample for the basis — never the
    full SVD (which would build a multi-GB U matrix and stall on big blobs).
    """
    C, H, W = emb.shape
    X = emb.reshape(C, -1).T.astype(np.float32)
    Xc = X - X.mean(0)
    fit = Xc if len(Xc) <= max_fit else Xc[np.random.default_rng(0).choice(len(Xc), max_fit, replace=False)]
    cov = fit.T @ fit                                   # (C, C)
    _, V = np.linalg.eigh(cov)                          # ascending eigenvalues
    comp = V[:, ::-1][:, :3]                            # top-3 components (C, 3)
    proj = Xc @ comp                                    # (N, 3)
    lo, hi = np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)
    return np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1).reshape(H, W, 3)


def pca_rgb_preview(embedding_tif, out_png):
    """Quick PCA->RGB PNG of an embedding raster, to eyeball that it worked (uses PIL)."""
    from PIL import Image
    with rasterio.open(embedding_tif) as s:
        emb = s.read()
    Image.fromarray((_pca_rgb(emb) * 255).astype(np.uint8)).save(out_png)
    return out_png


def plot_blob_boxes(blob_dir, out_png=None, max_labels=0):
    """One blob: RGB with box outlines (left) + per-pixel coverage heatmap (right).

    The heatmap shows how many boxes cover each pixel (1=single, 2=seam, 4=corner) — the
    clearest way to see the overlap that mean_mosaic averages over.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.colors import BoundaryNorm
    with rasterio.open(os.path.join(blob_dir, "blob.tif")) as s:
        rgb = s.read([1, 2, 3]).transpose(1, 2, 0)
    g = gpd.read_file(os.path.join(blob_dir, "boxes.fgb"))
    H, W = rgb.shape[:2]

    cover = np.zeros((H, W), np.int32)
    for r in g.itertuples():
        cover[r.row_px:r.row_px + r.h_px, r.col_px:r.col_px + r.w_px] += 1
    vmax = max(2, int(cover.max()))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(2 * W / 120 + 1, H / 120))
    axL.imshow(rgb)
    for i, r in enumerate(g.itertuples()):
        axL.add_patch(Rectangle((r.col_px, r.row_px), r.w_px, r.h_px,
                                fill=False, edgecolor=EDGE, linewidth=1.3))
        if max_labels and i < max_labels:
            axL.text(r.col_px + 4, r.row_px + 16, f"{r.col_px},{r.row_px}", color="white", fontsize=6)
    axL.legend(handles=[Patch(facecolor="none", edgecolor=EDGE,
                              label=f"DINO window ({int(g.w_px.iloc[0])} px native)")],
               loc="upper right", fontsize=7, framealpha=0.7)
    axL.set_title(f"{os.path.basename(blob_dir)} — {len(g)} boxes", fontsize=9)
    cmap = plt.get_cmap("viridis", vmax)
    im = axR.imshow(cover, cmap=cmap, norm=BoundaryNorm(np.arange(0.5, vmax + 1.5), cmap.N))
    fig.colorbar(im, ax=axR, ticks=range(1, vmax + 1), fraction=0.046, pad=0.04).set_label(
        "# boxes covering pixel (overlap)", fontsize=7)
    axR.set_title("coverage / overlap", fontsize=9)
    for ax in (axL, axR):
        ax.set_xticks([]); ax.set_yticks([])
    out_png = out_png or os.path.join(blob_dir, "boxes_overlay.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png


def plot_site_boxes_grid(blob_dirs, out_png, cols=6):
    """Thumbnail grid of every blob of a site with its boxes overlaid (title = site name)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    dirs = list(blob_dirs.values())
    rows = (len(dirs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.array(axes).reshape(-1)
    for ax, d in zip(axes, sorted(dirs)):
        with rasterio.open(os.path.join(d, "blob.tif")) as s:
            ax.imshow(s.read([1, 2, 3]).transpose(1, 2, 0))
        g = gpd.read_file(os.path.join(d, "boxes.fgb"))
        for r in g.itertuples():
            ax.add_patch(Rectangle((r.col_px, r.row_px), r.w_px, r.h_px,
                         fill=False, edgecolor=EDGE, linewidth=0.6))
        ax.set_title(f"{os.path.basename(d)} ({len(g)})", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(dirs):]:
        ax.axis("off")
    fig.suptitle(os.path.basename(os.path.dirname(out_png)), fontsize=11)  # site name only
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out_png, dpi=110); plt.close(fig)
    return out_png


def plot_blob_vs_embedding(blob_dir, out_png=None):
    """ASSESS: whole-blob RGB | PCA(3)->RGB of its embedding, side by side (GSD read from rasters)."""
    import matplotlib.pyplot as plt
    with rasterio.open(os.path.join(blob_dir, "blob.tif")) as s:
        rgb = s.read([1, 2, 3]).transpose(1, 2, 0); native = abs(s.transform.a)
    with rasterio.open(os.path.join(blob_dir, "embedding.tif")) as s:
        emb = s.read(); egsd = abs(s.transform.a)
    pca_rgb = _pca_rgb(emb)
    gh, gw = emb.shape[1], emb.shape[2]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6))
    a.imshow(rgb); a.set_title(f"{os.path.basename(blob_dir)} — RGB ({rgb.shape[1]}x{rgb.shape[0]} px @{native:.2f}m)")
    b.imshow(pca_rgb, interpolation="nearest")
    b.set_title(f"DINO embedding — PCA(3)->RGB ({gw}x{gh} patches @{egsd:.2f}m)")
    for ax in (a, b):
        ax.set_xticks([]); ax.set_yticks([])
    out_png = out_png or os.path.join(blob_dir, "blob_vs_embedding.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png


def cluster_blob(blob_dir, k=6, normalize=True, out_png=None):
    """RGB | PCA-RGB | KMeans(k) label map on the patch embeddings. Returns the PNG path.

    Unsupervised proxy for classification — shows whether DINO patches separate vegetation /
    ground / etc. Prints the per-cluster coverage fractions.
    """
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    with rasterio.open(os.path.join(blob_dir, "blob.tif")) as s:
        rgb = s.read([1, 2, 3]).transpose(1, 2, 0)
    with rasterio.open(os.path.join(blob_dir, "embedding.tif")) as s:
        emb = s.read()
    C, gh, gw = emb.shape
    X = emb.reshape(C, -1).T.astype(np.float32)
    if normalize:
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)   # cosine-ish
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    lab = km.labels_.reshape(gh, gw)
    frac = np.bincount(km.labels_, minlength=k) / km.labels_.size
    pca_rgb = _pca_rgb(emb)

    cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, 10))[:k])
    fig, (a, b, c2) = plt.subplots(1, 3, figsize=(19, 6))
    a.imshow(rgb); a.set_title(f"{os.path.basename(blob_dir)} — RGB")
    b.imshow(pca_rgb, interpolation="nearest"); b.set_title("DINO PCA(3)->RGB")
    c2.imshow(lab, cmap=cmap, vmin=0, vmax=k - 1, interpolation="nearest")
    c2.set_title(f"KMeans k={k} on patch embeddings")
    c2.legend(handles=[Patch(facecolor=cmap(i), label=f"c{i}: {frac[i]*100:.0f}%") for i in range(k)],
              loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    for ax in (a, b, c2):
        ax.set_xticks([]); ax.set_yticks([])
    out_png = out_png or os.path.join(blob_dir, f"cluster_k{k}.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"{os.path.basename(blob_dir)} KMeans k={k}: " + ", ".join(f"c{i}={frac[i]*100:.0f}%" for i in range(k)))
    return out_png
