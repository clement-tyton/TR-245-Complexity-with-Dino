"""Single-site pipeline: build blobs + boxes + metadata, embed via the activity, mean-blend.

`run_site` is the one-site entry point with a clean tqdm progress bar (activity output muted).
Per blob: blob.tif -> boxes.fgb -> DINO (muted) -> mean-blend -> embedding.tif (+ PCA png),
then the site box-grid + a blobs_metadata.parquet rollup.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import rasterio  # noqa: F401  (kept for parity / optional inspection)
from tqdm import tqdm

from . import config
from .core import (discover_blobs, blob_dir_ids, blob_to_geotiff, ensure_activity_size,
                   overlapping_bboxes, write_blob_metadata)
from .embedding import muted, run_activity_on_blob, mean_mosaic, clean_box_outputs
from .embedding_fast import embed_blob_fast
from .plots import pca_rgb_preview, plot_site_boxes_grid


def run_site(site_dir, embed=True, make_embed_png=False, make_grid=True, clean_intermediates=True,
             tile_patches=config.TILE_PATCHES, high_res=config.HIGH_RES, dino_model=config.DINO_MODEL,
             limit=None, uids=None, show_bar=True, fast=True, bf16=True, resume=True):
    """Build + (optionally) embed every blob of one site. Returns a summary dict.

    limit  : only the first N (largest) blobs.  uids: restrict to these key prefixes.
    show_bar: tqdm per-blob bar (True) or quiet (for the 2-GPU batch runner).
    fast   : COG-free in-RAM embedding (no per-box COG write/re-read); bf16: forward in bf16.
    make_embed_png: per-blob PCA preview (OFF by default — generate on demand via observe.py).
    resume : skip blobs that already have embedding.tif (cheap restart).
    Outputs live under WORK_DIR/<site_id>/<blob_id>/.
    """
    import json
    site_id = config.site_id_from_dir(site_dir)
    site_out = os.path.join(config.WORK_DIR, site_id)
    blobs = discover_blobs(site_dir)
    if uids:
        blobs = {u: m for u, m in blobs.items() if any(u.startswith(p) for p in uids)}
    if not blobs:
        print(f"site '{site_id}': no image tiles under {site_dir} — skipped")
        return {"site": site_id, "out": site_out, "blob_dirs": {}, "n_embedded": 0, "fails": []}

    did = blob_dir_ids(blobs)
    order = sorted(blobs.items(), key=lambda kv: -np.prod(kv[1].extent_px))[:limit]
    bdirs, bmeta, fails = {}, [], []
    it = tqdm(order, desc=site_id[:28], unit="blob", file=sys.__stderr__,
              dynamic_ncols=True, disable=not show_bar, position=0)
    # inner per-box bar (only when the outer bar is shown) so a slow blob isn't a frozen 0/N
    make_box_bar = None
    if show_bar and embed:
        def make_box_bar(n):
            return tqdm(total=n, desc="  boxes", unit="box", leave=False, position=1,
                        file=sys.__stderr__, dynamic_ncols=True)
    for uid, meta in it:
        d = os.path.join(site_out, did[uid])
        bdirs[uid] = d
        W, H = meta.extent_px
        if show_bar:
            it.set_postfix_str(f"{did[uid]} {W}x{H}")
        # resume: already-embedded blob -> reuse its metadata, skip all compute
        meta_json = os.path.join(d, "metadata.json")
        if resume and embed and os.path.exists(os.path.join(d, "embedding.tif")) and os.path.exists(meta_json):
            bmeta.append(json.load(open(meta_json)))
            continue
        try:
            blob_to_geotiff(meta, os.path.join(d, "blob.tif"))
            ensure_activity_size(os.path.join(d, "blob.tif"))   # pad tiny blobs to one window
            gdf = overlapping_bboxes(meta, os.path.join(d, "boxes.fgb"), site=site_id,
                                     tile_patches=tile_patches, high_res=high_res)
            if embed:
                emb_tif = os.path.join(d, "embedding.tif")
                if fast:   # COG-free in-RAM blend (no per-box COG write/re-read)
                    embed_blob_fast(d, emb_tif, dino_model=dino_model, high_res=high_res,
                                    bf16=bf16, make_box_bar=make_box_bar)
                else:      # original path via the activity (writes per-box COGs, then re-reads)
                    _, cogs = run_activity_on_blob(d, dino_model=dino_model, high_res=high_res,
                                                   make_box_bar=make_box_bar)
                    with muted():
                        mean_mosaic(cogs, emb_tif)
                    if clean_intermediates:
                        clean_box_outputs(d)
                if make_embed_png:
                    pca_rgb_preview(emb_tif, os.path.join(d, "embedding_pca.png"))
            bmeta.append(write_blob_metadata(meta, d, gdf, site=site_id,
                                             dino_model=dino_model, high_res=high_res))
        except Exception as e:
            fails.append((did[uid], str(e)))
            it.write(f"  ! {did[uid]} FAILED: {e}")

    n_patches = n_boxes = 0
    if bmeta:
        roll = pd.json_normalize(bmeta).drop(columns=["source_tiles_npz"])
        roll.to_parquet(os.path.join(site_out, "blobs_metadata.parquet"))
        n_patches, n_boxes = int(roll["patch_grid.n_patches"].sum()), int(roll["bbox.n_bbox"].sum())
    if make_grid and bdirs:
        plot_site_boxes_grid(bdirs, os.path.join(site_out, "_boxes_grid.png"))
    n_emb = sum(os.path.exists(os.path.join(x, "embedding.tif")) for x in bdirs.values())
    print(f"site '{site_id}': {len(bdirs)} blobs | {n_emb} embedded | {len(fails)} failed | "
          f"{n_patches} patches | {n_boxes} boxes  ->  {site_out}")
    return {"site": site_id, "out": site_out, "blob_dirs": bdirs, "n_embedded": n_emb,
            "n_blobs": len(bdirs), "fails": fails}
