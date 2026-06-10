"""Persistence (store): per-site output dirs, the embed loop, and the manifest.

Named 'store' (not 'io') to avoid clashing with the stdlib ``io`` module on sys.path.

Output layout (per site):
  <emb_root>/<site_id>/patches/cell_XXXX.npz      patch grids (gh,gw,C)
  <emb_root>/<site_id>/{cls_vecs.npy, cells.fgb}  per-site CLS + QGIS view
  <emb_root>/cells/site_id=<site_id>/cells.parquet  Hive partition (geometry + cls + id)

DuckDB later, ALL sites at once (site_id comes from the partition path):
  duckdb.sql("SELECT site_id, cell_id, cls FROM
      read_parquet('<emb_root>/cells/**/*.parquet', hive_partitioning=true)")
"""
from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm

import config  # noqa: F401
from dino import embed_cell_tokens


def site_emb_dirs(site_id, emb_root=config.EMB_ROOT):
    """Create + return (patch_dir, part_dir) for a site under the embeddings root."""
    patch_dir = os.path.join(emb_root, site_id, "patches")
    part_dir = os.path.join(emb_root, "cells", f"site_id={site_id}")   # Hive partition
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(part_dir, exist_ok=True)
    return patch_dir, part_dir


def embed_grid(act, model, device, grid_w, webmap_path, patch_dir, upsample=None,
               show_bar=True, desc=""):
    """Embed every cell of grid_w; write cell_XXXX.npz (patch_grid (gh,gw,C)). Keep ALL patches.

    Returns (npz_paths, cls_vecs (n_cells, C)). cls_vecs row i <-> cell_XXXX.npz <-> grid_w row i.
    """
    geoms = list(grid_w.geometry)
    npz_paths, cls_list = [], []
    it = tqdm(geoms, desc=desc[:22] or "cells", unit="cell") if show_bar else geoms
    for i, geom in enumerate(it):
        _, emb, cls, _ = embed_cell_tokens(act, model, device, tuple(geom.bounds),
                                           webmap_path, upsample=upsample)   # emb = (C, gh, gw)
        out = os.path.join(patch_dir, f"cell_{i:04d}.npz")
        np.savez_compressed(out, patch_grid=emb.transpose(1, 2, 0).astype(np.float32))  # (gh,gw,C)
        npz_paths.append(out)
        cls_list.append(cls)
    cls_vecs = np.asarray(cls_list, dtype=np.float32) if cls_list else np.zeros((0, 0), np.float32)
    return npz_paths, cls_vecs


def write_manifest(grid_w, npz_paths, cls_vecs, site_id, part_dir, emb_root=config.EMB_ROOT):
    """Stable cell_id linking geometry <-> patch_grid file <-> CLS (order-safe, CRS kept).

    Writes the Hive-partitioned GeoParquet (the source of truth), a FlatGeobuf for QGIS, and
    cls_vecs.npy. Hive partition dir encodes site_id, so DuckDB reads it from the path.
    """
    manifest = grid_w.reset_index(drop=True)[["geometry"]].copy()
    manifest.insert(0, "cell_id", np.arange(len(manifest)))               # cell_id == cell_XXXX.npz == cls row
    manifest["patch_npz"] = [os.path.relpath(p, emb_root) for p in npz_paths]  # relative to dataset root
    manifest["cls"] = cls_vecs.tolist()                                  # CLS vector inline (self-contained)
    manifest.to_parquet(os.path.join(part_dir, "cells.parquet"))         # Hive-partitioned GeoParquet
    manifest[["cell_id", "patch_npz", "geometry"]].to_file(
        os.path.join(emb_root, site_id, "cells.fgb"), driver="FlatGeobuf")  # QGIS view (reorders on read)
    np.save(os.path.join(emb_root, site_id, "cls_vecs.npy"), cls_vecs)    # cls_vecs[cell_id], order-stable
    return manifest


def write_site_meta(site_id, part_dir, ginfo, npz_paths, *, dino_model, high_res, upsample,
                    emb_root=config.EMB_ROOT):
    """Make the embeddings SELF-DESCRIBING: a meta.json next to cells.parquet capturing the
    grid/upscale geometry, the model, and the git commit. Without this, the only record of the
    upsample / patch size / embedding dim is the patch_grid shape + cell bbox (derivable) or the
    stdout log (lost). patch_ground_m = cell_ground_m / gw. Written twice (Hive partition dir +
    per-site dir) so it's found from either side. Best-effort on commit; never fails the run."""
    import datetime
    import json
    import subprocess
    gh, gw, c = (None, None, None)
    if npz_paths:
        sh = np.load(npz_paths[0])["patch_grid"].shape                    # (gh, gw, C)
        gh, gw, c = int(sh[0]), int(sh[1]), int(sh[2])
    cell_m = ginfo.get("cell_ground_m")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
                                capture_output=True, text=True).stdout.strip() or None
    except Exception:
        commit = None
    meta = {
        "site_id": site_id,
        "dino_model": dino_model, "high_res": bool(high_res), "upsample": upsample,
        "dino_dtype": config.DINO_DTYPE,
        "embed_dim": c, "patch_grid": [gh, gw],
        "patch_px": ginfo.get("patch_px"), "native_res_m": ginfo.get("native_res_m"),
        "cell_ground_m": cell_m,
        "patch_ground_m": (cell_m / gw if (cell_m and gw) else None),
        "n_cells": ginfo.get("n_cells", len(npz_paths)),
        "git_commit": commit,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "ginfo": ginfo,
    }
    for d in (part_dir, os.path.join(emb_root, site_id)):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    return meta
