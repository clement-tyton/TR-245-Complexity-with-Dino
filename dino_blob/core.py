"""Blob construction, bounding boxes, and per-blob metadata.

A "blob" is one source image reconstituted from its 384-px training tiles. This module:
  - discovers blobs from a site's tiles (any naming convention),
  - writes each blob as a georeferenced RGB GeoTIFF (the activity's raster input),
  - emits the overlapping bbox grid the activity embeds (one box per DINO window),
  - records per-blob metadata (source tiles, patch grid, box count, resolution).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from collections import Counter
from uuid import uuid4

import numpy as np
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import box

from .config import (TILE_RE, RGB_KEYS, HIGH_RES_T, MED_RES_T, DINO_MODEL, HIGH_RES,
                     TILE_PATCHES, OVERLAP_CELLS, activity_params)


# --------------------------------------------------------------------------- discovery
class BlobMeta:
    """One source image (a blob): its tiles, geotransform and SRID."""

    def __init__(self, uuid):
        self.uuid = uuid
        self.cells = {}            # (x, y) -> (w, h, path)
        self.gt = None             # 9-value affine from the npz
        self.srid = None

    @property
    def extent_px(self):
        W = max(x + w for (x, y), (w, h, _) in self.cells.items())
        H = max(y + h for (x, y), (w, h, _) in self.cells.items())
        return W, H

    @property
    def native_res(self):
        return max(abs(self.gt[0]), abs(self.gt[4]))


def _clean_key(stem: str) -> str:
    """Blob key from a tile stem; handles 'image_<uuid>' (29Metals) and '<name>_part' (most sites)."""
    if stem.startswith("image_"):
        stem = stem[len("image_"):]
    if stem.endswith("_part"):
        stem = stem[:-len("_part")]
    return stem


def discover_blobs(tiles_dir: str, splits=("train", "val")) -> dict:
    """Group image tiles under tiles_dir's split subfolders into blobs (headers only).

    Site-agnostic: matches <key>_<col>_<row>.npz for any naming. Only train/val are scanned
    (masks share names but live in *annot dirs, so they're never read as images).
    """
    blobs: dict[str, BlobMeta] = {}
    for s in splits:
        for f in glob.glob(os.path.join(tiles_dir, s, "*.npz")):
            m = TILE_RE.search(os.path.basename(f))
            if not m:
                continue
            uid, x, y = _clean_key(m.group(1)), int(m.group(2)), int(m.group(3))
            with np.load(f, allow_pickle=True) as npz:
                h, w = npz[RGB_KEYS[0]].shape
                gt = np.asarray(npz["GEO_TRANSFORM"], float)
                srid = int(npz["SRID"][0])
            b = blobs.setdefault(uid, BlobMeta(uid))
            b.cells[(x, y)] = (w, h, f)
            if b.gt is None:
                b.gt, b.srid = gt, srid
    return blobs


def blob_dir_ids(blobs: dict) -> dict:
    """Map each blob key -> a folder id: short uid[:8] when unique, else the full sanitized key.
    Prevents the prefix collisions named tiles (TurnerCamp_TD01/02...) would otherwise cause."""
    san = {u: re.sub(r"[^A-Za-z0-9._-]", "_", u) for u in blobs}
    short = {u: san[u][:8] for u in blobs}
    clash = {k for k, c in Counter(short.values()).items() if c > 1}
    return {u: (san[u] if short[u] in clash else short[u]) for u in blobs}


# ----------------------------------------------------------------------- reconstruction
def blob_array(meta: BlobMeta) -> np.ndarray:
    """Reconstitute the blob RGB as an HWC uint8 array at native resolution."""
    W, H = meta.extent_px
    canvas = np.zeros((H, W, 3), np.uint8)
    for (x, y), (w, h, path) in meta.cells.items():
        with np.load(path, allow_pickle=True) as npz:
            rgb = np.stack([npz[k] for k in RGB_KEYS], -1)
        canvas[y:y + h, x:x + w] = np.clip(rgb, 0, 255).astype(np.uint8)
    return canvas


def blob_transform(meta: BlobMeta) -> Affine:
    g = meta.gt
    return Affine(g[0], g[1], g[2], g[3], g[4], g[5])


def blob_to_geotiff(meta: BlobMeta, out_tif: str) -> str:
    """Write the reconstituted blob as a tiled, georeferenced 3-band RGB GeoTIFF."""
    arr = blob_array(meta)
    os.makedirs(os.path.dirname(os.path.abspath(out_tif)), exist_ok=True)
    with rasterio.open(out_tif, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=3, dtype="uint8", crs=f"EPSG:{meta.srid}",
                       transform=blob_transform(meta), compress="DEFLATE",
                       tiled=True, blockxsize=256, blockysize=256) as dst:  # activity needs tiled
        for i in range(3):
            dst.write(arr[:, :, i], i + 1)
        dst.descriptions = RGB_KEYS
    return out_tif


def ensure_activity_size(blob_tif: str):
    """Pad a blob.tif (edge) up to one DINO window if the blob is smaller (tiny satellite blobs).

    Keeps the transform so bboxes on the original extent still align; the activity trims the
    embedding back to the real extent. patch_size is read from the tif so it matches the activity.
    Returns (was_padded, target_size_px).
    """
    with rasterio.open(blob_tif) as s:
        res = max(abs(s.transform.a), abs(s.transform.e))
        H, W, crs, tf, desc = s.height, s.width, s.crs, s.transform, s.descriptions
        arr = s.read()
    patch = 1024 if res < HIGH_RES_T else (512 if res < MED_RES_T else 256)
    target = math.ceil(max(H, W) / patch) * patch
    if H >= target and W >= target:
        return False, target
    ph, pw = max(0, target - H), max(0, target - W)
    arr = np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="edge")   # original stays top-left
    with rasterio.open(blob_tif, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=3, dtype="uint8", crs=crs, transform=tf, compress="DEFLATE",
                       tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(arr)
        dst.descriptions = desc
    return True, target


# -------------------------------------------------------------------------- bounding box
def overlapping_bboxes(meta: BlobMeta, out_fgb: str, site=None, tile_patches=TILE_PATCHES,
                       overlap_cells=OVERLAP_CELLS, high_res=HIGH_RES) -> gpd.GeoDataFrame:
    """Overlapping boxes covering the blob, snapped to the embed grid so outputs average cleanly.

    Each box is one DINO window (tile_patches * patch_size native px). World coords are inset
    half a pixel so the activity's floor/ceil window lands on exactly NxN px. `site` is stamped
    on every feature for traceability. Returns the GeoDataFrame (also written to out_fgb).
    """
    res = meta.native_res
    p = activity_params(res, high_res)
    tile_native = tile_patches * p["patch_size"]
    cell_native = round(p["embed_gsd"] / res)          # native px per embed cell
    overlap_native = overlap_cells * cell_native
    stride = tile_native - overlap_native
    if stride <= 0:
        raise ValueError(f"overlap_cells={overlap_cells} too large for tile {tile_native}px")
    W, H = meta.extent_px
    tf = blob_transform(meta)

    def starts(extent):
        s = list(range(0, max(1, extent - tile_native + 1), stride)) or [0]
        if s[-1] != extent - tile_native:
            s.append(max(0, extent - tile_native))
        return sorted({(v // cell_native) * cell_native for v in s})

    rows = []
    for y0 in starts(H):
        for x0 in starts(W):
            w = min(tile_native, W - x0)
            h = min(tile_native, H - y0)
            x_min, y_max = tf * (x0 + 0.5, y0 + 0.5)
            x_max, y_min = tf * (x0 + w - 0.5, y0 + h - 0.5)
            rows.append({"site": site, "uuid": meta.uuid, "project_object_id": uuid4().hex,
                         "chunk_id": f"{meta.uuid[:8]}_{x0}_{y0}",
                         "col_px": x0, "row_px": y0, "w_px": w, "h_px": h,
                         "geometry": box(min(x_min, x_max), min(y_min, y_max),
                                         max(x_min, x_max), max(y_min, y_max))})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{meta.srid}")
    os.makedirs(os.path.dirname(os.path.abspath(out_fgb)), exist_ok=True)
    gdf.to_file(out_fgb, driver="FlatGeobuf")
    return gdf


# ----------------------------------------------------------------------------- metadata
def write_blob_metadata(meta: BlobMeta, blob_dir: str, gdf, site,
                        dino_model=DINO_MODEL, high_res=HIGH_RES) -> dict:
    """Write blob_dir/metadata.json (source tiles, boxes, patch grid). Returns the dict.

    Patch grid is read from embedding.tif when present (actual), else predicted from the GSD.
    """
    W, H = meta.extent_px
    p = activity_params(meta.native_res, high_res)
    src_tiles = sorted(os.path.basename(path) for (_, _), (_, _, path) in meta.cells.items())

    emb_path = os.path.join(blob_dir, "embedding.tif")
    if os.path.exists(emb_path):
        with rasterio.open(emb_path) as s:
            gh, gw, embed_dim, grid_src = s.height, s.width, s.count, "embedding.tif"
    else:
        ratio = p["embed_gsd"] / meta.native_res
        embed_dim = 4096 if dino_model == "dinov3_vit7b16" else 1024
        gh, gw, grid_src = round(H / ratio), round(W / ratio), "predicted"

    md = {
        "site": site,
        "uuid": meta.uuid,
        "n_source_tiles": len(src_tiles),
        "source_tiles_npz": src_tiles,
        "native_resolution_m": round(meta.native_res, 4),
        "srid": meta.srid,
        "blob_native_px": {"width": W, "height": H},
        "activity": {"dino_model": dino_model, "high_res": high_res,
                     "patch_size": p["patch_size"], "upsample": p["upsample"],
                     "embed_gsd_m": round(p["embed_gsd"], 4)},
        "bbox": {"n_bbox": int(len(gdf)), "bbox_px": int(gdf.w_px.iloc[0]),
                 "overlap_cells": OVERLAP_CELLS,
                 "overlap_px": OVERLAP_CELLS * round(p["embed_gsd"] / meta.native_res)},
        "patch_grid": {"grid_h": int(gh), "grid_w": int(gw), "n_patches": int(gh * gw),
                       "embed_dim": int(embed_dim), "source": grid_src},
    }
    with open(os.path.join(blob_dir, "metadata.json"), "w") as fh:
        json.dump(md, fh, indent=2)
    return md
