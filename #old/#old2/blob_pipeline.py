"""Site-agnostic pipeline to embed reconstituted blobs with the DINOv3 *activity*.

The activity does ONE monolithic forward per bbox and UPSCALES by native resolution
(x4 at >=0.15 m), so a whole blob would explode the model input (a 2055 px blob -> 2304
padded -> x4 = 9216 px). The tractable route is: tile each blob into small OVERLAPPING
bboxes, run the activity per bbox, then mean-blend the georeferenced outputs.

EVERYTHING here is derived from each site's own .npz metadata — nothing is hardcoded to
one site: native resolution, CRS/SRID, blob extent and the activity's resolution policy
are all computed. Mirrors the activity exactly:

    res thresholds : HIGH=0.07  MED=0.15  LOW=0.3
    patch_size     : res<0.07 ->1024 ; <0.15 ->512 ; else 256
    upsample       : res<0.07 ->1    ; <0.15 ->2   ; else 4     (x2 more if high_res)
    embed GSD      : native_res * 16 / upsample
    model input/bbox: ceil(bbox_native/patch_size)*patch_size * upsample

Pipeline per blob:
    1. reconstitute -> georeferenced RGB GeoTIFF  (blob_to_geotiff)
    2. overlapping, embed-grid-aligned bboxes      -> boxes.fgb  (overlapping_bboxes)
    3. [run the activity: rasters=[blob.tif], bbox=boxes.fgb] -> 1 COG per bbox
    4. mean_mosaic(cogs) -> one blob embedding raster

Usage:
    from blob_pipeline import discover_blobs, prepare_blob, mean_mosaic
    blobs = discover_blobs("/.../<site>/.../v2_tytonai_rg")
    for uid, meta in blobs.items():
        prepare_blob(meta, f"work/{uid}")        # writes blob.tif + boxes.fgb
    # ... run the activity on each ...
    mean_mosaic(glob("work/<uid>/cogs/*.tif"), f"work/<uid>/embedding.tif")
"""
from __future__ import annotations
import glob
import math
import os
import re
from collections import defaultdict
from uuid import uuid4
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import box

TILE_RE = re.compile(r"image_(.+?)_(\d+)_(\d+)\.npz")
RGB_KEYS = ("RED", "GREEN", "BLUE")
# activity resolution thresholds (mirror src/dinov3_embedding/main.py)
HIGH_RES, MED_RES, LOW_RES = 0.07, 0.15, 0.3
DINO_PATCH = 16  # ViT /16


# --------------------------------------------------------------------------- discovery
@dataclass
class BlobMeta:
    uuid: str
    cells: dict = field(default_factory=dict)  # (x,y) -> (w, h, path)
    gt: np.ndarray = None                      # 9-value affine from the npz
    srid: int = None

    @property
    def extent_px(self):
        W = max(x + w for (x, y), (w, h, _) in self.cells.items())
        H = max(y + h for (x, y), (w, h, _) in self.cells.items())
        return W, H

    @property
    def native_res(self):
        return max(abs(self.gt[0]), abs(self.gt[4]))


def discover_blobs(tiles_dir: str, splits=("train", "trainannot", "val", "valannot")) -> dict:
    """Group image_*.npz under tiles_dir into blobs. Reads npz headers only. Site-agnostic.

    `splits` are optional subfolders; files directly under tiles_dir are scanned too.
    Only image_* (not mask_*) tiles are used.
    """
    search_dirs = [tiles_dir] + [os.path.join(tiles_dir, s) for s in splits]
    blobs: dict[str, BlobMeta] = {}
    for d in search_dirs:
        for f in glob.glob(os.path.join(d, "image_*.npz")):
            m = TILE_RE.search(os.path.basename(f))
            if not m:
                continue
            uid, x, y = m.group(1), int(m.group(2)), int(m.group(3))
            with np.load(f, allow_pickle=True) as npz:
                h, w = npz[RGB_KEYS[0]].shape
                gt = np.asarray(npz["GEO_TRANSFORM"], float)
                srid = int(npz["SRID"][0])
            b = blobs.setdefault(uid, BlobMeta(uid))
            b.cells[(x, y)] = (w, h, f)
            if b.gt is None:
                b.gt, b.srid = gt, srid
    return blobs


def blob_array(meta: BlobMeta) -> np.ndarray:
    """Reconstitute the blob RGB as HWC uint8 at native resolution."""
    W, H = meta.extent_px
    canvas = np.zeros((H, W, 3), np.uint8)
    for (x, y), (w, h, path) in meta.cells.items():
        with np.load(path, allow_pickle=True) as npz:
            rgb = np.stack([npz[k] for k in RGB_KEYS], -1)
        canvas[y:y + h, x:x + w] = np.clip(rgb, 0, 255).astype(np.uint8)
    return canvas


# --------------------------------------------------------------------- activity policy
def activity_params(native_res: float, high_res: bool = False) -> dict:
    """patch_size / upsample / embed_gsd for a given native resolution. Mirrors the activity."""
    if native_res < HIGH_RES:
        patch_size, upsample = 1024, 1
    elif native_res < MED_RES:
        patch_size, upsample = 512, 2
    else:
        patch_size, upsample = 256, 4
    if high_res:
        upsample *= 2
    embed_gsd = native_res * DINO_PATCH / upsample
    return {"patch_size": patch_size, "upsample": upsample, "embed_gsd": embed_gsd}


def blob_transform(meta: BlobMeta) -> Affine:
    """rasterio Affine for the blob, from its own geotransform (internally consistent)."""
    g = meta.gt
    return Affine(g[0], g[1], g[2], g[3], g[4], g[5])


# ------------------------------------------------------------------------- step 1: tif
def blob_to_geotiff(meta: BlobMeta, out_tif: str) -> str:
    """Write the reconstituted blob as a georeferenced 3-band RGB GeoTIFF."""
    arr = blob_array(meta)            # HWC
    os.makedirs(os.path.dirname(os.path.abspath(out_tif)), exist_ok=True)
    with rasterio.open(
        out_tif, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=3, dtype="uint8",
        crs=f"EPSG:{meta.srid}", transform=blob_transform(meta),
        compress="DEFLATE", tiled=True, blockxsize=256, blockysize=256,  # activity needs tiled TIFF
    ) as dst:
        for i in range(3):
            dst.write(arr[:, :, i], i + 1)
        dst.descriptions = RGB_KEYS
    return out_tif


# ------------------------------------------------------------------ step 2: bbox tiles
def overlapping_bboxes(
    meta: BlobMeta, out_fgb: str, *, site: str | None = None,
    tile_patches: int = 1, overlap_cells: int = 16, high_res: bool = False,
) -> gpd.GeoDataFrame:
    """Overlapping bboxes covering the blob, aligned to the embedding grid. Site-agnostic.

    tile_patches : bbox native size = tile_patches * patch_size  (1 -> smallest/cheapest pass)
    overlap_cells: overlap between neighbours, in EMBED cells (converted to native px)
    Returns the GeoDataFrame (also written to out_fgb). Geometry in world coords (EPSG:srid).
    """
    res = meta.native_res
    p = activity_params(res, high_res)
    patch = p["patch_size"]
    tile_native = tile_patches * patch
    cell_native = round(p["embed_gsd"] / res)          # native px per embed cell
    overlap_native = overlap_cells * cell_native
    stride = tile_native - overlap_native
    if stride <= 0:
        raise ValueError(f"overlap_cells={overlap_cells} too large for tile {tile_native}px")

    W, H = meta.extent_px
    tf = blob_transform(meta)

    def starts(extent):
        s = list(range(0, max(1, extent - tile_native + 1), stride)) or [0]
        if s[-1] != extent - tile_native:                # ensure last tile hugs the edge
            s.append(max(0, extent - tile_native))
        # snap to embed-cell grid so outputs align for clean averaging
        return sorted({(v // cell_native) * cell_native for v in s})

    rows = []
    for y0 in starts(H):
        for x0 in starts(W):
            w = min(tile_native, W - x0)
            h = min(tile_native, H - y0)
            # pixel rect -> world bbox via the blob transform, inset by half a pixel so the
            # activity's floor(xmin)/ceil(xmax) window math lands on EXACTLY w x h px
            x_min, y_max = tf * (x0 + 0.5, y0 + 0.5)
            x_max, y_min = tf * (x0 + w - 0.5, y0 + h - 0.5)
            rows.append({
                "site": site,                             # provenance: which site this box came from
                "uuid": meta.uuid,
                "project_object_id": uuid4().hex,         # activity reads this -> training_area_id
                "chunk_id": f"{meta.uuid[:8]}_{x0}_{y0}",
                "col_px": x0, "row_px": y0, "w_px": w, "h_px": h,
                "geometry": box(min(x_min, x_max), min(y_min, y_max),
                                max(x_min, x_max), max(y_min, y_max)),
            })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{meta.srid}")
    os.makedirs(os.path.dirname(os.path.abspath(out_fgb)), exist_ok=True)
    gdf.to_file(out_fgb, driver="FlatGeobuf")
    return gdf


def prepare_blob(meta: BlobMeta, out_dir: str, **bbox_kw) -> dict:
    """Write blob.tif + boxes.fgb for one blob; return paths + the derived params."""
    os.makedirs(out_dir, exist_ok=True)
    tif = blob_to_geotiff(meta, os.path.join(out_dir, "blob.tif"))
    gdf = overlapping_bboxes(meta, os.path.join(out_dir, "boxes.fgb"), **bbox_kw)
    return {"raster": tif, "bbox": os.path.join(out_dir, "boxes.fgb"),
            "n_tiles": len(gdf), **activity_params(meta.native_res)}


# ------------------------------------------------------------- step 4: mean-blend merge
def mean_mosaic(cog_paths: list[str], out_tif: str) -> str:
    """Mean-blend a set of georeferenced embedding COGs (same CRS + GSD, grid-aligned).

    Accumulates a per-pixel sum and count across all rasters, divides -> mean in overlaps.
    Site-agnostic: union extent, band count, dtype and grid are inferred from the inputs.
    """
    if not cog_paths:
        raise ValueError("no COGs to merge")
    with rasterio.open(cog_paths[0]) as s0:
        crs, count = s0.crs, s0.count
        a = s0.transform
        px, py = a.a, a.e
    # union bounds
    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    for p in cog_paths:
        with rasterio.open(p) as s:
            b = s.bounds
            xmins.append(b.left); ymins.append(b.bottom)
            xmaxs.append(b.right); ymaxs.append(b.top)
    xmin, ymax = min(xmins), max(ymaxs)
    W = int(round((max(xmaxs) - xmin) / abs(px)))
    H = int(round((ymax - min(ymins)) / abs(py)))
    out_tf = Affine(px, 0, xmin, 0, py, ymax)

    acc = np.zeros((count, H, W), np.float64)
    cnt = np.zeros((H, W), np.float64)
    for p in cog_paths:
        with rasterio.open(p) as s:
            data = s.read()
            col = int(round((s.bounds.left - xmin) / abs(px)))
            row = int(round((ymax - s.bounds.top) / abs(py)))
            h, w = data.shape[1], data.shape[2]
            acc[:, row:row + h, col:col + w] += data
            cnt[row:row + h, col:col + w] += 1
    cnt = np.maximum(cnt, 1)
    mean = (acc / cnt).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(out_tif)), exist_ok=True)
    with rasterio.open(
        out_tif, "w", driver="COG", height=H, width=W, count=count,
        dtype="float32", crs=crs, transform=out_tf, compress="ZSTD",
    ) as dst:
        dst.write(mean)
    return out_tif


if __name__ == "__main__":
    import sys
    train_root = "/home/clement/local_copy_train_data"
    tiles_dir = sys.argv[1] if len(sys.argv) > 1 else \
        f"{train_root}/29Metals/29M_2451_GG_manned/10cm/v2_tytonai_rg"
    out_root = sys.argv[2] if len(sys.argv) > 2 else "blob_work"
    # site identity from the path -> namespaces outputs and is stamped on every box
    site_id = os.path.relpath(tiles_dir, train_root).replace(os.sep, "__")
    site_out = os.path.join(out_root, site_id)
    blobs = discover_blobs(tiles_dir)
    print(f"{len(blobs)} blobs in site '{site_id}'")
    for uid, meta in sorted(blobs.items(), key=lambda kv: -np.prod(kv[1].extent_px)):
        W, H = meta.extent_px
        info = prepare_blob(meta, os.path.join(site_out, uid[:8]), site=site_id)
        print(f"  {uid[:8]} {W}x{H}px res={meta.native_res:.3f} "
              f"patch={info['patch_size']} ups={info['upsample']} "
              f"embGSD={info['embed_gsd']:.2f} -> {info['n_tiles']} bboxes")
