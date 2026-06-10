# %% [markdown]
# replsession_onesite.py — one-site REPL: study area -> grid -> DINO (step by step)
# ===================================================================================
# Plan (whiteboard):
#   1) Build the STUDY AREA induced by the tiles union -> convex hull   <- this file now
#   2) Crop that study area with the TWebMap boundaries
#   3) Build the bbox->grid (tile size depends on site resolution)
#   4) DINO embeddings on the grid (maybe control intersection with annotations)
#   5) a) intersect embedding clusters with annotations & interpret
#      b) site-complexity metrics (site level / grid level / distribution)
#
# Tiles overlap and are dense (no holes inside the hull), so one convex-hull area per
# site replaces the old per-blob grouping. Run cell by cell; stop and check each visual.

# %% CELL 1 — imports + pick a site -------------------------------------------------
import glob
import os

# --- env the dinov3 activity reads (MUST be set before importing it) ---------------
os.environ["DINO_WEIGHTS_FOLDER"] = os.environ.get(
    "DINO_WEIGHTS_FOLDER",
    "/home/clement/Desktop/projets/1_Core_tyton_AI/tytonai-python-activities/"
    "dinov3_embedding/test_data/dinov3_weights")
os.environ.setdefault("S3_FILE_BUCKET", "")          # empty -> S3Mock uses plain local files
os.environ.setdefault("SAVE_DEBUG_IMG", "false")

import numpy as np
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

import math

import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as tf_from_bounds
from bbox_to_tile_grid.tilegrid import create_adaptive_grid          # bbox->grid activity
from dinov3_embedding.main import HIGH_RES, MED_RES                   # resolution thresholds (m/px)

import asyncio
import contextlib

from tytonai.test.s3_mock import S3Mock
from dinov3_embedding.io_schema.model import Input
from dinov3_embedding.main import Dinov3Embedding


SITE_DIR = "/home/clement/local_copy_train_data/BHP Creeks 2022/Manned Bens Oasis Post Dry/10cm/v2_tytonai_rg"
PIC = "outputs/pictures"
os.makedirs(PIC, exist_ok=True)
print("site:", SITE_DIR.split("/local_copy_train_data/")[-1])


# %% CELL 2 — read the tile bounding boxes (prerequisite) ---------------------------
def read_tile_bboxes(site_dir, splits=("train", "val")):
    """One row per tile = its real-world bbox (from GEO_TRANSFORM + tile shape).

    GEO_TRANSFORM = [px, 0, ox, 0, -px, oy, ...]; bbox = (ox, oy - h*px, ox + w*px, oy).
    Returns a GeoDataFrame (tile id, split, path, w, h, geometry) in the tiles' CRS.
    """
    rows, srid = [], None
    for s in splits:
        for f in glob.glob(os.path.join(site_dir, s, "*.npz")):
            with np.load(f, allow_pickle=True) as d:
                gt = np.asarray(d["GEO_TRANSFORM"], float)
                h, w = d["RED"].shape
                srid = int(d["SRID"][0])
            ox, oy, px, py = gt[2], gt[5], gt[0], gt[4]          # py is negative
            geom = box(ox, oy + h * py, ox + w * px, oy)         # (xmin, ymin, xmax, ymax)
            rows.append({"tile": os.path.basename(f)[:-4], "split": s, "path": f,
                         "w": int(w), "h": int(h), "geometry": geom})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{srid}")


tiles = read_tile_bboxes(SITE_DIR)
xmin, ymin, xmax, ymax = tiles.total_bounds
print(f"{len(tiles)} tiles | CRS {tiles.crs} | extent {xmax-xmin:.0f} x {ymax-ymin:.0f} m")
print(f"  tile sizes (w x h) seen: {sorted(set(zip(tiles.w, tiles.h)))[:6]} ...")
print(tiles[["tile", "w", "h"]].head(4).to_string(index=False))


# %% CELL 3 — visual helper: see all tile bboxes in world coords --------------------
def plot_tiles(gdf, out_png=os.path.join(PIC, "01_tile_bboxes.png"), title=None):
    """Draw every tile bbox (outline) in world coordinates."""
    fig, ax = plt.subplots(figsize=(11, 11))
    gdf.boundary.plot(ax=ax, color="#1f77b4", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title(title or f"{len(gdf)} tile bboxes — {os.path.basename(os.path.dirname(os.path.dirname(SITE_DIR)))}")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out_png


print("->", plot_tiles(tiles))


# %% CELL 4 — STEP 1: crop the tiles by the webmap extent ---------------------------
# Crop the tile union to the webmap raster's FULL EXTENT (its bounding rectangle).
# The webmap is the site's official "whole extent" deliverable, so its bounds define
# the area we keep; tiles (or parts of tiles) outside it are dropped/clipped.
WEBMAP = "/mnt/spatial/DeepThought/SiteData/BHP Creeks 2022/Manned Bens Oasis Post Dry/Raster/WholeExtent/10cm/RGB_webmap.tif"


def webmap_extent(webmap_path, dst_crs):
    """Webmap raster's full extent as a 1-row GeoDataFrame (reprojected to dst_crs)."""
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(webmap_path) as r:
        src_crs, b = r.crs, r.bounds
    bb = transform_bounds(src_crs, dst_crs, *b) if src_crs else tuple(b)
    return gpd.GeoDataFrame({"src_crs": [str(src_crs)]}, geometry=[box(*bb)], crs=dst_crs)


def crop_tiles_to_webmap(gdf, webmap_path):
    """Clip tile bboxes to the webmap extent. Returns (clipped_tiles, extent_gdf)."""
    ext = webmap_extent(webmap_path, gdf.crs)
    clipped = gpd.clip(gdf, ext)                      # crops geometries to the rectangle
    return clipped, ext


tiles_clip, extent = crop_tiles_to_webmap(tiles, WEBMAP)
print(f"webmap CRS {extent.src_crs.iloc[0]} -> tiles CRS {tiles.crs}")
print(f"  tiles inside extent: {len(tiles_clip)}/{len(tiles)}  (dropped {len(tiles)-len(tiles_clip)})")


# %% CELL 5 — visual helper: tiles + webmap extent ----------------------------------
def plot_webmap_crop(gdf, clipped, ext, out_png=os.path.join(PIC, "02_webmap_crop.png")):
    """Original tiles (grey), kept tiles (blue), webmap extent rectangle (orange)."""
    fig, ax = plt.subplots(figsize=(11, 11))
    gdf.boundary.plot(ax=ax, color="#cccccc", linewidth=0.4)
    clipped.boundary.plot(ax=ax, color="#1f77b4", linewidth=0.5)
    ext.boundary.plot(ax=ax, color="#ff7f0e", linewidth=2)
    ax.set_aspect("equal")
    ax.set_title(f"webmap-extent crop — {len(clipped)}/{len(gdf)} tiles kept")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out_png


print("->", plot_webmap_crop(tiles, tiles_clip, extent))


# %% CELL 6 — STEP 2: study area = convex hull of the CROPPED tiles union ---------
def study_area(gdf):
    """The site's STUDY AREA: convex hull of the union of the (cropped) tile bboxes.

    Tiles overlap and pack densely along the mapped corridor, so their union is a
    single (possibly concave) footprint; its convex hull is the area the model was
    trained on. Returns a 1-row GeoDataFrame (geometry + area/perimeter, same CRS).
    """
    union = gdf.geometry.union_all()                 # dissolve all tile bboxes into one footprint
    hull = union.convex_hull
    fill = union.area / hull.area                     # how "convex" the layout is (1 = no overhang)
    return gpd.GeoDataFrame(
        {"area_km2": [hull.area / 1e6], "perim_km": [hull.length / 1e3], "fill_ratio": [fill]},
        geometry=[hull], crs=gdf.crs,
    )


area = study_area(tiles_clip)
print(area.drop(columns="geometry").to_string(index=False))
print(f"  hull covers {area.area_km2.iloc[0]:.2f} km^2 | tiles fill {area.fill_ratio.iloc[0]*100:.0f}% of it")


# %% CELL 7 — visual helper: cropped tiles + study-area hull ----------------------
def plot_study_area(gdf, hull_gdf, out_png=os.path.join(PIC, "03_study_area.png"), title=None):
    """Tile bboxes (blue) under the convex-hull study area (green)."""
    fig, ax = plt.subplots(figsize=(11, 11))
    hull_gdf.plot(ax=ax, facecolor="#2ca02c", edgecolor="#2ca02c", alpha=0.10, linewidth=2)
    gdf.boundary.plot(ax=ax, color="#1f77b4", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title(title or f"study area (hull) — {len(gdf)} tiles, {hull_gdf.area_km2.iloc[0]:.2f} km^2")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out_png


print("->", plot_study_area(tiles_clip, area))


# Ok now we will use the bbox to grid activity from this github repo


# %% CELL 8 — STEP 3: tile the study area into a grid (cell size from site resolution)
# bbox_to_tile_grid over the study-area hull. Cell size (px) = the DINO patch_size for
# this site's native resolution (config.activity_params), so each grid cell is exactly
# one DINO window -> step 4 runs the embedding per cell with no resampling.


def dino_patch_size(res):
    """DINOv3 patch size (px) for a native resolution — mirrors the activity's policy."""
    return 1024 if res < HIGH_RES else 512 if res < MED_RES else 256


def snap_bbox_to_patch(bounds, transform, patch):
    """Expand a world bbox OUTWARD to span an integer number of patch-sized cells,
    aligned to the raster's pixel grid -> every grid cell is then EXACTLY patch x patch.
    """
    inv = ~transform
    cols, rows = zip(*[inv * (bounds[i], bounds[j]) for i in (0, 2) for j in (1, 3)])
    c0, r0 = math.floor(min(cols)), math.floor(min(rows))
    c1 = c0 + math.ceil((math.ceil(max(cols)) - c0) / patch) * patch   # extend to a multiple of patch
    r1 = r0 + math.ceil((math.ceil(max(rows)) - r0) / patch) * patch
    (x0, y0), (x1, y1) = transform * (c0, r0), transform * (c1, r1)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def cell_data_coverage(webmap_path, geoms, n=16):
    """Fraction of each cell that has actual RGB data (any of R/G/B != 0), via tiny decimated
    windowed reads. We read ONLY bands 1-3 and IGNORE the alpha band on purpose: alpha can
    flag pixels transparent where real imagery still exists. geoms must be in the raster CRS."""
    covs = []
    with rasterio.open(webmap_path) as r:
        for g in geoms:
            win = from_bounds(*g.bounds, transform=r.transform)
            a = r.read((1, 2, 3), window=win, boundless=True, fill_value=0, out_shape=(3, n, n))
            covs.append(float((a != 0).any(axis=0).mean()))
    return np.array(covs)


def build_tile_grid(study_gdf, tiles_gdf, webmap_path, tile_patches=1, min_data_cov=0.02):
    """Perfect patch x patch grid over the study area; only cells that matter are kept.

    1) snap the hull's bbox outward to an exact multiple of patch -> all cells are full squares;
    2) keep cells intersecting >=1 training tile (inside the study area);
    3) drop cells with no RGB data (all-black) so DINO never runs on void — alpha ignored.
    Returns (kept-grid in study CRS, info dict).
    """
    with rasterio.open(webmap_path) as r:
        gt, wcrs, res = r.transform, r.crs, abs(r.transform.a)
    patch = dino_patch_size(res) * tile_patches
    bbox = snap_bbox_to_patch(study_gdf.to_crs(wcrs).total_bounds, gt, patch)
    grid = create_adaptive_grid(bbox, None, gt, wcrs, patch, patch, fixed_size=True)  # full square grid
    grid = (grid.set_crs(wcrs) if grid.crs is None else grid).reset_index(drop=True)
    n_full = len(grid)
    # (a) inside the study area: intersect >=1 training tile
    tiles_w = tiles_gdf.to_crs(wcrs)
    hit = gpd.sjoin(grid, tiles_w[["geometry"]], predicate="intersects", how="inner")
    grid = grid.loc[sorted(hit.index.unique())].reset_index(drop=True)
    n_tiles = len(grid)
    # (b) webmap actually has RGB imagery there (drop all-black voids like cell 0; alpha ignored)
    cov = cell_data_coverage(webmap_path, grid.geometry)
    grid = grid.loc[cov >= min_data_cov].reset_index(drop=True)
    cell_px = sorted({int(round((g.bounds[2] - g.bounds[0]) / res)) for g in grid.geometry})
    grid = grid.to_crs(study_gdf.crs)
    info = {"native_res_m": round(res, 4), "patch_px": patch, "cell_ground_m": round(patch * res, 1),
            "cell_sizes_px": cell_px, "n_cells_full": n_full, "after_tile_filter": n_tiles,
            "n_cells": len(grid), "dropped_void": n_tiles - len(grid)}
    return grid, info


grid, ginfo = build_tile_grid(area, tiles_clip, WEBMAP)
print(ginfo)


# %% CELL 9 — visual helper: grid (red) over study area + tiles ----------------------
def plot_grid(tiles_gdf, hull_gdf, grid_gdf, out_png=os.path.join(PIC, "04_tile_grid.png")):
    """Tiles (blue) + study-area hull (green) + the tile grid (red)."""
    fig, ax = plt.subplots(figsize=(11, 11))
    hull_gdf.plot(ax=ax, facecolor="#2ca02c", edgecolor="#2ca02c", alpha=0.08, linewidth=1.5)
    tiles_gdf.boundary.plot(ax=ax, color="#1f77b4", linewidth=0.3)
    grid_gdf.boundary.plot(ax=ax, color="red", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_title(f"tile grid — {len(grid_gdf)} cells @ {ginfo['patch_px']}px ({ginfo['cell_ground_m']} m)")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_png


print("->", plot_grid(tiles_clip, area, grid))


# %% CELL 10 — STEP 4: DINO embedding per grid cell (dinov3_embedding activity) ------
# Self-contained: set the activity up ONCE (raster = webmap, boxes = grid) under S3Mock,
# load the model via the activity itself, then run its OWN helpers per cell
# (read_image_bands -> create_embedding [FP32] -> _trim_output). Test on ONE cell first.

DINO_MODEL = "dinov3_vitl16"


@contextlib.contextmanager
def muted():
    """Silence the activity's per-cell prints."""
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        yield


def setup_activity(webmap_path, grid_gdf, out_fgb=os.path.join(PIC, "grid.fgb"),
                   dino_model=DINO_MODEL, high_res=False):
    """Instantiate the activity once + load its model. Returns (act, model, device, grid_in_raster_crs)."""
    
    # webmap_path = WEBMAP
    # grid_gdf = grid
    # out_fgb=os.path.join(PIC, "grid.fgb")
    with rasterio.open(webmap_path) as r:
        wcrs = r.crs
    grid_w = grid_gdf.to_crs(wcrs)                      # cell bounds must be in the raster CRS
    grid_w.to_file(out_fgb, driver="FlatGeobuf")
    inp = Input.model_validate({"bbox": out_fgb, "dino_model": dino_model, "high_res": high_res,
                                "rasters": [{"bands": ["RED", "GREEN", "BLUE"], "raster_file": webmap_path}]})
    # the activity reads via an object store rooted at working_dir; root it at "/" so the
    # absolute /mnt/... webmap path resolves (default cwd would make it cwd-relative -> 404).
    act = Dinov3Embedding(inp, "", S3Mock(working_dir="/"))
    model, device = asyncio.run(act.load_model())      # the activity's own loader
    return act, model, device, grid_w


def embed_cell(act, model, device, bbox, webmap_path=WEBMAP):
    """Embed EXACTLY the bbox — read the cell verbatim (no activity padding/overlap), then
    upscale+embed with the activity's model. The RGB matches show_bbox(bbox) pixel-for-pixel
    and the embedding covers ONLY this cell (no neighbour context). -> (rgb HWC, emb CHW, tf).

    (We bypass act.read_image_bands, which pads each box with a context margin and trims the
    embedding back; here there's nothing to trim because we never padded.)
    """
    with rasterio.open(webmap_path) as r:
        rgb = r.read((1, 2, 3), window=from_bounds(*bbox, transform=r.transform),
                     boundless=True, fill_value=0).transpose(1, 2, 0)
    emb = act.create_embedding(model, device, rgb)        # upscales by upsample, embeds (FP32)
    tf = tf_from_bounds(*bbox, emb.shape[2], emb.shape[1])  # embedding geotransform for this cell
    return rgb, emb, tf


act, model, device, grid_w = setup_activity(WEBMAP, grid)
print(f"activity ready | patch {act.patch_size}px | upsample {act.patch_upsample_factor} | {len(grid_w)} cells")

rgb0, emb0, tf0 = embed_cell(act, model, device, tuple(grid_w.geometry.iloc[0].bounds))
print(f"cell 0: rgb {rgb0.shape} -> embedding {emb0.shape}")


# %% CELL 10a — check: rgb == exact cell, embedding covers only this cell --------------
# rgb0 IS the verbatim cell read now -> matches show_bbox(bbox). The forward upscales it by
# `upsample` internally, so patches = rgb*upsample/16 and embed_gsd = native_res*16/upsample.
with rasterio.open(WEBMAP) as r:
    direct = r.read((1, 2, 3), window=from_bounds(*grid_w.geometry.iloc[0].bounds, transform=r.transform),
                    boundless=True, fill_value=0)
gh, gw = emb0.shape[1:]
embed_gsd = ginfo["cell_ground_m"] / gh
print(f"rgb {rgb0.shape[:2]} == direct {direct.shape[1:]} ? {rgb0.shape[:2] == direct.shape[1:]} "
      f"| patches {gh}x{gw} | embed_gsd {embed_gsd:.3f} m")


# %% CELL 10b — quick helper: show the webmap imagery under ANY bbox (no DINO) -------
def show_bbox(bbox, webmap_path=WEBMAP, out_png=os.path.join(PIC, "bbox_rgb.png")):
    """Read + display the webmap RGB under one bbox (in the raster CRS). No model needed.
    e.g. show_bbox(grid_w.geometry.iloc[5].bounds)  or  show_bbox(grid_w.total_bounds)."""
    with rasterio.open(webmap_path) as r:
        rgb = r.read((1, 2, 3), window=from_bounds(*bbox, transform=r.transform),
                     boundless=True, fill_value=0).transpose(1, 2, 0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb); ax.axis("off")
    ax.set_title(f"{rgb.shape[:2]} @ {[round(v) for v in bbox]}")
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_png


print("->", show_bbox(grid_w.geometry.iloc[0].bounds))


# %% CELL 11 — visual: one cell, RGB vs PCA-RGB of its DINO embedding ----------------
def pca_rgb(emb, max_fit=200_000):
    """(C,H,W) embedding -> (H,W,3) in [0,1] from its top-3 PCA axes (2-98% stretch)."""
    C, H, W = emb.shape
    X = emb.reshape(C, -1).T.astype(np.float32)
    Xc = X - X.mean(0)
    fit = Xc if len(Xc) <= max_fit else Xc[np.random.default_rng(0).choice(len(Xc), max_fit, replace=False)]
    _, V = np.linalg.eigh(fit.T @ fit)                 # covariance/eigh route (no giant SVD)
    proj = Xc @ V[:, ::-1][:, :3]
    lo, hi = np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)
    return np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1).reshape(H, W, 3)


def plot_cell(rgb, emb, out_png=os.path.join(PIC, "05_cell0_embedding.png")):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(rgb); axs[0].set_title(f"RGB (webmap) {rgb.shape[:2]}")
    axs[1].imshow(pca_rgb(emb)); axs[1].set_title(f"DINO embedding PCA-RGB {emb.shape}")
    for a in axs:
        a.axis("off")
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_png


print("->", plot_cell(rgb0, emb0))


# %% CELL 12 — STEP 4 (full grid): embed EVERY cell, keep ALL patches + the CLS token --
# Loop the whole grid with a progress bar. ONE forward per cell via forward_features gives
# BOTH outputs: the full (C, gh, gw) patch grid -> georeferenced tif (disk, reusable), and
# the CLS token (C,) -> a single global descriptor per cell, stacked into cls_vecs.npy.
from tqdm import tqdm


def embed_cell_tokens(act, model, device, bbox, webmap_path=WEBMAP, upsample=None):
    """Exact-cell read -> one forward -> (rgb, patch grid (C,gh,gw), cls (C,), transform).

    upsample: forward resize factor. None -> act.patch_upsample_factor (2 here). Pass 4 to
    test the 2048 upscaling (512*4) -> 128x128 patch grid, embed_gsd ~0.39 m, ~4x GPU mem.
    """
    import torch
    from PIL import Image
    from dinov3_embedding.main import make_transform
    up = act.patch_upsample_factor if upsample is None else upsample
    with rasterio.open(webmap_path) as r:
        rgb = r.read((1, 2, 3), window=from_bounds(*bbox, transform=r.transform),
                     boundless=True, fill_value=0).transpose(1, 2, 0)
    x = make_transform(rgb.shape[0] * up)(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.inference_mode():
        f = model.forward_features(x)               # dict: x_norm_clstoken + x_norm_patchtokens
    cls = f["x_norm_clstoken"][0].float().cpu().numpy()             # (C,)
    pt = f["x_norm_patchtokens"][0].float().cpu().numpy()           # (gh*gw, C)
    g = int(round(pt.shape[0] ** 0.5))
    patch = pt.reshape(g, g, -1).transpose(2, 0, 1)                 # (C, gh, gw)
    return rgb, patch, cls, tf_from_bounds(*bbox, patch.shape[2], patch.shape[1])

# LOOP for one site -> per-site subdir + a Hive-partitioned parquet dataset (DuckDB-ready).
# Patch grids are stored as .npz (key "patch_grid", shape (gh, gw, C)) to match dino_seg's
# fast loaders / GPU PCA at scale; geometry + CLS + the links live in the partitioned parquet.
#
# Layout:
#   outputs/embeddings/<site_id>/patches/cell_XXXX.npz      patch grids (gh,gw,C)
#   outputs/embeddings/<site_id>/{cls_vecs.npy, cells.fgb}  per-site CLS + QGIS view
#   outputs/embeddings/cells/site_id=<site_id>/cells.parquet  Hive partition (geometry+cls+id)
import re

EMB_ROOT = "/mnt/ai/DeepThought/dino_embeddings"   # shared net store (123 TB free)
site_id = re.sub(r"[^0-9A-Za-z]+", "_", os.path.relpath(SITE_DIR, "/home/clement/local_copy_train_data")).strip("_")
PATCH_DIR = os.path.join(EMB_ROOT, site_id, "patches")
PART_DIR = os.path.join(EMB_ROOT, "cells", f"site_id={site_id}")
os.makedirs(PATCH_DIR, exist_ok=True)
os.makedirs(PART_DIR, exist_ok=True)

npz_paths = []
cls_vecs = np.zeros((len(grid_w), emb0.shape[0]), np.float32)
for i, geom in enumerate(tqdm(grid_w.geometry, desc=site_id[:22], unit="cell")):
    _, emb, cls, _ = embed_cell_tokens(act, model, device, tuple(geom.bounds))    # emb = (C, gh, gw)
    out = os.path.join(PATCH_DIR, f"cell_{i:04d}.npz")
    np.savez_compressed(out, patch_grid=emb.transpose(1, 2, 0).astype(np.float32))  # (gh, gw, C) dino_seg fmt
    npz_paths.append(out)
    cls_vecs[i] = cls

# manifest: stable cell_id linking geometry <-> patch_grid file <-> CLS (order-safe, CRS kept).
# Hive partition dir encodes site_id, so DuckDB reads it from the path (no duplicate column).
manifest = grid_w.reset_index(drop=True)[["geometry"]].copy()
manifest.insert(0, "cell_id", np.arange(len(manifest)))                 # cell_id == cell_XXXX.npz == cls row
manifest["patch_npz"] = [os.path.relpath(p, EMB_ROOT) for p in npz_paths]   # path relative to the dataset root
manifest["cls"] = cls_vecs.tolist()                                     # CLS vector inline (self-contained)
manifest.to_parquet(os.path.join(PART_DIR, "cells.parquet"))           # Hive-partitioned GeoParquet
manifest[["cell_id", "patch_npz", "geometry"]].to_file(
    os.path.join(EMB_ROOT, site_id, "cells.fgb"), driver="FlatGeobuf")  # QGIS view (reorders on read)
np.save(os.path.join(EMB_ROOT, site_id, "cls_vecs.npy"), cls_vecs)      # cls_vecs[cell_id], order-stable
print(f"[{site_id}] {len(npz_paths)} cells -> {PATCH_DIR}  |  {PART_DIR}/cells.parquet")
# DuckDB later, ALL sites at once (site_id comes from the partition path):
#   import duckdb; duckdb.sql("SELECT site_id, cell_id, cls FROM "
#       "read_parquet('outputs/embeddings/cells/**/*.parquet', hive_partitioning=true)")


# %% CELL 13a — GPU PCA (the ONE PCA used everywhere): fit once, project on GPU ---------
# dino_seg's fast transform: shapes from the .npy header (no body decompress), normalize +
# matmul on GPU, fp16 output with per-tile views (stored once). Works on our .npz
# "patch_grid" (gh,gw,C). 3 components -> the RGB web-map; 256 -> KMeans/BSP at scale.
import zipfile
from pathlib import Path

from numpy.lib import format as npy_format


class GPUPCA:
    """Fitted PCA (.mean_/.components_/.n_components_) for transform_all_tiles. Basis =
    covariance/eigh on a random patch subsample (the batch PCA). For a true all-patch fit,
    accumulate mean + X^T X over every cell, or use sklearn IncrementalPCA."""
    def __init__(self, npz_paths, n_components=256, normalize=True, max_fit=500_000, seed=0):
        rng = np.random.default_rng(seed)
        per = max(1, max_fit // len(npz_paths))
        fit = []
        for p in tqdm(npz_paths, desc="PCA fit", unit="tile"):
            a = np.load(p)["patch_grid"]
            a = a.reshape(-1, a.shape[-1]).astype(np.float32)
            if normalize:
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
            fit.append(a[rng.choice(len(a), min(len(a), per), replace=False)])
        X = np.concatenate(fit); self.mean_ = X.mean(0)
        _, V = np.linalg.eigh((X - self.mean_).T @ (X - self.mean_))
        self.components_ = np.ascontiguousarray(V[:, ::-1][:, :n_components].T)   # (n_components, C)
        self.n_components_ = n_components


def _patch_grid_shape_fast(path: Path) -> tuple[int, ...]:
    """Read patch_grid.npy's shape from the .npz without decompressing the array body."""
    with zipfile.ZipFile(path) as zf:
        with zf.open("patch_grid.npy") as member:
            ver = npy_format.read_magic(member)
            if ver == (1, 0):
                shape, _, _ = npy_format.read_array_header_1_0(member)
            elif ver == (2, 0):
                shape, _, _ = npy_format.read_array_header_2_0(member)
            else:
                shape = npy_format.read_array(member).shape
    return shape


def transform_all_tiles(files, pca, *, normalize=True, show_progress=True, key_fn=None,
                        device="cuda", out_dtype=np.float16):
    """Project every cached tile into PCA space (single decompress per file; normalize +
    project on GPU; fp16 flat output with per-tile views). Returns (per_tile, flat, shapes, names)."""
    import torch
    files = [Path(f) for f in files]
    key_fn = key_fn or (lambda f: f.stem)
    shapes, names = [], []
    bar = tqdm(files, desc="indexing shapes", unit="tile") if show_progress else files
    for f in bar:
        sh = _patch_grid_shape_fast(f)
        shapes.append((int(sh[0]), int(sh[1]))); names.append(key_fn(f))
    if len(set(names)) != len(names):
        dup = next(n for n in names if names.count(n) > 1)
        raise ValueError(f"Duplicate key from key_fn: {dup!r}. For cross-site use a namespaced key_fn.")
    total = sum(h * w for h, w in shapes)
    n_components = int(pca.n_components_)
    dev = torch.device(device)
    mean_t = torch.as_tensor(np.asarray(pca.mean_), dtype=torch.float32, device=dev)
    comp_t = torch.as_tensor(np.asarray(pca.components_), dtype=torch.float32, device=dev)
    per_tile_pca, flat_pca, off = {}, np.empty((total, n_components), dtype=out_dtype), 0
    it = (tqdm(zip(files, names, shapes), total=len(files), desc="PCA transform")
          if show_progress else zip(files, names, shapes))
    for f, name, (gh, gw) in it:
        pg = np.load(f)["patch_grid"]; pg = pg.reshape(-1, pg.shape[-1])
        t = torch.from_numpy(np.ascontiguousarray(pg)).to(dev, dtype=torch.float32)
        if normalize:
            t = t / (t.norm(dim=1, keepdim=True) + 1e-6)
        z = (t - mean_t) @ comp_t.T
        n = z.shape[0]
        flat_pca[off:off + n] = z.cpu().numpy().astype(out_dtype, copy=False)
        per_tile_pca[name] = flat_pca[off:off + n].reshape(gh, gw, n_components)
        off += n
    return per_tile_pca, flat_pca, shapes, names

# scale (KMeans/BSP): pca = GPUPCA(npz_paths, 256); per_tile, flat, _, names = transform_all_tiles(npz_paths, pca)


# %% CELL 13 — site PCA-RGB mosaic via the GPU PCA (3 components) ----------------------
# Projects every patch with the SAME GPU PCA (n_components=3), 2-98% stretch over all
# patches, paints each cell into a site array by its bbox. -> (canvas HxWx3, transform, gsd).
def site_pca_canvas(npz_paths, geoms, pca=None, device="cuda"):
    with np.load(npz_paths[0]) as d0:
        gh, gw, C = d0["patch_grid"].shape
    b = [g.bounds for g in geoms]
    gsd = (b[0][2] - b[0][0]) / gw
    xmin, ymax = min(x[0] for x in b), max(x[3] for x in b)
    xmax, ymin = max(x[2] for x in b), min(x[1] for x in b)
    W, H = int(round((xmax - xmin) / gsd)), int(round((ymax - ymin) / gsd))
    pca = pca or GPUPCA(npz_paths, n_components=3, max_fit=300_000)
    per_tile, flat, _, _ = transform_all_tiles(npz_paths, pca, device=device,
                                               out_dtype=np.float32, show_progress=False)
    lo, hi = np.percentile(flat, 2, 0), np.percentile(flat, 98, 0)
    canvas = np.zeros((H, W, 3), np.float32)
    for p, g in zip(npz_paths, geoms):                    # paint each cell's projected patches
        tile = per_tile[Path(p).stem]; ch, cw = tile.shape[:2]
        col, row = int(round((g.bounds[0] - xmin) / gsd)), int(round((ymax - g.bounds[3]) / gsd))
        canvas[row:row + ch, col:col + cw] = np.clip((tile - lo) / (hi - lo + 1e-6), 0, 1)
    return canvas, rasterio.Affine(gsd, 0, xmin, 0, -gsd, ymax), gsd


def plot_site_pca(npz_paths, geoms, out_png):
    canvas, _, gsd = site_pca_canvas(npz_paths, geoms)
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.imshow(canvas); ax.axis("off")
    ax.set_title(f"site patch-level PCA-RGB — {len(npz_paths)} cells @ {gsd:.2f} m")
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png


print("->", plot_site_pca(npz_paths, list(grid_w.geometry), os.path.join(EMB_ROOT, site_id, "site_patch_pca.png")))


# %% CELL 14 — georeferenced RGB-PCA web-map GeoTIFF (open in QGIS) --------------------
# Same canvas as CELL 13, written as a 3-band uint8 GeoTIFF (CRS + transform) -> QGIS.
def build_pca_webmap(npz_paths, geoms, crs, out_tif):
    canvas, transform, gsd = site_pca_canvas(npz_paths, geoms)
    arr = (canvas * 255).astype(np.uint8).transpose(2, 0, 1)    # band-major; 0 = nodata
    os.makedirs(os.path.dirname(os.path.abspath(out_tif)), exist_ok=True)
    with rasterio.open(out_tif, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=3,
                       dtype="uint8", crs=crs, transform=transform, photometric="RGB",
                       compress="DEFLATE", tiled=True, blockxsize=256, blockysize=256, nodata=0) as dst:
        dst.write(arr)
    print(f"  {arr.shape[2]}x{arr.shape[1]}px @ {gsd:.2f} m | CRS {crs} -> {out_tif}")
    return out_tif


print("->", build_pca_webmap(npz_paths, list(grid_w.geometry), grid_w.crs,
                             os.path.join(EMB_ROOT, site_id, "dino_pca_webmap.tif")))


#  TO  DO :
# a new webmap with dino RGB
# a new map witch kmeans clustering
# a new map with PCA clustering Spli , Binary Space partitioning
# alignment with annotations ? upsamplig he tiles back to the site resolution ?
# TO DO a grid of the representation showing the step of the [rpcess ?]