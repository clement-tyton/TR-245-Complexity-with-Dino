"""Geometry transforms: training tiles -> webmap crop -> study-area hull -> patch grid.

Pure geometry + rasterio reads (no DINO/torch). The patch-size policy lives in config.
"""
from __future__ import annotations

import glob
import math
import os
import zipfile

import numpy as np
import geopandas as gpd
from shapely.geometry import box
from numpy.lib import format as npy_format
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from bbox_to_tile_grid.tilegrid import create_adaptive_grid          # bbox->grid activity
from tqdm import tqdm

import config


def _npz_member_shape(zf, name):
    """Shape of an array inside a .npz from its .npy header — no body decompression."""
    with zf.open(name + ".npy") as m:
        ver = npy_format.read_magic(m)
        rd = npy_format.read_array_header_1_0 if ver == (1, 0) else npy_format.read_array_header_2_0
        shape, _, _ = rd(m)
    return shape


def _npz_member_array(zf, name):
    """Full (small) array from a .npz member, via one open on an existing ZipFile handle."""
    with zf.open(name + ".npy") as m:
        return npy_format.read_array(m, allow_pickle=True)


def read_tile_bboxes(site_dir, splits=("train", "val"), show_bar=True):
    """One row per tile = its real-world bbox (from GEO_TRANSFORM + tile shape).

    GEO_TRANSFORM = [px, 0, ox, 0, -px, oy, ...]; bbox = (ox, oy - h*px, ox + w*px, oy).
    Returns a GeoDataFrame (tile id, split, path, w, h, geometry) in the tiles' CRS.
    Reads only the small GEO_TRANSFORM/SRID arrays + RED's header shape (no band decompression)
    -> fast even over /mnt. Shows a tqdm bar (the per-tile network reads can be slow).
    """
    files = [(s, f) for s in splits for f in glob.glob(os.path.join(site_dir, s, "*.npz"))]
    rows, srid = [], None
    for s, f in tqdm(files, desc="read tiles", unit="tile", disable=not show_bar):
        with zipfile.ZipFile(f) as zf:                       # ONE open per tile (over /mnt)
            h, w = _npz_member_shape(zf, "RED")              # header only — no band decompress
            gt = np.asarray(_npz_member_array(zf, "GEO_TRANSFORM"), float)   # tiny arrays
            srid = int(_npz_member_array(zf, "SRID")[0])
        ox, oy, px, py = gt[2], gt[5], gt[0], gt[4]          # py is negative
        geom = box(ox, oy + h * py, ox + w * px, oy)         # (xmin, ymin, xmax, ymax)
        rows.append({"tile": os.path.basename(f)[:-4], "split": s, "path": f,
                     "w": int(w), "h": int(h), "geometry": geom})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{srid}")


def webmap_extent(webmap_path, dst_crs):
    """Webmap raster's full extent as a 1-row GeoDataFrame (reprojected to dst_crs)."""
    with rasterio.open(webmap_path) as r:
        src_crs, b = r.crs, r.bounds
    bb = transform_bounds(src_crs, dst_crs, *b) if src_crs else tuple(b)
    return gpd.GeoDataFrame({"src_crs": [str(src_crs)]}, geometry=[box(*bb)], crs=dst_crs)


def crop_tiles_to_webmap(gdf, webmap_path):
    """Clip tile bboxes to the webmap extent. Returns (clipped_tiles, extent_gdf)."""
    ext = webmap_extent(webmap_path, gdf.crs)
    clipped = gpd.clip(gdf, ext)                      # crops geometries to the rectangle
    return clipped, ext


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


def build_tile_grid(study_gdf, tiles_gdf, webmap_path, tile_patches=config.TILE_PATCHES,
                    high_res=config.HIGH_RES, min_data_cov=config.MIN_DATA_COV):
    """Perfect patch x patch grid over the study area; only cells that matter are kept.

    1) snap the hull's bbox outward to an exact multiple of patch -> all cells are full squares;
    2) keep cells intersecting >=1 training tile (inside the study area);
    3) drop cells with no RGB data (all-black) so DINO never runs on void — alpha ignored.
    Returns (kept-grid in study CRS, info dict). patch_size from config.activity_params(res).
    """
    with rasterio.open(webmap_path) as r:
        gt, wcrs, res = r.transform, r.crs, abs(r.transform.a)
    patch = config.activity_params(res, high_res)["patch_size"] * tile_patches
    bbox = snap_bbox_to_patch(study_gdf.to_crs(wcrs).total_bounds, gt, patch)
    grid = create_adaptive_grid(bbox, None, gt, wcrs, patch, patch, fixed_size=True)  # full square grid
    grid = (grid.set_crs(wcrs) if grid.crs is None else grid).reset_index(drop=True)
    n_full = len(grid)
    # (a) inside the study area: intersect >=1 training tile
    tiles_w = tiles_gdf.to_crs(wcrs)
    hit = gpd.sjoin(grid, tiles_w[["geometry"]], predicate="intersects", how="inner")
    grid = grid.loc[sorted(hit.index.unique())].reset_index(drop=True)
    n_tiles = len(grid)
    # (b) webmap actually has RGB imagery there (drop all-black voids; alpha ignored)
    cov = cell_data_coverage(webmap_path, grid.geometry)
    grid = grid.loc[cov >= min_data_cov].reset_index(drop=True)
    cell_px = sorted({int(round((g.bounds[2] - g.bounds[0]) / res)) for g in grid.geometry})
    grid = grid.to_crs(study_gdf.crs)
    info = {"native_res_m": round(res, 4), "patch_px": patch, "cell_ground_m": round(patch * res, 1),
            "cell_sizes_px": cell_px, "n_cells_full": n_full, "after_tile_filter": n_tiles,
            "n_cells": len(grid), "dropped_void": n_tiles - len(grid)}
    return grid, info
