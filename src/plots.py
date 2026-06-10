"""Matplotlib QA helpers — each saves a PNG and returns its path. No module globals.

These mirror the REPL's per-step visual checkpoints; pass gdfs / paths / titles explicitly.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds

import config
from pca import pca_rgb, site_pca_canvas, webmap_data_mask


def _png(name):
    return os.path.join(config.PIC_DIR, name)


def plot_tiles(gdf, out_png=None, title=None):
    """Draw every tile bbox (outline) in world coordinates."""
    out_png = out_png or _png("01_tile_bboxes.png")
    fig, ax = plt.subplots(figsize=(11, 11))
    gdf.boundary.plot(ax=ax, color="#1f77b4", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title(title or f"{len(gdf)} tile bboxes")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out_png


def plot_webmap_crop(gdf, clipped, ext, out_png=None):
    """Original tiles (grey), kept tiles (blue), webmap extent rectangle (orange)."""
    out_png = out_png or _png("02_webmap_crop.png")
    fig, ax = plt.subplots(figsize=(11, 11))
    gdf.boundary.plot(ax=ax, color="#cccccc", linewidth=0.4)
    clipped.boundary.plot(ax=ax, color="#1f77b4", linewidth=0.5)
    ext.boundary.plot(ax=ax, color="#ff7f0e", linewidth=2)
    ax.set_aspect("equal")
    ax.set_title(f"webmap-extent crop — {len(clipped)}/{len(gdf)} tiles kept")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out_png


def plot_study_area(gdf, hull_gdf, out_png=None, title=None):
    """Tile bboxes (blue) under the convex-hull study area (green)."""
    out_png = out_png or _png("03_study_area.png")
    fig, ax = plt.subplots(figsize=(11, 11))
    hull_gdf.plot(ax=ax, facecolor="#2ca02c", edgecolor="#2ca02c", alpha=0.10, linewidth=2)
    gdf.boundary.plot(ax=ax, color="#1f77b4", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title(title or f"study area (hull) — {len(gdf)} tiles, {hull_gdf.area_km2.iloc[0]:.2f} km^2")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out_png


def plot_grid(tiles_gdf, hull_gdf, grid_gdf, info=None, out_png=None):
    """Tiles (blue) + study-area hull (green) + the tile grid (red). info = build_tile_grid dict."""
    out_png = out_png or _png("04_tile_grid.png")
    info = info or {}
    fig, ax = plt.subplots(figsize=(11, 11))
    hull_gdf.plot(ax=ax, facecolor="#2ca02c", edgecolor="#2ca02c", alpha=0.08, linewidth=1.5)
    tiles_gdf.boundary.plot(ax=ax, color="#1f77b4", linewidth=0.3)
    grid_gdf.boundary.plot(ax=ax, color="red", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_title(f"tile grid — {len(grid_gdf)} cells @ {info.get('patch_px','?')}px "
                 f"({info.get('cell_ground_m','?')} m)")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_png


def show_bbox(bbox, webmap_path, out_png=None):
    """Read + display the webmap RGB under one bbox (in the raster CRS). No model needed."""
    out_png = out_png or _png("bbox_rgb.png")
    with rasterio.open(webmap_path) as r:
        rgb = r.read((1, 2, 3), window=from_bounds(*bbox, transform=r.transform),
                     boundless=True, fill_value=0).transpose(1, 2, 0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb); ax.axis("off")
    ax.set_title(f"{rgb.shape[:2]} @ {[round(v) for v in bbox]}")
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_png


def plot_cell(rgb, emb, out_png=None):
    """One cell: RGB (webmap) vs PCA-RGB of its DINO embedding."""
    out_png = out_png or _png("05_cell0_embedding.png")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(rgb); axs[0].set_title(f"RGB (webmap) {rgb.shape[:2]}")
    axs[1].imshow(pca_rgb(emb)); axs[1].set_title(f"DINO embedding PCA-RGB {emb.shape}")
    for a in axs:
        a.axis("off")
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_png


def plot_qa_grid(tiles, tiles_clip, extent, area, grid, info=None, out_png=None, title=None):
    """One 2x2 control image per site: (1) tiles, (2) webmap-extent crop, (3) study-area hull,
    (4) the tile grid — the same four steps as the separate 01/02/03/04 PNGs, stacked."""
    out_png = out_png or _png("qa_steps.png")
    info = info or {}
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    a0, a1, a2, a3 = axs.ravel()

    tiles.boundary.plot(ax=a0, color="#1f77b4", linewidth=0.5)
    a0.set_title(f"1. tiles — {len(tiles)}")

    tiles.boundary.plot(ax=a1, color="#cccccc", linewidth=0.4)
    tiles_clip.boundary.plot(ax=a1, color="#1f77b4", linewidth=0.5)
    extent.boundary.plot(ax=a1, color="#ff7f0e", linewidth=2)
    a1.set_title(f"2. webmap-extent crop — {len(tiles_clip)}/{len(tiles)} kept")

    area.plot(ax=a2, facecolor="#2ca02c", edgecolor="#2ca02c", alpha=0.10, linewidth=2)
    tiles_clip.boundary.plot(ax=a2, color="#1f77b4", linewidth=0.5)
    a2.set_title(f"3. study area — {area.area_km2.iloc[0]:.2f} km^2")

    area.plot(ax=a3, facecolor="#2ca02c", edgecolor="#2ca02c", alpha=0.08, linewidth=1.5)
    tiles_clip.boundary.plot(ax=a3, color="#1f77b4", linewidth=0.3)
    grid.boundary.plot(ax=a3, color="red", linewidth=0.8)
    a3.set_title(f"4. grid — {len(grid)} cells @ {info.get('patch_px','?')}px "
                 f"({info.get('cell_ground_m','?')} m)")

    for a in axs.ravel():
        a.set_aspect("equal"); a.set_xlabel("easting (m)"); a.set_ylabel("northing (m)")
    fig.suptitle(title or "site QA — tiles -> crop -> study area -> grid", fontsize=14)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_png


def plot_site_pca(npz_paths, geoms, out_png, webmap_path=None):
    """Site patch-level PCA-RGB mosaic (PNG). If webmap_path is given, no-data shows white
    (matching the transparent nodata in the QGIS GeoTIFF)."""
    canvas, transform, gsd = site_pca_canvas(npz_paths, geoms)
    if webmap_path:
        H, W = canvas.shape[:2]
        canvas = canvas.copy()
        canvas[~webmap_data_mask(webmap_path, transform, H, W)] = 1.0   # no-data -> white
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.imshow(canvas); ax.axis("off")
    ax.set_title(f"site patch-level PCA-RGB — {len(npz_paths)} cells @ {gsd:.2f} m")
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png
