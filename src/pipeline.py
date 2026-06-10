"""One-site orchestrator: training tiles -> grid -> DINO embeddings -> manifest + PCA webmap.

Usage:
    python src/pipeline.py "/home/clement/local_copy_train_data/BHP Creeks 2022/.../v2_tytonai_rg"
or:
    from pipeline import run_site; run_site(site_dir)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # make src/ modules importable

import config
import transforms
import dino
import store as sink   # src/store.py (persistence) — named 'store' to avoid the stdlib 'io' clash
import pca


def run_site(site_dir, *, rgb_path=None, res=None, out_root=config.EMB_ROOT,
             dino_model=config.DINO_MODEL, high_res=config.HIGH_RES,
             tile_patches=config.TILE_PATCHES, min_data_cov=config.MIN_DATA_COV,
             upsample=None, make_plots=False, make_webmap=True, show_bar=True) -> dict:
    """Build + embed every grid cell of one site. Returns a summary dict.

    rgb_path: the webmap RGB raster. If None, resolved from config/site_rgb_paths.json via the
    site's '<Project>/<Site>' key (res picked from sites_to_resolutions.json if res is None).
    Outputs under out_root/<site_id>/ + the Hive-partitioned out_root/cells/site_id=<id>/.
    """
    site_id = config.site_id_from_dir(site_dir)
    if rgb_path is None:
        rgb_path = config.resolve_rgb(config.site_key_from_dir(site_dir), res)["rgb_path"]

    tiles = transforms.read_tile_bboxes(site_dir)
    tiles_clip, extent = transforms.crop_tiles_to_webmap(tiles, rgb_path)
    area = transforms.study_area(tiles_clip)
    grid, ginfo = transforms.build_tile_grid(area, tiles_clip, rgb_path,
                                             tile_patches=tile_patches, high_res=high_res,
                                             min_data_cov=min_data_cov)
    print(f"[{site_id}] {len(tiles)} tiles -> {ginfo['n_cells']} cells {ginfo}")

    patch_dir, part_dir = sink.site_emb_dirs(site_id, out_root)

    # per-site 2x2 QA control image (tiles -> crop -> study area -> grid), always written
    import plots
    qa_png = plots.plot_qa_grid(tiles, tiles_clip, extent, area, grid, info=ginfo,
                                out_png=os.path.join(out_root, site_id, "qa_steps.png"),
                                title=f"{site_id} — {ginfo['n_cells']} cells")
    print(f"  QA -> {qa_png}")

    act, model, device, grid_w = dino.setup_activity(rgb_path, grid, dino_model=dino_model,
                                                     high_res=high_res)
    npz_paths, cls_vecs = sink.embed_grid(act, model, device, grid_w, rgb_path, patch_dir,
                                          upsample=upsample, show_bar=show_bar, desc=site_id)
    sink.write_manifest(grid_w, npz_paths, cls_vecs, site_id, part_dir, emb_root=out_root)

    webmap_tif = None
    if make_webmap and npz_paths:
        webmap_tif = os.path.join(out_root, site_id, "dino_pca_webmap.tif")
        pca.build_pca_webmap(npz_paths, list(grid_w.geometry), grid_w.crs, webmap_tif)

    if make_plots:
        import plots
        plots.plot_tiles(tiles)
        plots.plot_webmap_crop(tiles, tiles_clip, extent)
        plots.plot_study_area(tiles_clip, area)
        plots.plot_grid(tiles_clip, area, grid, info=ginfo)
        if npz_paths:
            plots.plot_site_pca(npz_paths, list(grid_w.geometry),
                                os.path.join(out_root, site_id, "site_patch_pca.png"))

    print(f"[{site_id}] done: {len(npz_paths)} cells -> {patch_dir}  |  {part_dir}/cells.parquet")
    return {"site_id": site_id, "n_tiles": len(tiles), "n_cells": len(grid_w), "ginfo": ginfo,
            "patch_dir": patch_dir, "part_dir": part_dir, "webmap_tif": webmap_tif,
            "qa_png": qa_png, "npz_paths": npz_paths}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Embed one site's grid with DINOv3 + PCA webmap.")
    ap.add_argument("site_dir", help="…/<Project>/<Site>/<res>/v2_tytonai_rg")
    ap.add_argument("--rgb-path", default=None, help="override the webmap raster (else resolved from config)")
    ap.add_argument("--out-root", default=config.EMB_ROOT)
    ap.add_argument("--upsample", type=int, default=None, help="forward upscale (2=1024, 4=2048)")
    ap.add_argument("--no-webmap", dest="make_webmap", action="store_false")
    ap.add_argument("--plots", dest="make_plots", action="store_true")
    a = ap.parse_args()
    run_site(a.site_dir, rgb_path=a.rgb_path, out_root=a.out_root, upsample=a.upsample,
             make_webmap=a.make_webmap, make_plots=a.make_plots)
