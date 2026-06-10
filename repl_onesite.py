# %% [markdown]
# repl_onesite.py — one-site REPL (thin driver over src/)
# =======================================================
# Step-by-step, cell-by-cell, with a visual checkpoint after each stage. All logic lives in
# src/ (transforms, dino, pca, io, plots); this file just calls it so interactive and batch
# (pipeline.run_site) can never diverge. Run cell by cell.

# %% CELL 1 — path + imports + pick a site ------------------------------------------
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config            # noqa: E402  (sets the DINO env on import — must come first)
import transforms        # noqa: E402
import dino              # noqa: E402
import pca               # noqa: E402
import plots             # noqa: E402
import store as sink     # noqa: E402  (src/store.py — persistence)

SITE_DIR = "/home/clement/local_copy_train_data/BHP Creeks 2022/Manned Bens Oasis Post Dry/10cm/v2_tytonai_rg"
site_id = config.site_id_from_dir(SITE_DIR)
WEBMAP = config.resolve_rgb(config.site_key_from_dir(SITE_DIR))["rgb_path"]   # resolved from config/
print("site:", site_id, "\nwebmap:", WEBMAP)


# %% CELL 2 — read tile bboxes ------------------------------------------------------
tiles = transforms.read_tile_bboxes(SITE_DIR)
xmin, ymin, xmax, ymax = tiles.total_bounds
print(f"{len(tiles)} tiles | CRS {tiles.crs} | extent {xmax-xmin:.0f} x {ymax-ymin:.0f} m")
print("->", plots.plot_tiles(tiles))


# %% CELL 3 — crop tiles to the webmap extent ---------------------------------------
tiles_clip, extent = transforms.crop_tiles_to_webmap(tiles, WEBMAP)
print(f"tiles inside extent: {len(tiles_clip)}/{len(tiles)}")
print("->", plots.plot_webmap_crop(tiles, tiles_clip, extent))


# %% CELL 4 — study area = convex hull ----------------------------------------------
area = transforms.study_area(tiles_clip)
print(area.drop(columns="geometry").to_string(index=False))
print("->", plots.plot_study_area(tiles_clip, area))


# %% CELL 5 — build the patch grid --------------------------------------------------
grid, ginfo = transforms.build_tile_grid(area, tiles_clip, WEBMAP)
print(ginfo)
print("->", plots.plot_grid(tiles_clip, area, grid, info=ginfo))
# 2x2 control image of all four steps for this site
print("->", plots.plot_qa_grid(tiles, tiles_clip, extent, area, grid, info=ginfo,
                               out_png=os.path.join(config.EMB_ROOT, site_id, "qa_steps.png"),
                               title=f"{site_id} — {ginfo['n_cells']} cells"))


# %% CELL 6 — set up the activity + sanity-embed ONE cell ---------------------------
act, model, device, grid_w = dino.setup_activity(WEBMAP, grid)
print(f"activity ready | patch {act.patch_size}px | upsample {act.patch_upsample_factor} | {len(grid_w)} cells")
rgb0, emb0, tf0 = dino.embed_cell(act, model, device, tuple(grid_w.geometry.iloc[0].bounds), WEBMAP)
print(f"cell 0: rgb {rgb0.shape} -> embedding {emb0.shape}")
print("->", plots.show_bbox(grid_w.geometry.iloc[0].bounds, WEBMAP))
print("->", plots.plot_cell(rgb0, emb0))


# %% CELL 7 — embed the WHOLE grid + write manifest ---------------------------------
patch_dir, part_dir = sink.site_emb_dirs(site_id)
npz_paths, cls_vecs = sink.embed_grid(act, model, device, grid_w, WEBMAP, patch_dir, desc=site_id)
sink.write_manifest(grid_w, npz_paths, cls_vecs, site_id, part_dir)
print(f"[{site_id}] {len(npz_paths)} cells -> {patch_dir}")


# %% CELL 8 — site PCA-RGB mosaic (PNG) + georeferenced webmap (QGIS) ----------------
print("->", plots.plot_site_pca(npz_paths, list(grid_w.geometry),
                                os.path.join(config.EMB_ROOT, site_id, "site_patch_pca.png")))
print("->", pca.build_pca_webmap(npz_paths, list(grid_w.geometry), grid_w.crs,
                                 os.path.join(config.EMB_ROOT, site_id, "dino_pca_webmap.tif")))


# %% CELL 9 — (scale) fit a 256-d GPU PCA + project all patches ----------------------
# pca_256 = pca.GPUPCA(npz_paths, n_components=256)
# per_tile, flat, shapes, names = pca.transform_all_tiles(npz_paths, pca_256, device=device.type)
# -> flat: (total_patches, 256) fp16 for KMeans/BSP; per_tile[name]: (gh, gw, 256) view

#  TO DO:
#  - kmeans / BSP clustering map
#  - alignment with annotations
#  - the 7B @ 2048 comparison on the A6000 (store PCA-reduced patches)
