"""Extract per-chunk bounding boxes for a site and save them as a FlatGeobuf (.fgb).

A "chunk" can be:
  - "blob"  : one box per reconstituted source image (uuid)         [default]
  - "tile"  : one box per native 384 tile
  - <int>   : square chunks of N px (e.g. 768/1024) tiled over each blob, row-major,
              remainder clamped to the blob edge (last row/col may be smaller)

World coords come from each tile's GEO_TRANSFORM (a 3x3 affine flattened to 9 vals):
    wx = ox + col*gt[0] + row*gt[1]
    wy = oy + col*gt[3] + row*gt[4]
with the pixel (col, row) = filename offset (X, Y) added to the blob origin.

!! Caveat on THIS dataset: every blob of a site shares the SAME GEO_TRANSFORM origin
   (placeholder), so all blob boxes overlap in world space. The function is correct and
   reusable on sites with valid per-image geotransforms; on broken sites use granularity
   in pixel space or fix the geotransform upstream. The function reports the overlap.

Usage:
    from chunk_bboxes import chunks_to_fgb
    gdf = chunks_to_fgb(SITE_DIR, "chunks.fgb", granularity="blob")
    gdf = chunks_to_fgb(SITE_DIR, "tiles768.fgb", granularity=768)

CLI:
    .venv/bin/python chunk_bboxes.py <site_dir> <out.fgb> [blob|tile|<int>]
"""
from __future__ import annotations
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import geopandas as gpd
from shapely.geometry import box

PAT = re.compile(r"image_(.+)_(\d+)_(\d+)\.npz")


def _scan_blobs(site_dir: str) -> dict:
    """uuid -> dict(cells={(X,Y):(w,h)}, gt=9-affine, srid). One pass, reads npz headers."""
    blobs = defaultdict(lambda: {"cells": {}, "gt": None, "srid": None})
    for split in ("train", "val"):
        for f in glob.glob(os.path.join(site_dir, split, "*.npz")):
            m = PAT.search(os.path.basename(f))
            if not m:
                continue
            uid, x, y = m.group(1), int(m.group(2)), int(m.group(3))
            d = np.load(f, allow_pickle=True)
            h, w = d["RED"].shape
            b = blobs[uid]
            b["cells"][(x, y)] = (w, h)
            if b["gt"] is None:
                b["gt"] = np.asarray(d["GEO_TRANSFORM"], float)
                b["srid"] = int(d["SRID"][0])
    return blobs


def _extent(cells: dict) -> tuple[int, int]:
    """blob pixel size (W, H) using real (possibly truncated) tile sizes."""
    W = max(x + w for (x, y), (w, h) in cells.items())
    H = max(y + h for (x, y), (w, h) in cells.items())
    return W, H


def _px_to_world(gt: np.ndarray, col: float, row: float) -> tuple[float, float]:
    wx = gt[2] + col * gt[0] + row * gt[1]
    wy = gt[5] + col * gt[3] + row * gt[4]
    return wx, wy


def _pixel_box_to_world(gt: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    """axis-aligned world box covering pixel rectangle [x0:x1, y0:y1]."""
    corners = [_px_to_world(gt, c, r) for c, r in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return box(min(xs), min(ys), max(xs), max(ys))


def _chunk_rects(W: int, H: int, granularity, cells):
    """yield (col_px, row_px, w_px, h_px) pixel rectangles for the requested granularity."""
    if granularity == "blob":
        yield (0, 0, W, H)
    elif granularity == "tile":
        for (x, y), (w, h) in sorted(cells.items()):
            yield (x, y, w, h)
    elif isinstance(granularity, int):
        S = granularity
        for y0 in range(0, H, S):
            for x0 in range(0, W, S):
                w = min(S, W - x0)
                h = min(S, H - y0)
                yield (x0, y0, w, h)
    else:
        raise ValueError(f"granularity must be 'blob', 'tile' or int, got {granularity!r}")


def chunks_to_fgb(site_dir: str, out_fgb: str, granularity="blob") -> gpd.GeoDataFrame:
    """Build per-chunk world bboxes for a site and write them to a FlatGeobuf file.

    granularity: "blob" | "tile" | int(px). Returns the GeoDataFrame (also written to disk).
    """
    blobs = _scan_blobs(site_dir)
    if not blobs:
        raise FileNotFoundError(f"no image_*.npz tiles under {site_dir}")

    srids = {b["srid"] for b in blobs.values()}
    if len(srids) != 1:
        raise ValueError(f"mixed SRIDs in site: {srids}")
    srid = srids.pop()

    rows = []
    for uid, b in blobs.items():
        gt, cells = b["gt"], b["cells"]
        W, H = _extent(cells)
        for i, (x0, y0, w, h) in enumerate(_chunk_rects(W, H, granularity, cells)):
            rows.append({
                "uuid": uid,
                "chunk_id": f"{uid[:8]}_{x0}_{y0}",
                "col_px": x0, "row_px": y0, "w_px": w, "h_px": h,
                "geometry": _pixel_box_to_world(gt, x0, y0, x0 + w, y0 + h),
            })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{srid}")
    os.makedirs(os.path.dirname(os.path.abspath(out_fgb)), exist_ok=True)
    gdf.to_file(out_fgb, driver="FlatGeobuf")

    # honest sanity report: do boxes overlap (broken/placeholder geotransform)?
    n = len(gdf)
    uniq_origins = {(round(b["gt"][2], 3), round(b["gt"][5], 3)) for b in blobs.values()}
    print(f"wrote {n} '{granularity}' chunks from {len(blobs)} blobs -> {out_fgb}")
    print(f"  CRS=EPSG:{srid}  distinct blob origins={len(uniq_origins)}")
    if len(uniq_origins) == 1 and granularity == "blob":
        print("  WARNING: all blobs share one origin -> blob boxes overlap in world space "
              "(geotransform is a placeholder on this dataset).")
    return gdf


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/clement/local_copy_train_data/29Metals/29M_2451_GG_manned/10cm/v2_tytonai_rg"
    out = sys.argv[2] if len(sys.argv) > 2 else "chunks_blob.fgb"
    gran = sys.argv[3] if len(sys.argv) > 3 else "blob"
    if isinstance(gran, str) and gran.isdigit():
        gran = int(gran)
    chunks_to_fgb(site, out, gran)
