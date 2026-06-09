"""Run the DINOv3 activity on a blob and mean-blend its per-box embeddings.

Importing `.config` first guarantees the activity's env (weights folder, local S3) is set
before the activity package is imported (lazily, inside run_activity_on_blob).
"""
from __future__ import annotations

import contextlib
import glob
import os
import shutil

import numpy as np
import rasterio
from rasterio.transform import Affine

from . import config  # noqa: F401  (import for side effect: sets activity env vars)


@contextlib.contextmanager
def muted():
    """Silence the activity's chatty stdout+stderr. (tqdm writes to sys.__stderr__, so it survives.)"""
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        yield


_MODEL_CACHE = {}   # dino_model -> (model, device); loaded once per process (model load is ~1s)


def _cached_model(dino_model):
    """Load the DINOv3 model once and reuse it (lets us run boxes one-by-one for a sub-bar
    without paying a reload each time)."""
    if dino_model not in _MODEL_CACHE:
        import asyncio
        from tytonai.test.s3_mock import S3Mock
        from dinov3_embedding.io_schema.model import Input
        from dinov3_embedding.main import Dinov3Embedding
        inp = Input.model_validate({"bbox": "x.fgb", "dino_model": dino_model, "high_res": False,
                                    "rasters": [{"bands": ["RED", "GREEN", "BLUE"], "raster_file": "x.tif"}]})
        with muted():
            _MODEL_CACHE[dino_model] = asyncio.run(Dinov3Embedding(inp, "", S3Mock()).load_model())
    return _MODEL_CACHE[dino_model]


def _make_activity(bbox_fgb, dino_model, high_res, model=None):
    from tytonai.test.s3_mock import S3Mock
    from dinov3_embedding.io_schema.model import Input
    from dinov3_embedding.main import Dinov3Embedding
    inp = Input.model_validate({"bbox": bbox_fgb, "dino_model": dino_model, "high_res": high_res,
                                "rasters": [{"bands": ["RED", "GREEN", "BLUE"], "raster_file": "blob.tif"}]})
    act = Dinov3Embedding(inp, "", S3Mock())
    if model is not None:
        async def _cached():
            return model
        act.load_model = _cached     # reuse the preloaded model instead of reloading
    return act


def run_activity_on_blob(blob_dir, dino_model=config.DINO_MODEL, high_res=config.HIGH_RES,
                         make_box_bar=None):
    """Run Dinov3Embedding on blob_dir/{blob.tif, boxes.fgb}. Returns (result, [cog_paths]).

    If make_box_bar(n_boxes) is given (a tqdm factory), boxes are run ONE AT A TIME with a
    cached model so an inner per-box progress bar can tick — otherwise all boxes go in one
    activity call. Same COGs either way (the activity writes one per box regardless).
    """
    prev = os.getcwd()
    try:
        os.chdir(blob_dir)  # S3Mock reads/writes relative to cwd
        for f in glob.glob("*.tif"):
            if f != "blob.tif":
                os.remove(f)  # clear stale per-box outputs

        if make_box_bar is None:
            with muted():
                result = _make_activity("boxes.fgb", dino_model, high_res).start()
        else:
            import geopandas as gpd
            model = _cached_model(dino_model)
            g = gpd.read_file("boxes.fgb")
            bar = make_box_bar(len(g))
            for i in range(len(g)):
                g.iloc[[i]].to_file("_one.fgb", driver="FlatGeobuf")
                with muted():
                    _make_activity("_one.fgb", dino_model, high_res, model=model).start()
                bar.update(1)
            bar.close()
            if os.path.exists("_one.fgb"):
                os.remove("_one.fgb")
            result = None

        cogs = sorted(os.path.abspath(f) for f in glob.glob("*.tif") if f != "blob.tif")
        return result, cogs
    finally:
        os.chdir(prev)


def mean_mosaic(cog_paths, out_tif):
    """Mean-blend georeferenced embedding COGs (same CRS+GSD, grid-aligned): sum/count -> mean."""
    if not cog_paths:
        raise ValueError("no COGs to merge")
    with rasterio.open(cog_paths[0]) as s0:
        crs, count, a = s0.crs, s0.count, s0.transform
        px, py = a.a, a.e
    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    for p in cog_paths:
        with rasterio.open(p) as s:
            b = s.bounds
            xmins.append(b.left); ymins.append(b.bottom); xmaxs.append(b.right); ymaxs.append(b.top)
    xmin, ymax = min(xmins), max(ymaxs)
    W = int(round((max(xmaxs) - xmin) / abs(px)))
    H = int(round((ymax - min(ymins)) / abs(py)))
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
    mean = (acc / np.maximum(cnt, 1)).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(out_tif)), exist_ok=True)
    with rasterio.open(out_tif, "w", driver="COG", height=H, width=W, count=count,
                       dtype="float32", crs=crs, transform=Affine(px, 0, xmin, 0, py, ymax),
                       compress="ZSTD") as dst:
        dst.write(mean)
    return out_tif


def clean_box_outputs(blob_dir):
    """Drop per-box COGs + S3Mock uuid dirs after the blend (keep blob/boxes/embedding/meta/png)."""
    for f in glob.glob(os.path.join(blob_dir, "*.tif")):
        if os.path.basename(f) not in ("blob.tif", "embedding.tif"):
            os.remove(f)
    for d in glob.glob(os.path.join(blob_dir, "*-*-*-*-*")):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
