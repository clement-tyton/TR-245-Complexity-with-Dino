"""COG-free fast embedding: same DINO inputs as the activity, but accumulated in RAM.

The activity writes one COG per box (~1.8 s each) and mean_mosaic re-reads them. Here we
reuse the activity's EXACT preprocessing (its read_image_bands + _trim_output, and a faithful
copy of create_embedding) but keep each box's trimmed embedding in memory and accumulate the
mean directly — writing only the final per-blob embedding.tif.

FP32 (bf16=False) reproduces the activity's embeddings byte-for-byte (modulo float
accumulation order). bf16=True runs the forward under autocast (faster, ~identical values).
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import rasterio
from rasterio.transform import Affine
import geopandas as gpd

from . import config
from .embedding import _cached_model, muted


def _embed_array(model, device, img_arr, upsample, bf16):
    """Faithful copy of the activity's create_embedding (optionally bf16 forward). -> (C, gh, gw)."""
    import torch
    from PIL import Image
    from dinov3_embedding.main import make_transform
    x = make_transform(img_arr.shape[0] * upsample)(Image.fromarray(img_arr)).unsqueeze(0).to(device)
    with torch.inference_mode():
        if bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
        else:
            f = model.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
    return f.squeeze(0).permute(1, 2, 0).float().cpu().numpy().transpose(2, 0, 1)


def embed_blob_fast(blob_dir, out_tif, dino_model=config.DINO_MODEL, high_res=config.HIGH_RES,
                    bf16=False, make_box_bar=None, nodata_frac=0.98):
    """Embed one blob (blob.tif + boxes.fgb) without per-box COGs. Writes out_tif, returns it.

    Boxes that are >= nodata_frac all-black (holes from missing tiles) are skipped — no GPU
    spent and those cells stay zero-vectors (maskable downstream) instead of garbage embeddings.
    Returns None if EVERY box is nodata.
    """
    from tytonai.test.s3_mock import S3Mock
    from dinov3_embedding.io_schema.model import Input
    from dinov3_embedding.main import Dinov3Embedding

    model, device = _cached_model(dino_model)
    prev = os.getcwd()
    try:
        os.chdir(blob_dir)
        bboxes = [tuple(geom.bounds) for geom in gpd.read_file("boxes.fgb").geometry]
        with rasterio.open("blob.tif") as s:
            crs = s.crs
        inp = Input.model_validate({"bbox": "boxes.fgb", "dino_model": dino_model, "high_res": high_res,
                                    "rasters": [{"bands": ["RED", "GREEN", "BLUE"], "raster_file": "blob.tif"}]})
        act = Dinov3Embedding(inp, "", S3Mock())
        upsample = act.patch_upsample_factor
        patch_size = act.patch_size

        async def _collect():
            tiles, skipped = [], 0
            bar = make_box_bar(len(bboxes)) if make_box_bar else None
            for bbox in bboxes:
                with muted():   # silence the activity's per-box read/_trim_output prints
                    img_data, window, out_window, tf = await act.read_image_bands(bbox, patch_size)
                    prepared = np.concatenate([b.img_data for b in img_data.bands], axis=0).transpose(1, 2, 0)
                    if (prepared == 0).all(axis=2).mean() >= nodata_frac:   # hole -> skip, no embed
                        skipped += 1
                    else:
                        emb = _embed_array(model, device, prepared, upsample, bf16)
                        trimmed = act._trim_output(emb, window, out_window)   # (C, th, tw)
                        px = (tf.a * out_window.width) / trimmed.shape[2]     # = embed_gsd
                        py = (tf.e * out_window.height) / trimmed.shape[1]
                        tiles.append((trimmed, tf.c, tf.f, px, py))           # array + world origin + gsd
                if bar:                                                       # bar writes to __stderr__
                    bar.update(1)
            if bar:
                bar.close()
            return tiles, skipped

        tiles, skipped = asyncio.run(_collect())
        if not tiles:                                  # whole blob is nodata -> nothing to write
            return None

        # union + mean-blend in RAM (same math as mean_mosaic, no disk round-trip)
        px, py = tiles[0][3], tiles[0][4]
        xmin = min(t[1] for t in tiles)
        ymax = max(t[2] for t in tiles)
        xmax = max(t[1] + t[0].shape[2] * px for t in tiles)
        ymin = min(t[2] + t[0].shape[1] * py for t in tiles)   # py < 0
        W = int(round((xmax - xmin) / abs(px)))
        H = int(round((ymax - ymin) / abs(py)))
        C = tiles[0][0].shape[0]
        acc = np.zeros((C, H, W), np.float64)
        cnt = np.zeros((H, W), np.float64)
        for trimmed, left, top, _, _ in tiles:
            col = int(round((left - xmin) / abs(px)))
            row = int(round((ymax - top) / abs(py)))
            h, w = trimmed.shape[1], trimmed.shape[2]
            acc[:, row:row + h, col:col + w] += trimmed
            cnt[row:row + h, col:col + w] += 1
        mean = (acc / np.maximum(cnt, 1)).astype(np.float32)

        os.makedirs(os.path.dirname(os.path.abspath(out_tif)), exist_ok=True)
        with rasterio.open(out_tif, "w", driver="COG", height=H, width=W, count=C, dtype="float32",
                           crs=crs, transform=Affine(px, 0, xmin, 0, py, ymax), compress="ZSTD") as dst:
            dst.write(mean)
        return out_tif
    finally:
        os.chdir(prev)
