"""DINOv3 activity setup + per-cell embedding.

This is the ONLY module that touches the dinov3_embedding activity / torch model. All
activity/torch imports are LAZY inside the functions so that ``import config`` (which sets
the env vars) always runs first — the activity package is never imported at module load.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as tf_from_bounds

import config


@contextlib.contextmanager
def muted():
    """Silence the activity's per-cell prints."""
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        yield


def pick_dino_model(min_gb_7b=config.VRAM_GB_FOR_7B, verbose=True):
    """Choose dinov3_vit7b16 if the GPU has >= min_gb_7b of VRAM, else dinov3_vitl16.
    Prints the GPU + chosen model. Falls back to ViT-L if there's no CUDA device."""
    import torch
    if not torch.cuda.is_available():
        if verbose:
            print("GPU: none (CPU) -> dinov3_vitl16")
        return "dinov3_vitl16"
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1e9
    model = "dinov3_vit7b16" if gb >= min_gb_7b else "dinov3_vitl16"
    if verbose:
        print(f"GPU: {p.name} ({gb:.0f} GB VRAM) -> {model} "
              f"({'fits 7B' if model.endswith('7b16') else f'< {min_gb_7b:.0f} GB, using ViT-L'})")
    return model


def _trust_torch_load():
    """PyTorch >=2.6 defaults torch.load(weights_only=True); the DINOv3 7B checkpoint is in the
    legacy .tar format, which weights_only can't read -> load fails. The activity calls torch.load
    without weights_only, so we flip the default to False (safe: trusted company GCS weights, and
    the result only feeds model.load_state_dict). Idempotent — guarded so repeated per-site calls
    don't nest wrappers."""
    import torch
    if getattr(torch.load, "_trusted", False):
        return
    _orig = torch.load
    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)
    _load._trusted = True
    torch.load = _load


def setup_activity(webmap_path, grid_gdf, out_fgb=None,
                   dino_model=config.DINO_MODEL, high_res=config.HIGH_RES):
    """Instantiate the activity once + load its model. Returns (act, model, device, grid_in_raster_crs).

    The grid is reprojected to the raster CRS and written to a FlatGeobuf (the activity's bbox
    input). S3Mock(working_dir="/") roots the object store at "/" so absolute /mnt/... webmap
    paths resolve (a default mock would make the path cwd-relative -> 404).
    """
    import asyncio
    from tytonai.test.s3_mock import S3Mock
    from dinov3_embedding.io_schema.model import Input
    from dinov3_embedding.main import Dinov3Embedding

    if dino_model in (None, "auto"):
        dino_model = pick_dino_model()        # choose 7B vs ViT-L by GPU VRAM (+ print)
    out_fgb = out_fgb or os.path.join(config.PIC_DIR, "grid.fgb")
    with rasterio.open(webmap_path) as r:
        wcrs = r.crs
    grid_w = grid_gdf.to_crs(wcrs)                      # cell bounds must be in the raster CRS
    grid_w.to_file(out_fgb, driver="FlatGeobuf")
    inp = Input.model_validate({"bbox": out_fgb, "dino_model": dino_model, "high_res": high_res,
                                "rasters": [{"bands": ["RED", "GREEN", "BLUE"], "raster_file": webmap_path}]})
    act = Dinov3Embedding(inp, "", S3Mock(working_dir="/"))
    _trust_torch_load()                                # PyTorch 2.6 weights_only fix (legacy .tar 7B)
    model, device = asyncio.run(act.load_model())      # the activity's own loader
    return act, model, device, grid_w


def embed_cell(act, model, device, bbox, webmap_path):
    """Embed EXACTLY the bbox — read the cell verbatim (no activity padding/overlap), then
    upscale+embed with the activity's model. The RGB matches the raw webmap read pixel-for-pixel
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


def embed_cell_tokens(act, model, device, bbox, webmap_path, upsample=None):
    """Exact-cell read -> one forward -> (rgb, patch grid (C,gh,gw), cls (C,), transform).

    One forward via forward_features yields BOTH the patch tokens AND the CLS token.
    upsample: forward resize factor. None -> act.patch_upsample_factor (2 at 10cm). Pass 4 to
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
