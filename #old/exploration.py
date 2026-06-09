"""Standalone DINOv3 ViT-L/16 embedding over a folder of image tiles, at NATIVE resolution.

No TytonAI activity framework, no S3, no geotransform/window math. The whole point is to
avoid the production pipeline's resolution-driven upscaling (which forces a 1024 px window
and a 2x upsample -> 2048 px DINO input, and even crashes on tiles smaller than patch_size).
Here each tile is fed to DINO at its native size (default 384, a multiple of 16), batched.

Pipeline:
  1. Build the ViT-L/16 backbone once and load local pretrained weights.
  2. Stream tiles from disk with a DataLoader (CPU workers do the IO + normalization).
  3. Forward each batch through DINO, mean-pool the patch grid -> one vector per tile.
  4. Write all vectors into a single memmapped .npy [N, embed_dim] + a paths.txt sidecar.

Requires: torch, the `dinov3` package (FAIR), rasterio, numpy, tqdm.

Example:
    python embed_tiles_standalone.py \
        --tiles-dir /path/to/120k_tiles \
        --weights ./test_data/dinov3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
        --out-dir ./embeddings_out \
        --size 384 --batch-size 32 --workers 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

# DINOv3 backbone — same class the activity uses.
from dinov3.models.vision_transformer import DinoVisionTransformer

# Normalization used by the activity's make_transform (satellite pretrain stats).
MEAN = (0.430, 0.411, 0.296)
STD = (0.213, 0.156, 0.143)
PATCH_SIZE = 16
EMBED_DIM = 1024  # ViT-L/16
TILE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_vitl16(device: torch.device) -> DinoVisionTransformer:
    """Recreate the exact ViT-L/16 config from dinov3_embedding.backbones (no activity import)."""
    return DinoVisionTransformer(
        img_size=224,
        patch_size=PATCH_SIZE,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=EMBED_DIM,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=True,
        device=device,
    )


def load_model(weights: Path, device: torch.device) -> DinoVisionTransformer:
    """Build the backbone and load pretrained weights once."""
    model = build_vitl16(device)
    state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


class TileDataset(Dataset):
    """Reads RGB tiles from disk and returns normalized CHW tensors.

    The heavy work here is disk IO, which the DataLoader parallelizes across worker
    processes so the GPU never waits on file reads.
    """

    def __init__(self, paths: list[Path], size: int):
        self.paths = paths
        # ToImage -> Resize(size,size) -> float[0,1] -> normalize. Resize is a no-op for
        # tiles already at `size`; it guarantees a uniform shape so batching works.
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((size, size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=MEAN, std=STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with rasterio.open(path) as src:
            arr = src.read(indexes=[1, 2, 3])  # CHW, first 3 bands = RGB
        arr = np.transpose(arr, (1, 2, 0))  # HWC for ToImage
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return self.transform(arr), idx


def list_tiles(tiles_dir: Path) -> list[Path]:
    """Find all tile files under tiles_dir (recursive), sorted for stable ordering."""
    paths = sorted(p for p in tiles_dir.rglob("*") if p.suffix.lower() in TILE_EXTS)
    if not paths:
        msg = f"No tiles with extensions {sorted(TILE_EXTS)} found under {tiles_dir}"
        raise FileNotFoundError(msg)
    return paths


@torch.inference_mode()
def embed(
    model: DinoVisionTransformer,
    loader: DataLoader,
    out: np.memmap,
    device: torch.device,
    *,
    amp: bool,
) -> None:
    """Run every batch through DINO, mean-pool the patch grid, write into `out`."""
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp)
    for batch, idxs in tqdm(loader, desc="embedding", unit="batch"):
        batch = batch.to(device, non_blocking=True)
        with autocast:
            # reshape=True -> [B, embed_dim, h_patches, w_patches]; norm=True -> final LN.
            feats = model.get_intermediate_layers(batch, n=1, reshape=True, norm=True)[0]
            pooled = feats.mean(dim=(2, 3))  # global average over patches -> [B, embed_dim]
        out[idxs.numpy()] = pooled.float().cpu().numpy()



parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tiles-dir", type=Path, required=True, help="Folder of tiles (recursive)")
parser.add_argument("--weights", type=Path, required=True, help="Path to the .pth weights")
parser.add_argument("--out-dir", type=Path, required=True, help="Where to write outputs")
parser.add_argument("--size", type=int, default=384, help="DINO input size (multiple of 16)")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--workers", type=int, default=8, help="DataLoader IO workers")
parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu, ...")
parser.add_argument("--amp", action="store_true", help="Use bf16 autocast (faster, less VRAM)")
args = parser.parse_args()

if args.size % PATCH_SIZE != 0:
    msg = f"--size {args.size} must be a multiple of {PATCH_SIZE} for a /16 ViT"
    raise SystemExit(msg)

device = torch.device(args.device)
args.out_dir.mkdir(parents=True, exist_ok=True)

paths = list_tiles(args.tiles_dir)
print(f"Found {len(paths)} tiles. Loading model on {device} ...")

model = load_model(args.weights, device)

# Memmap output so 120k vectors never need to fit in RAM at once.
out_path = args.out_dir / "embeddings.npy"
out = np.lib.format.open_memmap(
    out_path, mode="w+", dtype=np.float32, shape=(len(paths), EMBED_DIM)
)
(args.out_dir / "paths.txt").write_text("\n".join(str(p) for p in paths))

dataset = TileDataset(paths, args.size)
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=args.workers > 0,
)

start = time.perf_counter()
embed(model, loader, out, device, amp=args.amp)
out.flush()
elapsed = time.perf_counter() - start
print(
    f"Done: {len(paths)} tiles in {elapsed:.1f}s "
    f"({len(paths) / elapsed:.1f} tiles/s) -> {out_path}"
)

