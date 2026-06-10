# %% [markdown]
# repl_multisites.py — multi-site driver (thin, over src/)
# =======================================================
# Edit SITES, dry-run the resolution (which sites have a webmap + tiles), then RUN.
# Resume-safe: re-running CELL 4 skips sites whose cells.parquet already exists.

# %% CELL 1 — path + imports --------------------------------------------------------
import os
import sys

# Single GPU: run everything on this physical device. MUST be set before any torch import
# (our modules import torch lazily, so setting it here is enough). Edit/remove as needed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# Inference dtype. "bf16" casts the model+input to bfloat16: weights ~half VRAM (7B: 25->12.5GB)
# and activations halved -> fixes OOM at high upscaling. Outputs are still stored fp32, and the
# upscaling (upsample/high_res) is untouched. Set to "fp32" to revert. MUST be set before import config.
os.environ.setdefault("DINO_DTYPE", "bf16")

try:                     # running as a file / in VS Code
    _ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:        # bare REPL paste (no __file__) -> cwd must be the repo root
    _ROOT = "."
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _ROOT)   # repo root -> import download_weights
sys.path.insert(0, _SRC)
for _m in ("config", "transforms", "dino", "pca", "plots", "store", "pipeline"):
    sys.modules.pop(_m, None)   # reload fresh from src/ on re-run

import config            # noqa: E402  (sets the DINO env on import — must come first)
import pipeline          # noqa: E402


# %% CELL 2 — choose the sites ------------------------------------------------------
ALL = pipeline.all_site_keys()                    # every site in config/sites_to_resolutions.json
print(f"{len(ALL)} sites in the catalog")
SITES = ALL                                       # or a hand-picked list, e.g.:
# SITES = ["BHP Creeks 2022/Manned Bens Oasis Post Dry", "BHP_Rehab_2024/MWER2024"]


# %% CELL 3 — dry-run: which sites resolve (webmap + tiles) vs skip — no embedding ---
ok, miss = [], []
for k in SITES:
    try:
        config.resolve_rgb(k)
        config.resolve_tiles(k)                   # checks the /mnt ObjectData tiles dir exists
        ok.append(k)
    except Exception as e:
        miss.append((k, str(e)[:70]))
print(f"resolvable: {len(ok)} | unresolved: {len(miss)}")
for k, e in miss:
    print("  skip", k, "|", e)


# %% CELL 3b — ensure DINO weights are downloaded PROPERLY (robust; skip if already complete) ---
# WITHOUT this, CELL 4 -> setup_activity falls back to the activity's own downloader (128 parallel
# range reads + full pre-allocation), which on a GCS timeout leaves a CORRUPT .pth that its
# existence-only check then serves forever. Our downloader is sequential, resumable, size-validated
# and returns instantly if the file is already complete.
import asyncio                  # noqa: E402
import dino                     # noqa: E402  (picks 7B vs ViT-L by this GPU's VRAM)
import download_weights as dw   # noqa: E402  (repo-root module)
_model = dino.pick_dino_model()                 # prints the GPU + chosen model
asyncio.run(dw.download(_model))                # verbose, resumable; no-op if already on disk


# %% CELL 4 — RUN (resume-safe; finished sites are skipped) --------------------------
res = pipeline.run_sites(ok, make_webmap=True, make_plots=False)
print("embedded/ok:", len(res["done"]), "| skipped/failed:", len(res["fails"]))


# %% CELL 5 — (optional) 2-GPU on the server: sites split modulo across GPU 0 and 1 -----
# Option A — ONE command, one process per GPU (output interleaves):
# pipeline.run_all_gpus(gpus=(0, 1), site_keys=ok, make_webmap=True)
#
# Option B — TWO terminals (clean per-GPU progress bars), each its own GPU + shard:
#   CUDA_VISIBLE_DEVICES=0 python src/pipeline.py --all-sites --shard 0/2
#   CUDA_VISIBLE_DEVICES=1 python src/pipeline.py --all-sites --shard 1/2
# (or from the CLI in one go:  python src/pipeline.py --all-sites --gpus 0,1)
# Both are resume-safe and split sites[i::2], so the two GPUs never touch the same site.
