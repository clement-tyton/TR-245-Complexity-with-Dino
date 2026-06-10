"""Verbose, resumable DINOv3 weights downloader (GCS public bucket — no HF token, no auth).

Why this exists: the activity's own downloader (dinov3_embedding/download.py) fans out up to
128 parallel 100 MB range requests and PRE-ALLOCATES the full file before writing. One GCS
timeout then leaves a full-size but TRUNCATED .pth on disk, and its existence-only check
(`if local_file.exists()`) serves that corrupt file forever -> "PytorchStreamReader failed ...
failed finding central directory". This script instead:

  - prints the bucket / object / size up front (so you SEE what it's doing),
  - downloads sequentially in small chunks with a live progress bar + MB/s,
  - writes to <file>.part and RESUMES from wherever a previous run stopped,
  - retries each chunk on timeout (backoff) instead of nuking the whole download,
  - validates COMPLETENESS by size (the 7B checkpoint is legacy torch .tar, not a zip, so a
    format check would wrongly reject it) before renaming to the final name.

Usage (repo root, project venv), default = the 7B; pass vitl for the small one:
    .venv/bin/python download_weights.py            # dinov3_vit7b16
    .venv/bin/python download_weights.py vitl       # dinov3_vitl16
Tunables: WEIGHTS_CHUNK_MB (default 32), WEIGHTS_MAX_RETRIES (default 8).
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import config  # noqa: F401,E402  -> sets DINO_WEIGHTS_FOLDER (+ all DINO env) on import

from dinov3_embedding.download import (  # noqa: E402  reuse the lib's bucket/path constants
    DINO_WEIGHTS_FOLDER, GCS_BUCKET, LARGE_MODEL, SMALL_MODEL,
    LARGE_MODEL_PATH, SMALL_MODEL_PATH,
)
from obstore.store import GCSStore  # noqa: E402

CHUNK = int(os.getenv("WEIGHTS_CHUNK_MB", "32")) * 1024 * 1024
MAX_RETRIES = int(os.getenv("WEIGHTS_MAX_RETRIES", "8"))
TIMEOUT = os.getenv("WEIGHTS_TIMEOUT", "300s")          # per-request timeout (obstore default ~30s)
CONNECT_TIMEOUT = os.getenv("WEIGHTS_CONNECT_TIMEOUT", "30s")


def _human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024


async def download(model: str) -> Path:
    model_path = LARGE_MODEL_PATH if model == LARGE_MODEL else SMALL_MODEL_PATH
    final = Path(DINO_WEIGHTS_FOLDER) / model_path
    part = final.with_suffix(final.suffix + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)

    print(f"weights folder : {DINO_WEIGHTS_FOLDER}")
    print(f"GCS bucket     : {GCS_BUCKET}  (public, anonymous — no token)")
    print(f"object         : {model_path}")

    store = GCSStore(GCS_BUCKET, skip_signature=True,
                     client_options={"timeout": TIMEOUT, "connect_timeout": CONNECT_TIMEOUT})
    print(f"timeout        : {TIMEOUT} per request (connect {CONNECT_TIMEOUT})")
    size = (await store.head_async(model_path))["size"]

    # Validate by COMPLETENESS (size), not format: the 7B checkpoint is legacy torch .tar, not a
    # zip, so zipfile.is_zipfile would wrongly reject a perfectly good file. A truncated download
    # is shorter than the remote object; a complete one matches it byte-for-byte.
    if final.exists():
        ok = final.stat().st_size == size
        print(f"already on disk: {final}  ({_human(final.stat().st_size)} / {_human(size)}) "
              f"-> {'complete, nothing to do' if ok else 'TRUNCATED — deleting'}")
        if ok:
            return final
        final.unlink()

    done = part.stat().st_size if part.exists() else 0
    if done > size:                                          # stale/oversized .part -> restart clean
        print(f"  .part ({_human(done)}) > remote ({_human(size)}) -> discarding"); part.unlink(); done = 0
    print(f"remote size    : {_human(size)}")
    print(f"resuming from  : {_human(done)} ({100*done/size:.1f}%)\n" if done else "starting fresh\n")

    t0 = time.monotonic()
    with part.open("r+b" if part.exists() else "wb") as f:
        f.seek(done)
        while done < size:
            end = min(done + CHUNK, size)
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    buf = await store.get_range_async(model_path, start=done, end=end)
                    break
                except Exception as e:                       # timeout/body error -> retry this chunk only
                    if attempt == MAX_RETRIES:
                        print(f"\nchunk {done}-{end} failed {MAX_RETRIES}x: {str(e)[:100]}")
                        print(f"-> partial saved at {part} ; re-run to resume from here.")
                        raise
                    wait = min(2 ** attempt, 30)
                    print(f"  chunk @{_human(done)} retry {attempt}/{MAX_RETRIES} in {wait}s "
                          f"({str(e)[:60]})")
                    await asyncio.sleep(wait)
            f.write(bytes(buf)); f.flush()
            done = end
            mbps = (done / (time.monotonic() - t0 + 1e-9)) / 1e6
            print(f"\r  {_human(done)}/{_human(size)} ({100*done/size:5.1f}%)  {mbps:5.1f} MB/s",
                  end="", flush=True)
    print()

    if part.stat().st_size != size:                          # completeness guard before promoting
        raise RuntimeError(f"size mismatch {part.stat().st_size} != {size}; .part kept, re-run to resume.")
    part.rename(final)
    print(f"done -> {final}  ({_human(final.stat().st_size)}, {time.monotonic()-t0:.0f}s)")
    return final


if __name__ == "__main__":
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "7b"
    asyncio.run(download(SMALL_MODEL if which in ("vitl", "l", "small") else LARGE_MODEL))
