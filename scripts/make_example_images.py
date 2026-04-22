"""Generate the 2-3 synthetic example images checked into data/example/.

Run once on repo setup — the tifs are small enough to check in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def _blobs(shape: tuple[int, int], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = shape
    img = np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n):
        cy, cx = rng.integers(20, h - 20), rng.integers(20, w - 20)
        sigma = rng.uniform(4, 7)
        intensity = rng.uniform(0.6, 1.0)
        img += intensity * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "example"
    out.mkdir(parents=True, exist_ok=True)
    chip = "C1-S1"
    ts = "2026-04-20T12-00-00"
    shape = (256, 256)
    channels = {
        "brightfield": (60, 1),
        "calcein": (50, 2),
        "pi": (10, 3),
        "hoechst": (60, 4),
    }
    for name, (n, seed) in channels.items():
        arr = _blobs(shape, n, seed)
        path = out / f"{chip}_{ts}_{name}.tif"
        tifffile.imwrite(path.as_posix(), arr)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
