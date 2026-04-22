"""Nuclear counting from Hoechst/DAPI channel via Laplacian-of-Gaussian blob detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default physical pixel size on the gantry camera (mm/px). Override at call time.
DEFAULT_PX_MM = 0.00145  # ~1.45 µm/px at the current mag.


@dataclass
class NuclearCount:
    total_cells: int
    density_per_mm2: float


def _normalize(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        # Hoechst is commonly imaged on a blue channel; take the max for robustness.
        image = image.max(axis=2)
    arr = image.astype(np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax <= vmin:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def count_nuclei(
    hoechst: np.ndarray,
    min_sigma: float = 2.0,
    max_sigma: float = 6.0,
    num_sigma: int = 5,
    threshold: float = 0.08,
    px_mm: float = DEFAULT_PX_MM,
) -> NuclearCount:
    """Count nuclei using scikit-image's LoG blob detector.

    ``px_mm`` maps pixel size to millimetres so we can report density per mm².
    """
    from skimage.feature import blob_log

    norm = _normalize(hoechst)
    blobs = blob_log(
        norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
    )
    count = int(len(blobs))

    h, w = norm.shape[:2]
    area_mm2 = (h * px_mm) * (w * px_mm)
    density = (count / area_mm2) if area_mm2 > 0 else 0.0
    return NuclearCount(total_cells=count, density_per_mm2=density)
