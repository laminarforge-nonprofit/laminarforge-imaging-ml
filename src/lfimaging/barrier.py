"""Barrier integrity score from phase-contrast or brightfield.

Fraction of the imaging field covered by an intact monolayer. Holes are gaps in the
segmentation mask, detected by edge+fill morphology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BarrierScore:
    pct_coverage: float
    hole_count: int
    mean_hole_area: float


def score_barrier(image: np.ndarray, min_hole_area: int = 50) -> BarrierScore:
    """Return coverage %, hole count, and mean hole area (in pixels)."""
    from scipy import ndimage as ndi
    from skimage import filters, morphology

    if image.ndim == 3:
        image = image.mean(axis=2)
    arr = image.astype(np.float32)
    if arr.max() > 0:
        arr = arr / arr.max()

    # Edges of the monolayer tend to be high-frequency; blur first.
    blurred = filters.gaussian(arr, sigma=2.0)
    thr = filters.threshold_otsu(blurred) if blurred.size else 0.5
    tissue = blurred > thr
    tissue = morphology.remove_small_holes(tissue, area_threshold=min_hole_area)
    tissue = morphology.closing(tissue, morphology.disk(3))

    coverage = float(np.count_nonzero(tissue)) / max(1, tissue.size) * 100.0

    # Holes are background components fully surrounded by tissue. Label background,
    # drop the single component touching the image border.
    bg = ~tissue
    labeled, n_components = ndi.label(bg)
    if n_components == 0:
        return BarrierScore(pct_coverage=coverage, hole_count=0, mean_hole_area=0.0)

    border_labels = set(
        np.unique(
            np.concatenate(
                [labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]]
            )
        ).tolist()
    )
    hole_areas: list[int] = []
    for lbl in range(1, n_components + 1):
        if lbl in border_labels:
            continue
        area = int(np.count_nonzero(labeled == lbl))
        if area >= min_hole_area:
            hole_areas.append(area)

    hole_count = len(hole_areas)
    mean_area = float(np.mean(hole_areas)) if hole_areas else 0.0
    return BarrierScore(pct_coverage=coverage, hole_count=hole_count, mean_hole_area=mean_area)
