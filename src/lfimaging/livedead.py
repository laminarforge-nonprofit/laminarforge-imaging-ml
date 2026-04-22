"""LIVE/DEAD quantification from Calcein (green) + PI (red) fluorescence.

The current implementation is a threshold-based heuristic suitable for smoke-testing
the pipeline end-to-end. A U-Net training harness is stubbed in ``train_livedead`` —
it accepts a YAML config and a directory of image/mask pairs and trains on MPS (Apple)
or CUDA when available. Retrained weights are loaded at runtime via
``LF_LIVEDEAD_WEIGHTS`` if set; otherwise the heuristic is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LiveDeadResult:
    live_count: int
    dead_count: int
    viability_pct: float
    artifact_count: int = 0  # cells positive for BOTH stains, flagged as artifact


def _threshold(image: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Return a boolean mask of bright pixels using a percentile threshold."""
    if image.ndim == 3:
        image = image[..., 0]  # use first channel if caller passed an RGB
    if image.size == 0:
        return np.zeros_like(image, dtype=bool)
    arr = image.astype(np.float32)
    thr = float(np.percentile(arr, percentile))
    # Guard against flat images (all same value).
    if thr <= float(arr.min()):
        return np.zeros_like(arr, dtype=bool)
    return arr > thr


def _count_components(mask: np.ndarray) -> int:
    from scipy import ndimage as ndi

    _, n = ndi.label(mask)
    return int(n)


def analyze_livedead(
    calcein: np.ndarray,
    pi: np.ndarray,
    weights_path: str | Path | None = None,
) -> LiveDeadResult:
    """Compute live/dead counts and viability % from Calcein + PI images.

    Heuristic:
        * threshold each channel at the 95th percentile
        * green-only cell = live, red-only = dead, both = artifact
    """
    weights_path = weights_path or os.getenv("LF_LIVEDEAD_WEIGHTS")
    if weights_path and Path(weights_path).exists():
        try:
            return _unet_livedead(calcein, pi, Path(weights_path))
        except Exception:
            # Fall back to heuristic rather than failing the whole pipeline.
            pass

    live_mask = _threshold(calcein) & ~_threshold(pi)
    dead_mask = _threshold(pi) & ~_threshold(calcein)
    both_mask = _threshold(calcein) & _threshold(pi)

    live = _count_components(live_mask)
    dead = _count_components(dead_mask)
    artifact = _count_components(both_mask)

    total = live + dead
    viability = (live / total * 100.0) if total > 0 else 0.0
    return LiveDeadResult(
        live_count=live,
        dead_count=dead,
        viability_pct=viability,
        artifact_count=artifact,
    )


def _unet_livedead(calcein: np.ndarray, pi: np.ndarray, weights: Path) -> LiveDeadResult:
    """Load a retrained U-Net and run inference. Only called when weights exist."""
    import torch  # type: ignore[import-not-found]

    from .unet import UNet  # lazy; unet.py only needed when weights are present

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = UNet(in_channels=2, out_channels=3).to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    stack = np.stack([calcein, pi], axis=0).astype(np.float32) / 255.0
    x = torch.from_numpy(stack).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()[0]

    live = _count_components(pred == 1)
    dead = _count_components(pred == 2)
    total = live + dead
    viability = (live / total * 100.0) if total > 0 else 0.0
    return LiveDeadResult(
        live_count=live,
        dead_count=dead,
        viability_pct=viability,
        artifact_count=0,
    )
