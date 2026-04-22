"""LaminarForge imaging ML pipeline.

Processes organ-chip imaging gantry output into quantitative per-chip metrics:
confluence, cell count, LIVE/DEAD ratio, nuclear density, barrier integrity.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .pipeline import ChipMetrics, process_chip  # noqa: F401
