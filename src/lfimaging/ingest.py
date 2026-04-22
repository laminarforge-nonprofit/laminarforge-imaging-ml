"""Ingest stage — read images from a local directory or S3 URI.

An ImageBundle groups every channel captured for a single chip at a single timestamp
(brightfield, calcein, pi, hoechst, phase). Downstream stages expect channel arrays
as uint8 or uint16 numpy arrays (H,W) or (H,W,C).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import tifffile
from imageio.v3 import imread

# Known channels we expect from the gantry imaging subsystem (A-07C2AA94).
KNOWN_CHANNELS = ("brightfield", "calcein", "pi", "hoechst", "phase")

# Filenames from the gantry look like:
#   C1-S2_2026-04-20T12-30-00_calcein.tif
#   C3-S1_2026-04-20T12-30-00_brightfield.png
_FILENAME_RE = re.compile(
    r"^(?P<chip>[A-Z]\d+-S\d+)_"
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_"
    r"(?P<channel>[a-z]+)\.(?:tif|tiff|png|jpg|jpeg)$",
    re.IGNORECASE,
)


@dataclass
class ImageBundle:
    """A set of channel images for one chip at one timestamp."""

    chip_id: str
    timestamp: datetime
    channels: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int] | None:
        for arr in self.channels.values():
            return arr.shape[:2]
        return None

    def has(self, channel: str) -> bool:
        return channel in self.channels and self.channels[channel] is not None

    def get(self, channel: str) -> np.ndarray | None:
        return self.channels.get(channel)


def _load_image(path: Path) -> np.ndarray:
    """Load a single image into a numpy array. TIFFs go through tifffile."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        arr = tifffile.imread(path.as_posix())
    else:
        arr = imread(path.as_posix())
    return np.asarray(arr)


def parse_filename(name: str) -> tuple[str, datetime, str] | None:
    """Parse a gantry filename into (chip_id, timestamp, channel) or None if it doesn't match."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    chip = m.group("chip")
    ts = datetime.strptime(m.group("ts"), "%Y-%m-%dT%H-%M-%S")
    channel = m.group("channel").lower()
    return chip, ts, channel


def discover_local(root: Path) -> list[Path]:
    """Walk a directory for supported image files."""
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts)


def load_bundle_from_dir(directory: Path | str, chip_id: str | None = None) -> ImageBundle:
    """Load a single ImageBundle from a directory, inferring metadata from filenames.

    If ``chip_id`` is provided, only files matching that chip are loaded. Otherwise
    the first chip encountered wins (one-bundle-per-directory convention).
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(directory)

    bundle_chip: str | None = None
    bundle_ts: datetime | None = None
    channels: dict[str, np.ndarray] = {}

    for path in discover_local(directory):
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        chip, ts, channel = parsed
        if chip_id is not None and chip != chip_id:
            continue
        if bundle_chip is None:
            bundle_chip = chip
            bundle_ts = ts
        elif chip != bundle_chip:
            continue
        channels[channel] = _load_image(path)

    if bundle_chip is None or bundle_ts is None:
        raise ValueError(
            f"No gantry-conformant images found under {directory}. "
            "Expected filenames like 'C1-S2_2026-04-20T12-30-00_calcein.tif'."
        )

    return ImageBundle(chip_id=bundle_chip, timestamp=bundle_ts, channels=channels)


def load_bundles_from_s3(uri: str, profile: str = "laminarforge-dev") -> Iterable[ImageBundle]:
    """Lazy S3 loader — yields ImageBundles grouped by (chip_id, timestamp).

    Uses boto3 with the ``laminarforge-dev`` profile by default. Objects are expected
    at s3://bucket/prefix/<gantry-filename>.
    """
    import boto3  # local import keeps boto optional for non-S3 runs

    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")

    grouped: dict[tuple[str, datetime], dict[str, np.ndarray]] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key.rsplit("/", 1)[-1]
            parsed_name = parse_filename(name)
            if parsed_name is None:
                continue
            chip, ts, channel = parsed_name
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            arr = _load_image_bytes(name, body) if body else None
            if arr is None:
                continue
            grouped.setdefault((chip, ts), {})[channel] = arr

    for (chip, ts), channels in sorted(grouped.items(), key=lambda kv: kv[0][1]):
        yield ImageBundle(chip_id=chip, timestamp=ts, channels=channels)


def _load_image_bytes(name: str, body: bytes) -> np.ndarray:
    """Load an image from an in-memory byte string."""
    import io

    suffix = Path(name).suffix.lower()
    buf = io.BytesIO(body)
    if suffix in (".tif", ".tiff"):
        return np.asarray(tifffile.imread(buf))
    return np.asarray(imread(buf))
