"""Training harness for the LIVE/DEAD U-Net.

YAML schema::

    images_dir: /path/to/two-channel-tifs
    masks_dir:  /path/to/label-masks    # 0 bg, 1 live, 2 dead
    output:     models/livedead_v1.pt
    epochs:     50
    batch_size: 8
    lr:         1e-3
    device:     mps                     # mps | cuda | cpu (auto if missing)

Run via ``lf-imaging train-livedead --config config.yaml``. Gracefully exits if
required labeled data is missing and prints the gap — acceptable for v0 per the
ticket spec (heuristic ships even without retrained weights).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _autodetect_device() -> str:
    import torch  # type: ignore[import-not-found]

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(config_path: Path) -> None:
    cfg = _load_yaml(Path(config_path))
    images_dir = Path(cfg.get("images_dir", ""))
    masks_dir = Path(cfg.get("masks_dir", ""))
    if not images_dir.exists() or not masks_dir.exists():
        log.error(
            "Labeled data missing: images_dir=%s masks_dir=%s. "
            "Shipping heuristic-only LIVE/DEAD and skipping training.",
            images_dir,
            masks_dir,
        )
        return

    try:
        import tifffile
        import torch  # type: ignore[import-not-found]
        from torch.utils.data import DataLoader, Dataset  # type: ignore[import-not-found]
    except ImportError:
        log.error("Install the [ml] extras first: pip install -e '.[ml]'")
        return

    from .unet import UNet

    device = cfg.get("device") or _autodetect_device()
    epochs = int(cfg.get("epochs", 50))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("lr", 1e-3))
    output = Path(cfg.get("output", "models/livedead_v1.pt"))

    class LiveDeadDataset(Dataset):  # type: ignore[misc]
        def __init__(self, images: Path, masks: Path) -> None:
            self.pairs = sorted(
                (p, masks / p.name) for p in images.glob("*.tif") if (masks / p.name).exists()
            )

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int) -> tuple[Any, Any]:
            img_path, mask_path = self.pairs[idx]
            img = tifffile.imread(img_path.as_posix()).astype("float32") / 255.0
            mask = tifffile.imread(mask_path.as_posix()).astype("int64")
            # Expect (2, H, W) two-channel TIF: calcein + pi.
            if img.ndim == 2:
                img = img[None]
            x = torch.from_numpy(img)
            y = torch.from_numpy(mask)
            return x, y

    dataset = LiveDeadDataset(images_dir, masks_dir)
    if len(dataset) == 0:
        log.error("No image/mask pairs found under %s", images_dir)
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = UNet(in_channels=2, out_channels=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    log.info("training on %d samples, device=%s, epochs=%d", len(dataset), device, epochs)
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        log.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, total / max(1, len(loader)))

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model._model.state_dict(), output)  # noqa: SLF001
    log.info("wrote weights -> %s", output)
