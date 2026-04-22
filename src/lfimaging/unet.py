"""Minimal U-Net used by the LIVE/DEAD training harness.

Three output channels: background / live / dead. Kept small so training is feasible
on an M4 Max via MPS without remote GPU access.
"""

from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]

        return torch
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install the [ml] extras to use U-Net: pip install -e '.[ml]'") from e


class UNet:
    """Lightweight U-Net wrapper. Real implementation is constructed lazily."""

    def __init__(self, in_channels: int = 2, out_channels: int = 3, base: int = 32) -> None:
        torch = _require_torch()
        nn = torch.nn

        def block(c_in: int, c_out: int) -> Any:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        class _UNet(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.enc1 = block(in_channels, base)
                self.enc2 = block(base, base * 2)
                self.enc3 = block(base * 2, base * 4)
                self.bottleneck = block(base * 4, base * 8)
                self.pool = nn.MaxPool2d(2)
                self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
                self.dec3 = block(base * 8, base * 4)
                self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
                self.dec2 = block(base * 4, base * 2)
                self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
                self.dec1 = block(base * 2, base)
                self.head = nn.Conv2d(base, out_channels, 1)

            def forward(self, x: Any) -> Any:
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                b = self.bottleneck(self.pool(e3))
                d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                return self.head(d1)

        self._model = _UNet()

    def to(self, device: str) -> UNet:
        self._model = self._model.to(device)
        return self

    def eval(self) -> UNet:
        self._model.eval()
        return self

    def load_state_dict(self, state: Any) -> None:
        self._model.load_state_dict(state)

    def parameters(self) -> Any:
        return self._model.parameters()

    def __call__(self, x: Any) -> Any:
        return self._model(x)
