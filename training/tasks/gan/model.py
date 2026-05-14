from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from training.utils.input_spec import out_size_to_chw, resolve_input_spec


@dataclass
class GANModelConfig:
    latent_dim: int = 128
    hidden_dim: int = 256


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GANSystem(nn.Module):
    """Composite GAN module that contains E/G/D and utility helpers."""

    def __init__(self, channels: int, height: int, width: int, cfg: GANModelConfig):
        super().__init__()
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.image_dim = int(channels * height * width)
        self.latent_dim = int(cfg.latent_dim)

        self.encoder = Encoder(self.image_dim, self.latent_dim, cfg.hidden_dim)
        self.generator = Generator(self.latent_dim, self.image_dim, cfg.hidden_dim)
        self.discriminator = Discriminator(self.image_dim, cfg.hidden_dim)

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1)

    def _reshape(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.view(flat.shape[0], self.channels, self.height, self.width)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._flatten(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._reshape(self.generator(z))

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(self._flatten(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def generate(self, num_samples: int, device: torch.device | str | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(int(num_samples), self.latent_dim, device=device)
        with torch.no_grad():
            return self.decode(z)


def _get_value(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def build_model(cfg: dict[str, Any]) -> GANSystem:
    model_cfg = _get_value(cfg, "model", {})
    gan_cfg = _get_value(model_cfg, "gan", {})

    out_size, layout = resolve_input_spec(cfg)
    channels, height, width = out_size_to_chw(out_size, layout)

    model_config = GANModelConfig(
        latent_dim=int(_get_value(gan_cfg, "latent_dim", 128)),
        hidden_dim=int(_get_value(gan_cfg, "hidden_dim", 256)),
    )

    model = GANSystem(channels=channels, height=height, width=width, cfg=model_config)
    setattr(model, "input_size", out_size)
    setattr(model, "input_layout", "CHW")
    return model
