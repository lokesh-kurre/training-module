from __future__ import annotations

from typing import Any

import torch
import torch.optim as optim

from training.tasks.base import BaseTask, TaskStepOutput
from training.tasks.registry import register_task
from training.utils.importer import get_obj_by_name
from training.utils.input_spec import resolve_input_spec


@register_task("gan")
class GANTask(BaseTask):
    """GAN training task with Encoder/Generator/Discriminator composite model."""

    def build_model(self) -> Any:
        model_builder = get_obj_by_name(
            self.cfg.get("trainer", {}).get(
                "model_builder",
                "training.tasks.gan.model.build_model",
            )
        )
        return model_builder(self.cfg)

    def build_losses(self) -> Any:
        loss_builder = get_obj_by_name(
            self.cfg.get("trainer", {}).get(
                "loss_builder",
                "training.tasks.gan.losses.build_loss",
            )
        )
        return loss_builder(self.cfg)

    def build_metrics(self) -> Any:
        metrics_fn = self.cfg.get("trainer", {}).get(
            "metrics_fn",
            "training.tasks.gan.metrics.compute_metrics",
        )
        if isinstance(metrics_fn, str):
            return get_obj_by_name(metrics_fn)
        return metrics_fn

    def get_optimizers(self) -> dict[str, optim.Optimizer] | None:
        """Return separate D and G optimizers for GAN training (alternating updates)."""
        optimizer_cfg = self.cfg.get("optimizer", {})
        optimizer_name = optimizer_cfg.get("name", "adam").lower()
        lr = float(optimizer_cfg.get("lr", 0.0002))
        weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

        if optimizer_name == "adam":
            opt_d = optim.Adam(self.model.discriminator.parameters(), lr=lr, weight_decay=weight_decay)
            opt_g = optim.Adam(
                list(self.model.encoder.parameters()) + list(self.model.generator.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            opt_d = optim.AdamW(self.model.discriminator.parameters(), lr=lr, weight_decay=weight_decay)
            opt_g = optim.AdamW(
                list(self.model.encoder.parameters()) + list(self.model.generator.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            momentum = float(optimizer_cfg.get("momentum", 0.9))
            opt_d = optim.SGD(self.model.discriminator.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            opt_g = optim.SGD(
                list(self.model.encoder.parameters()) + list(self.model.generator.parameters()),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return {"discriminator": opt_d, "generator": opt_g}

    def _to_chw(self, images: torch.Tensor) -> torch.Tensor:
        _out_size, layout = resolve_input_spec(self.cfg)
        if layout == "HWC" and images.ndim == 4:
            images = images.permute(0, 3, 1, 2).contiguous()
        return images.float()

    def _to_minus1_plus1(self, images: torch.Tensor) -> torch.Tensor:
        # If input appears ImageNet-normalized, denormalize first.
        img_min = float(images.min().item())
        img_max = float(images.max().item())
        if img_min < -0.2 or img_max > 1.2:
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, -1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, -1, 1, 1)
            images = images * std + mean

        images = images.clamp(0.0, 1.0)
        return (images * 2.0) - 1.0

    def _extract_latent_code(self, batch: Any) -> torch.Tensor | None:
        """Extract latent code from batch if provided (for conditional generation).

        Batch format can be:
        - (images,): standard format, latent is None
        - (images, latent): conditional generation
        - (content, style, latent): style transfer format
        """
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            return None

        latent = None
        # (content, style, latent)
        if len(batch) >= 3 and torch.is_tensor(batch[2]):
            latent = batch[2]
        # (images, latent)
        elif (
            len(batch) >= 2
            and torch.is_tensor(batch[1])
            and getattr(batch[1], "ndim", 0) == 2
            and torch.is_floating_point(batch[1])
        ):
            latent = batch[1]

        if latent is None:
            return None

        if latent.ndim != 2:
            return None

        latent_dim = int(self.model.latent_dim)
        if latent.shape[1] > latent_dim:
            latent = latent[:, :latent_dim]
        elif latent.shape[1] < latent_dim:
            pad = torch.zeros(latent.shape[0], latent_dim - latent.shape[1], device=latent.device, dtype=latent.dtype)
            latent = torch.cat([latent, pad], dim=1)

        return latent

    def _step(self, batch: Any, phase: str = "joint") -> TaskStepOutput:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = self._to_minus1_plus1(self._to_chw(images))

        z_enc = self.model.encode(images)
        recon = self.model.decode(z_enc)

        # Check if batch provides latent code (for conditional/style transfer generation)
        z_provided = self._extract_latent_code(batch)
        if z_provided is not None:
            z_noise = z_provided.to(images.device).float()
        else:
            z_noise = torch.randn(images.shape[0], self.model.latent_dim, device=images.device)
        fake = self.model.decode(z_noise)

        d_real_logits = self.model.discriminate(images)
        d_fake_logits = self.model.discriminate(fake.detach())
        d_recon_logits = self.model.discriminate(recon.detach())

        adv_loss_fn = self.loss_fn["adv"]
        recon_loss_fn = self.loss_fn["recon"]

        real_target = torch.ones_like(d_real_logits)
        fake_target = torch.zeros_like(d_fake_logits)

        d_loss_real = adv_loss_fn(d_real_logits, real_target)
        d_loss_fake = adv_loss_fn(d_fake_logits, fake_target)
        d_loss_recon = adv_loss_fn(d_recon_logits, fake_target)
        d_loss = (d_loss_real + d_loss_fake + d_loss_recon) / 3.0

        g_fake_logits = self.model.discriminate(fake)
        g_recon_logits = self.model.discriminate(recon)
        g_adv_loss = 0.5 * (
            adv_loss_fn(g_fake_logits, torch.ones_like(g_fake_logits))
            + adv_loss_fn(g_recon_logits, torch.ones_like(g_recon_logits))
        )
        recon_l1 = recon_loss_fn(recon, images)

        joint_loss = (
            self.loss_fn["weight_d"] * d_loss
            + self.loss_fn["weight_g"] * g_adv_loss
            + self.loss_fn["weight_recon"] * recon_l1
        )

        d_obj_loss = self.loss_fn["weight_d"] * d_loss
        g_obj_loss = self.loss_fn["weight_g"] * g_adv_loss + self.loss_fn["weight_recon"] * recon_l1

        if phase == "discriminator":
            step_loss = d_obj_loss
        elif phase == "generator":
            step_loss = g_obj_loss
        else:
            step_loss = joint_loss

        outputs = {
            "d_real_logits": d_real_logits.detach(),
            "d_fake_logits": d_fake_logits.detach(),
            "d_recon_logits": d_recon_logits.detach(),
            "d_loss": d_loss.detach(),
            "g_adv_loss": g_adv_loss.detach(),
            "recon_l1": recon_l1.detach(),
            "joint_loss": joint_loss.detach(),
            "d_obj_loss": d_obj_loss.detach(),
            "g_obj_loss": g_obj_loss.detach(),
            "fake_samples": fake.detach(),
            "recon_samples": recon.detach(),
            "real_samples": images.detach(),
        }
        metrics = self.metrics_fn(outputs, self.cfg)
        return TaskStepOutput(loss=step_loss, outputs=outputs, metrics=metrics)

    def training_step(self, batch: Any) -> TaskStepOutput:
        return self._step(batch, phase="joint")

    def training_step_discriminator(self, batch: Any) -> TaskStepOutput:
        return self._step(batch, phase="discriminator")

    def training_step_generator(self, batch: Any) -> TaskStepOutput:
        return self._step(batch, phase="generator")

    def validation_step(self, batch: Any) -> TaskStepOutput:
        return self._step(batch)

    def predict_step(self, batch: Any) -> Any:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = self._to_minus1_plus1(self._to_chw(images))
        return self.model(images)
