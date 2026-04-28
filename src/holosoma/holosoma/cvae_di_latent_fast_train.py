# python src/holosoma/holosoma/cvae_di_latent_fast_train.py \
#   --data-dir /home/rllab/haechan/holosoma/logs/WholeBodyTracking/cvae_suitcase/telemetry \
#   --ir-window-body-source all \
#   --ir-cvae-checkpoint /home/rllab/haechan/holosoma/logs/CVAE/0416_ir_all_64/best.pt \
#   --condition-dim 16 \
#   --encoder-type temporal_gru \
#   --best-val-metric val_value_mae \
#   --decoder-value-loss-weight 100 \
#   --decoder-value-loss-type l1 \
#   --decoder-value-eval true \
#   --decoder-eval-samples 8

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from loguru import logger
from torch import nn

from holosoma.cvae_di_train import (
    DEFAULT_CLIP_MODEL_ID,
    DEFAULT_CONDITION_TEXT,
    DEFAULT_DATA_DIR,
    DEFAULT_IR_CVAE_CHECKPOINT,
    DEFAULT_OUTPUT_ROOT,
    TelemetryMetadata,
    TextConditionProjector,
    compute_target_latent_normalization_stats,
    denormalize_target_latents,
    denormalize_target_logvar,
    encode_ir_latent_targets,
    extract_episode_paired_windows,
    flatten_episode_split,
    gaussian_kl_divergence,
    iterate_batch_indices,
    normalize_target_latents,
    normalize_target_logvar,
)
from holosoma.cvae_ir_train import (
    CLIPTextFeatureExtractor,
    U_WINDOW_BODY_SOURCE_CHOICES,
    clone_state_dict_to_cpu,
    compute_metric_differences,
    configure_cuda_backend,
    create_run_paths,
    init_wandb,
    load_cvae as load_ir_cvae,
    normalize_ir_window_body_source,
    rename_metric_prefix,
    resolve_device,
    save_config,
    set_seed,
    split_episode_indices,
    str_to_bool,
)
from holosoma.utils.safe_torch_import import torch


BEST_VAL_METRIC_CHOICES = (
    "val_loss",
    "val_distribution_kl",
    "val_moment_loss",
    "val_mu_rmse",
    "val_std_rmse",
    "val_value_mae",
    "val_value_rmse",
)
DECODER_VALUE_LOSS_CHOICES = ("l1", "mse", "smooth_l1")
ENCODER_TYPE_CHOICES = ("temporal_gru", "fast_cnn")


@dataclass
class TrainConfig:
    data_dir: str = DEFAULT_DATA_DIR
    condition_text: str = DEFAULT_CONDITION_TEXT
    ir_window_body_source: str = "all"
    output_root: str = DEFAULT_OUTPUT_ROOT
    run_name: str = "cvae-di-latent-fast"
    ir_cvae_checkpoint: str = DEFAULT_IR_CVAE_CHECKPOINT
    latent_dim: int = 64
    encoder_type: str = "temporal_gru"
    condition_dim: int = 16
    conv_channels: tuple[int, int, int] = (32, 64, 64)
    hidden_dim: int = 256
    batch_size: int = 8192
    epochs: int = 5000
    learning_rate: float = 3e-4
    distribution_kl_weight: float = 1.0
    moment_loss_weight: float = 0.1
    decoder_value_loss_weight: float = 100.0
    decoder_train_samples: int = 2
    decoder_value_loss_type: str = "l1"
    best_val_metric: str = "val_value_mae"
    decoder_value_eval: bool = True
    decoder_eval_samples: int = 8
    min_feature_std: float = 1e-4
    min_target_latent_std: float = 1e-3
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-2
    val_improvement_min_delta: float = 1e-4
    lr_plateau_patience: int = 100
    lr_plateau_factor: float = 0.5
    min_learning_rate: float = 1e-5
    early_stop_patience: int = 500
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 100
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    wandb_enabled: bool = True
    wandb_project: str = "CVAE"
    wandb_entity: str | None = None
    wandb_group: str = "cvae_di_latent_fast"
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ("cvae", "di", "latent", "fast", "depth_window")
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID
    clip_cache_dir: str | None = None
    clip_local_files_only: bool = True
    clip_quiet_load: bool = True


class FastDepthWindowLatentEncoder(nn.Module):
    """Fast depth-window encoder that treats the temporal window as image channels."""

    def __init__(
        self,
        input_shape: Sequence[int],
        text_feature_dim: int,
        condition_dim: int,
        latent_dim: int,
        hidden_dim: int,
        conv_channels: Sequence[int],
        *,
        logvar_clamp_min: float = -10.0,
        logvar_clamp_max: float = 10.0,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"Depth input shape must have length 3, got {input_shape}")
        if len(conv_channels) != 3:
            raise ValueError(f"Expected conv_channels=(c1, c2, c3), got {conv_channels}")

        self.window_size = int(input_shape[0])
        self.height = int(input_shape[1])
        self.width = int(input_shape[2])
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        c1, c2, c3 = (int(value) for value in conv_channels)
        self.conv_channels = (c1, c2, c3)
        self.logvar_clamp_min = float(logvar_clamp_min)
        self.logvar_clamp_max = float(logvar_clamp_max)

        self.features = nn.Sequential(
            nn.Conv2d(self.window_size, c1, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 5)),
        )
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 5, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
        )
        self.text_projector = TextConditionProjector(text_feature_dim, self.condition_dim)
        self.latent_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.condition_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
        )
        self.mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected depth batch shape [B, T, H, W], got {tuple(x.shape)}")
        batch_size, window_size, height, width = x.shape
        if (window_size, height, width) != (self.window_size, self.height, self.width):
            raise ValueError(
                f"Expected depth batch spatial shape {(self.window_size, self.height, self.width)}, "
                f"got {(window_size, height, width)}"
            )
        if text_features.ndim != 2 or text_features.shape[0] != batch_size:
            raise ValueError(
                f"Expected text feature batch shape [B, D] with B={batch_size}, got {tuple(text_features.shape)}"
            )
        hidden = self.trunk(self.features(x))
        condition = self.text_projector(text_features)
        hidden = self.latent_head(torch.cat([hidden, condition], dim=-1))
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        logvar = torch.clamp(logvar, min=self.logvar_clamp_min, max=self.logvar_clamp_max)
        return mu, logvar


class TemporalDepthWindowLatentEncoder(nn.Module):
    """Frame CNN + GRU encoder for depth windows where temporal motion matters."""

    def __init__(
        self,
        input_shape: Sequence[int],
        text_feature_dim: int,
        condition_dim: int,
        latent_dim: int,
        hidden_dim: int,
        conv_channels: Sequence[int],
        *,
        logvar_clamp_min: float = -10.0,
        logvar_clamp_max: float = 10.0,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"Depth input shape must have length 3, got {input_shape}")
        if len(conv_channels) != 3:
            raise ValueError(f"Expected conv_channels=(c1, c2, c3), got {conv_channels}")

        self.window_size = int(input_shape[0])
        self.height = int(input_shape[1])
        self.width = int(input_shape[2])
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        c1, c2, c3 = (int(value) for value in conv_channels)
        self.conv_channels = (c1, c2, c3)
        self.logvar_clamp_min = float(logvar_clamp_min)
        self.logvar_clamp_max = float(logvar_clamp_max)

        self.frame_features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 5)),
        )
        self.frame_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 5, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
        )
        self.temporal_encoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.text_projector = TextConditionProjector(text_feature_dim, self.condition_dim)
        self.latent_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.condition_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
        )
        self.mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected depth batch shape [B, T, H, W], got {tuple(x.shape)}")
        batch_size, window_size, height, width = x.shape
        if (window_size, height, width) != (self.window_size, self.height, self.width):
            raise ValueError(
                f"Expected depth batch spatial shape {(self.window_size, self.height, self.width)}, "
                f"got {(window_size, height, width)}"
            )
        if text_features.ndim != 2 or text_features.shape[0] != batch_size:
            raise ValueError(
                f"Expected text feature batch shape [B, D] with B={batch_size}, got {tuple(text_features.shape)}"
            )

        frames = x.reshape(batch_size * window_size, 1, height, width)
        frame_features = self.frame_projection(self.frame_features(frames))
        frame_features = frame_features.reshape(batch_size, window_size, self.hidden_dim)
        _, hidden = self.temporal_encoder(frame_features)
        temporal_feature = hidden[-1]
        condition = self.text_projector(text_features)
        hidden = self.latent_head(torch.cat([temporal_feature, condition], dim=-1))
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        logvar = torch.clamp(logvar, min=self.logvar_clamp_min, max=self.logvar_clamp_max)
        return mu, logvar


def create_depth_encoder(
    *,
    encoder_type: str,
    input_shape: Sequence[int],
    text_feature_dim: int,
    condition_dim: int,
    latent_dim: int,
    hidden_dim: int,
    conv_channels: Sequence[int],
) -> nn.Module:
    if encoder_type == "fast_cnn":
        return FastDepthWindowLatentEncoder(
            input_shape=input_shape,
            text_feature_dim=text_feature_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            conv_channels=conv_channels,
        )
    if encoder_type == "temporal_gru":
        return TemporalDepthWindowLatentEncoder(
            input_shape=input_shape,
            text_feature_dim=text_feature_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            conv_channels=conv_channels,
        )
    raise ValueError(f"encoder_type must be one of {ENCODER_TYPE_CHOICES}, got {encoder_type}.")


def posterior_moment_loss(
    target_mu: torch.Tensor,
    target_logvar: torch.Tensor,
    predicted_mu: torch.Tensor,
    predicted_logvar: torch.Tensor,
) -> torch.Tensor:
    target_std = torch.exp(0.5 * torch.clamp(target_logvar, min=-20.0, max=20.0))
    predicted_std = torch.exp(0.5 * torch.clamp(predicted_logvar, min=-20.0, max=20.0))
    mu_loss = (predicted_mu - target_mu).pow(2).mean()
    std_loss = (predicted_std - target_std).pow(2).mean()
    return mu_loss + std_loss


def compute_loss_terms(
    target_mu: torch.Tensor,
    target_logvar: torch.Tensor,
    predicted_mu: torch.Tensor,
    predicted_logvar: torch.Tensor,
    *,
    distribution_kl_weight: float,
    moment_loss_weight: float,
) -> dict[str, torch.Tensor]:
    distribution_kl = gaussian_kl_divergence(target_mu, target_logvar, predicted_mu, predicted_logvar)
    moment_loss = posterior_moment_loss(target_mu, target_logvar, predicted_mu, predicted_logvar)
    total = distribution_kl_weight * distribution_kl + moment_loss_weight * moment_loss
    return {
        "loss": total,
        "distribution_kl": distribution_kl,
        "moment_loss": moment_loss,
    }


def make_eval_eps(shape: torch.Size, *, device: str, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, device=device, generator=generator)


def ground_truth_value_loss(
    decoded_ir: torch.Tensor,
    ground_truth_ir: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "l1":
        return (decoded_ir - ground_truth_ir).abs().mean()
    if loss_type == "mse":
        return (decoded_ir - ground_truth_ir).pow(2).mean()
    if loss_type == "smooth_l1":
        return nn.functional.smooth_l1_loss(decoded_ir, ground_truth_ir)
    raise ValueError(f"Unknown decoder value loss type: {loss_type}")


@torch.no_grad()
def evaluate_ir_decoder_oracle(
    target_mu: torch.Tensor,
    target_logvar: torch.Tensor,
    target_ir_windows: torch.Tensor,
    *,
    ir_decoder_model: Any,
    ir_decoder_text_feature: torch.Tensor,
    ir_feature_mean: torch.Tensor,
    ir_feature_std: torch.Tensor,
    batch_size: int,
    decoder_eval_samples: int,
    device: str,
    prefix: str,
    use_mean_z: bool,
) -> dict[str, float | int]:
    if target_mu.shape[0] == 0:
        return {
            f"{prefix}_num_samples": 0,
            f"{prefix}_value_mae": float("nan"),
            f"{prefix}_value_rmse": float("nan"),
            f"{prefix}_value_max_abs": float("nan"),
            f"{prefix}_value_dim_mae_max": float("nan"),
            f"{prefix}_value_dim_mae_p95": float("nan"),
            f"{prefix}_value_dim_rmse_max": float("nan"),
        }

    ir_decoder_model.eval()
    ir_decoder_text_feature = ir_decoder_text_feature.to(device=device, dtype=torch.float32)
    ir_feature_mean = ir_feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
    ir_feature_std = ir_feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
    decoder_eval_samples = 1 if use_mean_z else max(int(decoder_eval_samples), 1)

    total_value_abs_error = 0.0
    total_value_squared_error = 0.0
    total_value_elements = 0
    total_value_abs_by_dim: torch.Tensor | None = None
    total_value_sq_by_dim: torch.Tensor | None = None
    total_value_dim_count = 0
    value_max_abs = 0.0
    seen_samples = 0

    for batch_number, batch_indices in enumerate(
        iterate_batch_indices(int(target_mu.shape[0]), batch_size, shuffle=False, seed=0)
    ):
        batch_mu = target_mu.index_select(0, batch_indices).to(device=device, dtype=torch.float32, non_blocking=True)
        batch_logvar = target_logvar.index_select(0, batch_indices).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        batch_target_ir = target_ir_windows.index_select(0, batch_indices).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        batch_target_ir = batch_target_ir.reshape(batch_target_ir.shape[0], -1)
        batch_size_current = int(batch_mu.shape[0])
        batch_std = torch.exp(0.5 * torch.clamp(batch_logvar, min=-20.0, max=20.0))
        decoder_text = ir_decoder_text_feature.expand(batch_size_current, -1)
        batch_value_abs = 0.0
        batch_value_sq = 0.0
        batch_value_max_abs = 0.0
        batch_value_abs_by_dim = torch.zeros(batch_target_ir.shape[1], device=device)
        batch_value_sq_by_dim = torch.zeros(batch_target_ir.shape[1], device=device)

        for sample_index in range(decoder_eval_samples):
            if use_mean_z:
                sampled_z = batch_mu
            else:
                decoder_eps = make_eval_eps(
                    batch_mu.shape,
                    device=device,
                    seed=9876 + batch_number * decoder_eval_samples + sample_index,
                )
                sampled_z = batch_mu + batch_std * decoder_eps
            decoded_ir_normalized = ir_decoder_model.decode(sampled_z, decoder_text)
            decoded_ir = decoded_ir_normalized * ir_feature_std + ir_feature_mean
            value_error = decoded_ir - batch_target_ir
            value_abs = value_error.abs()
            batch_value_abs += value_abs.sum().item()
            batch_value_sq += value_error.square().sum().item()
            batch_value_abs_by_dim += value_abs.sum(dim=0)
            batch_value_sq_by_dim += value_error.square().sum(dim=0)
            batch_value_max_abs = max(batch_value_max_abs, float(value_abs.max().item()))

        total_value_abs_error += batch_value_abs / decoder_eval_samples
        total_value_squared_error += batch_value_sq / decoder_eval_samples
        total_value_elements += int(batch_target_ir.numel())
        if total_value_abs_by_dim is None:
            total_value_abs_by_dim = torch.zeros_like(batch_value_abs_by_dim, device="cpu")
            total_value_sq_by_dim = torch.zeros_like(batch_value_sq_by_dim, device="cpu")
        total_value_abs_by_dim += (batch_value_abs_by_dim / decoder_eval_samples).detach().cpu()
        total_value_sq_by_dim += (batch_value_sq_by_dim / decoder_eval_samples).detach().cpu()
        total_value_dim_count += batch_size_current
        value_max_abs = max(value_max_abs, batch_value_max_abs)
        seen_samples += batch_size_current

    value_dim_mae = total_value_abs_by_dim / total_value_dim_count
    value_dim_rmse = torch.sqrt(total_value_sq_by_dim / total_value_dim_count)
    return {
        f"{prefix}_num_samples": int(seen_samples),
        f"{prefix}_value_mae": total_value_abs_error / total_value_elements,
        f"{prefix}_value_rmse": math.sqrt(total_value_squared_error / total_value_elements),
        f"{prefix}_value_max_abs": value_max_abs,
        f"{prefix}_value_dim_mae_max": float(value_dim_mae.max().item()),
        f"{prefix}_value_dim_mae_p95": float(torch.quantile(value_dim_mae, 0.95).item()),
        f"{prefix}_value_dim_rmse_max": float(value_dim_rmse.max().item()),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    depth_windows: torch.Tensor,
    target_mu: torch.Tensor,
    target_logvar: torch.Tensor,
    *,
    base_text_feature: torch.Tensor,
    batch_size: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    target_latent_mean: torch.Tensor,
    target_latent_std: torch.Tensor,
    target_ir_windows: torch.Tensor | None = None,
    ir_decoder_model: Any | None = None,
    ir_decoder_text_feature: torch.Tensor | None = None,
    ir_feature_mean: torch.Tensor | None = None,
    ir_feature_std: torch.Tensor | None = None,
    decoder_eval_samples: int = 1,
    distribution_kl_weight: float,
    moment_loss_weight: float,
    device: str,
    prefix: str,
) -> dict[str, float | int]:
    if depth_windows.shape[0] == 0:
        return {
            f"{prefix}_num_samples": 0,
            f"{prefix}_loss": float("nan"),
            f"{prefix}_distribution_kl": float("nan"),
            f"{prefix}_moment_loss": float("nan"),
            f"{prefix}_mu_mae": float("nan"),
            f"{prefix}_mu_rmse": float("nan"),
            f"{prefix}_std_rmse": float("nan"),
            f"{prefix}_value_mae": float("nan"),
            f"{prefix}_value_rmse": float("nan"),
            f"{prefix}_value_max_abs": float("nan"),
            f"{prefix}_value_dim_mae_max": float("nan"),
            f"{prefix}_value_dim_mae_p95": float("nan"),
            f"{prefix}_value_dim_rmse_max": float("nan"),
        }

    feature_mean_device = feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
    feature_std_device = feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
    target_latent_mean_device = target_latent_mean.to(device=device, dtype=torch.float32)
    target_latent_std_device = target_latent_std.to(device=device, dtype=torch.float32)

    model.eval()
    total_loss = 0.0
    total_distribution_kl = 0.0
    total_moment_loss = 0.0
    total_mu_abs = 0.0
    total_mu_sq = 0.0
    total_std_sq = 0.0
    total_values = 0
    total_value_abs_error = 0.0
    total_value_squared_error = 0.0
    total_value_elements = 0
    total_value_abs_by_dim: torch.Tensor | None = None
    total_value_sq_by_dim: torch.Tensor | None = None
    total_value_dim_count = 0
    value_max_abs = 0.0
    seen_samples = 0
    decoder_eval_enabled = (
        target_ir_windows is not None
        and ir_decoder_model is not None
        and ir_decoder_text_feature is not None
        and ir_feature_mean is not None
        and ir_feature_std is not None
    )
    decoder_eval_samples = max(int(decoder_eval_samples), 1)
    if decoder_eval_enabled:
        ir_decoder_model.eval()
        ir_decoder_text_feature = ir_decoder_text_feature.to(device=device, dtype=torch.float32)
        ir_feature_mean = ir_feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
        ir_feature_std = ir_feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)

    for batch_number, batch_indices in enumerate(
        iterate_batch_indices(int(depth_windows.shape[0]), batch_size, shuffle=False, seed=0)
    ):
        batch_depth = depth_windows.index_select(0, batch_indices).to(device=device, dtype=torch.float32, non_blocking=True)
        batch_target_mu = target_mu.index_select(0, batch_indices).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        batch_target_logvar = target_logvar.index_select(0, batch_indices).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if decoder_eval_enabled:
            batch_target_ir = target_ir_windows.index_select(0, batch_indices).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            batch_target_ir = batch_target_ir.reshape(batch_target_ir.shape[0], -1)
        else:
            batch_target_ir = None
        batch_size_current = int(batch_depth.shape[0])

        batch_depth = (batch_depth - feature_mean_device) / feature_std_device
        batch_text = base_text_feature.expand(batch_size_current, -1)
        batch_target_mu_normalized = normalize_target_latents(
            batch_target_mu,
            target_latent_mean_device,
            target_latent_std_device,
        )
        batch_target_logvar_normalized = normalize_target_logvar(batch_target_logvar, target_latent_std_device)

        predicted_mu_normalized, predicted_logvar_normalized = model(batch_depth, batch_text)
        loss_terms = compute_loss_terms(
            batch_target_mu_normalized,
            batch_target_logvar_normalized,
            predicted_mu_normalized,
            predicted_logvar_normalized,
            distribution_kl_weight=distribution_kl_weight,
            moment_loss_weight=moment_loss_weight,
        )
        if not torch.isfinite(loss_terms["loss"]):
            raise RuntimeError("Evaluation loss became non-finite.")

        predicted_mu = denormalize_target_latents(
            predicted_mu_normalized,
            target_latent_mean_device,
            target_latent_std_device,
        )
        predicted_logvar = denormalize_target_logvar(predicted_logvar_normalized, target_latent_std_device)
        predicted_std = torch.exp(0.5 * torch.clamp(predicted_logvar, min=-20.0, max=20.0))
        target_std = torch.exp(0.5 * torch.clamp(batch_target_logvar, min=-20.0, max=20.0))

        mu_error = predicted_mu - batch_target_mu
        std_error = predicted_std - target_std

        total_loss += loss_terms["loss"].item() * batch_size_current
        total_distribution_kl += loss_terms["distribution_kl"].item() * batch_size_current
        total_moment_loss += loss_terms["moment_loss"].item() * batch_size_current
        total_mu_abs += mu_error.abs().sum().item()
        total_mu_sq += mu_error.square().sum().item()
        total_std_sq += std_error.square().sum().item()
        total_values += int(batch_target_mu.numel())
        if decoder_eval_enabled and batch_target_ir is not None:
            batch_value_abs = 0.0
            batch_value_sq = 0.0
            batch_value_max_abs = 0.0
            batch_value_abs_by_dim = torch.zeros(batch_target_ir.shape[1], device=device)
            batch_value_sq_by_dim = torch.zeros(batch_target_ir.shape[1], device=device)
            decoder_text = ir_decoder_text_feature.expand(batch_size_current, -1)
            for sample_index in range(decoder_eval_samples):
                decoder_eps = make_eval_eps(
                    batch_target_mu.shape,
                    device=device,
                    seed=4321 + batch_number * decoder_eval_samples + sample_index,
                )
                sampled_z = predicted_mu + predicted_std * decoder_eps
                decoded_ir_normalized = ir_decoder_model.decode(sampled_z, decoder_text)
                decoded_ir = decoded_ir_normalized * ir_feature_std + ir_feature_mean
                value_error = decoded_ir - batch_target_ir
                value_abs = value_error.abs()
                batch_value_abs += value_abs.sum().item()
                batch_value_sq += value_error.square().sum().item()
                batch_value_abs_by_dim += value_abs.sum(dim=0)
                batch_value_sq_by_dim += value_error.square().sum(dim=0)
                batch_value_max_abs = max(batch_value_max_abs, float(value_abs.max().item()))
            total_value_abs_error += batch_value_abs / decoder_eval_samples
            total_value_squared_error += batch_value_sq / decoder_eval_samples
            total_value_elements += int(batch_target_ir.numel())
            if total_value_abs_by_dim is None:
                total_value_abs_by_dim = torch.zeros_like(batch_value_abs_by_dim, device="cpu")
                total_value_sq_by_dim = torch.zeros_like(batch_value_sq_by_dim, device="cpu")
            total_value_abs_by_dim += (batch_value_abs_by_dim / decoder_eval_samples).detach().cpu()
            total_value_sq_by_dim += (batch_value_sq_by_dim / decoder_eval_samples).detach().cpu()
            total_value_dim_count += batch_size_current
            value_max_abs = max(value_max_abs, batch_value_max_abs)
        seen_samples += batch_size_current

    distribution_kl = total_distribution_kl / seen_samples
    if total_value_elements > 0:
        value_mae = total_value_abs_error / total_value_elements
        value_rmse = math.sqrt(total_value_squared_error / total_value_elements)
        value_dim_mae = total_value_abs_by_dim / total_value_dim_count
        value_dim_rmse = torch.sqrt(total_value_sq_by_dim / total_value_dim_count)
        value_dim_mae_max = float(value_dim_mae.max().item())
        value_dim_mae_p95 = float(torch.quantile(value_dim_mae, 0.95).item())
        value_dim_rmse_max = float(value_dim_rmse.max().item())
    else:
        value_mae = float("nan")
        value_rmse = float("nan")
        value_max_abs = float("nan")
        value_dim_mae_max = float("nan")
        value_dim_mae_p95 = float("nan")
        value_dim_rmse_max = float("nan")
    return {
        f"{prefix}_num_samples": int(seen_samples),
        f"{prefix}_loss": total_loss / seen_samples,
        f"{prefix}_distribution_kl": distribution_kl,
        f"{prefix}_moment_loss": total_moment_loss / seen_samples,
        f"{prefix}_mu_mae": total_mu_abs / total_values,
        f"{prefix}_mu_rmse": math.sqrt(total_mu_sq / total_values),
        f"{prefix}_std_rmse": math.sqrt(total_std_sq / total_values),
        f"{prefix}_value_mae": value_mae,
        f"{prefix}_value_rmse": value_rmse,
        f"{prefix}_value_max_abs": value_max_abs,
        f"{prefix}_value_dim_mae_max": value_dim_mae_max,
        f"{prefix}_value_dim_mae_p95": value_dim_mae_p95,
        f"{prefix}_value_dim_rmse_max": value_dim_rmse_max,
    }


def make_checkpoint_payload(
    *,
    config: TrainConfig,
    input_shape: tuple[int, int, int],
    num_samples: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    target_latent_mean: torch.Tensor,
    target_latent_std: torch.Tensor,
    text_feature_dim: int,
    telemetry_metadata: TelemetryMetadata,
    model: nn.Module,
    checkpoint_type: str,
    epoch: int,
    val_loss: float,
    val_selection_metric: str,
    val_selection_score: float,
    ir_alignment_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_type": "depth_window_latent_encoder",
        "encoder_type": config.encoder_type,
        "checkpoint_type": checkpoint_type,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_selection_metric": val_selection_metric,
        "val_selection_score": val_selection_score,
        "config": asdict(config),
        "input_shape": [input_shape[0], input_shape[1], input_shape[2]],
        "num_samples": int(num_samples),
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "target_latent_mean": target_latent_mean.cpu(),
        "target_latent_std": target_latent_std.cpu(),
        "text_feature_dim": int(text_feature_dim),
        "condition_text": config.condition_text,
        "telemetry": asdict(telemetry_metadata),
        "clip": {
            "model_id": config.clip_model_id,
            "cache_dir": config.clip_cache_dir,
            "local_files_only": config.clip_local_files_only,
        },
        "ir_alignment": ir_alignment_metadata,
        "encoder_state_dict": model.state_dict(),
    }


def save_encoder_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    config: TrainConfig,
    input_shape: tuple[int, int, int],
    num_samples: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    target_latent_mean: torch.Tensor,
    target_latent_std: torch.Tensor,
    text_feature_dim: int,
    telemetry_metadata: TelemetryMetadata,
    checkpoint_type: str,
    epoch: int,
    val_loss: float,
    val_selection_metric: str,
    val_selection_score: float,
    ir_alignment_metadata: dict[str, Any],
) -> None:
    payload = make_checkpoint_payload(
        config=config,
        input_shape=input_shape,
        num_samples=num_samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_latent_mean=target_latent_mean,
        target_latent_std=target_latent_std,
        text_feature_dim=text_feature_dim,
        telemetry_metadata=telemetry_metadata,
        model=model,
        checkpoint_type=checkpoint_type,
        epoch=epoch,
        val_loss=val_loss,
        val_selection_metric=val_selection_metric,
        val_selection_score=val_selection_score,
        ir_alignment_metadata=ir_alignment_metadata,
    )
    torch.save(payload, checkpoint_path)


def validate_config(config: TrainConfig) -> None:
    config.ir_window_body_source = normalize_ir_window_body_source(config.ir_window_body_source)
    if config.best_val_metric not in BEST_VAL_METRIC_CHOICES:
        raise ValueError(f"best_val_metric must be one of {BEST_VAL_METRIC_CHOICES}, got {config.best_val_metric}.")
    if len(config.conv_channels) != 3:
        raise ValueError(f"conv_channels must have three values, got {config.conv_channels}.")
    if config.latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {config.latent_dim}.")
    if config.condition_dim <= 0:
        raise ValueError(f"condition_dim must be positive, got {config.condition_dim}.")
    if config.hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}.")
    if config.encoder_type not in ENCODER_TYPE_CHOICES:
        raise ValueError(f"encoder_type must be one of {ENCODER_TYPE_CHOICES}, got {config.encoder_type}.")
    if config.decoder_eval_samples <= 0:
        raise ValueError(f"decoder_eval_samples must be positive, got {config.decoder_eval_samples}.")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}.")
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.epochs}.")
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}.")
    if config.distribution_kl_weight < 0 or config.moment_loss_weight < 0:
        raise ValueError("Loss weights must be non-negative.")
    if config.decoder_value_loss_weight < 0:
        raise ValueError(f"decoder_value_loss_weight must be non-negative, got {config.decoder_value_loss_weight}.")
    if config.decoder_train_samples <= 0:
        raise ValueError(f"decoder_train_samples must be positive, got {config.decoder_train_samples}.")
    if config.decoder_value_loss_type not in DECODER_VALUE_LOSS_CHOICES:
        raise ValueError(
            f"decoder_value_loss_type must be one of {DECODER_VALUE_LOSS_CHOICES}, "
            f"got {config.decoder_value_loss_type}."
        )
    if config.best_val_metric in {"val_value_mae", "val_value_rmse"} and not config.decoder_value_eval:
        raise ValueError(f"best_val_metric={config.best_val_metric} requires decoder_value_eval=true.")
    if config.max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be positive, got {config.max_grad_norm}.")
    if config.val_improvement_min_delta < 0:
        raise ValueError(f"val_improvement_min_delta must be non-negative, got {config.val_improvement_min_delta}.")
    if config.lr_plateau_patience < 0:
        raise ValueError(f"lr_plateau_patience must be non-negative, got {config.lr_plateau_patience}.")
    if not 0.0 < config.lr_plateau_factor < 1.0:
        raise ValueError(f"lr_plateau_factor must be in (0, 1), got {config.lr_plateau_factor}.")
    if config.min_learning_rate <= 0:
        raise ValueError(f"min_learning_rate must be positive, got {config.min_learning_rate}.")
    if config.early_stop_patience < 0:
        raise ValueError(f"early_stop_patience must be non-negative, got {config.early_stop_patience}.")


def train_encoder(config: TrainConfig) -> Path:
    validate_config(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    configure_cuda_backend(device)
    if str(device).startswith("cuda"):
        cuda_device = torch.device(device)
        cuda_index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
        logger.info(f"Using CUDA device for fast latent training: {torch.cuda.get_device_name(cuda_index)} ({device})")
    else:
        logger.info(f"Using device for fast latent training: {device}")

    run_paths = create_run_paths(config)
    save_config(config, run_paths)
    wandb = None
    metrics_history: list[dict[str, float | int]] = []

    try:
        episodes, telemetry_metadata = extract_episode_paired_windows(
            Path(config.data_dir),
            ir_window_body_source=config.ir_window_body_source,
        )
        split_indices = split_episode_indices(len(episodes), config.val_ratio, config.test_ratio, config.seed)
        train_ir_np, train_depth_np, train_episode_ids = flatten_episode_split(
            episodes,
            split_indices["train"],
            telemetry_metadata.ir_window_shape,
            telemetry_metadata.depth_input_shape,
        )
        val_ir_np, val_depth_np, val_episode_ids = flatten_episode_split(
            episodes,
            split_indices["val"],
            telemetry_metadata.ir_window_shape,
            telemetry_metadata.depth_input_shape,
        )
        test_ir_np, test_depth_np, test_episode_ids = flatten_episode_split(
            episodes,
            split_indices["test"],
            telemetry_metadata.ir_window_shape,
            telemetry_metadata.depth_input_shape,
        )
        del episodes

        train_depth = torch.from_numpy(train_depth_np)
        val_depth = torch.from_numpy(val_depth_np)
        test_depth = torch.from_numpy(test_depth_np)
        train_ir = torch.from_numpy(train_ir_np)
        val_ir = torch.from_numpy(val_ir_np)
        test_ir = torch.from_numpy(test_ir_np)
        del train_depth_np, val_depth_np, test_depth_np, train_ir_np, val_ir_np, test_ir_np

        feature_mean = train_depth.mean(dim=0)
        feature_std = train_depth.std(dim=0).clamp_min(config.min_feature_std)

        ir_target_mu_train, ir_target_logvar_train, ir_payload = encode_ir_latent_targets(
            ir_checkpoint_path=config.ir_cvae_checkpoint,
            ir_windows=train_ir,
            condition_text=config.condition_text,
            ir_window_body_source=config.ir_window_body_source,
            batch_size=config.batch_size,
            device=device,
        )
        ir_target_mu_val, ir_target_logvar_val, _ = encode_ir_latent_targets(
            ir_checkpoint_path=config.ir_cvae_checkpoint,
            ir_windows=val_ir,
            condition_text=config.condition_text,
            ir_window_body_source=config.ir_window_body_source,
            batch_size=config.batch_size,
            device=device,
        )
        ir_target_mu_test, ir_target_logvar_test, _ = encode_ir_latent_targets(
            ir_checkpoint_path=config.ir_cvae_checkpoint,
            ir_windows=test_ir,
            condition_text=config.condition_text,
            ir_window_body_source=config.ir_window_body_source,
            batch_size=config.batch_size,
            device=device,
        )
        if config.decoder_value_loss_weight <= 0:
            del train_ir

        target_latent_dim = int(ir_target_mu_train.shape[1])
        if target_latent_dim != config.latent_dim:
            raise ValueError(
                f"Depth encoder latent_dim={config.latent_dim} must match frozen IR-CVAE latent dim={target_latent_dim}."
            )

        target_latent_mean, target_latent_std = compute_target_latent_normalization_stats(
            ir_target_mu_train,
            min_std=config.min_target_latent_std,
        )
        ir_target_mu_train_normalized = normalize_target_latents(
            ir_target_mu_train,
            target_latent_mean,
            target_latent_std,
        )
        ir_target_logvar_train_normalized = normalize_target_logvar(ir_target_logvar_train, target_latent_std)

        clip_text = CLIPTextFeatureExtractor(
            model_id=config.clip_model_id,
            device=device,
            cache_dir=config.clip_cache_dir,
            local_files_only=config.clip_local_files_only,
            quiet_load=config.clip_quiet_load,
        )
        logger.info(
            f"Loaded CLIP text encoder for fast latent student: model={config.clip_model_id}, "
            f"projected_condition_dim={config.condition_dim}, local_files_only={config.clip_local_files_only}"
        )
        base_text_feature = clip_text.encode([config.condition_text]).to(device=device, dtype=torch.float32)
        if not torch.isfinite(base_text_feature).all():
            raise RuntimeError("CLIP text features for fast latent student contain non-finite values.")
        text_feature_dim = int(base_text_feature.shape[-1])

        ir_decoder_model = None
        ir_decoder_text_feature = None
        ir_decoder_feature_mean = None
        ir_decoder_feature_std = None
        if config.decoder_value_eval or config.decoder_value_loss_weight > 0:
            ir_decoder_model, ir_decoder_payload = load_ir_cvae(config.ir_cvae_checkpoint, device=device)
            ir_decoder_model.eval()
            ir_decoder_model.requires_grad_(False)
            decoder_clip_cfg = ir_decoder_payload["clip"]
            decoder_text_extractor = CLIPTextFeatureExtractor(
                model_id=decoder_clip_cfg["model_id"],
                device=device,
                cache_dir=decoder_clip_cfg["cache_dir"],
                local_files_only=decoder_clip_cfg["local_files_only"],
                quiet_load=True,
            )
            decoder_text_string = config.condition_text or ir_decoder_payload["condition_text"]
            ir_decoder_text_feature = decoder_text_extractor.encode([decoder_text_string]).to(
                device=device,
                dtype=torch.float32,
            )
            ir_decoder_feature_mean = ir_decoder_payload["feature_mean"]
            ir_decoder_feature_std = ir_decoder_payload["feature_std"]
            logger.info(
                "Enabled frozen IR decoder value path: "
                f"decoder_value_loss_weight={config.decoder_value_loss_weight}, "
                f"decoder_value_loss_type={config.decoder_value_loss_type}, "
                f"decoder_train_samples={config.decoder_train_samples}, "
                f"decoder_eval_samples={config.decoder_eval_samples}, "
                f"target_ir_window_shape={tuple(ir_decoder_payload['input_shape'])}"
            )

        model = create_depth_encoder(
            encoder_type=config.encoder_type,
            input_shape=telemetry_metadata.depth_input_shape,
            text_feature_dim=text_feature_dim,
            condition_dim=config.condition_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            conv_channels=config.conv_channels,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        feature_mean_device = feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
        feature_std_device = feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
        target_latent_mean_device = target_latent_mean.to(device=device, dtype=torch.float32)
        target_latent_std_device = target_latent_std.to(device=device, dtype=torch.float32)
        ir_decoder_feature_mean_device = (
            ir_decoder_feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
            if ir_decoder_feature_mean is not None
            else None
        )
        ir_decoder_feature_std_device = (
            ir_decoder_feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
            if ir_decoder_feature_std is not None
            else None
        )

        ir_alignment_metadata = {
            "ir_cvae_checkpoint": config.ir_cvae_checkpoint,
            "target_model_type": ir_payload.get("model_type"),
            "target_condition_text": ir_payload.get("condition_text"),
            "target_latent_dim": target_latent_dim,
            "target_distribution": "posterior_mu_logvar",
            "ir_window_shape": list(ir_payload["input_shape"]),
            "ir_window_body_source": config.ir_window_body_source,
            "selected_telemetry_ir_window_shape": list(telemetry_metadata.ir_window_shape),
            "target_latent_mean_abs_mean": float(target_latent_mean.abs().mean().item()),
            "target_latent_std_mean": float(target_latent_std.mean().item()),
            "target_latent_std_min": float(target_latent_std.min().item()),
            "target_logvar_mean": float(ir_target_logvar_train.mean().item()),
            "target_logvar_min": float(ir_target_logvar_train.min().item()),
            "decoder_value_loss_weight": config.decoder_value_loss_weight,
            "decoder_train_samples": config.decoder_train_samples,
            "decoder_value_loss_type": config.decoder_value_loss_type,
        }

        logger.info(
            f"Training fast depth-window latent encoder on train/val/test windows = "
            f"{train_depth.shape[0]}/{val_depth.shape[0]}/{test_depth.shape[0]}, "
            f"depth_window_shape={telemetry_metadata.depth_input_shape}, latent_dim={config.latent_dim}, "
            f"encoder_type={config.encoder_type}, conv_channels={config.conv_channels}, hidden_dim={config.hidden_dim}, "
            f"condition_dim={config.condition_dim}, clip_model={config.clip_model_id}, "
            f"device={device}, seed={config.seed}, ir_window_body_source={config.ir_window_body_source}"
        )
        logger.info(
            "Training objective: match frozen IR-CVAE posterior from depth image only. "
            f"loss = {config.distribution_kl_weight}*distribution_kl "
            f"+ {config.moment_loss_weight}*posterior moment loss "
            f"+ {config.decoder_value_loss_weight}*frozen decoder ground-truth "
            f"{config.decoder_value_loss_type} value loss. "
            f"best_val_metric={config.best_val_metric}, learning_rate={config.learning_rate}, "
            f"lr_plateau_patience={config.lr_plateau_patience}, early_stop_patience={config.early_stop_patience}"
        )

        wandb = init_wandb(config, run_paths)
        teacher_oracle_metrics: dict[str, float | int] = {}
        if (
            ir_decoder_model is not None
            and ir_decoder_text_feature is not None
            and ir_decoder_feature_mean is not None
            and ir_decoder_feature_std is not None
        ):
            teacher_oracle_metrics.update(
                evaluate_ir_decoder_oracle(
                    ir_target_mu_val,
                    ir_target_logvar_val,
                    val_ir,
                    ir_decoder_model=ir_decoder_model,
                    ir_decoder_text_feature=ir_decoder_text_feature,
                    ir_feature_mean=ir_decoder_feature_mean,
                    ir_feature_std=ir_decoder_feature_std,
                    batch_size=config.batch_size,
                    decoder_eval_samples=config.decoder_eval_samples,
                    device=device,
                    prefix="teacher_val_sample",
                    use_mean_z=False,
                )
            )
            teacher_oracle_metrics.update(
                evaluate_ir_decoder_oracle(
                    ir_target_mu_val,
                    ir_target_logvar_val,
                    val_ir,
                    ir_decoder_model=ir_decoder_model,
                    ir_decoder_text_feature=ir_decoder_text_feature,
                    ir_feature_mean=ir_decoder_feature_mean,
                    ir_feature_std=ir_decoder_feature_std,
                    batch_size=config.batch_size,
                    decoder_eval_samples=config.decoder_eval_samples,
                    device=device,
                    prefix="teacher_val_mean",
                    use_mean_z=True,
                )
            )
            teacher_oracle_metrics.update(
                evaluate_ir_decoder_oracle(
                    ir_target_mu_test,
                    ir_target_logvar_test,
                    test_ir,
                    ir_decoder_model=ir_decoder_model,
                    ir_decoder_text_feature=ir_decoder_text_feature,
                    ir_feature_mean=ir_decoder_feature_mean,
                    ir_feature_std=ir_decoder_feature_std,
                    batch_size=config.batch_size,
                    decoder_eval_samples=config.decoder_eval_samples,
                    device=device,
                    prefix="teacher_test_sample",
                    use_mean_z=False,
                )
            )
            teacher_oracle_metrics.update(
                evaluate_ir_decoder_oracle(
                    ir_target_mu_test,
                    ir_target_logvar_test,
                    test_ir,
                    ir_decoder_model=ir_decoder_model,
                    ir_decoder_text_feature=ir_decoder_text_feature,
                    ir_feature_mean=ir_decoder_feature_mean,
                    ir_feature_std=ir_decoder_feature_std,
                    batch_size=config.batch_size,
                    decoder_eval_samples=config.decoder_eval_samples,
                    device=device,
                    prefix="teacher_test_mean",
                    use_mean_z=True,
                )
            )
            logger.info(
                "Frozen IR decoder oracle value error: "
                f"val_sample_mae={teacher_oracle_metrics['teacher_val_sample_value_mae']:.6f}, "
                f"val_mean_mae={teacher_oracle_metrics['teacher_val_mean_value_mae']:.6f}, "
                f"test_sample_mae={teacher_oracle_metrics['teacher_test_sample_value_mae']:.6f}, "
                f"test_mean_mae={teacher_oracle_metrics['teacher_test_mean_value_mae']:.6f}"
            )
            if wandb is not None and wandb.run is not None:
                wandb.log({f"oracle/{key}": value for key, value in teacher_oracle_metrics.items()}, step=0)
        best_val_score = float("inf")
        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state: dict[str, torch.Tensor] | None = None
        last_model_state: dict[str, torch.Tensor] | None = None
        epochs_since_best = 0

        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_distribution_kl = 0.0
            epoch_moment_loss = 0.0
            epoch_decoder_value_loss = 0.0
            seen_samples = 0

            for batch_indices in iterate_batch_indices(
                int(train_depth.shape[0]),
                config.batch_size,
                shuffle=True,
                seed=config.seed + epoch,
            ):
                batch_depth = train_depth.index_select(0, batch_indices).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_target_mu = ir_target_mu_train_normalized.index_select(0, batch_indices).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_target_logvar = ir_target_logvar_train_normalized.index_select(0, batch_indices).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                if config.decoder_value_loss_weight > 0:
                    batch_target_ir = train_ir.index_select(0, batch_indices).to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    batch_target_ir = batch_target_ir.reshape(batch_target_ir.shape[0], -1)
                else:
                    batch_target_ir = None
                batch_size_current = int(batch_depth.shape[0])
                batch_depth = (batch_depth - feature_mean_device) / feature_std_device
                batch_text = base_text_feature.expand(batch_size_current, -1)

                predicted_mu, predicted_logvar = model(batch_depth, batch_text)
                loss_terms = compute_loss_terms(
                    batch_target_mu,
                    batch_target_logvar,
                    predicted_mu,
                    predicted_logvar,
                    distribution_kl_weight=config.distribution_kl_weight,
                    moment_loss_weight=config.moment_loss_weight,
                )
                batch_ground_truth_value_loss = torch.zeros((), device=device)
                if config.decoder_value_loss_weight > 0:
                    if (
                        ir_decoder_model is None
                        or ir_decoder_text_feature is None
                        or ir_decoder_feature_mean_device is None
                        or ir_decoder_feature_std_device is None
                        or batch_target_ir is None
                    ):
                        raise RuntimeError("Frozen IR decoder is required for decoder value loss.")
                    predicted_mu_value = denormalize_target_latents(
                        predicted_mu,
                        target_latent_mean_device,
                        target_latent_std_device,
                    )
                    predicted_logvar_value = denormalize_target_logvar(predicted_logvar, target_latent_std_device)
                    predicted_std_value = torch.exp(0.5 * torch.clamp(predicted_logvar_value, min=-20.0, max=20.0))
                    decoder_text = ir_decoder_text_feature.expand(batch_size_current, -1)
                    for _ in range(config.decoder_train_samples):
                        sampled_z = predicted_mu_value + predicted_std_value * torch.randn_like(predicted_std_value)
                        decoded_ir_normalized = ir_decoder_model.decode(sampled_z, decoder_text)
                        decoded_ir = (
                            decoded_ir_normalized * ir_decoder_feature_std_device
                            + ir_decoder_feature_mean_device
                        )
                        batch_ground_truth_value_loss = batch_ground_truth_value_loss + ground_truth_value_loss(
                            decoded_ir,
                            batch_target_ir,
                            loss_type=config.decoder_value_loss_type,
                        )
                    batch_ground_truth_value_loss = batch_ground_truth_value_loss / config.decoder_train_samples

                loss = loss_terms["loss"] + config.decoder_value_loss_weight * batch_ground_truth_value_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss became non-finite.")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item() * batch_size_current
                epoch_distribution_kl += loss_terms["distribution_kl"].item() * batch_size_current
                epoch_moment_loss += loss_terms["moment_loss"].item() * batch_size_current
                epoch_decoder_value_loss += batch_ground_truth_value_loss.item() * batch_size_current
                seen_samples += batch_size_current

            train_metrics = {
                "train_num_samples": int(seen_samples),
                "train_loss": epoch_loss / seen_samples,
                "train_distribution_kl": epoch_distribution_kl / seen_samples,
                "train_moment_loss": epoch_moment_loss / seen_samples,
                "train_decoder_value_loss": (
                    epoch_decoder_value_loss / seen_samples
                    if config.decoder_value_loss_weight > 0
                    else float("nan")
                ),
            }
            val_metrics = evaluate_model(
                model,
                val_depth,
                ir_target_mu_val,
                ir_target_logvar_val,
                base_text_feature=base_text_feature,
                batch_size=config.batch_size,
                feature_mean=feature_mean,
                feature_std=feature_std,
                target_latent_mean=target_latent_mean,
                target_latent_std=target_latent_std,
                target_ir_windows=val_ir,
                ir_decoder_model=ir_decoder_model,
                ir_decoder_text_feature=ir_decoder_text_feature,
                ir_feature_mean=ir_decoder_feature_mean,
                ir_feature_std=ir_decoder_feature_std,
                decoder_eval_samples=config.decoder_eval_samples,
                distribution_kl_weight=config.distribution_kl_weight,
                moment_loss_weight=config.moment_loss_weight,
                device=device,
                prefix="val",
            )
            current_val_loss = float(val_metrics["val_loss"])
            current_val_selection_score = float(val_metrics[config.best_val_metric])
            if not math.isfinite(current_val_selection_score):
                raise RuntimeError(f"Validation selection metric {config.best_val_metric} is not finite.")

            improved = current_val_selection_score < best_val_score - config.val_improvement_min_delta
            if improved:
                best_val_score = current_val_selection_score
                best_val_loss = current_val_loss
                best_epoch = epoch
                epochs_since_best = 0
                best_model_state = clone_state_dict_to_cpu(model)
                save_encoder_checkpoint(
                    run_paths.best_checkpoint_path,
                    model=model,
                    config=config,
                    input_shape=telemetry_metadata.depth_input_shape,
                    num_samples=train_depth.shape[0],
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    target_latent_mean=target_latent_mean,
                    target_latent_std=target_latent_std,
                    text_feature_dim=text_feature_dim,
                    telemetry_metadata=telemetry_metadata,
                    checkpoint_type="best",
                    epoch=epoch,
                    val_loss=current_val_loss,
                    val_selection_metric=config.best_val_metric,
                    val_selection_score=current_val_selection_score,
                    ir_alignment_metadata=ir_alignment_metadata,
                )
            else:
                epochs_since_best += 1

            if (
                not improved
                and config.lr_plateau_patience > 0
                and epochs_since_best > 0
                and epochs_since_best % config.lr_plateau_patience == 0
            ):
                old_lr = float(optimizer.param_groups[0]["lr"])
                new_lr = max(old_lr * config.lr_plateau_factor, config.min_learning_rate)
                if new_lr < old_lr:
                    for group in optimizer.param_groups:
                        group["lr"] = new_lr
                    logger.info(
                        f"Reduced learning rate after {epochs_since_best} epochs without validation improvement: "
                        f"{old_lr:.6g} -> {new_lr:.6g}"
                    )
            current_learning_rate = float(optimizer.param_groups[0]["lr"])

            epoch_metrics = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "train/loss": train_metrics["train_loss"],
                "train/distribution_kl": train_metrics["train_distribution_kl"],
                "train/moment_loss": train_metrics["train_moment_loss"],
                "train/decoder_value_loss": train_metrics["train_decoder_value_loss"],
                "val/loss": val_metrics["val_loss"],
                "val/distribution_kl": val_metrics["val_distribution_kl"],
                "val/moment_loss": val_metrics["val_moment_loss"],
                "val/mu_mae": val_metrics["val_mu_mae"],
                "val/mu_rmse": val_metrics["val_mu_rmse"],
                "val/std_rmse": val_metrics["val_std_rmse"],
                "val/value_mae": val_metrics["val_value_mae"],
                "val/value_rmse": val_metrics["val_value_rmse"],
                "val/value_max_abs": val_metrics["val_value_max_abs"],
                "config/decoder_value_eval": config.decoder_value_eval,
                "config/decoder_value_loss_weight": config.decoder_value_loss_weight,
                "config/decoder_train_samples": config.decoder_train_samples,
                "config/decoder_eval_samples": config.decoder_eval_samples,
                "train/learning_rate": current_learning_rate,
                "val/selection_score": current_val_selection_score,
                "best_val_loss": best_val_loss,
                "best_val_score": best_val_score,
                "best_epoch": best_epoch,
                "epochs_since_best": epochs_since_best,
            }
            metrics_history.append(epoch_metrics)

            if wandb is not None and wandb.run is not None:
                wandb.log(epoch_metrics, step=epoch)

            if epoch % config.log_interval == 0 or epoch == 1 or epoch == config.epochs:
                logger.info(
                    f"epoch={epoch:04d} "
                    f"train_loss={train_metrics['train_loss']:.6f} "
                    f"train_distribution_kl={train_metrics['train_distribution_kl']:.6f} "
                    f"train_moment={train_metrics['train_moment_loss']:.6f} "
                    f"train_value_loss={train_metrics['train_decoder_value_loss']:.6f} "
                    f"val_loss={val_metrics['val_loss']:.6f} "
                    f"val_distribution_kl={val_metrics['val_distribution_kl']:.6f} "
                    f"val_moment={val_metrics['val_moment_loss']:.6f} "
                    f"val_value_mae={val_metrics['val_value_mae']:.6f} "
                    f"val_mu_rmse={val_metrics['val_mu_rmse']:.6f} "
                    f"val_std_rmse={val_metrics['val_std_rmse']:.6f} "
                    f"val_value_rmse={val_metrics['val_value_rmse']:.6f} "
                    f"val_select={current_val_selection_score:.6f} "
                    f"best_val_score={best_val_score:.6f} "
                    f"lr={current_learning_rate:.6g} "
                    f"epochs_since_best={epochs_since_best} "
                    f"best_val_metric={config.best_val_metric}"
                )

            if config.early_stop_patience > 0 and epochs_since_best >= config.early_stop_patience:
                logger.info(
                    f"Early stopping at epoch={epoch} because {config.best_val_metric} did not improve for "
                    f"{epochs_since_best} epochs. best_epoch={best_epoch}, best_val_score={best_val_score:.6f}"
                )
                break

        last_model_state = clone_state_dict_to_cpu(model)
        last_epoch = int(metrics_history[-1]["epoch"])
        final_val_loss = float(metrics_history[-1]["val_loss"])
        final_val_selection_score = float(metrics_history[-1]["val/selection_score"])
        save_encoder_checkpoint(
            run_paths.last_checkpoint_path,
            model=model,
            config=config,
            input_shape=telemetry_metadata.depth_input_shape,
            num_samples=train_depth.shape[0],
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_latent_mean=target_latent_mean,
            target_latent_std=target_latent_std,
            text_feature_dim=text_feature_dim,
            telemetry_metadata=telemetry_metadata,
            checkpoint_type="last",
            epoch=last_epoch,
            val_loss=final_val_loss,
            val_selection_metric=config.best_val_metric,
            val_selection_score=final_val_selection_score,
            ir_alignment_metadata=ir_alignment_metadata,
        )

        if best_model_state is None:
            best_model_state = clone_state_dict_to_cpu(model)
            best_epoch = config.epochs
            best_val_score = final_val_selection_score
            best_val_loss = final_val_loss

        model.load_state_dict(best_model_state)
        best_test_metrics = evaluate_model(
            model,
            test_depth,
            ir_target_mu_test,
            ir_target_logvar_test,
            base_text_feature=base_text_feature,
            batch_size=config.batch_size,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_latent_mean=target_latent_mean,
            target_latent_std=target_latent_std,
            target_ir_windows=test_ir,
            ir_decoder_model=ir_decoder_model,
            ir_decoder_text_feature=ir_decoder_text_feature,
            ir_feature_mean=ir_decoder_feature_mean,
            ir_feature_std=ir_decoder_feature_std,
            decoder_eval_samples=config.decoder_eval_samples,
            distribution_kl_weight=config.distribution_kl_weight,
            moment_loss_weight=config.moment_loss_weight,
            device=device,
            prefix="test",
        )
        model.load_state_dict(last_model_state)
        last_test_metrics = evaluate_model(
            model,
            test_depth,
            ir_target_mu_test,
            ir_target_logvar_test,
            base_text_feature=base_text_feature,
            batch_size=config.batch_size,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_latent_mean=target_latent_mean,
            target_latent_std=target_latent_std,
            target_ir_windows=test_ir,
            ir_decoder_model=ir_decoder_model,
            ir_decoder_text_feature=ir_decoder_text_feature,
            ir_feature_mean=ir_decoder_feature_mean,
            ir_feature_std=ir_decoder_feature_std,
            decoder_eval_samples=config.decoder_eval_samples,
            distribution_kl_weight=config.distribution_kl_weight,
            moment_loss_weight=config.moment_loss_weight,
            device=device,
            prefix="test",
        )
        test_differences = compute_metric_differences(last_test_metrics, best_test_metrics)
        best_test_metrics_named = rename_metric_prefix(best_test_metrics, "test_", "best_test_")
        last_test_metrics_named = rename_metric_prefix(last_test_metrics, "test_", "last_test_")

        summary = {
            "seed": config.seed,
            "depth_input_shape": list(telemetry_metadata.depth_input_shape),
            "num_episodes": len(train_episode_ids) + len(val_episode_ids) + len(test_episode_ids),
            "num_train_episodes": len(train_episode_ids),
            "num_val_episodes": len(val_episode_ids),
            "num_test_episodes": len(test_episode_ids),
            "num_train_windows": int(train_depth.shape[0]),
            "num_val_windows": int(val_depth.shape[0]),
            "num_test_windows": int(test_depth.shape[0]),
            "telemetry": asdict(telemetry_metadata),
            "train_episode_ids": train_episode_ids,
            "val_episode_ids": val_episode_ids,
            "test_episode_ids": test_episode_ids,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_metric": config.best_val_metric,
            "best_val_score": best_val_score,
            "decoder_value_eval": bool(config.decoder_value_eval and ir_decoder_model is not None),
            "decoder_value_loss_weight": config.decoder_value_loss_weight,
            "decoder_train_samples": config.decoder_train_samples,
            "decoder_value_loss_type": config.decoder_value_loss_type,
            "decoder_eval_samples": config.decoder_eval_samples,
            "teacher_oracle_metrics": teacher_oracle_metrics,
            "best_checkpoint": str(run_paths.best_checkpoint_path),
            "last_checkpoint": str(run_paths.last_checkpoint_path),
            "ir_alignment": ir_alignment_metadata,
            "best_test_metrics": best_test_metrics_named,
            "last_test_metrics": last_test_metrics_named,
            "test_metric_differences": test_differences,
            "history": metrics_history,
        }
        with run_paths.metrics_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        logger.info(f"Saved best fast latent checkpoint to: {run_paths.best_checkpoint_path}")
        logger.info(f"Saved last fast latent checkpoint to: {run_paths.last_checkpoint_path}")
        logger.info(f"Saved split and metric summary to: {run_paths.metrics_path}")
        logger.info(
            f"Test comparison: best_distribution_kl={best_test_metrics_named['best_test_distribution_kl']:.6f}, "
            f"last_distribution_kl={last_test_metrics_named['last_test_distribution_kl']:.6f}, "
            f"best_value_rmse={best_test_metrics_named['best_test_value_rmse']:.6f}, "
            f"last_value_rmse={last_test_metrics_named['last_test_value_rmse']:.6f}"
        )

        if wandb is not None and wandb.run is not None:
            final_log = {
                "epoch": last_epoch,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_score": best_val_score,
                "best_test/loss": best_test_metrics_named["best_test_loss"],
                "best_test/distribution_kl": best_test_metrics_named["best_test_distribution_kl"],
                "best_test/moment_loss": best_test_metrics_named["best_test_moment_loss"],
                "best_test/mu_rmse": best_test_metrics_named["best_test_mu_rmse"],
                "best_test/std_rmse": best_test_metrics_named["best_test_std_rmse"],
                "best_test/value_mae": best_test_metrics_named["best_test_value_mae"],
                "best_test/value_rmse": best_test_metrics_named["best_test_value_rmse"],
                "best_test/value_max_abs": best_test_metrics_named["best_test_value_max_abs"],
                "last_test/loss": last_test_metrics_named["last_test_loss"],
                "last_test/distribution_kl": last_test_metrics_named["last_test_distribution_kl"],
                "last_test/moment_loss": last_test_metrics_named["last_test_moment_loss"],
                "last_test/mu_rmse": last_test_metrics_named["last_test_mu_rmse"],
                "last_test/std_rmse": last_test_metrics_named["last_test_std_rmse"],
                "last_test/value_mae": last_test_metrics_named["last_test_value_mae"],
                "last_test/value_rmse": last_test_metrics_named["last_test_value_rmse"],
                "last_test/value_max_abs": last_test_metrics_named["last_test_value_max_abs"],
                **best_test_metrics_named,
                **last_test_metrics_named,
                **test_differences,
            }
            wandb.log(final_log, step=last_epoch)
            wandb.save(str(run_paths.config_path), base_path=str(run_paths.run_dir))
            wandb.save(str(run_paths.best_checkpoint_path), base_path=str(run_paths.run_dir))
            wandb.save(str(run_paths.last_checkpoint_path), base_path=str(run_paths.run_dir))
            wandb.save(str(run_paths.metrics_path), base_path=str(run_paths.run_dir))

        return run_paths.best_checkpoint_path
    finally:
        if wandb is not None and wandb.run is not None:
            wandb.finish()


def load_encoder(checkpoint_path: str, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    config_dict = payload["config"]
    encoder = create_depth_encoder(
        encoder_type=config_dict.get("encoder_type", payload.get("encoder_type", "fast_cnn")),
        input_shape=tuple(payload["input_shape"]),
        text_feature_dim=payload["text_feature_dim"],
        condition_dim=config_dict["condition_dim"],
        latent_dim=config_dict["latent_dim"],
        hidden_dim=config_dict["hidden_dim"],
        conv_channels=tuple(config_dict["conv_channels"]),
    )
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()
    return encoder, payload


@torch.no_grad()
def encode_depth_window_to_latent_distribution(
    checkpoint_path: str,
    depth_window: np.ndarray | list,
    condition_text: str | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder, payload = load_encoder(checkpoint_path, device=device)
    depth_window_array = np.asarray(depth_window, dtype=np.float32)
    expected_shape = tuple(payload["input_shape"])
    if tuple(depth_window_array.shape) != expected_shape:
        raise ValueError(f"Expected depth_window shape {expected_shape}, got {depth_window_array.shape}")

    x = torch.tensor(depth_window_array, dtype=torch.float32, device=device).unsqueeze(0)
    feature_mean = payload["feature_mean"].to(device=device, dtype=torch.float32).unsqueeze(0)
    feature_std = payload["feature_std"].to(device=device, dtype=torch.float32).unsqueeze(0)
    x = (x - feature_mean) / feature_std
    clip_cfg = payload["clip"]
    text_extractor = CLIPTextFeatureExtractor(
        model_id=clip_cfg["model_id"],
        device=device,
        cache_dir=clip_cfg["cache_dir"],
        local_files_only=clip_cfg["local_files_only"],
        quiet_load=True,
    )
    text_string = condition_text or payload["condition_text"]
    text_features = text_extractor.encode([text_string]).to(device=device, dtype=torch.float32)
    mu_normalized, logvar_normalized = encoder(x, text_features)
    target_latent_mean = payload["target_latent_mean"].to(device=device, dtype=torch.float32)
    target_latent_std = payload["target_latent_std"].to(device=device, dtype=torch.float32)
    mu = denormalize_target_latents(mu_normalized, target_latent_mean, target_latent_std)
    logvar = denormalize_target_logvar(logvar_normalized, target_latent_std)
    return mu.squeeze(0).cpu(), logvar.squeeze(0).cpu()


@torch.no_grad()
def encode_depth_window_to_latent(
    checkpoint_path: str,
    depth_window: np.ndarray | list,
    condition_text: str | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    mu, logvar = encode_depth_window_to_latent_distribution(
        checkpoint_path,
        depth_window,
        condition_text=condition_text,
        device=device,
    )
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a fast depth-window encoder to match frozen IR-CVAE posterior latents."
    )
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--condition-text", type=str, default=TrainConfig.condition_text)
    parser.add_argument(
        "--ir-window-body-source",
        type=str,
        default=TrainConfig.ir_window_body_source,
        choices=U_WINDOW_BODY_SOURCE_CHOICES,
    )
    parser.add_argument("--output-root", type=str, default=TrainConfig.output_root)
    parser.add_argument("--run-name", type=str, default=TrainConfig.run_name)
    parser.add_argument("--ir-cvae-checkpoint", type=str, default=TrainConfig.ir_cvae_checkpoint)
    parser.add_argument("--latent-dim", type=int, default=TrainConfig.latent_dim)
    parser.add_argument("--encoder-type", type=str, default=TrainConfig.encoder_type, choices=ENCODER_TYPE_CHOICES)
    parser.add_argument("--condition-dim", type=int, default=TrainConfig.condition_dim)
    parser.add_argument("--conv-channel-1", type=int, default=TrainConfig.conv_channels[0])
    parser.add_argument("--conv-channel-2", type=int, default=TrainConfig.conv_channels[1])
    parser.add_argument("--conv-channel-3", type=int, default=TrainConfig.conv_channels[2])
    parser.add_argument("--hidden-dim", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--distribution-kl-weight", type=float, default=TrainConfig.distribution_kl_weight)
    parser.add_argument("--moment-loss-weight", type=float, default=TrainConfig.moment_loss_weight)
    parser.add_argument("--decoder-value-loss-weight", type=float, default=TrainConfig.decoder_value_loss_weight)
    parser.add_argument("--decoder-train-samples", type=int, default=TrainConfig.decoder_train_samples)
    parser.add_argument(
        "--decoder-value-loss-type",
        type=str,
        default=TrainConfig.decoder_value_loss_type,
        choices=DECODER_VALUE_LOSS_CHOICES,
    )
    parser.add_argument("--decoder-value-eval", type=str_to_bool, default=TrainConfig.decoder_value_eval)
    parser.add_argument("--decoder-eval-samples", type=int, default=TrainConfig.decoder_eval_samples)
    parser.add_argument(
        "--best-val-metric",
        type=str,
        default=TrainConfig.best_val_metric,
        choices=BEST_VAL_METRIC_CHOICES,
        help="Validation metric minimized when saving best.pt.",
    )
    parser.add_argument("--min-feature-std", type=float, default=TrainConfig.min_feature_std)
    parser.add_argument("--min-target-latent-std", type=float, default=TrainConfig.min_target_latent_std)
    parser.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--val-improvement-min-delta", type=float, default=TrainConfig.val_improvement_min_delta)
    parser.add_argument("--lr-plateau-patience", type=int, default=TrainConfig.lr_plateau_patience)
    parser.add_argument("--lr-plateau-factor", type=float, default=TrainConfig.lr_plateau_factor)
    parser.add_argument("--min-learning-rate", type=float, default=TrainConfig.min_learning_rate)
    parser.add_argument("--early-stop-patience", type=int, default=TrainConfig.early_stop_patience)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--val-ratio", type=float, default=TrainConfig.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=TrainConfig.test_ratio)
    parser.add_argument("--wandb-enabled", type=str_to_bool, default=TrainConfig.wandb_enabled)
    parser.add_argument("--wandb-project", type=str, default=TrainConfig.wandb_project)
    parser.add_argument("--wandb-entity", type=str, default=TrainConfig.wandb_entity)
    parser.add_argument("--wandb-group", type=str, default=TrainConfig.wandb_group)
    parser.add_argument("--wandb-mode", type=str, default=TrainConfig.wandb_mode, choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", nargs="*", default=list(TrainConfig.wandb_tags))
    parser.add_argument("--clip-model-id", type=str, default=TrainConfig.clip_model_id)
    parser.add_argument("--clip-cache-dir", type=str, default=TrainConfig.clip_cache_dir)
    parser.add_argument("--clip-local-files-only", type=str_to_bool, default=TrainConfig.clip_local_files_only)
    args = parser.parse_args()

    wandb_enabled = bool(args.wandb_enabled) and args.wandb_mode != "disabled"
    wandb_mode = "offline" if args.wandb_mode == "disabled" else args.wandb_mode
    return TrainConfig(
        data_dir=args.data_dir,
        condition_text=args.condition_text,
        ir_window_body_source=normalize_ir_window_body_source(args.ir_window_body_source),
        output_root=args.output_root,
        run_name=args.run_name,
        ir_cvae_checkpoint=args.ir_cvae_checkpoint,
        latent_dim=args.latent_dim,
        encoder_type=args.encoder_type,
        condition_dim=args.condition_dim,
        conv_channels=(args.conv_channel_1, args.conv_channel_2, args.conv_channel_3),
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        distribution_kl_weight=args.distribution_kl_weight,
        moment_loss_weight=args.moment_loss_weight,
        decoder_value_loss_weight=args.decoder_value_loss_weight,
        decoder_train_samples=args.decoder_train_samples,
        decoder_value_loss_type=args.decoder_value_loss_type,
        best_val_metric=args.best_val_metric,
        decoder_value_eval=bool(args.decoder_value_eval),
        decoder_eval_samples=args.decoder_eval_samples,
        min_feature_std=args.min_feature_std,
        min_target_latent_std=args.min_target_latent_std,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        val_improvement_min_delta=args.val_improvement_min_delta,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        min_learning_rate=args.min_learning_rate,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        device=args.device,
        log_interval=args.log_interval,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        wandb_enabled=wandb_enabled,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_mode=wandb_mode,
        wandb_tags=tuple(args.wandb_tags),
        clip_model_id=args.clip_model_id,
        clip_cache_dir=args.clip_cache_dir,
        clip_local_files_only=bool(args.clip_local_files_only),
    )


def main() -> None:
    config = parse_args()
    best_checkpoint = train_encoder(config)
    logger.info(f"Finished fast latent training. Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
