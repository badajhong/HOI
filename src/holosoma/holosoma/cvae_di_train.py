# python src/holosoma/holosoma/cvae_di_train.py \
#   --data-dir /home/rllab/haechan/holosoma/logs/WholeBodyTracking/cvae_suitcase/telemetry \
#   --ir-window-body-source all \
#   --ir-cvae-checkpoint /home/rllab/haechan/holosoma/logs/CVAE/0416_ir_all_64/best.pt \
#   --best-val-metric val_value_rmse \
#   --learning-rate 3e-4 \
#   --weight-decay 1e-2 \
#   --decoder-value-loss-weight 100 \
#   --decoder-train-samples 1 \
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

from holosoma.cvae_ir_train import (
    CLIPTextFeatureExtractor,
    U_WINDOW_BODY_SOURCE_CHOICES,
    _select_ir_window_body_source,
    _selected_component_names_for_body_source,
    clone_state_dict_to_cpu,
    compute_metric_differences,
    configure_cuda_backend,
    create_run_paths,
    init_wandb,
    load_cvae as load_ir_cvae,
    load_encoder as load_ir_encoder,
    normalize_ir_window_body_source,
    rename_metric_prefix,
    resolve_device,
    sample_latent_z,
    save_config,
    set_seed,
    split_episode_indices,
    str_to_bool,
)
from holosoma.utils.safe_torch_import import torch


DEFAULT_DATA_DIR = "/home/rllab/haechan/holosoma/logs/WholeBodyTracking/cvae_suitcase/telemetry"
DEFAULT_OUTPUT_ROOT = "/home/rllab/haechan/holosoma/logs/CVAE/"
DEFAULT_CONDITION_TEXT = "Push the suitcase, and set it back down."
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_IR_CVAE_CHECKPOINT = "/home/rllab/haechan/holosoma/logs/CVAE/0416_ir_all_64/best.pt"
BEST_VAL_METRIC_CHOICES = (
    "val_loss",
    "val_loss_distribution_kl",
    "val_value_mae",
    "val_value_rmse",
)


@dataclass
class TrainConfig:
    data_dir: str = DEFAULT_DATA_DIR
    condition_text: str = DEFAULT_CONDITION_TEXT
    ir_window_body_source: str = "all"
    output_root: str = DEFAULT_OUTPUT_ROOT
    run_name: str = "cvae-di"
    ir_cvae_checkpoint: str = DEFAULT_IR_CVAE_CHECKPOINT
    latent_dim: int = 64
    hidden_dims: tuple[int, int] = (128, 256)
    condition_dim: int = 16
    batch_size: int = 8192
    epochs: int = 10000
    learning_rate: float = 3e-4
    best_val_metric: str = "val_value_rmse"
    decoder_value_eval: bool = True
    decoder_value_loss_weight: float = 100.0
    decoder_train_samples: int = 1
    decoder_eval_samples: int = 8
    min_feature_std: float = 1e-4
    min_target_latent_std: float = 1e-3
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-2
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 100
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    wandb_enabled: bool = True
    wandb_project: str = "CVAE"
    wandb_entity: str | None = None
    wandb_group: str = "cvae_di"
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ("cvae", "di", "depth_window", "cnn", "temporal")
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID
    clip_cache_dir: str | None = None
    clip_local_files_only: bool = True
    clip_quiet_load: bool = True


@dataclass
class EpisodePairedWindows:
    episode_id: str
    ir_windows: np.ndarray
    depth_windows: np.ndarray


@dataclass
class TelemetryMetadata:
    depth_input_shape: tuple[int, int, int]
    ir_window_shape: tuple[int, int]
    ir_window_body_source: str = "all"
    depth_resolution: tuple[int, int] | None = None
    ir_t_mode: str | None = None
    ir_t_components: tuple[str, ...] = ()
    ir_t_dim: int | None = None
    source_episode_id: str | None = None


class TextConditionProjector(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.Tanh(),
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.net(text_features)


class SharedFrameCNN(nn.Module):
    def __init__(self, frame_feature_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 5)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 5, frame_feature_dim),
            nn.LayerNorm(frame_feature_dim),
            nn.Tanh(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(frames))


class DepthWindowLatentEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        text_feature_dim: int,
        condition_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        *,
        logvar_clamp_min: float = -10.0,
        logvar_clamp_max: float = 10.0,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"Depth input shape must have length 3, got {input_shape}")
        if len(hidden_dims) != 2:
            raise ValueError(f"Expected hidden_dims=(frame_feature_dim, temporal_hidden_dim), got {hidden_dims}")

        self.window_size = int(input_shape[0])
        self.height = int(input_shape[1])
        self.width = int(input_shape[2])
        self.frame_feature_dim = int(hidden_dims[0])
        self.temporal_hidden_dim = int(hidden_dims[1])
        self.text_projector = TextConditionProjector(text_feature_dim, condition_dim)
        self.frame_encoder = SharedFrameCNN(self.frame_feature_dim)
        self.temporal_encoder = nn.GRU(
            input_size=self.frame_feature_dim,
            hidden_size=self.temporal_hidden_dim,
            batch_first=True,
        )
        self.latent_head = nn.Sequential(
            nn.Linear(self.temporal_hidden_dim + condition_dim, self.temporal_hidden_dim),
            nn.LayerNorm(self.temporal_hidden_dim),
            nn.Tanh(),
        )
        self.mu = nn.Linear(self.temporal_hidden_dim, latent_dim)
        self.logvar = nn.Linear(self.temporal_hidden_dim, latent_dim)
        self.logvar_clamp_min = float(logvar_clamp_min)
        self.logvar_clamp_max = float(logvar_clamp_max)

    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected depth batch shape [B, T, H, W], got {tuple(x.shape)}")
        batch_size, window_size, height, width = x.shape
        if (window_size, height, width) != (self.window_size, self.height, self.width):
            raise ValueError(
                f"Expected depth batch spatial shape {(self.window_size, self.height, self.width)}, "
                f"got {(window_size, height, width)}"
            )

        frame_tensor = x.reshape(batch_size * window_size, 1, height, width)
        frame_features = self.frame_encoder(frame_tensor)
        frame_features = frame_features.reshape(batch_size, window_size, self.frame_feature_dim)

        _, hidden = self.temporal_encoder(frame_features)
        temporal_feature = hidden[-1]
        condition = self.text_projector(text_features)
        latent_hidden = self.latent_head(torch.cat([temporal_feature, condition], dim=-1))
        mu = self.mu(latent_hidden)
        logvar = self.logvar(latent_hidden)
        logvar = torch.clamp(logvar, min=self.logvar_clamp_min, max=self.logvar_clamp_max)
        return mu, logvar


def iterate_batch_indices(num_samples: int, batch_size: int, *, shuffle: bool, seed: int):
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        indices = torch.randperm(num_samples, generator=generator)
    else:
        indices = torch.arange(num_samples, dtype=torch.long)

    for start in range(0, num_samples, batch_size):
        yield indices[start : start + batch_size]


def extract_episode_paired_windows(
    data_dir: Path,
    ir_window_body_source: str = "all",
) -> tuple[list[EpisodePairedWindows], TelemetryMetadata]:
    ir_window_body_source = normalize_ir_window_body_source(ir_window_body_source)
    json_paths = sorted(data_dir.glob("episode_env*_idx*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No episode JSON files found under: {data_dir}")

    logger.info(
        f"Scanning {len(json_paths)} telemetry JSON files under: {data_dir} "
        f"with ir_window_body_source='{ir_window_body_source}'"
    )

    episodes: list[EpisodePairedWindows] = []
    expected_ir_shape: tuple[int, int] | None = None
    expected_depth_shape: tuple[int, int, int] | None = None
    total_windows = 0
    metadata: TelemetryMetadata | None = None

    for file_index, json_path in enumerate(json_paths, start=1):
        if file_index == 1 or file_index % 10 == 0 or file_index == len(json_paths):
            logger.info(f"Reading paired ir_window/depth_window values from file {file_index}/{len(json_paths)}: {json_path.name}")

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        entries = payload.get("entries", [])
        if not isinstance(entries, list) or not entries:
            continue

        episode_ir_windows: list[np.ndarray] = []
        episode_depth_windows: list[np.ndarray] = []
        for entry_index, entry in enumerate(entries):
            if "ir_window" not in entry or "depth_window" not in entry:
                raise ValueError(
                    f"Expected both ir_window and depth_window in {json_path} entry {entry_index}, "
                    f"available keys={sorted(entry.keys())}."
                )

            ir_window = np.asarray(entry["ir_window"], dtype=np.float32)
            ir_window = _select_ir_window_body_source(ir_window, ir_window_body_source)
            depth_window = np.asarray(entry["depth_window"], dtype=np.float32)

            if ir_window.ndim != 2:
                raise ValueError(
                    f"Expected ir_window to have rank 2, got shape {ir_window.shape} "
                    f"in {json_path} entry {entry_index}."
                )
            if depth_window.ndim != 3:
                raise ValueError(
                    f"Expected depth_window to have rank 3, got shape {depth_window.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            current_ir_shape = (int(ir_window.shape[0]), int(ir_window.shape[1]))
            current_depth_shape = (int(depth_window.shape[0]), int(depth_window.shape[1]), int(depth_window.shape[2]))
            if expected_ir_shape is None:
                expected_ir_shape = current_ir_shape
            elif current_ir_shape != expected_ir_shape:
                raise ValueError(
                    f"Inconsistent ir_window shape. Expected {expected_ir_shape}, got {ir_window.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            if expected_depth_shape is None:
                expected_depth_shape = current_depth_shape
            elif current_depth_shape != expected_depth_shape:
                raise ValueError(
                    f"Inconsistent depth_window shape. Expected {expected_depth_shape}, got {depth_window.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            episode_ir_windows.append(ir_window)
            episode_depth_windows.append(depth_window)

        if not episode_ir_windows:
            continue

        episodes.append(
            EpisodePairedWindows(
                episode_id=json_path.stem,
                ir_windows=np.stack(episode_ir_windows, axis=0),
                depth_windows=np.stack(episode_depth_windows, axis=0),
            )
        )
        total_windows += len(episode_ir_windows)

        if metadata is None:
            assert expected_ir_shape is not None
            assert expected_depth_shape is not None
            ir_t_components_value = payload.get("ir_t_components")
            ir_t_components = ()
            if isinstance(ir_t_components_value, list):
                ir_t_components = tuple(str(value) for value in ir_t_components_value)

            ir_t_mode_value = payload.get("ir_t_mode")
            ir_t_mode = str(ir_t_mode_value) if isinstance(ir_t_mode_value, str) else None

            ir_t_dim_value = payload.get("ir_t_dim")
            ir_t_dim = int(ir_t_dim_value) if isinstance(ir_t_dim_value, int | float) else None
            if ir_t_components and len(ir_t_components) != expected_ir_shape[1]:
                selected_components = _selected_component_names_for_body_source(
                    ir_t_components,
                    input_feature_dim=expected_ir_shape[1],
                    body_source=ir_window_body_source,
                )
                if len(selected_components) == expected_ir_shape[1]:
                    logger.info(
                        f"Telemetry metadata has {len(ir_t_components)} original ir_t components; "
                        f"using {ir_window_body_source}-selected metadata with {len(selected_components)} components."
                    )
                    ir_t_components = selected_components
                    ir_t_dim = expected_ir_shape[1]
            if ir_window_body_source != "all" and ir_t_dim is not None and ir_t_dim != expected_ir_shape[1]:
                logger.info(
                    f"Telemetry metadata has original ir_t_dim={ir_t_dim}; "
                    f"using {ir_window_body_source}-selected ir_t_dim={expected_ir_shape[1]}."
                )
                ir_t_dim = expected_ir_shape[1]
            if ir_window_body_source != "all" and ir_t_mode is not None:
                suffix = f"_{ir_window_body_source}_only"
                if not ir_t_mode.endswith(suffix):
                    ir_t_mode = f"{ir_t_mode}{suffix}"
            if ir_t_dim is not None and ir_t_dim != expected_ir_shape[1]:
                raise ValueError(
                    f"Telemetry metadata ir_t_dim={ir_t_dim} does not match extracted ir_window feature dim="
                    f"{expected_ir_shape[1]} for {json_path}."
                )
            if ir_t_components and len(ir_t_components) != expected_ir_shape[1]:
                raise ValueError(
                    f"Telemetry metadata lists {len(ir_t_components)} ir_t components, but extracted ir_window "
                    f"feature dim is {expected_ir_shape[1]} for {json_path}."
                )

            depth_resolution_value = payload.get("depth_resolution")
            depth_resolution = None
            if isinstance(depth_resolution_value, list) and len(depth_resolution_value) == 2:
                depth_resolution = (int(depth_resolution_value[0]), int(depth_resolution_value[1]))
                # IR telemetry stores camera depth_resolution as metadata, but
                # the nested depth_window list itself is the training tensor.
                # Do not transpose/flip here; keep the saved [T, H, W] layout.

            metadata = TelemetryMetadata(
                depth_input_shape=expected_depth_shape,
                ir_window_shape=expected_ir_shape,
                ir_window_body_source=ir_window_body_source,
                depth_resolution=depth_resolution,
                ir_t_mode=ir_t_mode,
                ir_t_components=ir_t_components,
                ir_t_dim=ir_t_dim,
                source_episode_id=json_path.stem,
            )

    if not episodes or metadata is None:
        raise ValueError(f"No valid paired ir_window/depth_window entries found under: {data_dir}")

    logger.info(
        f"Loaded {total_windows} paired samples from {len(episodes)} episode files with "
        f"ir_window_shape={metadata.ir_window_shape}, depth_window_shape={metadata.depth_input_shape}, "
        f"ir_window_body_source='{metadata.ir_window_body_source}'"
    )
    if metadata.ir_t_mode is not None:
        logger.info(
            f"Telemetry metadata: ir_t_mode={metadata.ir_t_mode}, "
            f"ir_t_dim={metadata.ir_t_dim or metadata.ir_window_shape[1]}, "
            f"depth_resolution={metadata.depth_resolution}, "
            f"source={metadata.source_episode_id}"
        )

    return episodes, metadata


def flatten_episode_split(
    episodes: Sequence[EpisodePairedWindows],
    indices: Sequence[int],
    ir_window_shape: tuple[int, int],
    depth_window_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected = [episodes[index] for index in indices]
    episode_ids = [episode.episode_id for episode in selected]
    if not selected:
        return (
            np.empty((0, ir_window_shape[0], ir_window_shape[1]), dtype=np.float32),
            np.empty((0, depth_window_shape[0], depth_window_shape[1], depth_window_shape[2]), dtype=np.float32),
            episode_ids,
        )

    stacked_ir = np.concatenate([episode.ir_windows for episode in selected], axis=0).astype(np.float32, copy=False)
    stacked_depth = np.concatenate([episode.depth_windows for episode in selected], axis=0).astype(np.float32, copy=False)
    return stacked_ir, stacked_depth, episode_ids


@torch.no_grad()
def encode_ir_latent_targets(
    *,
    ir_checkpoint_path: str,
    ir_windows: torch.Tensor,
    condition_text: str,
    ir_window_body_source: str,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    ir_encoder, ir_payload = load_ir_encoder(ir_checkpoint_path, device=device)
    ir_encoder.eval()
    ir_encoder.requires_grad_(False)
    expected_shape = tuple(ir_payload["input_shape"])
    if tuple(ir_windows.shape[1:]) != expected_shape:
        raise ValueError(
            f"IR-CVAE checkpoint expects ir_window shape {expected_shape}, but telemetry provides "
            f"{tuple(ir_windows.shape[1:])} after --ir-window-body-source={ir_window_body_source}. "
            "Use an IR-CVAE checkpoint trained with the same u-window body source, or change "
            "--ir-window-body-source to match the checkpoint."
        )

    clip_cfg = ir_payload["clip"]
    target_text_extractor = CLIPTextFeatureExtractor(
        model_id=clip_cfg["model_id"],
        device=device,
        cache_dir=clip_cfg["cache_dir"],
        local_files_only=clip_cfg["local_files_only"],
        quiet_load=True,
    )
    text_string = condition_text or ir_payload["condition_text"]
    base_text_feature = target_text_extractor.encode([text_string]).to(device=device, dtype=torch.float32)
    if not torch.isfinite(base_text_feature).all():
        raise RuntimeError("CLIP text features for frozen IR-CVAE conditioning contain non-finite values.")

    feature_mean = ir_payload["feature_mean"].to(device=device, dtype=torch.float32).unsqueeze(0)
    feature_std = ir_payload["feature_std"].to(device=device, dtype=torch.float32).unsqueeze(0)
    latent_dim = int(ir_payload["config"]["latent_dim"])
    num_samples = int(ir_windows.shape[0])
    target_mu = torch.empty((num_samples, latent_dim), dtype=torch.float32)
    target_logvar = torch.empty((num_samples, latent_dim), dtype=torch.float32)

    logger.info(
        f"Encoding {num_samples} paired ir_window samples with frozen IR-CVAE from {ir_checkpoint_path} "
        "to build IR posterior distribution targets."
    )

    for batch_indices in iterate_batch_indices(num_samples, batch_size, shuffle=False, seed=0):
        batch_ir = ir_windows.index_select(0, batch_indices).to(device=device, dtype=torch.float32, non_blocking=True)
        batch_ir = batch_ir.reshape(batch_ir.shape[0], -1)
        batch_ir = (batch_ir - feature_mean) / feature_std
        if not torch.isfinite(batch_ir).all():
            raise RuntimeError("Non-finite values detected in normalized ir_window batch before frozen IR-CVAE encoding.")
        batch_text = base_text_feature.expand(batch_ir.shape[0], -1)
        mu, logvar = ir_encoder(batch_ir, batch_text)
        if not (torch.isfinite(mu).all() and torch.isfinite(logvar).all()):
            raise RuntimeError(
                "Frozen IR-CVAE produced non-finite posterior targets. "
                "Please verify the checkpoint, CLIP text conditioning, and ir_window normalization."
            )
        target_mu[batch_indices] = mu.detach().cpu()
        target_logvar[batch_indices] = logvar.detach().cpu()

    return target_mu, target_logvar, ir_payload


def compute_target_latent_normalization_stats(
    target_latents: torch.Tensor,
    *,
    min_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_latents.ndim != 2:
        raise ValueError(f"Expected latent targets with shape [N, D], got {tuple(target_latents.shape)}")

    target_latent_mean = target_latents.mean(dim=0)
    target_latent_std = target_latents.std(dim=0).clamp_min(min_std)
    return target_latent_mean, target_latent_std


def normalize_target_latents(
    target_latents: torch.Tensor,
    target_latent_mean: torch.Tensor,
    target_latent_std: torch.Tensor,
) -> torch.Tensor:
    return (target_latents - target_latent_mean.unsqueeze(0)) / target_latent_std.unsqueeze(0)


def denormalize_target_latents(
    normalized_latents: torch.Tensor,
    target_latent_mean: torch.Tensor,
    target_latent_std: torch.Tensor,
) -> torch.Tensor:
    return normalized_latents * target_latent_std.unsqueeze(0) + target_latent_mean.unsqueeze(0)


def normalize_target_logvar(target_logvar: torch.Tensor, target_latent_std: torch.Tensor) -> torch.Tensor:
    return target_logvar - 2.0 * torch.log(target_latent_std.clamp_min(1e-6)).unsqueeze(0)


def denormalize_target_logvar(normalized_logvar: torch.Tensor, target_latent_std: torch.Tensor) -> torch.Tensor:
    return normalized_logvar + 2.0 * torch.log(target_latent_std.clamp_min(1e-6)).unsqueeze(0)


def gaussian_kl_divergence(
    target_mu: torch.Tensor,
    target_logvar: torch.Tensor,
    predicted_mu: torch.Tensor,
    predicted_logvar: torch.Tensor,
) -> torch.Tensor:
    target_logvar = torch.clamp(target_logvar, min=-20.0, max=20.0)
    predicted_logvar = torch.clamp(predicted_logvar, min=-20.0, max=20.0)
    target_var = torch.exp(target_logvar)
    predicted_var = torch.exp(predicted_logvar).clamp_min(1e-8)
    kl_per_dim = 0.5 * (
        predicted_logvar
        - target_logvar
        + (target_var + (target_mu - predicted_mu).pow(2)) / predicted_var
        - 1.0
    )
    return kl_per_dim.mean()


def make_encoder_checkpoint_payload(
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
    model: DepthWindowLatentEncoder,
    checkpoint_type: str,
    epoch: int,
    val_loss_total: float,
    val_selection_metric: str,
    val_selection_score: float,
    ir_alignment_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_type": "depth_window_temporal_encoder",
        "checkpoint_type": checkpoint_type,
        "epoch": epoch,
        "val_loss_total": val_loss_total,
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
    model: DepthWindowLatentEncoder,
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
    val_loss_total: float,
    val_selection_metric: str,
    val_selection_score: float,
    ir_alignment_metadata: dict[str, Any],
) -> None:
    payload = make_encoder_checkpoint_payload(
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
        val_loss_total=val_loss_total,
        val_selection_metric=val_selection_metric,
        val_selection_score=val_selection_score,
        ir_alignment_metadata=ir_alignment_metadata,
    )
    torch.save(payload, checkpoint_path)


@torch.no_grad()
def evaluate_model(
    model: DepthWindowLatentEncoder,
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
    decoder_value_loss_weight: float = 0.0,
    decoder_eval_samples: int = 1,
    device: str,
    prefix: str,
) -> dict[str, float | int]:
    if depth_windows.shape[0] == 0:
        return {
            f"{prefix}_num_samples": 0,
            f"{prefix}_loss": float("nan"),
            f"{prefix}_loss_distribution_kl": float("nan"),
            f"{prefix}_value_mae": float("nan"),
            f"{prefix}_value_rmse": float("nan"),
            f"{prefix}_value_max_abs": float("nan"),
        }

    mean_device = feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
    std_device = feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
    target_latent_mean_device = target_latent_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
    target_latent_std_device = target_latent_std.to(device=device, dtype=torch.float32).unsqueeze(0)

    model.eval()
    total_distribution_kl = 0.0
    total_value_abs_error = 0.0
    total_value_squared_error = 0.0
    total_value_elements = 0
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

    for batch_indices in iterate_batch_indices(int(depth_windows.shape[0]), batch_size, shuffle=False, seed=0):
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

        batch_depth = (batch_depth - mean_device) / std_device
        batch_target_mu_normalized = normalize_target_latents(
            batch_target_mu,
            target_latent_mean_device.squeeze(0),
            target_latent_std_device.squeeze(0),
        )
        batch_target_logvar_normalized = normalize_target_logvar(
            batch_target_logvar,
            target_latent_std_device.squeeze(0),
        )
        batch_text = base_text_feature.expand(batch_size_current, -1)
        if not torch.isfinite(batch_depth).all():
            raise RuntimeError("Non-finite values detected in normalized depth_window batch during evaluation.")
        if not (torch.isfinite(batch_target_mu).all() and torch.isfinite(batch_target_logvar).all()):
            raise RuntimeError("Non-finite values detected in frozen IR latent targets during evaluation.")

        predicted_mu_normalized, predicted_logvar_normalized = model(batch_depth, batch_text)
        predicted_mu = denormalize_target_latents(
            predicted_mu_normalized,
            target_latent_mean_device.squeeze(0),
            target_latent_std_device.squeeze(0),
        )
        predicted_logvar = denormalize_target_logvar(
            predicted_logvar_normalized,
            target_latent_std_device.squeeze(0),
        )
        if not (
            torch.isfinite(predicted_mu_normalized).all()
            and torch.isfinite(predicted_logvar_normalized).all()
            and torch.isfinite(predicted_mu).all()
            and torch.isfinite(predicted_logvar).all()
        ):
            raise RuntimeError("Depth-window encoder produced non-finite outputs during evaluation.")

        distribution_kl = gaussian_kl_divergence(
            batch_target_mu_normalized,
            batch_target_logvar_normalized,
            predicted_mu_normalized,
            predicted_logvar_normalized,
        )
        total_distribution_kl += distribution_kl.item() * batch_size_current
        if decoder_eval_enabled and batch_target_ir is not None:
            batch_value_abs = 0.0
            batch_value_sq = 0.0
            batch_value_max_abs = 0.0
            decoder_text = ir_decoder_text_feature.expand(batch_size_current, -1)
            for _ in range(decoder_eval_samples):
                sampled_z = sample_latent_z(predicted_mu, predicted_logvar)
                decoded_ir_normalized = ir_decoder_model.decode(sampled_z, decoder_text)
                decoded_ir = decoded_ir_normalized * ir_feature_std + ir_feature_mean
                value_error = decoded_ir - batch_target_ir
                value_abs = value_error.abs()
                batch_value_abs += value_abs.sum().item()
                batch_value_sq += value_error.square().sum().item()
                batch_value_max_abs = max(batch_value_max_abs, float(value_abs.max().item()))
            total_value_abs_error += batch_value_abs / decoder_eval_samples
            total_value_squared_error += batch_value_sq / decoder_eval_samples
            total_value_elements += int(batch_target_ir.numel())
            value_max_abs = max(value_max_abs, batch_value_max_abs)
        seen_samples += batch_size_current

    if total_value_elements > 0:
        mean_squared_value_error = total_value_squared_error / total_value_elements
        value_mae = total_value_abs_error / total_value_elements
        value_rmse = math.sqrt(mean_squared_value_error)
    else:
        mean_squared_value_error = float("nan")
        value_mae = float("nan")
        value_rmse = float("nan")
        value_max_abs = float("nan")

    distribution_kl_mean = total_distribution_kl / seen_samples
    total_loss = distribution_kl_mean
    if math.isfinite(mean_squared_value_error):
        total_loss += decoder_value_loss_weight * mean_squared_value_error

    return {
        f"{prefix}_num_samples": int(seen_samples),
        f"{prefix}_loss": total_loss,
        f"{prefix}_loss_distribution_kl": distribution_kl_mean,
        f"{prefix}_value_mae": value_mae,
        f"{prefix}_value_rmse": value_rmse,
        f"{prefix}_value_max_abs": value_max_abs,
    }


def train_encoder(config: TrainConfig) -> Path:
    config.ir_window_body_source = normalize_ir_window_body_source(config.ir_window_body_source)
    if config.best_val_metric not in BEST_VAL_METRIC_CHOICES:
        raise ValueError(
            f"Unknown best_val_metric='{config.best_val_metric}'. "
            f"Choose one of: {', '.join(BEST_VAL_METRIC_CHOICES)}"
        )
    if config.decoder_eval_samples <= 0:
        raise ValueError(f"decoder_eval_samples must be positive, got {config.decoder_eval_samples}.")
    if config.decoder_train_samples <= 0:
        raise ValueError(f"decoder_train_samples must be positive, got {config.decoder_train_samples}.")
    if config.decoder_value_loss_weight < 0:
        raise ValueError(f"decoder_value_loss_weight must be non-negative, got {config.decoder_value_loss_weight}.")
    set_seed(config.seed)
    device = resolve_device(config.device)
    configure_cuda_backend(device)
    if str(device).startswith("cuda"):
        cuda_device = torch.device(device)
        cuda_index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
        logger.info(f"Using CUDA device for depth-window encoder training: {torch.cuda.get_device_name(cuda_index)} ({device})")
    else:
        logger.info(f"Using device for depth-window encoder training: {device}")

    run_paths = create_run_paths(config)
    save_config(config, run_paths)
    wandb = None
    metrics_history: list[dict[str, float | int]] = []

    try:
        data_dir = Path(config.data_dir)
        episodes, telemetry_metadata = extract_episode_paired_windows(
            data_dir,
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

        logger.info(
            f"Episode split with seed={config.seed}: train={len(train_episode_ids)} episodes, "
            f"val={len(val_episode_ids)} episodes, test={len(test_episode_ids)} episodes"
        )
        logger.info(
            f"Window split: train={train_depth_np.shape[0]}, val={val_depth_np.shape[0]}, test={test_depth_np.shape[0]}"
        )

        train_depth = torch.from_numpy(train_depth_np)
        val_depth = torch.from_numpy(val_depth_np)
        test_depth = torch.from_numpy(test_depth_np)
        train_ir = torch.from_numpy(train_ir_np)
        val_ir = torch.from_numpy(val_ir_np)
        test_ir = torch.from_numpy(test_ir_np)

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
        del train_ir_np, val_ir_np, test_ir_np

        target_latent_dim = int(ir_target_mu_train.shape[1])
        if target_latent_dim != config.latent_dim:
            raise ValueError(
                f"Depth encoder latent_dim={config.latent_dim} must match frozen IR-CVAE latent dim={target_latent_dim}. "
                "Set --latent-dim to the same value as the IR checkpoint."
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
            f"Loaded CLIP text encoder for depth-window encoder: model={config.clip_model_id}, "
            f"local_files_only={config.clip_local_files_only}, quiet_load={config.clip_quiet_load}"
        )
        base_text_feature = clip_text.encode([config.condition_text]).to(device=device, dtype=torch.float32)
        if not torch.isfinite(base_text_feature).all():
            raise RuntimeError("CLIP text features for depth-window encoder contain non-finite values.")
        text_feature_dim = int(base_text_feature.shape[-1])

        ir_decoder_model = None
        ir_decoder_text_feature = None
        ir_decoder_feature_mean = None
        ir_decoder_feature_std = None
        ir_decoder_feature_mean_device = None
        ir_decoder_feature_std_device = None
        if config.decoder_value_eval or config.decoder_value_loss_weight > 0:
            try:
                ir_decoder_model, ir_decoder_payload = load_ir_cvae(config.ir_cvae_checkpoint, device=device)
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
                ir_decoder_feature_mean_device = ir_decoder_feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
                ir_decoder_feature_std_device = ir_decoder_feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
                logger.info(
                    "Enabled frozen IR decoder value path: "
                    f"decoder_value_loss_weight={config.decoder_value_loss_weight}, "
                    f"decoder_train_samples={config.decoder_train_samples}, "
                    f"decoder_eval_samples={config.decoder_eval_samples}, "
                    f"target_ir_window_shape={tuple(ir_decoder_payload['input_shape'])}"
                )
            except ValueError as error:
                if config.decoder_value_loss_weight > 0:
                    raise RuntimeError(
                        "decoder_value_loss_weight > 0 requires an IR-CVAE checkpoint with a frozen decoder."
                    ) from error
                logger.warning(f"Disabled frozen IR decoder value validation: {error}")

        wandb = init_wandb(config, run_paths)

        model = DepthWindowLatentEncoder(
            input_shape=telemetry_metadata.depth_input_shape,
            text_feature_dim=text_feature_dim,
            condition_dim=config.condition_dim,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        feature_mean_device = feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
        feature_std_device = feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
        target_latent_mean_device = target_latent_mean.to(device=device, dtype=torch.float32)
        target_latent_std_device = target_latent_std.to(device=device, dtype=torch.float32)

        logger.info(
            f"Training depth-window encoder on train/val/test windows = "
            f"{train_depth.shape[0]}/{val_depth.shape[0]}/{test_depth.shape[0]}, "
            f"depth_window_shape={telemetry_metadata.depth_input_shape}, latent_dim={config.latent_dim}, "
            f"encoder_hidden_dims={config.hidden_dims} (frame_feature_dim, temporal_hidden_dim), "
            f"device={device}, seed={config.seed}, clip_model={config.clip_model_id}, "
            f"ir_window_body_source={config.ir_window_body_source}"
        )
        logger.info(
            f"Frozen IR target checkpoint={config.ir_cvae_checkpoint}, "
            f"target_latent_dim={target_latent_dim}, target_distribution=posterior_mu_logvar"
        )
        logger.info(
            "Training objective: KL(teacher IR posterior || depth posterior) in normalized latent space. "
            f"Optional value objective adds {config.decoder_value_loss_weight} * frozen-decoder value loss "
            f"with {config.decoder_train_samples} sampled z draw(s). "
            f"Stabilizers: learning_rate={config.learning_rate}, min_feature_std={config.min_feature_std}, "
            f"min_target_latent_std={config.min_target_latent_std}, max_grad_norm={config.max_grad_norm}, "
            f"best_val_metric={config.best_val_metric}, decoder_value_eval={config.decoder_value_eval}"
        )
        if telemetry_metadata.ir_t_mode is not None:
            logger.info(
                f"Telemetry metadata: ir_t_mode={telemetry_metadata.ir_t_mode}, "
                f"ir_t_dim={telemetry_metadata.ir_t_dim or telemetry_metadata.ir_window_shape[1]}, "
                f"depth_resolution={telemetry_metadata.depth_resolution}"
            )
        logger.info(
            "Depth data stays on CPU and only minibatches move to GPU so the encoder can stay lightweight "
            "while training over the full telemetry set."
        )

        best_val_score = float("inf")
        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state: dict[str, torch.Tensor] | None = None
        last_model_state: dict[str, torch.Tensor] | None = None
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
        }

        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_distribution_kl = 0.0
            epoch_decoder_value = 0.0
            epoch_loss = 0.0
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
                batch_target_mu_normalized = ir_target_mu_train_normalized.index_select(0, batch_indices).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_target_logvar_normalized = ir_target_logvar_train_normalized.index_select(0, batch_indices).to(
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
                if not torch.isfinite(batch_depth).all():
                    raise RuntimeError("Non-finite values detected in normalized depth_window batch.")
                if not (
                    torch.isfinite(batch_target_mu_normalized).all()
                    and torch.isfinite(batch_target_logvar_normalized).all()
                ):
                    raise RuntimeError("Non-finite values detected in frozen IR latent targets.")

                predicted_mu_normalized, predicted_logvar_normalized = model(batch_depth, batch_text)
                if not (
                    torch.isfinite(predicted_mu_normalized).all()
                    and torch.isfinite(predicted_logvar_normalized).all()
                ):
                    raise RuntimeError(
                        "Depth-window encoder produced non-finite outputs during training. "
                        "Try a smaller learning rate or inspect the current batch statistics."
                    )

                distribution_kl = gaussian_kl_divergence(
                    batch_target_mu_normalized,
                    batch_target_logvar_normalized,
                    predicted_mu_normalized,
                    predicted_logvar_normalized,
                )
                decoder_value_loss = torch.zeros((), device=device)
                if config.decoder_value_loss_weight > 0:
                    if (
                        ir_decoder_model is None
                        or ir_decoder_text_feature is None
                        or ir_decoder_feature_mean_device is None
                        or ir_decoder_feature_std_device is None
                        or batch_target_ir is None
                    ):
                        raise RuntimeError("Frozen IR decoder is required for decoder value loss.")
                    predicted_mu = denormalize_target_latents(
                        predicted_mu_normalized,
                        target_latent_mean_device,
                        target_latent_std_device,
                    )
                    predicted_logvar = denormalize_target_logvar(
                        predicted_logvar_normalized,
                        target_latent_std_device,
                    )
                    decoder_text = ir_decoder_text_feature.expand(batch_size_current, -1)
                    for _ in range(config.decoder_train_samples):
                        sampled_z = sample_latent_z(predicted_mu, predicted_logvar)
                        decoded_ir_normalized = ir_decoder_model.decode(sampled_z, decoder_text)
                        decoded_ir = (
                            decoded_ir_normalized * ir_decoder_feature_std_device
                            + ir_decoder_feature_mean_device
                        )
                        decoder_value_loss = decoder_value_loss + (decoded_ir - batch_target_ir).pow(2).mean()
                    decoder_value_loss = decoder_value_loss / config.decoder_train_samples

                batch_loss = distribution_kl + config.decoder_value_loss_weight * decoder_value_loss
                if not torch.isfinite(batch_loss):
                    raise RuntimeError(
                        "Training loss became non-finite. "
                        "This usually means the frozen IR targets or the current batch activations contain NaN/Inf."
                    )

                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()

                epoch_distribution_kl += distribution_kl.item() * batch_size_current
                epoch_decoder_value += decoder_value_loss.item() * batch_size_current
                epoch_loss += batch_loss.item() * batch_size_current
                seen_samples += batch_size_current

            train_decoder_value_loss = (
                epoch_decoder_value / seen_samples if config.decoder_value_loss_weight > 0 else float("nan")
            )
            train_metrics = {
                "train_num_samples": int(seen_samples),
                "train_loss": epoch_loss / seen_samples,
                "train_loss_distribution_kl": epoch_distribution_kl / seen_samples,
                "train_value_rmse": math.sqrt(train_decoder_value_loss)
                if math.isfinite(train_decoder_value_loss)
                else float("nan"),
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
                decoder_value_loss_weight=config.decoder_value_loss_weight,
                decoder_eval_samples=config.decoder_eval_samples,
                device=device,
                prefix="val",
            )
            current_val_loss = float(val_metrics["val_loss"])
            current_val_selection_score = float(val_metrics[config.best_val_metric])
            if not math.isfinite(current_val_selection_score):
                raise RuntimeError(
                    f"Validation selection metric {config.best_val_metric} is not finite. "
                    "If you selected val_value_mae or val_value_rmse, use an IR-CVAE checkpoint that contains "
                    "'cvae_state_dict' from the updated cvae_ir_train.py."
                )

            if current_val_selection_score < best_val_score:
                best_val_score = current_val_selection_score
                best_val_loss = current_val_loss
                best_epoch = epoch
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
                    val_loss_total=current_val_loss,
                    val_selection_metric=config.best_val_metric,
                    val_selection_score=current_val_selection_score,
                    ir_alignment_metadata=ir_alignment_metadata,
                )

            epoch_metrics = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "train/loss": train_metrics["train_loss"],
                "train/distribution_kl": train_metrics["train_loss_distribution_kl"],
                "train/value_rmse": train_metrics["train_value_rmse"],
                "val/loss": val_metrics["val_loss"],
                "val/distribution_kl": val_metrics["val_loss_distribution_kl"],
                "val/value_mae": val_metrics["val_value_mae"],
                "val/value_rmse": val_metrics["val_value_rmse"],
                "val/value_max_abs": val_metrics["val_value_max_abs"],
                "config/decoder_value_eval": config.decoder_value_eval,
                "config/decoder_value_loss_weight": config.decoder_value_loss_weight,
                "config/decoder_train_samples": config.decoder_train_samples,
                "config/decoder_eval_samples": config.decoder_eval_samples,
                "val/selection_score": current_val_selection_score,
                "best_val_loss": best_val_loss,
                "best_val_score": best_val_score,
                "best_epoch": best_epoch,
            }
            metrics_history.append(epoch_metrics)

            if wandb is not None and wandb.run is not None:
                wandb.log(epoch_metrics, step=epoch)

            if epoch % config.log_interval == 0 or epoch == 1 or epoch == config.epochs:
                logger.info(
                    f"epoch={epoch:04d} "
                    f"train_loss={train_metrics['train_loss']:.6f} "
                    f"train_kl={train_metrics['train_loss_distribution_kl']:.6f} "
                    f"train_value_rmse={train_metrics['train_value_rmse']:.6f} "
                    f"val_loss={val_metrics['val_loss']:.6f} "
                    f"val_kl={val_metrics['val_loss_distribution_kl']:.6f} "
                    f"val_value_mae={val_metrics['val_value_mae']:.6f} "
                    f"val_value_rmse={val_metrics['val_value_rmse']:.6f} "
                    f"val_select={current_val_selection_score:.6f} "
                    f"best_val_loss={best_val_loss:.6f} "
                    f"best_val_score={best_val_score:.6f} "
                    f"best_val_metric={config.best_val_metric}"
                )

        last_model_state = clone_state_dict_to_cpu(model)
        final_val_total = float(metrics_history[-1]["val_loss"])
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
            epoch=config.epochs,
            val_loss_total=final_val_total,
            val_selection_metric=config.best_val_metric,
            val_selection_score=final_val_selection_score,
            ir_alignment_metadata=ir_alignment_metadata,
        )

        if best_model_state is None:
            best_model_state = clone_state_dict_to_cpu(model)
            best_epoch = config.epochs
            best_val_score = final_val_selection_score
            best_val_loss = final_val_total
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
                epoch=best_epoch,
                val_loss_total=final_val_total,
                val_selection_metric=config.best_val_metric,
                val_selection_score=final_val_selection_score,
                ir_alignment_metadata=ir_alignment_metadata,
            )

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
            decoder_value_loss_weight=config.decoder_value_loss_weight,
            decoder_eval_samples=config.decoder_eval_samples,
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
            decoder_value_loss_weight=config.decoder_value_loss_weight,
            decoder_eval_samples=config.decoder_eval_samples,
            device=device,
            prefix="test",
        )

        test_differences = compute_metric_differences(last_test_metrics, best_test_metrics)
        best_test_metrics_named = rename_metric_prefix(best_test_metrics, "test_", "best_test_")
        last_test_metrics_named = rename_metric_prefix(last_test_metrics, "test_", "last_test_")

        summary = {
            "seed": config.seed,
            "depth_input_shape": list(telemetry_metadata.depth_input_shape),
            "ir_window_shape": list(telemetry_metadata.ir_window_shape),
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
            "decoder_eval_samples": config.decoder_eval_samples,
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

        logger.info(f"Saved best encoder checkpoint to: {run_paths.best_checkpoint_path}")
        logger.info(f"Saved last encoder checkpoint to: {run_paths.last_checkpoint_path}")
        logger.info(f"Saved split and metric summary to: {run_paths.metrics_path}")
        logger.info(
            f"Test comparison: best_loss={best_test_metrics_named['best_test_loss']:.6f}, "
            f"last_loss={last_test_metrics_named['last_test_loss']:.6f}, "
            f"delta(last-best)={test_differences.get('test_loss_last_minus_best', float('nan')):.6f}, "
            f"best_value_rmse={best_test_metrics_named['best_test_value_rmse']:.6f}, "
            f"last_value_rmse={last_test_metrics_named['last_test_value_rmse']:.6f}"
        )

        if wandb is not None and wandb.run is not None:
            final_log = {
                "epoch": config.epochs,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_score": best_val_score,
                "best_test/loss": best_test_metrics_named["best_test_loss"],
                "best_test/distribution_kl": best_test_metrics_named["best_test_loss_distribution_kl"],
                "best_test/value_mae": best_test_metrics_named["best_test_value_mae"],
                "best_test/value_rmse": best_test_metrics_named["best_test_value_rmse"],
                "best_test/value_max_abs": best_test_metrics_named["best_test_value_max_abs"],
                "last_test/loss": last_test_metrics_named["last_test_loss"],
                "last_test/distribution_kl": last_test_metrics_named["last_test_loss_distribution_kl"],
                "last_test/value_mae": last_test_metrics_named["last_test_value_mae"],
                "last_test/value_rmse": last_test_metrics_named["last_test_value_rmse"],
                "last_test/value_max_abs": last_test_metrics_named["last_test_value_max_abs"],
                "compare/test_loss_last_minus_best": test_differences.get("test_loss_last_minus_best", float("nan")),
                "compare/test_loss_distribution_kl_last_minus_best": test_differences.get(
                    "test_loss_distribution_kl_last_minus_best",
                    float("nan"),
                ),
                "compare/test_value_mae_last_minus_best": test_differences.get(
                    "test_value_mae_last_minus_best",
                    float("nan"),
                ),
                "compare/test_value_rmse_last_minus_best": test_differences.get(
                    "test_value_rmse_last_minus_best",
                    float("nan"),
                ),
                **best_test_metrics_named,
                **last_test_metrics_named,
                **test_differences,
            }
            wandb.log(final_log, step=config.epochs)
            wandb.save(str(run_paths.config_path), base_path=str(run_paths.run_dir))
            wandb.save(str(run_paths.best_checkpoint_path), base_path=str(run_paths.run_dir))
            wandb.save(str(run_paths.last_checkpoint_path), base_path=str(run_paths.run_dir))
            wandb.save(str(run_paths.metrics_path), base_path=str(run_paths.run_dir))

        return run_paths.best_checkpoint_path
    finally:
        if wandb is not None and wandb.run is not None:
            wandb.finish()


def load_encoder(checkpoint_path: str, device: str = "cpu") -> tuple[DepthWindowLatentEncoder, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    config_dict = payload["config"]
    encoder = DepthWindowLatentEncoder(
        input_shape=tuple(payload["input_shape"]),
        text_feature_dim=payload["text_feature_dim"],
        condition_dim=config_dict["condition_dim"],
        hidden_dims=tuple(config_dict["hidden_dims"]),
        latent_dim=config_dict["latent_dim"],
        logvar_clamp_min=config_dict.get("logvar_clamp_min", -10.0),
        logvar_clamp_max=config_dict.get("logvar_clamp_max", 10.0),
    )
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()
    return encoder, payload


@torch.no_grad()
def encode_depth_window_to_latent(
    checkpoint_path: str,
    depth_window: np.ndarray | list,
    condition_text: str | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    encoder, payload = load_encoder(checkpoint_path, device=device)
    depth_window_array = np.asarray(depth_window, dtype=np.float32)
    expected_shape = tuple(payload["input_shape"])
    if tuple(depth_window_array.shape) != expected_shape:
        raise ValueError(f"Expected depth_window shape {expected_shape}, got {depth_window_array.shape}")

    clip_cfg = payload["clip"]
    text_extractor = CLIPTextFeatureExtractor(
        model_id=clip_cfg["model_id"],
        device=device,
        cache_dir=clip_cfg["cache_dir"],
        local_files_only=clip_cfg["local_files_only"],
        quiet_load=True,
    )
    text_string = condition_text or payload["condition_text"]
    text_features = text_extractor.encode([text_string])

    x = torch.tensor(depth_window_array, dtype=torch.float32, device=device).unsqueeze(0)
    feature_mean = payload["feature_mean"].to(device=device, dtype=torch.float32)
    feature_std = payload["feature_std"].to(device=device, dtype=torch.float32)
    x = (x - feature_mean.unsqueeze(0)) / feature_std.unsqueeze(0)

    mu, logvar = encoder(x, text_features)
    target_latent_mean = payload.get("target_latent_mean")
    target_latent_std = payload.get("target_latent_std")
    if target_latent_mean is not None and target_latent_std is not None:
        target_latent_std = target_latent_std.to(device=device, dtype=torch.float32)
        mu = denormalize_target_latents(
            mu,
            target_latent_mean.to(device=device, dtype=torch.float32),
            target_latent_std,
        )
        logvar = denormalize_target_logvar(logvar, target_latent_std)
    return sample_latent_z(mu, logvar).squeeze(0).cpu()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a lightweight CNN+temporal depth-window encoder aligned to frozen IR-CVAE latents."
    )
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--condition-text", type=str, default=TrainConfig.condition_text)
    parser.add_argument(
        "--ir-window-body-source",
        type=str,
        default=TrainConfig.ir_window_body_source,
        choices=U_WINDOW_BODY_SOURCE_CHOICES,
        help="Which body subset to use from telemetry ir_window when building frozen IR-CVAE targets.",
    )
    parser.add_argument("--output-root", type=str, default=TrainConfig.output_root)
    parser.add_argument("--run-name", type=str, default=TrainConfig.run_name)
    parser.add_argument("--ir-cvae-checkpoint", type=str, default=TrainConfig.ir_cvae_checkpoint)
    parser.add_argument("--latent-dim", type=int, default=TrainConfig.latent_dim)
    parser.add_argument("--hidden-dim-1", type=int, default=TrainConfig.hidden_dims[0])
    parser.add_argument("--hidden-dim-2", type=int, default=TrainConfig.hidden_dims[1])
    parser.add_argument("--condition-dim", type=int, default=TrainConfig.condition_dim)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument(
        "--best-val-metric",
        type=str,
        default=TrainConfig.best_val_metric,
        choices=BEST_VAL_METRIC_CHOICES,
        help="Validation metric minimized when saving best.pt.",
    )
    parser.add_argument("--decoder-value-eval", type=str_to_bool, default=TrainConfig.decoder_value_eval)
    parser.add_argument("--decoder-value-loss-weight", type=float, default=TrainConfig.decoder_value_loss_weight)
    parser.add_argument("--decoder-train-samples", type=int, default=TrainConfig.decoder_train_samples)
    parser.add_argument("--decoder-eval-samples", type=int, default=TrainConfig.decoder_eval_samples)
    parser.add_argument("--min-feature-std", type=float, default=TrainConfig.min_feature_std)
    parser.add_argument("--min-target-latent-std", type=float, default=TrainConfig.min_target_latent_std)
    parser.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
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
        hidden_dims=(args.hidden_dim_1, args.hidden_dim_2),
        condition_dim=args.condition_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        best_val_metric=args.best_val_metric,
        decoder_value_eval=bool(args.decoder_value_eval),
        decoder_value_loss_weight=args.decoder_value_loss_weight,
        decoder_train_samples=args.decoder_train_samples,
        decoder_eval_samples=args.decoder_eval_samples,
        min_feature_std=args.min_feature_std,
        min_target_latent_std=args.min_target_latent_std,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
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
    logger.info(f"Finished training depth-window encoder. Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
