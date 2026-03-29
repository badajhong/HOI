from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from loguru import logger
from torch import nn
from torch.nn import functional as F

from holosoma.cvae_ir_train import (
    CLIPTextFeatureExtractor,
    clone_state_dict_to_cpu,
    compute_metric_differences,
    configure_cuda_backend,
    create_run_paths,
    init_wandb,
    kl_divergence,
    load_encoder as load_ir_encoder,
    rename_metric_prefix,
    resolve_device,
    save_config,
    set_seed,
    split_episode_indices,
    str_to_bool,
)
from holosoma.utils.safe_torch_import import torch


DEFAULT_DATA_DIR = "/home/rllab/haechan/holosoma/logs/WholeBodyTracking/20260327_150430-g1_29dof_wbt_manager-ir/telemetry"
DEFAULT_OUTPUT_ROOT = "/home/rllab/haechan/holosoma/logs/CVAE/"
DEFAULT_CONDITION_TEXT = "Push the suitcase, and set it back down."
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_IR_CVAE_CHECKPOINT = "/home/rllab/haechan/holosoma/logs/CVAE/20260328_024744-cvae-ir/best.pt"


@dataclass
class TrainConfig:
    data_dir: str = DEFAULT_DATA_DIR
    condition_text: str = DEFAULT_CONDITION_TEXT
    output_root: str = DEFAULT_OUTPUT_ROOT
    run_name: str = "cvae-di"
    ir_cvae_checkpoint: str = DEFAULT_IR_CVAE_CHECKPOINT
    latent_dim: int = 32
    hidden_dims: tuple[int, int] = (64, 128)
    condition_dim: int = 64
    batch_size: int = 2048
    epochs: int = 5000
    learning_rate: float = 1e-4
    contrastive_weight: float = 0
    smooth_l1_beta: float = 1
    contrastive_temperature: float = 0.07
    min_feature_std: float = 1e-4
    min_target_latent_std: float = 1e-3
    max_grad_norm: float = 1.0
    logvar_clamp_min: float = -10.0
    logvar_clamp_max: float = 10.0
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 10
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
    u_windows: np.ndarray
    depth_windows: np.ndarray


@dataclass
class TelemetryMetadata:
    depth_input_shape: tuple[int, int, int]
    u_window_shape: tuple[int, int]
    depth_resolution: tuple[int, int] | None = None
    u_t_mode: str | None = None
    u_t_components: tuple[str, ...] = ()
    u_t_dim: int | None = None
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


def compose_total_loss_value(
    smooth_l1_loss: float,
    contrastive_loss: float,
    *,
    contrastive_weight: float,
) -> float:
    return smooth_l1_loss + contrastive_weight * contrastive_loss


def iterate_batch_indices(num_samples: int, batch_size: int, *, shuffle: bool, seed: int):
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        indices = torch.randperm(num_samples, generator=generator)
    else:
        indices = torch.arange(num_samples, dtype=torch.long)

    for start in range(0, num_samples, batch_size):
        yield indices[start : start + batch_size]


def extract_episode_paired_windows(data_dir: Path) -> tuple[list[EpisodePairedWindows], TelemetryMetadata]:
    json_paths = sorted(data_dir.glob("episode_env*_idx*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No episode JSON files found under: {data_dir}")

    logger.info(f"Scanning {len(json_paths)} telemetry JSON files under: {data_dir}")

    episodes: list[EpisodePairedWindows] = []
    expected_u_shape: tuple[int, int] | None = None
    expected_depth_shape: tuple[int, int, int] | None = None
    total_windows = 0
    metadata: TelemetryMetadata | None = None

    for file_index, json_path in enumerate(json_paths, start=1):
        if file_index == 1 or file_index % 10 == 0 or file_index == len(json_paths):
            logger.info(f"Reading paired u_window/depth_window values from file {file_index}/{len(json_paths)}: {json_path.name}")

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        entries = payload.get("entries", [])
        if not isinstance(entries, list) or not entries:
            continue

        episode_u_windows: list[np.ndarray] = []
        episode_depth_windows: list[np.ndarray] = []
        for entry_index, entry in enumerate(entries):
            if "u_window" not in entry or "depth_window" not in entry:
                raise ValueError(
                    f"Expected both u_window and depth_window in {json_path} entry {entry_index}, "
                    f"available keys={sorted(entry.keys())}."
                )

            u_window = np.asarray(entry["u_window"], dtype=np.float32)
            depth_window = np.asarray(entry["depth_window"], dtype=np.float32)

            if u_window.ndim != 2:
                raise ValueError(
                    f"Expected u_window to have rank 2, got shape {u_window.shape} "
                    f"in {json_path} entry {entry_index}."
                )
            if depth_window.ndim != 3:
                raise ValueError(
                    f"Expected depth_window to have rank 3, got shape {depth_window.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            current_u_shape = (int(u_window.shape[0]), int(u_window.shape[1]))
            current_depth_shape = (int(depth_window.shape[0]), int(depth_window.shape[1]), int(depth_window.shape[2]))
            if expected_u_shape is None:
                expected_u_shape = current_u_shape
            elif current_u_shape != expected_u_shape:
                raise ValueError(
                    f"Inconsistent u_window shape. Expected {expected_u_shape}, got {u_window.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            if expected_depth_shape is None:
                expected_depth_shape = current_depth_shape
            elif current_depth_shape != expected_depth_shape:
                raise ValueError(
                    f"Inconsistent depth_window shape. Expected {expected_depth_shape}, got {depth_window.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            episode_u_windows.append(u_window)
            episode_depth_windows.append(depth_window)

        if not episode_u_windows:
            continue

        episodes.append(
            EpisodePairedWindows(
                episode_id=json_path.stem,
                u_windows=np.stack(episode_u_windows, axis=0),
                depth_windows=np.stack(episode_depth_windows, axis=0),
            )
        )
        total_windows += len(episode_u_windows)

        if metadata is None:
            assert expected_u_shape is not None
            assert expected_depth_shape is not None
            u_t_components_value = payload.get("u_t_components")
            u_t_components = ()
            if isinstance(u_t_components_value, list):
                u_t_components = tuple(str(value) for value in u_t_components_value)

            u_t_mode_value = payload.get("u_t_mode")
            u_t_mode = str(u_t_mode_value) if isinstance(u_t_mode_value, str) else None

            u_t_dim_value = payload.get("u_t_dim")
            u_t_dim = int(u_t_dim_value) if isinstance(u_t_dim_value, int | float) else None
            if u_t_dim is not None and u_t_dim != expected_u_shape[1]:
                raise ValueError(
                    f"Telemetry metadata u_t_dim={u_t_dim} does not match extracted u_window feature dim="
                    f"{expected_u_shape[1]} for {json_path}."
                )

            depth_resolution_value = payload.get("depth_resolution")
            depth_resolution = None
            if isinstance(depth_resolution_value, list) and len(depth_resolution_value) == 2:
                depth_resolution = (int(depth_resolution_value[0]), int(depth_resolution_value[1]))
                spatial_hw = expected_depth_shape[1:]
                spatial_wh = (spatial_hw[1], spatial_hw[0])
                if depth_resolution not in (spatial_hw, spatial_wh):
                    raise ValueError(
                        f"Telemetry depth_resolution={depth_resolution} does not match extracted depth_window spatial "
                        f"shape={expected_depth_shape[1:]} for {json_path}."
                    )
                if depth_resolution == spatial_wh:
                    logger.info(
                        f"Telemetry depth_resolution={depth_resolution} is stored as (width, height); "
                        f"depth_window tensors use (height, width)={spatial_hw}."
                    )

            metadata = TelemetryMetadata(
                depth_input_shape=expected_depth_shape,
                u_window_shape=expected_u_shape,
                depth_resolution=depth_resolution,
                u_t_mode=u_t_mode,
                u_t_components=u_t_components,
                u_t_dim=u_t_dim,
                source_episode_id=json_path.stem,
            )

    if not episodes or metadata is None:
        raise ValueError(f"No valid paired u_window/depth_window entries found under: {data_dir}")

    logger.info(
        f"Loaded {total_windows} paired samples from {len(episodes)} episode files with "
        f"u_window_shape={metadata.u_window_shape}, depth_window_shape={metadata.depth_input_shape}"
    )
    if metadata.u_t_mode is not None:
        logger.info(
            f"Telemetry metadata: u_t_mode={metadata.u_t_mode}, "
            f"u_t_dim={metadata.u_t_dim or metadata.u_window_shape[1]}, "
            f"depth_resolution={metadata.depth_resolution}, "
            f"source={metadata.source_episode_id}"
        )

    return episodes, metadata


def flatten_episode_split(
    episodes: Sequence[EpisodePairedWindows],
    indices: Sequence[int],
    u_window_shape: tuple[int, int],
    depth_window_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected = [episodes[index] for index in indices]
    episode_ids = [episode.episode_id for episode in selected]
    if not selected:
        return (
            np.empty((0, u_window_shape[0], u_window_shape[1]), dtype=np.float32),
            np.empty((0, depth_window_shape[0], depth_window_shape[1], depth_window_shape[2]), dtype=np.float32),
            episode_ids,
        )

    stacked_u = np.concatenate([episode.u_windows for episode in selected], axis=0).astype(np.float32, copy=False)
    stacked_depth = np.concatenate([episode.depth_windows for episode in selected], axis=0).astype(np.float32, copy=False)
    return stacked_u, stacked_depth, episode_ids


@torch.no_grad()
def encode_ir_latent_targets(
    *,
    ir_checkpoint_path: str,
    u_windows: torch.Tensor,
    condition_text: str,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ir_encoder, ir_payload = load_ir_encoder(ir_checkpoint_path, device=device)
    expected_shape = tuple(ir_payload["input_shape"])
    if tuple(u_windows.shape[1:]) != expected_shape:
        raise ValueError(
            f"IR-CVAE checkpoint expects u_window shape {expected_shape}, but telemetry provides {tuple(u_windows.shape[1:])}."
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
    num_samples = int(u_windows.shape[0])
    targets = torch.empty((num_samples, latent_dim), dtype=torch.float32)

    logger.info(
        f"Encoding {num_samples} paired u_window samples with frozen IR-CVAE from {ir_checkpoint_path} "
        f"to build latent contrastive targets."
    )

    for batch_indices in iterate_batch_indices(num_samples, batch_size, shuffle=False, seed=0):
        batch_u = u_windows.index_select(0, batch_indices).to(device=device, dtype=torch.float32, non_blocking=True)
        batch_u = batch_u.reshape(batch_u.shape[0], -1)
        batch_u = (batch_u - feature_mean) / feature_std
        if not torch.isfinite(batch_u).all():
            raise RuntimeError("Non-finite values detected in normalized u_window batch before frozen IR-CVAE encoding.")
        batch_text = base_text_feature.expand(batch_u.shape[0], -1)
        mu, _ = ir_encoder(batch_u, batch_text)
        if not torch.isfinite(mu).all():
            raise RuntimeError(
                "Frozen IR-CVAE produced non-finite latent targets. "
                "Please verify the checkpoint, CLIP text conditioning, and u_window normalization."
            )
        targets[batch_indices] = mu.detach().cpu()

    return targets, ir_payload


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


def symmetric_contrastive_loss(
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_norm = F.normalize(source_latent, dim=-1)
    target_norm = F.normalize(target_latent, dim=-1)
    logits = source_norm @ target_norm.t()
    logits = logits / max(temperature, 1e-6)
    labels = torch.arange(source_latent.shape[0], device=source_latent.device)
    loss_forward = F.cross_entropy(logits, labels)
    loss_backward = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_forward + loss_backward)
    cosine = torch.sum(source_norm * target_norm, dim=-1).mean()
    retrieval_top1 = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, cosine, retrieval_top1


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
    val_selection_loss: float,
    ir_alignment_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_type": "depth_window_temporal_encoder",
        "checkpoint_type": checkpoint_type,
        "epoch": epoch,
        "val_loss_total": val_loss_total,
        "val_selection_loss": val_selection_loss,
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
    val_selection_loss: float,
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
        val_selection_loss=val_selection_loss,
        ir_alignment_metadata=ir_alignment_metadata,
    )
    torch.save(payload, checkpoint_path)


@torch.no_grad()
def evaluate_model(
    model: DepthWindowLatentEncoder,
    depth_windows: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    base_text_feature: torch.Tensor,
    batch_size: int,
    smooth_l1_beta: float,
    contrastive_weight: float,
    contrastive_temperature: float,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    target_latent_mean: torch.Tensor,
    target_latent_std: torch.Tensor,
    device: str,
    prefix: str,
) -> dict[str, float | int]:
    if depth_windows.shape[0] == 0:
        return {
            f"{prefix}_num_samples": 0,
            f"{prefix}_loss_smooth_l1": float("nan"),
            f"{prefix}_loss_smooth_l1_normalized": float("nan"),
            f"{prefix}_loss_kl": float("nan"),
            f"{prefix}_loss_contrastive": float("nan"),
            f"{prefix}_loss_total": float("nan"),
            f"{prefix}_loss_total_optimized": float("nan"),
            f"{prefix}_latent_cosine": float("nan"),
            f"{prefix}_latent_retrieval_top1": float("nan"),
            f"{prefix}_latent_norm": float("nan"),
        }

    mean_device = feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
    std_device = feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
    target_latent_mean_device = target_latent_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
    target_latent_std_device = target_latent_std.to(device=device, dtype=torch.float32).unsqueeze(0)

    model.eval()
    total_smooth_l1 = 0.0
    total_smooth_l1_normalized = 0.0
    total_kl = 0.0
    total_contrastive = 0.0
    total_loss = 0.0
    total_loss_optimized = 0.0
    total_cosine = 0.0
    total_retrieval = 0.0
    total_latent_norm = 0.0
    seen_samples = 0

    for batch_indices in iterate_batch_indices(int(depth_windows.shape[0]), batch_size, shuffle=False, seed=0):
        batch_depth = depth_windows.index_select(0, batch_indices).to(device=device, dtype=torch.float32, non_blocking=True)
        batch_target_latent = target_latents.index_select(0, batch_indices).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        batch_size_current = int(batch_depth.shape[0])

        batch_depth = (batch_depth - mean_device) / std_device
        batch_target_latent_normalized = normalize_target_latents(
            batch_target_latent,
            target_latent_mean_device.squeeze(0),
            target_latent_std_device.squeeze(0),
        )
        batch_text = base_text_feature.expand(batch_size_current, -1)
        if not torch.isfinite(batch_depth).all():
            raise RuntimeError("Non-finite values detected in normalized depth_window batch during evaluation.")
        if not torch.isfinite(batch_target_latent).all():
            raise RuntimeError("Non-finite values detected in frozen IR latent targets during evaluation.")

        mu_normalized, logvar = model(batch_depth, batch_text)
        predicted_latent = denormalize_target_latents(
            mu_normalized,
            target_latent_mean_device.squeeze(0),
            target_latent_std_device.squeeze(0),
        )
        if not (torch.isfinite(mu_normalized).all() and torch.isfinite(logvar).all() and torch.isfinite(predicted_latent).all()):
            raise RuntimeError("Depth-window encoder produced non-finite outputs during evaluation.")

        smooth_l1_loss = F.smooth_l1_loss(predicted_latent, batch_target_latent, beta=smooth_l1_beta)
        smooth_l1_loss_normalized = F.smooth_l1_loss(mu_normalized, batch_target_latent_normalized, beta=smooth_l1_beta)
        kl_loss = kl_divergence(mu_normalized, logvar)
        contrastive_loss, cosine, retrieval_top1 = symmetric_contrastive_loss(
            predicted_latent,
            batch_target_latent,
            temperature=contrastive_temperature,
        )
        total_batch_loss = smooth_l1_loss + contrastive_weight * contrastive_loss
        total_batch_loss_optimized = smooth_l1_loss_normalized + contrastive_weight * contrastive_loss
        if not torch.isfinite(total_batch_loss_optimized):
            raise RuntimeError("Evaluation loss became non-finite.")

        total_smooth_l1 += smooth_l1_loss.item() * batch_size_current
        total_smooth_l1_normalized += smooth_l1_loss_normalized.item() * batch_size_current
        total_kl += kl_loss.item() * batch_size_current
        total_contrastive += contrastive_loss.item() * batch_size_current
        total_loss += total_batch_loss.item() * batch_size_current
        total_loss_optimized += total_batch_loss_optimized.item() * batch_size_current
        total_cosine += cosine.item() * batch_size_current
        total_retrieval += retrieval_top1.item() * batch_size_current
        total_latent_norm += torch.linalg.norm(predicted_latent, dim=-1).mean().item() * batch_size_current
        seen_samples += batch_size_current

    return {
        f"{prefix}_num_samples": int(seen_samples),
        f"{prefix}_loss_smooth_l1": total_smooth_l1 / seen_samples,
        f"{prefix}_loss_smooth_l1_normalized": total_smooth_l1_normalized / seen_samples,
        f"{prefix}_loss_kl": total_kl / seen_samples,
        f"{prefix}_loss_contrastive": total_contrastive / seen_samples,
        f"{prefix}_loss_total": total_loss / seen_samples,
        f"{prefix}_loss_total_optimized": total_loss_optimized / seen_samples,
        f"{prefix}_latent_cosine": total_cosine / seen_samples,
        f"{prefix}_latent_retrieval_top1": total_retrieval / seen_samples,
        f"{prefix}_latent_norm": total_latent_norm / seen_samples,
    }


def train_encoder(config: TrainConfig) -> Path:
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
        episodes, telemetry_metadata = extract_episode_paired_windows(data_dir)
        split_indices = split_episode_indices(len(episodes), config.val_ratio, config.test_ratio, config.seed)

        train_u_np, train_depth_np, train_episode_ids = flatten_episode_split(
            episodes,
            split_indices["train"],
            telemetry_metadata.u_window_shape,
            telemetry_metadata.depth_input_shape,
        )
        val_u_np, val_depth_np, val_episode_ids = flatten_episode_split(
            episodes,
            split_indices["val"],
            telemetry_metadata.u_window_shape,
            telemetry_metadata.depth_input_shape,
        )
        test_u_np, test_depth_np, test_episode_ids = flatten_episode_split(
            episodes,
            split_indices["test"],
            telemetry_metadata.u_window_shape,
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
        train_u = torch.from_numpy(train_u_np)
        val_u = torch.from_numpy(val_u_np)
        test_u = torch.from_numpy(test_u_np)

        feature_mean = train_depth.mean(dim=0)
        feature_std = train_depth.std(dim=0).clamp_min(config.min_feature_std)

        ir_target_latent_train, ir_payload = encode_ir_latent_targets(
            ir_checkpoint_path=config.ir_cvae_checkpoint,
            u_windows=train_u,
            condition_text=config.condition_text,
            batch_size=config.batch_size,
            device=device,
        )
        ir_target_latent_val, _ = encode_ir_latent_targets(
            ir_checkpoint_path=config.ir_cvae_checkpoint,
            u_windows=val_u,
            condition_text=config.condition_text,
            batch_size=config.batch_size,
            device=device,
        )
        ir_target_latent_test, _ = encode_ir_latent_targets(
            ir_checkpoint_path=config.ir_cvae_checkpoint,
            u_windows=test_u,
            condition_text=config.condition_text,
            batch_size=config.batch_size,
            device=device,
        )
        del train_u, val_u, test_u
        del train_u_np, val_u_np, test_u_np

        target_latent_dim = int(ir_target_latent_train.shape[1])
        if target_latent_dim != config.latent_dim:
            raise ValueError(
                f"Depth encoder latent_dim={config.latent_dim} must match frozen IR-CVAE latent dim={target_latent_dim}. "
                "Set --latent-dim to the same value as the IR checkpoint."
            )

        target_latent_mean, target_latent_std = compute_target_latent_normalization_stats(
            ir_target_latent_train,
            min_std=config.min_target_latent_std,
        )
        ir_target_latent_train_normalized = normalize_target_latents(
            ir_target_latent_train,
            target_latent_mean,
            target_latent_std,
        )

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

        wandb = init_wandb(config, run_paths)

        model = DepthWindowLatentEncoder(
            input_shape=telemetry_metadata.depth_input_shape,
            text_feature_dim=text_feature_dim,
            condition_dim=config.condition_dim,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim,
            logvar_clamp_min=config.logvar_clamp_min,
            logvar_clamp_max=config.logvar_clamp_max,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        feature_mean_device = feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
        feature_std_device = feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
        target_latent_mean_device = target_latent_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
        target_latent_std_device = target_latent_std.to(device=device, dtype=torch.float32).unsqueeze(0)

        logger.info(
            f"Training depth-window encoder on train/val/test windows = "
            f"{train_depth.shape[0]}/{val_depth.shape[0]}/{test_depth.shape[0]}, "
            f"depth_window_shape={telemetry_metadata.depth_input_shape}, latent_dim={config.latent_dim}, "
            f"encoder_hidden_dims={config.hidden_dims} (frame_feature_dim, temporal_hidden_dim), "
            f"device={device}, seed={config.seed}, clip_model={config.clip_model_id}"
        )
        logger.info(
            f"Frozen IR target checkpoint={config.ir_cvae_checkpoint}, "
            f"target_latent_dim={target_latent_dim}, contrastive_temperature={config.contrastive_temperature}"
        )
        logger.info(
            f"Training objective: SmoothL1(beta={config.smooth_l1_beta}) + "
            f"{config.contrastive_weight} * contrastive loss. "
            f"Primary regression objective is SmoothL1(beta={config.smooth_l1_beta}) on train-split normalized IR latents; "
            "validation SmoothL1 is still reported in the original latent space. "
            "Contrastive weight is fixed; KL is logged only. "
            f"Stabilizers: learning_rate={config.learning_rate}, min_feature_std={config.min_feature_std}, "
            f"min_target_latent_std={config.min_target_latent_std}, max_grad_norm={config.max_grad_norm}, "
            f"logvar_clamp=[{config.logvar_clamp_min}, {config.logvar_clamp_max}]"
        )
        if telemetry_metadata.u_t_mode is not None:
            logger.info(
                f"Telemetry metadata: u_t_mode={telemetry_metadata.u_t_mode}, "
                f"u_t_dim={telemetry_metadata.u_t_dim or telemetry_metadata.u_window_shape[1]}, "
                f"depth_resolution={telemetry_metadata.depth_resolution}"
            )
        logger.info(
            "Depth data stays on CPU and only minibatches move to GPU so the encoder can stay lightweight "
            "while training over the full telemetry set."
        )

        best_selection_loss = float("inf")
        best_epoch = 0
        best_model_state: dict[str, torch.Tensor] | None = None
        last_model_state: dict[str, torch.Tensor] | None = None
        ir_alignment_metadata = {
            "ir_cvae_checkpoint": config.ir_cvae_checkpoint,
            "target_model_type": ir_payload.get("model_type"),
            "target_condition_text": ir_payload.get("condition_text"),
            "target_latent_dim": target_latent_dim,
            "u_window_shape": list(ir_payload["input_shape"]),
            "target_latent_mean_abs_mean": float(target_latent_mean.abs().mean().item()),
            "target_latent_std_mean": float(target_latent_std.mean().item()),
            "target_latent_std_min": float(target_latent_std.min().item()),
        }

        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_smooth_l1 = 0.0
            epoch_smooth_l1_normalized = 0.0
            epoch_kl = 0.0
            epoch_contrastive = 0.0
            epoch_total = 0.0
            epoch_total_optimized = 0.0
            epoch_cosine = 0.0
            epoch_retrieval = 0.0
            epoch_latent_norm = 0.0
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
                batch_target_latent = ir_target_latent_train.index_select(0, batch_indices).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_target_latent_normalized = ir_target_latent_train_normalized.index_select(0, batch_indices).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_size_current = int(batch_depth.shape[0])

                batch_depth = (batch_depth - feature_mean_device) / feature_std_device
                batch_text = base_text_feature.expand(batch_size_current, -1)
                if not torch.isfinite(batch_depth).all():
                    raise RuntimeError("Non-finite values detected in normalized depth_window batch.")
                if not torch.isfinite(batch_target_latent).all():
                    raise RuntimeError("Non-finite values detected in frozen IR latent targets.")

                mu_normalized, logvar = model(batch_depth, batch_text)
                predicted_latent = denormalize_target_latents(
                    mu_normalized,
                    target_latent_mean_device.squeeze(0),
                    target_latent_std_device.squeeze(0),
                )
                if not (
                    torch.isfinite(mu_normalized).all()
                    and torch.isfinite(logvar).all()
                    and torch.isfinite(predicted_latent).all()
                ):
                    raise RuntimeError(
                        "Depth-window encoder produced non-finite outputs during training. "
                        "Try a smaller learning rate or inspect the current batch statistics."
                    )

                smooth_l1_loss = F.smooth_l1_loss(predicted_latent, batch_target_latent, beta=config.smooth_l1_beta)
                smooth_l1_loss_normalized = F.smooth_l1_loss(
                    mu_normalized,
                    batch_target_latent_normalized,
                    beta=config.smooth_l1_beta,
                )
                kl_loss = kl_divergence(mu_normalized, logvar)
                contrastive_loss, cosine, retrieval_top1 = symmetric_contrastive_loss(
                    predicted_latent,
                    batch_target_latent,
                    temperature=config.contrastive_temperature,
                )
                total_batch_loss = smooth_l1_loss + config.contrastive_weight * contrastive_loss
                total_batch_loss_optimized = smooth_l1_loss_normalized + config.contrastive_weight * contrastive_loss
                if not torch.isfinite(total_batch_loss_optimized):
                    raise RuntimeError(
                        "Training loss became non-finite. "
                        "This usually means the frozen IR targets or the current batch activations contain NaN/Inf."
                    )

                optimizer.zero_grad(set_to_none=True)
                total_batch_loss_optimized.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()

                epoch_smooth_l1 += smooth_l1_loss.item() * batch_size_current
                epoch_smooth_l1_normalized += smooth_l1_loss_normalized.item() * batch_size_current
                epoch_kl += kl_loss.item() * batch_size_current
                epoch_contrastive += contrastive_loss.item() * batch_size_current
                epoch_total += total_batch_loss.item() * batch_size_current
                epoch_total_optimized += total_batch_loss_optimized.item() * batch_size_current
                epoch_cosine += cosine.item() * batch_size_current
                epoch_retrieval += retrieval_top1.item() * batch_size_current
                epoch_latent_norm += torch.linalg.norm(predicted_latent, dim=-1).mean().item() * batch_size_current
                seen_samples += batch_size_current

            train_metrics = {
                "train_num_samples": int(seen_samples),
                "train_loss_smooth_l1": epoch_smooth_l1 / seen_samples,
                "train_loss_smooth_l1_normalized": epoch_smooth_l1_normalized / seen_samples,
                "train_loss_kl": epoch_kl / seen_samples,
                "train_loss_contrastive": epoch_contrastive / seen_samples,
                "train_loss_total": epoch_total / seen_samples,
                "train_loss_total_optimized": epoch_total_optimized / seen_samples,
                "train_latent_cosine": epoch_cosine / seen_samples,
                "train_latent_retrieval_top1": epoch_retrieval / seen_samples,
                "train_latent_norm": epoch_latent_norm / seen_samples,
            }
            val_metrics = evaluate_model(
                model,
                val_depth,
                ir_target_latent_val,
                base_text_feature=base_text_feature,
                batch_size=config.batch_size,
                smooth_l1_beta=config.smooth_l1_beta,
                contrastive_weight=config.contrastive_weight,
                contrastive_temperature=config.contrastive_temperature,
                feature_mean=feature_mean,
                feature_std=feature_std,
                target_latent_mean=target_latent_mean,
                target_latent_std=target_latent_std,
                device=device,
                prefix="val",
            )
            current_val_loss = float(val_metrics["val_loss_total"])
            current_val_selection_loss = compose_total_loss_value(
                float(val_metrics["val_loss_smooth_l1"]),
                float(val_metrics["val_loss_contrastive"]),
                contrastive_weight=config.contrastive_weight,
            )

            if current_val_selection_loss < best_selection_loss:
                best_selection_loss = current_val_selection_loss
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
                    val_selection_loss=current_val_selection_loss,
                    ir_alignment_metadata=ir_alignment_metadata,
                )

            epoch_metrics = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "train/smooth_l1_loss": train_metrics["train_loss_smooth_l1"],
                "train/smooth_l1_loss_normalized": train_metrics["train_loss_smooth_l1_normalized"],
                "train/kl_loss": train_metrics["train_loss_kl"],
                "train/contrastive_loss": train_metrics["train_loss_contrastive"],
                "train/total_loss": train_metrics["train_loss_total"],
                "train/total_loss_optimized": train_metrics["train_loss_total_optimized"],
                "train/latent_cosine": train_metrics["train_latent_cosine"],
                "train/latent_retrieval_top1": train_metrics["train_latent_retrieval_top1"],
                "train/latent_norm": train_metrics["train_latent_norm"],
                "val/smooth_l1_loss": val_metrics["val_loss_smooth_l1"],
                "val/smooth_l1_loss_normalized": val_metrics["val_loss_smooth_l1_normalized"],
                "val/kl_loss": val_metrics["val_loss_kl"],
                "val/contrastive_loss": val_metrics["val_loss_contrastive"],
                "val/total_loss": val_metrics["val_loss_total"],
                "val/total_loss_optimized": val_metrics["val_loss_total_optimized"],
                "val/selection_loss": current_val_selection_loss,
                "val/latent_cosine": val_metrics["val_latent_cosine"],
                "val/latent_retrieval_top1": val_metrics["val_latent_retrieval_top1"],
                "val/latent_norm": val_metrics["val_latent_norm"],
                "config/contrastive_weight": config.contrastive_weight,
                "best_val_selection_loss": best_selection_loss,
                "best_epoch": best_epoch,
            }
            metrics_history.append(epoch_metrics)

            if wandb is not None and wandb.run is not None:
                wandb.log(epoch_metrics, step=epoch)

            if epoch % config.log_interval == 0 or epoch == 1 or epoch == config.epochs:
                logger.info(
                    f"epoch={epoch:04d} "
                    f"train_smooth_l1={train_metrics['train_loss_smooth_l1']:.6f} "
                    f"train_smooth_l1_norm={train_metrics['train_loss_smooth_l1_normalized']:.6f} "
                    f"train_kl={train_metrics['train_loss_kl']:.6f} "
                    f"train_contrastive={train_metrics['train_loss_contrastive']:.6f} "
                    f"train_total={train_metrics['train_loss_total']:.6f} "
                    f"train_total_opt={train_metrics['train_loss_total_optimized']:.6f} "
                    f"val_smooth_l1={val_metrics['val_loss_smooth_l1']:.6f} "
                    f"val_smooth_l1_norm={val_metrics['val_loss_smooth_l1_normalized']:.6f} "
                    f"val_kl={val_metrics['val_loss_kl']:.6f} "
                    f"val_contrastive={val_metrics['val_loss_contrastive']:.6f} "
                    f"val_total={val_metrics['val_loss_total']:.6f} "
                    f"val_total_opt={val_metrics['val_loss_total_optimized']:.6f} "
                    f"val_select={current_val_selection_loss:.6f} "
                    f"cosine(train/val)={train_metrics['train_latent_cosine']:.4f}/{val_metrics['val_latent_cosine']:.4f} "
                    f"top1(train/val)={train_metrics['train_latent_retrieval_top1']:.4f}/{val_metrics['val_latent_retrieval_top1']:.4f} "
                    f"latent_norm(train/val)={train_metrics['train_latent_norm']:.4f}/{val_metrics['val_latent_norm']:.4f} "
                    f"lambda={config.contrastive_weight:.6f}"
                )

        last_model_state = clone_state_dict_to_cpu(model)
        final_val_total = float(metrics_history[-1]["val_loss_total"])
        final_val_selection_loss = float(metrics_history[-1]["val/selection_loss"])
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
            val_selection_loss=final_val_selection_loss,
            ir_alignment_metadata=ir_alignment_metadata,
        )

        if best_model_state is None:
            best_model_state = clone_state_dict_to_cpu(model)
            best_epoch = config.epochs
            best_selection_loss = final_val_selection_loss
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
                val_selection_loss=best_selection_loss,
                ir_alignment_metadata=ir_alignment_metadata,
            )

        model.load_state_dict(best_model_state)
        best_test_metrics = evaluate_model(
            model,
            test_depth,
            ir_target_latent_test,
            base_text_feature=base_text_feature,
            batch_size=config.batch_size,
            smooth_l1_beta=config.smooth_l1_beta,
            contrastive_weight=config.contrastive_weight,
            contrastive_temperature=config.contrastive_temperature,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_latent_mean=target_latent_mean,
            target_latent_std=target_latent_std,
            device=device,
            prefix="test",
        )

        model.load_state_dict(last_model_state)
        last_test_metrics = evaluate_model(
            model,
            test_depth,
            ir_target_latent_test,
            base_text_feature=base_text_feature,
            batch_size=config.batch_size,
            smooth_l1_beta=config.smooth_l1_beta,
            contrastive_weight=config.contrastive_weight,
            contrastive_temperature=config.contrastive_temperature,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_latent_mean=target_latent_mean,
            target_latent_std=target_latent_std,
            device=device,
            prefix="test",
        )

        test_differences = compute_metric_differences(last_test_metrics, best_test_metrics)
        best_test_metrics_named = rename_metric_prefix(best_test_metrics, "test_", "best_test_")
        last_test_metrics_named = rename_metric_prefix(last_test_metrics, "test_", "last_test_")

        summary = {
            "seed": config.seed,
            "depth_input_shape": list(telemetry_metadata.depth_input_shape),
            "u_window_shape": list(telemetry_metadata.u_window_shape),
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
            "best_val_selection_loss": best_selection_loss,
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
            f"Test comparison: best_cosine={best_test_metrics_named['best_test_latent_cosine']:.6f}, "
            f"last_cosine={last_test_metrics_named['last_test_latent_cosine']:.6f}, "
            f"delta(last-best)={test_differences.get('test_latent_cosine_last_minus_best', float('nan')):.6f}"
        )

        if wandb is not None and wandb.run is not None:
            final_log = {
                "epoch": config.epochs,
                "best_epoch": best_epoch,
                "best_val_selection_loss": best_selection_loss,
                "best_test/smooth_l1_loss": best_test_metrics_named["best_test_loss_smooth_l1"],
                "best_test/kl_loss": best_test_metrics_named["best_test_loss_kl"],
                "best_test/contrastive_loss": best_test_metrics_named["best_test_loss_contrastive"],
                "best_test/total_loss": best_test_metrics_named["best_test_loss_total"],
                "best_test/latent_cosine": best_test_metrics_named["best_test_latent_cosine"],
                "best_test/latent_retrieval_top1": best_test_metrics_named["best_test_latent_retrieval_top1"],
                "best_test/latent_norm": best_test_metrics_named["best_test_latent_norm"],
                "last_test/smooth_l1_loss": last_test_metrics_named["last_test_loss_smooth_l1"],
                "last_test/kl_loss": last_test_metrics_named["last_test_loss_kl"],
                "last_test/contrastive_loss": last_test_metrics_named["last_test_loss_contrastive"],
                "last_test/total_loss": last_test_metrics_named["last_test_loss_total"],
                "last_test/latent_cosine": last_test_metrics_named["last_test_latent_cosine"],
                "last_test/latent_retrieval_top1": last_test_metrics_named["last_test_latent_retrieval_top1"],
                "last_test/latent_norm": last_test_metrics_named["last_test_latent_norm"],
                "compare/test_total_loss_last_minus_best": test_differences.get("test_loss_total_last_minus_best", float("nan")),
                "compare/test_latent_cosine_last_minus_best": test_differences.get(
                    "test_latent_cosine_last_minus_best",
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
        logvar_clamp_min=config_dict["logvar_clamp_min"],
        logvar_clamp_max=config_dict["logvar_clamp_max"],
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

    mu, _ = encoder(x, text_features)
    target_latent_mean = payload.get("target_latent_mean")
    target_latent_std = payload.get("target_latent_std")
    if target_latent_mean is not None and target_latent_std is not None:
        mu = denormalize_target_latents(
            mu,
            target_latent_mean.to(device=device, dtype=torch.float32),
            target_latent_std.to(device=device, dtype=torch.float32),
        )
    return mu.squeeze(0).cpu()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a lightweight CNN+temporal depth-window encoder aligned to frozen IR-CVAE latents."
    )
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--condition-text", type=str, default=TrainConfig.condition_text)
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
    parser.add_argument("--contrastive-weight", type=float, default=TrainConfig.contrastive_weight)
    parser.add_argument("--smooth-l1-beta", type=float, default=TrainConfig.smooth_l1_beta)
    parser.add_argument("--contrastive-temperature", type=float, default=TrainConfig.contrastive_temperature)
    parser.add_argument("--min-feature-std", type=float, default=TrainConfig.min_feature_std)
    parser.add_argument("--min-target-latent-std", type=float, default=TrainConfig.min_target_latent_std)
    parser.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--logvar-clamp-min", type=float, default=TrainConfig.logvar_clamp_min)
    parser.add_argument("--logvar-clamp-max", type=float, default=TrainConfig.logvar_clamp_max)
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
        output_root=args.output_root,
        run_name=args.run_name,
        ir_cvae_checkpoint=args.ir_cvae_checkpoint,
        latent_dim=args.latent_dim,
        hidden_dims=(args.hidden_dim_1, args.hidden_dim_2),
        condition_dim=args.condition_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        contrastive_weight=args.contrastive_weight,
        smooth_l1_beta=args.smooth_l1_beta,
        contrastive_temperature=args.contrastive_temperature,
        min_feature_std=args.min_feature_std,
        min_target_latent_std=args.min_target_latent_std,
        max_grad_norm=args.max_grad_norm,
        logvar_clamp_min=args.logvar_clamp_min,
        logvar_clamp_max=args.logvar_clamp_max,
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
