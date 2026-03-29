from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from loguru import logger
from torch import nn

from holosoma.utils.experiment_paths import get_timestamp
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.wandb import get_wandb


DEFAULT_DATA_DIR = "/home/rllab/haechan/holosoma/logs/WholeBodyTracking/20260327_150430-g1_29dof_wbt_manager-ir/telemetry"
DEFAULT_OUTPUT_ROOT = "/home/rllab/haechan/holosoma/logs/CVAE/"
DEFAULT_CONDITION_TEXT = "Push the suitcase, and set it back down."
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


@dataclass
class TrainConfig:
    data_dir: str = DEFAULT_DATA_DIR
    condition_text: str = DEFAULT_CONDITION_TEXT
    output_root: str = DEFAULT_OUTPUT_ROOT
    run_name: str = "cvae-ir"
    latent_dim: int = 32
    hidden_dims: tuple[int, int] = (256, 128)
    condition_dim: int = 64
    batch_size: int = 512
    epochs: int = 5000
    learning_rate: float = 1e-3
    kl_weight: float = 1e-4
    weight_decay: float = 1e-6
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 10
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    wandb_enabled: bool = True
    wandb_project: str = "CVAE"
    wandb_entity: str | None = None
    wandb_group: str = "cvae_ir"
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ("cvae", "ir", "u_window")
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID
    clip_cache_dir: str | None = None
    clip_local_files_only: bool = True
    clip_quiet_load: bool = True


@dataclass
class EpisodeWindows:
    episode_id: str
    windows: np.ndarray


@dataclass
class TelemetryMetadata:
    input_shape: tuple[int, int]
    u_t_mode: str | None = None
    u_t_components: tuple[str, ...] = ()
    u_t_dim: int | None = None
    source_episode_id: str | None = None


@dataclass
class RunPaths:
    run_dir: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    config_path: Path
    metrics_path: Path
    wandb_dir: Path


class CLIPTextFeatureExtractor:
    def __init__(
        self,
        model_id: str,
        device: str,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        quiet_load: bool = True,
    ):
        try:
            from transformers import AutoTokenizer, CLIPTextModelWithProjection
        except ImportError as exc:
            raise ImportError(
                "transformers is required to encode text with CLIP. "
                "Install it in the current environment, for example: `pip install transformers`."
            ) from exc

        self.device = device
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.quiet_load = quiet_load
        if quiet_load:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
                self.model = CLIPTextModelWithProjection.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                ).to(device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.model = CLIPTextModelWithProjection.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        text_outputs = self.model(**inputs)
        if hasattr(text_outputs, "text_embeds") and text_outputs.text_embeds is not None:
            text_features = text_outputs.text_embeds
        elif hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
            text_features = text_outputs.pooler_output
        elif hasattr(text_outputs, "last_hidden_state") and text_outputs.last_hidden_state is not None:
            text_features = text_outputs.last_hidden_state[:, 0, :]
        else:
            raise RuntimeError("CLIP text model did not return text_embeds, pooler_output, or last_hidden_state.")

        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return text_features.float()


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


class UWindowEncoder(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, hidden_dims: Sequence[int], latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dims[1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(torch.cat([x, condition], dim=-1))
        return self.mu(hidden), self.logvar(hidden)


class UWindowDecoder(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], output_dim),
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, condition], dim=-1))


class UWindowCVAE(nn.Module):
    def __init__(self, input_dim: int, text_feature_dim: int, condition_dim: int, hidden_dims: Sequence[int], latent_dim: int):
        super().__init__()
        self.text_projector = TextConditionProjector(text_feature_dim, condition_dim)
        self.encoder = UWindowEncoder(input_dim, condition_dim, hidden_dims, latent_dim)
        self.decoder = UWindowDecoder(latent_dim, condition_dim, hidden_dims, input_dim)

    def encode(self, x: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.text_projector(text_features)
        return self.encoder(x, condition)

    def decode(self, z: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        condition = self.text_projector(text_features)
        return self.decoder(z, condition)

    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, text_features)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decode(z, text_features)
        return recon, mu, logvar


class SavedUWindowEncoder(nn.Module):
    def __init__(self, input_dim: int, text_feature_dim: int, condition_dim: int, hidden_dims: Sequence[int], latent_dim: int):
        super().__init__()
        self.text_projector = TextConditionProjector(text_feature_dim, condition_dim)
        self.encoder = UWindowEncoder(input_dim, condition_dim, hidden_dims, latent_dim)

    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.text_projector(text_features)
        return self.encoder(x, condition)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device == "auto":
        device = "cuda"

    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for CVAE training, but torch.cuda.is_available() is False. "
                "Please run in a CUDA-enabled environment or pass --device cpu explicitly."
            )
        return device

    return device


def configure_cuda_backend(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _extract_first_json_value_by_key(text: str, key: str) -> Any | None:
    decoder = json.JSONDecoder()
    token = f'"{key}"'
    key_index = text.find(token)
    if key_index == -1:
        return None

    colon_index = text.find(":", key_index + len(token))
    if colon_index == -1:
        raise ValueError(f"Malformed JSON while searching for key '{key}'.")

    value_start = colon_index + 1
    while value_start < len(text) and text[value_start].isspace():
        value_start += 1

    value, _ = decoder.raw_decode(text, value_start)
    return value


def _extract_u_windows_from_json_text(json_path: Path) -> list[np.ndarray]:
    text = json_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    token = '"u_window"'
    pos = 0
    extracted: list[np.ndarray] = []

    while True:
        key_index = text.find(token, pos)
        if key_index == -1:
            break

        colon_index = text.find(":", key_index + len(token))
        if colon_index == -1:
            raise ValueError(f"Malformed JSON while searching for u_window in: {json_path}")

        value_start = colon_index + 1
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1

        value, value_end = decoder.raw_decode(text, value_start)
        extracted.append(np.asarray(value, dtype=np.float32))
        pos = value_end

    return extracted


def _extract_telemetry_metadata_from_json_text(json_path: Path, input_shape: tuple[int, int]) -> TelemetryMetadata:
    text = json_path.read_text(encoding="utf-8")
    u_t_mode_value = _extract_first_json_value_by_key(text, "u_t_mode")
    u_t_components_value = _extract_first_json_value_by_key(text, "u_t_components")
    u_t_dim_value = _extract_first_json_value_by_key(text, "u_t_dim")

    u_t_mode = str(u_t_mode_value) if isinstance(u_t_mode_value, str) else None
    u_t_components: tuple[str, ...] = ()
    if isinstance(u_t_components_value, list):
        u_t_components = tuple(str(value) for value in u_t_components_value)
    u_t_dim = int(u_t_dim_value) if isinstance(u_t_dim_value, int | float) else None

    if u_t_dim is not None and u_t_dim != input_shape[1]:
        raise ValueError(
            f"Telemetry metadata u_t_dim={u_t_dim} does not match extracted u_window feature dim={input_shape[1]} "
            f"for {json_path}."
        )
    if u_t_components and len(u_t_components) != input_shape[1]:
        raise ValueError(
            f"Telemetry metadata lists {len(u_t_components)} u_t components, but extracted u_window feature dim is "
            f"{input_shape[1]} for {json_path}."
        )

    return TelemetryMetadata(
        input_shape=input_shape,
        u_t_mode=u_t_mode,
        u_t_components=u_t_components,
        u_t_dim=u_t_dim,
        source_episode_id=json_path.stem,
    )


def extract_episode_u_windows(data_dir: Path) -> tuple[list[EpisodeWindows], TelemetryMetadata]:
    json_paths = sorted(data_dir.glob("episode_env*_idx*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No episode JSON files found under: {data_dir}")

    logger.info(f"Scanning {len(json_paths)} telemetry JSON files under: {data_dir}")

    episodes: list[EpisodeWindows] = []
    expected_shape: tuple[int, int] | None = None
    total_windows = 0
    metadata_source_path: Path | None = None

    for file_index, json_path in enumerate(json_paths, start=1):
        if file_index == 1 or file_index % 10 == 0 or file_index == len(json_paths):
            logger.info(f"Reading u_window values from file {file_index}/{len(json_paths)}: {json_path.name}")

        file_windows = _extract_u_windows_from_json_text(json_path)
        if not file_windows:
            continue

        validated_windows: list[np.ndarray] = []
        for entry_index, u_window_array in enumerate(file_windows):
            if u_window_array.ndim != 2:
                raise ValueError(
                    f"Expected u_window to have rank 2, got shape {u_window_array.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            current_shape = (int(u_window_array.shape[0]), int(u_window_array.shape[1]))
            if expected_shape is None:
                expected_shape = current_shape
            elif current_shape != expected_shape:
                raise ValueError(
                    f"Inconsistent u_window shape. Expected {expected_shape}, got {u_window_array.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            validated_windows.append(u_window_array)

        episode_array = np.stack(validated_windows, axis=0)
        episodes.append(EpisodeWindows(episode_id=json_path.stem, windows=episode_array))
        total_windows += int(episode_array.shape[0])
        if metadata_source_path is None:
            metadata_source_path = json_path

    if not episodes or expected_shape is None:
        raise ValueError(f"No valid u_window entries found under: {data_dir}")

    assert metadata_source_path is not None
    telemetry_metadata = _extract_telemetry_metadata_from_json_text(metadata_source_path, expected_shape)

    logger.info(
        f"Loaded {total_windows} u_window samples from {len(episodes)} episode files with shape {expected_shape}"
    )
    if telemetry_metadata.u_t_mode is not None:
        logger.info(
            f"Telemetry metadata: u_t_mode={telemetry_metadata.u_t_mode}, "
            f"u_t_dim={telemetry_metadata.u_t_dim or expected_shape[1]}, "
            f"source={telemetry_metadata.source_episode_id}"
        )
    return episodes, telemetry_metadata


def split_episode_indices(num_episodes: int, val_ratio: float, test_ratio: float, seed: int) -> dict[str, np.ndarray]:
    if num_episodes < 3:
        raise ValueError(f"Need at least 3 episodes for train/val/test split, got {num_episodes}.")
    if val_ratio < 0.0 or test_ratio < 0.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"Invalid split ratios: val_ratio={val_ratio}, test_ratio={test_ratio}. Expected val+test < 1.0."
        )

    indices = np.arange(num_episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    num_test = int(round(num_episodes * test_ratio))
    num_val = int(round(num_episodes * val_ratio))

    if test_ratio > 0.0 and num_test == 0:
        num_test = 1
    if val_ratio > 0.0 and num_val == 0:
        num_val = 1

    while num_test + num_val >= num_episodes:
        if num_test >= num_val and num_test > 0:
            num_test -= 1
        elif num_val > 0:
            num_val -= 1
        else:
            break

    num_train = num_episodes - num_val - num_test
    if num_train <= 0:
        raise ValueError(
            f"Episode split leaves no training data: num_episodes={num_episodes}, num_val={num_val}, num_test={num_test}."
        )

    train_indices = np.sort(indices[:num_train])
    val_indices = np.sort(indices[num_train : num_train + num_val])
    test_indices = np.sort(indices[num_train + num_val :])
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def flatten_episode_split(
    episodes: Sequence[EpisodeWindows],
    indices: Sequence[int],
    sample_shape: tuple[int, int],
) -> tuple[np.ndarray, list[str]]:
    selected = [episodes[index] for index in indices]
    episode_ids = [episode.episode_id for episode in selected]
    if not selected:
        return np.empty((0, sample_shape[0], sample_shape[1]), dtype=np.float32), episode_ids

    stacked = np.concatenate([episode.windows for episode in selected], axis=0).astype(np.float32, copy=False)
    return stacked, episode_ids


def create_run_paths(config: TrainConfig) -> RunPaths:
    timestamp = get_timestamp()
    run_dir = Path(config.output_root) / f"{timestamp}-{config.run_name}"
    wandb_dir = run_dir / ".wandb"
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        best_checkpoint_path=run_dir / "best.pt",
        last_checkpoint_path=run_dir / "last.pt",
        config_path=run_dir / "train_config.json",
        metrics_path=run_dir / "metrics.json",
        wandb_dir=wandb_dir,
    )


def save_config(config: TrainConfig, run_paths: RunPaths) -> None:
    with run_paths.config_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)
    logger.info(f"Saved CVAE config to: {run_paths.config_path}")


def init_wandb(config: TrainConfig, run_paths: RunPaths) -> Any | None:
    if not config.wandb_enabled:
        return None

    wandb = get_wandb()
    wandb_kwargs: dict[str, Any] = {
        "project": config.wandb_project,
        "name": run_paths.run_dir.name,
        "group": config.wandb_group,
        "config": asdict(config),
        "dir": str(run_paths.wandb_dir),
        "mode": config.wandb_mode,
        "tags": list(config.wandb_tags),
    }
    if config.wandb_entity:
        wandb_kwargs["entity"] = config.wandb_entity

    wandb.init(**wandb_kwargs)
    if hasattr(wandb, "define_metric"):
        wandb.define_metric("epoch")
        for pattern in (
            "train/*",
            "val/*",
            "best_test/*",
            "last_test/*",
            "compare/*",
        ):
            wandb.define_metric(pattern, step_metric="epoch")
    logger.info(f"Initialized W&B run in: {run_paths.wandb_dir}")
    return wandb


def estimate_tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)


def move_split_to_device(
    x: torch.Tensor,
    text_features: torch.Tensor,
    *,
    device: str,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_device = torch.device(device)
    if x.device == target_device and text_features.device == target_device:
        return x, text_features

    total_bytes = estimate_tensor_bytes(x, text_features)
    logger.info(
        f"Moving {split_name} tensors to {device} "
        f"({total_bytes / (1024 ** 2):.2f} MiB including text conditions)."
    )
    return (
        x.to(device=device, dtype=torch.float32, non_blocking=True),
        text_features.to(device=device, dtype=torch.float32, non_blocking=True),
    )


def iterate_minibatches(
    x: torch.Tensor,
    text_features: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
):
    if x.shape[0] != text_features.shape[0]:
        raise ValueError(
            f"Mismatched sample counts for x ({x.shape[0]}) and text_features ({text_features.shape[0]})."
        )

    num_samples = int(x.shape[0])
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        indices = torch.randperm(num_samples, generator=generator)
        if x.is_cuda:
            indices = indices.to(device=x.device, non_blocking=True)
    else:
        index_device = x.device if x.is_cuda else torch.device("cpu")
        indices = torch.arange(num_samples, device=index_device)

    for start in range(0, num_samples, batch_size):
        batch_indices = indices[start : start + batch_size]
        if batch_indices.device != x.device:
            batch_indices = batch_indices.to(device=x.device, non_blocking=True)
        yield x.index_select(0, batch_indices), text_features.index_select(0, batch_indices)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_encoder_only(
    model: UWindowCVAE,
    *,
    input_dim: int,
    text_feature_dim: int,
    condition_dim: int,
    hidden_dims: Sequence[int],
    latent_dim: int,
) -> SavedUWindowEncoder:
    encoder_only = SavedUWindowEncoder(
        input_dim=input_dim,
        text_feature_dim=text_feature_dim,
        condition_dim=condition_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
    )
    encoder_only.text_projector.load_state_dict(model.text_projector.state_dict())
    encoder_only.encoder.load_state_dict(model.encoder.state_dict())
    encoder_only.eval()
    return encoder_only


def make_encoder_checkpoint_payload(
    *,
    config: TrainConfig,
    input_shape: tuple[int, int],
    flattened_dim: int,
    num_samples: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    text_feature_dim: int,
    telemetry_metadata: TelemetryMetadata,
    encoder_only: SavedUWindowEncoder,
    checkpoint_type: str,
    epoch: int,
    val_loss_total: float,
) -> dict[str, Any]:
    return {
        "model_type": "u_window_cvae_encoder",
        "checkpoint_type": checkpoint_type,
        "epoch": epoch,
        "val_loss_total": val_loss_total,
        "config": asdict(config),
        "input_shape": [input_shape[0], input_shape[1]],
        "flattened_dim": int(flattened_dim),
        "num_samples": int(num_samples),
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "text_feature_dim": int(text_feature_dim),
        "condition_text": config.condition_text,
        "telemetry": asdict(telemetry_metadata),
        "clip": {
            "model_id": config.clip_model_id,
            "cache_dir": config.clip_cache_dir,
            "local_files_only": config.clip_local_files_only,
        },
        "encoder_state_dict": encoder_only.state_dict(),
    }


def save_encoder_checkpoint(
    checkpoint_path: Path,
    *,
    model: UWindowCVAE,
    config: TrainConfig,
    input_shape: tuple[int, int],
    flattened_dim: int,
    num_samples: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    text_feature_dim: int,
    telemetry_metadata: TelemetryMetadata,
    checkpoint_type: str,
    epoch: int,
    val_loss_total: float,
) -> None:
    encoder_only = build_encoder_only(
        model,
        input_dim=flattened_dim,
        text_feature_dim=text_feature_dim,
        condition_dim=config.condition_dim,
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
    )
    payload = make_encoder_checkpoint_payload(
        config=config,
        input_shape=input_shape,
        flattened_dim=flattened_dim,
        num_samples=num_samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
        text_feature_dim=text_feature_dim,
        telemetry_metadata=telemetry_metadata,
        encoder_only=encoder_only,
        checkpoint_type=checkpoint_type,
        epoch=epoch,
        val_loss_total=val_loss_total,
    )
    torch.save(payload, checkpoint_path)


@torch.no_grad()
def evaluate_model(
    model: UWindowCVAE,
    x: torch.Tensor,
    text_features: torch.Tensor,
    *,
    batch_size: int,
    kl_weight: float,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    device: str,
    prefix: str,
) -> dict[str, float | int]:
    if x.shape[0] == 0:
        return {
            f"{prefix}_num_samples": 0,
            f"{prefix}_loss_reconstruction": float("nan"),
            f"{prefix}_loss_kl": float("nan"),
            f"{prefix}_loss_total": float("nan"),
            f"{prefix}_value_mae": float("nan"),
            f"{prefix}_value_rmse": float("nan"),
            f"{prefix}_value_max_abs": float("nan"),
        }

    recon_loss_fn = nn.MSELoss(reduction="mean")
    mean_device = feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
    std_device = feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)

    model.eval()
    total_recon = 0.0
    total_kl = 0.0
    total_loss = 0.0
    total_abs = 0.0
    total_sq = 0.0
    max_abs = 0.0
    seen_samples = 0
    seen_values = 0

    for batch_x, batch_text in iterate_minibatches(
        x,
        text_features,
        batch_size,
        shuffle=False,
        seed=0,
    ):
        batch_x = batch_x.to(device=device, non_blocking=True)
        batch_text = batch_text.to(device=device, non_blocking=True)
        batch_size_current = batch_x.shape[0]

        recon, mu, logvar = model(batch_x, batch_text)
        recon_loss = recon_loss_fn(recon, batch_x)
        kl_loss = kl_divergence(mu, logvar)
        total_batch_loss = recon_loss + kl_weight * kl_loss

        recon_denorm = recon * std_device + mean_device
        target_denorm = batch_x * std_device + mean_device
        abs_diff = (recon_denorm - target_denorm).abs()
        sq_diff = (recon_denorm - target_denorm).pow(2)

        total_recon += recon_loss.item() * batch_size_current
        total_kl += kl_loss.item() * batch_size_current
        total_loss += total_batch_loss.item() * batch_size_current
        total_abs += abs_diff.sum().item()
        total_sq += sq_diff.sum().item()
        max_abs = max(max_abs, float(abs_diff.max().item()))
        seen_samples += batch_size_current
        seen_values += abs_diff.numel()

    return {
        f"{prefix}_num_samples": int(seen_samples),
        f"{prefix}_loss_reconstruction": total_recon / seen_samples,
        f"{prefix}_loss_kl": total_kl / seen_samples,
        f"{prefix}_loss_total": total_loss / seen_samples,
        f"{prefix}_value_mae": total_abs / seen_values,
        f"{prefix}_value_rmse": math.sqrt(total_sq / seen_values),
        f"{prefix}_value_max_abs": max_abs,
    }


def rename_metric_prefix(metrics: dict[str, float | int], old_prefix: str, new_prefix: str) -> dict[str, float | int]:
    renamed: dict[str, float | int] = {}
    for key, value in metrics.items():
        if key.startswith(old_prefix):
            renamed[new_prefix + key[len(old_prefix) :]] = value
        else:
            renamed[key] = value
    return renamed


def compute_metric_differences(
    last_metrics: dict[str, float | int],
    best_metrics: dict[str, float | int],
) -> dict[str, float]:
    differences: dict[str, float] = {}
    shared_keys = sorted(set(last_metrics.keys()) & set(best_metrics.keys()))
    for key in shared_keys:
        last_value = last_metrics[key]
        best_value = best_metrics[key]
        if isinstance(last_value, (int, float)) and isinstance(best_value, (int, float)):
            differences[f"{key}_last_minus_best"] = float(last_value) - float(best_value)
    return differences


def train_encoder(config: TrainConfig) -> Path:
    set_seed(config.seed)
    device = resolve_device(config.device)
    configure_cuda_backend(device)
    if str(device).startswith("cuda"):
        cuda_device = torch.device(device)
        cuda_index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
        logger.info(f"Using CUDA device for CVAE training: {torch.cuda.get_device_name(cuda_index)} ({device})")
    else:
        logger.info(f"Using device for CVAE training: {device}")
    run_paths = create_run_paths(config)
    save_config(config, run_paths)
    wandb = None
    metrics_history: list[dict[str, float | int]] = []

    try:
        data_dir = Path(config.data_dir)
        episodes, telemetry_metadata = extract_episode_u_windows(data_dir)
        sample_shape = telemetry_metadata.input_shape
        split_indices = split_episode_indices(len(episodes), config.val_ratio, config.test_ratio, config.seed)

        train_windows_np, train_episode_ids = flatten_episode_split(episodes, split_indices["train"], sample_shape)
        val_windows_np, val_episode_ids = flatten_episode_split(episodes, split_indices["val"], sample_shape)
        test_windows_np, test_episode_ids = flatten_episode_split(episodes, split_indices["test"], sample_shape)

        logger.info(
            f"Episode split with seed={config.seed}: train={len(train_episode_ids)} episodes, "
            f"val={len(val_episode_ids)} episodes, test={len(test_episode_ids)} episodes"
        )
        logger.info(
            f"Window split: train={train_windows_np.shape[0]}, val={val_windows_np.shape[0]}, test={test_windows_np.shape[0]}"
        )

        flattened_dim = sample_shape[0] * sample_shape[1]
        x_train_raw = torch.tensor(train_windows_np.reshape(train_windows_np.shape[0], -1), dtype=torch.float32)
        x_val_raw = torch.tensor(val_windows_np.reshape(val_windows_np.shape[0], -1), dtype=torch.float32)
        x_test_raw = torch.tensor(test_windows_np.reshape(test_windows_np.shape[0], -1), dtype=torch.float32)

        feature_mean = x_train_raw.mean(dim=0)
        feature_std = x_train_raw.std(dim=0).clamp_min(1e-6)

        x_train = (x_train_raw - feature_mean) / feature_std
        x_val = (x_val_raw - feature_mean) / feature_std
        x_test = (x_test_raw - feature_mean) / feature_std

        clip_text = CLIPTextFeatureExtractor(
            model_id=config.clip_model_id,
            device=device,
            cache_dir=config.clip_cache_dir,
            local_files_only=config.clip_local_files_only,
            quiet_load=config.clip_quiet_load,
        )
        logger.info(
            f"Loaded CLIP text encoder: model={config.clip_model_id}, local_files_only={config.clip_local_files_only}, "
            f"quiet_load={config.clip_quiet_load}"
        )
        base_text_feature = clip_text.encode([config.condition_text]).to(device=device, dtype=torch.float32)
        text_feature_dim = int(base_text_feature.shape[-1])

        train_text = base_text_feature.repeat(x_train.shape[0], 1)
        val_text = base_text_feature.repeat(x_val.shape[0], 1)
        test_text = base_text_feature.repeat(x_test.shape[0], 1)

        x_train, train_text = move_split_to_device(x_train, train_text, device=device, split_name="train")
        x_val, val_text = move_split_to_device(x_val, val_text, device=device, split_name="val")
        x_test, test_text = move_split_to_device(x_test, test_text, device=device, split_name="test")

        wandb = init_wandb(config, run_paths)

        model = UWindowCVAE(
            input_dim=flattened_dim,
            text_feature_dim=text_feature_dim,
            condition_dim=config.condition_dim,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        reconstruction_loss_fn = nn.MSELoss(reduction="mean")
        feature_mean_device = feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
        feature_std_device = feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)

        logger.info(
            f"Training CVAE on train/val/test windows = {x_train.shape[0]}/{x_val.shape[0]}/{x_test.shape[0]}, "
            f"input_shape={sample_shape}, latent_dim={config.latent_dim}, device={device}, seed={config.seed}, "
            f"clip_model={config.clip_model_id}"
        )
        if telemetry_metadata.u_t_mode is not None:
            logger.info(
                f"Training on telemetry u_t_mode={telemetry_metadata.u_t_mode}, "
                f"u_t_dim={telemetry_metadata.u_t_dim or sample_shape[1]}, "
                f"u_t_components={list(telemetry_metadata.u_t_components) if telemetry_metadata.u_t_components else 'n/a'}"
            )

        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state: dict[str, torch.Tensor] | None = None
        last_model_state: dict[str, torch.Tensor] | None = None

        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_recon = 0.0
            epoch_kl = 0.0
            epoch_total = 0.0
            epoch_abs = 0.0
            epoch_sq = 0.0
            epoch_max_abs = 0.0
            seen_samples = 0
            seen_values = 0

            for batch_x, batch_text in iterate_minibatches(
                x_train,
                train_text,
                config.batch_size,
                shuffle=True,
                seed=config.seed + epoch,
            ):
                batch_x = batch_x.to(device=device, non_blocking=True)
                batch_text = batch_text.to(device=device, non_blocking=True)
                batch_size_current = batch_x.shape[0]

                recon, mu, logvar = model(batch_x, batch_text)
                recon_loss = reconstruction_loss_fn(recon, batch_x)
                kl_loss = kl_divergence(mu, logvar)
                total_batch_loss = recon_loss + config.kl_weight * kl_loss

                optimizer.zero_grad(set_to_none=True)
                total_batch_loss.backward()
                optimizer.step()

                recon_denorm = recon * feature_std_device + feature_mean_device
                target_denorm = batch_x * feature_std_device + feature_mean_device
                abs_diff = (recon_denorm - target_denorm).abs()
                sq_diff = (recon_denorm - target_denorm).pow(2)

                epoch_recon += recon_loss.item() * batch_size_current
                epoch_kl += kl_loss.item() * batch_size_current
                epoch_total += total_batch_loss.item() * batch_size_current
                epoch_abs += abs_diff.sum().item()
                epoch_sq += sq_diff.sum().item()
                epoch_max_abs = max(epoch_max_abs, float(abs_diff.max().item()))
                seen_samples += batch_size_current
                seen_values += abs_diff.numel()

            train_metrics = {
                "train_num_samples": int(seen_samples),
                "train_loss_reconstruction": epoch_recon / seen_samples,
                "train_loss_kl": epoch_kl / seen_samples,
                "train_loss_total": epoch_total / seen_samples,
                "train_value_mae": epoch_abs / seen_values,
                "train_value_rmse": math.sqrt(epoch_sq / seen_values),
                "train_value_max_abs": epoch_max_abs,
            }
            val_metrics = evaluate_model(
                model,
                x_val,
                val_text,
                batch_size=config.batch_size,
                kl_weight=config.kl_weight,
                feature_mean=feature_mean,
                feature_std=feature_std,
                device=device,
                prefix="val",
            )

            current_val_loss = float(val_metrics["val_loss_total"])
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                best_model_state = clone_state_dict_to_cpu(model)
                save_encoder_checkpoint(
                    run_paths.best_checkpoint_path,
                    model=model,
                    config=config,
                    input_shape=sample_shape,
                    flattened_dim=flattened_dim,
                    num_samples=x_train.shape[0],
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    text_feature_dim=text_feature_dim,
                    telemetry_metadata=telemetry_metadata,
                    checkpoint_type="best",
                    epoch=epoch,
                    val_loss_total=current_val_loss,
                )

            epoch_metrics = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "train/reconstruction_loss": train_metrics["train_loss_reconstruction"],
                "train/kl_loss": train_metrics["train_loss_kl"],
                "train/total_loss": train_metrics["train_loss_total"],
                "train/value_mae": train_metrics["train_value_mae"],
                "train/value_rmse": train_metrics["train_value_rmse"],
                "val/reconstruction_loss": val_metrics["val_loss_reconstruction"],
                "val/kl_loss": val_metrics["val_loss_kl"],
                "val/total_loss": val_metrics["val_loss_total"],
                "val/value_mae": val_metrics["val_value_mae"],
                "val/value_rmse": val_metrics["val_value_rmse"],
                "best_val_loss_total": best_val_loss,
                "best_epoch": best_epoch,
            }
            metrics_history.append(epoch_metrics)

            if wandb is not None and wandb.run is not None:
                wandb.log(epoch_metrics, step=epoch)

            if epoch % config.log_interval == 0 or epoch == 1 or epoch == config.epochs:
                logger.info(
                    f"epoch={epoch:04d} "
                    f"train_recon={train_metrics['train_loss_reconstruction']:.6f} "
                    f"train_kl={train_metrics['train_loss_kl']:.6f} "
                    f"train_total={train_metrics['train_loss_total']:.6f} "
                    f"val_recon={val_metrics['val_loss_reconstruction']:.6f} "
                    f"val_kl={val_metrics['val_loss_kl']:.6f} "
                    f"val_total={val_metrics['val_loss_total']:.6f} "
                    f"best_val={best_val_loss:.6f} "
                    f"train_mae={train_metrics['train_value_mae']:.6f} "
                    f"val_mae={val_metrics['val_value_mae']:.6f}"
                )

        last_model_state = clone_state_dict_to_cpu(model)
        save_encoder_checkpoint(
            run_paths.last_checkpoint_path,
            model=model,
            config=config,
            input_shape=sample_shape,
            flattened_dim=flattened_dim,
            num_samples=x_train.shape[0],
            feature_mean=feature_mean,
            feature_std=feature_std,
            text_feature_dim=text_feature_dim,
            telemetry_metadata=telemetry_metadata,
            checkpoint_type="last",
            epoch=config.epochs,
            val_loss_total=float(metrics_history[-1]["val_loss_total"]),
        )

        if best_model_state is None:
            best_model_state = clone_state_dict_to_cpu(model)
            best_epoch = config.epochs
            best_val_loss = float(metrics_history[-1]["val_loss_total"])
            save_encoder_checkpoint(
                run_paths.best_checkpoint_path,
                model=model,
                config=config,
                input_shape=sample_shape,
                flattened_dim=flattened_dim,
                num_samples=x_train.shape[0],
                feature_mean=feature_mean,
                feature_std=feature_std,
                text_feature_dim=text_feature_dim,
                telemetry_metadata=telemetry_metadata,
                checkpoint_type="best",
                epoch=best_epoch,
                val_loss_total=best_val_loss,
            )

        model.load_state_dict(best_model_state)
        best_test_metrics = evaluate_model(
            model,
            x_test,
            test_text,
            batch_size=config.batch_size,
            kl_weight=config.kl_weight,
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=device,
            prefix="test",
        )

        model.load_state_dict(last_model_state)
        last_test_metrics = evaluate_model(
            model,
            x_test,
            test_text,
            batch_size=config.batch_size,
            kl_weight=config.kl_weight,
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=device,
            prefix="test",
        )

        test_differences = compute_metric_differences(last_test_metrics, best_test_metrics)
        best_test_metrics_named = rename_metric_prefix(best_test_metrics, "test_", "best_test_")
        last_test_metrics_named = rename_metric_prefix(last_test_metrics, "test_", "last_test_")

        summary = {
            "seed": config.seed,
            "input_shape": [sample_shape[0], sample_shape[1]],
            "num_episodes": len(episodes),
            "num_train_episodes": len(train_episode_ids),
            "num_val_episodes": len(val_episode_ids),
            "num_test_episodes": len(test_episode_ids),
            "num_train_windows": int(x_train.shape[0]),
            "num_val_windows": int(x_val.shape[0]),
            "num_test_windows": int(x_test.shape[0]),
            "telemetry": asdict(telemetry_metadata),
            "train_episode_ids": train_episode_ids,
            "val_episode_ids": val_episode_ids,
            "test_episode_ids": test_episode_ids,
            "best_epoch": best_epoch,
            "best_val_loss_total": best_val_loss,
            "best_checkpoint": str(run_paths.best_checkpoint_path),
            "last_checkpoint": str(run_paths.last_checkpoint_path),
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
            f"Test comparison: best_mae={best_test_metrics_named['best_test_value_mae']:.6f}, "
            f"last_mae={last_test_metrics_named['last_test_value_mae']:.6f}, "
            f"delta(last-best)={test_differences['test_value_mae_last_minus_best']:.6f}"
        )

        if wandb is not None and wandb.run is not None:
            final_log = {
                "epoch": config.epochs,
                "best_epoch": best_epoch,
                "best_val_loss_total": best_val_loss,
                "best_test/reconstruction_loss": best_test_metrics_named["best_test_loss_reconstruction"],
                "best_test/kl_loss": best_test_metrics_named["best_test_loss_kl"],
                "best_test/total_loss": best_test_metrics_named["best_test_loss_total"],
                "best_test/value_mae": best_test_metrics_named["best_test_value_mae"],
                "best_test/value_rmse": best_test_metrics_named["best_test_value_rmse"],
                "last_test/reconstruction_loss": last_test_metrics_named["last_test_loss_reconstruction"],
                "last_test/kl_loss": last_test_metrics_named["last_test_loss_kl"],
                "last_test/total_loss": last_test_metrics_named["last_test_loss_total"],
                "last_test/value_mae": last_test_metrics_named["last_test_value_mae"],
                "last_test/value_rmse": last_test_metrics_named["last_test_value_rmse"],
                "compare/test_total_loss_last_minus_best": test_differences.get("test_loss_total_last_minus_best", float("nan")),
                "compare/test_value_mae_last_minus_best": test_differences.get("test_value_mae_last_minus_best", float("nan")),
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


def load_encoder(checkpoint_path: str, device: str = "cpu") -> tuple[SavedUWindowEncoder, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    config_dict = payload["config"]
    encoder = SavedUWindowEncoder(
        input_dim=payload["flattened_dim"],
        text_feature_dim=payload["text_feature_dim"],
        condition_dim=config_dict["condition_dim"],
        hidden_dims=tuple(config_dict["hidden_dims"]),
        latent_dim=config_dict["latent_dim"],
    )
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()
    return encoder, payload


@torch.no_grad()
def encode_u_window_to_latent(
    checkpoint_path: str,
    u_window: np.ndarray | list,
    condition_text: str | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    encoder, payload = load_encoder(checkpoint_path, device=device)
    u_window_array = np.asarray(u_window, dtype=np.float32)
    expected_shape = tuple(payload["input_shape"])
    if tuple(u_window_array.shape) != expected_shape:
        raise ValueError(f"Expected u_window shape {expected_shape}, got {u_window_array.shape}")

    clip_cfg = payload["clip"]
    text_extractor = CLIPTextFeatureExtractor(
        model_id=clip_cfg["model_id"],
        device=device,
        cache_dir=clip_cfg["cache_dir"],
        local_files_only=clip_cfg["local_files_only"],
    )
    text_string = condition_text or payload["condition_text"]
    text_features = text_extractor.encode([text_string])

    x = torch.tensor(u_window_array.reshape(1, -1), dtype=torch.float32, device=device)
    feature_mean = payload["feature_mean"].to(device=device, dtype=torch.float32)
    feature_std = payload["feature_std"].to(device=device, dtype=torch.float32)
    x = (x - feature_mean.unsqueeze(0)) / feature_std.unsqueeze(0)

    mu, _ = encoder(x, text_features)
    return mu.squeeze(0).cpu()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a CLIP-conditioned CVAE encoder over IR telemetry u_window sequences.")
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--condition-text", type=str, default=TrainConfig.condition_text)
    parser.add_argument("--output-root", type=str, default=TrainConfig.output_root)
    parser.add_argument("--run-name", type=str, default=TrainConfig.run_name)
    parser.add_argument("--latent-dim", type=int, default=TrainConfig.latent_dim)
    parser.add_argument("--hidden-dim-1", type=int, default=TrainConfig.hidden_dims[0])
    parser.add_argument("--hidden-dim-2", type=int, default=TrainConfig.hidden_dims[1])
    parser.add_argument("--condition-dim", type=int, default=TrainConfig.condition_dim)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--kl-weight", type=float, default=TrainConfig.kl_weight)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=42)
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
        latent_dim=args.latent_dim,
        hidden_dims=(args.hidden_dim_1, args.hidden_dim_2),
        condition_dim=args.condition_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        log_interval=args.log_interval,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        wandb_enabled=wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_mode=wandb_mode,
        wandb_tags=tuple(args.wandb_tags),
        clip_model_id=args.clip_model_id,
        clip_cache_dir=args.clip_cache_dir,
        clip_local_files_only=bool(args.clip_local_files_only),
    )


def main() -> None:
    config = parse_args()
    train_encoder(config)


if __name__ == "__main__":
    main()
