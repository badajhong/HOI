# python src/holosoma/holosoma/cvae_ir_train.py \
# --data-dir /home/rllab/haechan/holosoma/logs/WholeBodyTracking/cvae_suitcase/telemetry \
# --ir-window-body-source all

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


DEFAULT_DATA_DIR = "/home/rllab/haechan/holosoma/logs/WholeBodyTracking/cvae_suitcase/telemetry"
DEFAULT_OUTPUT_ROOT = "/home/rllab/haechan/holosoma/logs/CVAE/"
DEFAULT_CONDITION_TEXT = "Push the suitcase, and set it back down."
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
U_WINDOW_BODY_SOURCE_CHOICES = ("all", "hands", "pelvis")


@dataclass
class TrainConfig:
    data_dir: str = DEFAULT_DATA_DIR
    condition_text: str = DEFAULT_CONDITION_TEXT
    ir_window_body_source: str = "all"
    output_root: str = DEFAULT_OUTPUT_ROOT
    run_name: str = "cvae-ir"
    latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 128)
    condition_dim: int = 16
    batch_size: int = 4096
    epochs: int = 30000
    learning_rate: float = 1e-3
    kl_weight: float = 1e-6
    kl_warmup_epochs: int = 1000
    """Linear KL weight warmup: ramp from 0 to kl_weight over this many epochs. 0 disables warmup."""
    lr_scheduler: str = "cosine_warmup"
    """LR scheduler: 'none', 'cosine', or 'cosine_warmup'. 'cosine' uses CosineAnnealingLR; 'cosine_warmup' adds linear warmup."""
    lr_warmup_epochs: int = 0
    """Number of warmup epochs for 'cosine_warmup' scheduler."""
    lr_min_factor: float = 0.01
    """Minimum LR as fraction of initial LR for cosine scheduler."""
    grad_clip_norm: float = 0
    """Max gradient norm for clipping. 0 disables clipping."""
    activation: str = "relu"
    """Activation function: 'relu', 'elu', or 'gelu'. GELU recommended for better gradient flow."""
    use_layer_norm: bool = True
    """Add LayerNorm after each hidden layer. Recommended for deeper networks (3+ hidden layers)."""
    dropout: float = 0
    """Dropout rate in MLP hidden layers. 0 disables. Recommended 0.05-0.1 for larger models."""
    free_bits: float = 0
    """Minimum KL per latent dimension (nats). Prevents posterior collapse while allowing looser regularization. Recommended 0.1-0.5."""
    recon_loss_type: str = "mse"
    """Reconstruction loss: 'mse', 'l1', 'huber', or 'mse_l1' (combined). 'mse_l1' often gives sharper reconstructions."""
    recon_loss_space: str = "original"
    """Space for reconstruction loss: 'normalized' optimizes z-scored data; 'original' directly optimizes raw-value RMSE/MAE."""
    l1_weight: float = 0.1
    """Weight for L1 component when recon_loss_type='mse_l1'."""
    min_feature_std: float = 1e-4
    """Minimum std used during feature normalization; avoids exploding nearly constant dimensions."""
    eval_latent_samples: int = 8
    """Number of z samples used to Monte-Carlo estimate validation/test reconstruction metrics."""
    best_val_metric: str = "val_value_rmse"
    """Validation metric used to save best checkpoint. Use val_value_rmse/mae when reconstruction quality matters most."""
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 100
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    wandb_enabled: bool = True
    wandb_project: str = "CVAE"
    wandb_entity: str | None = None
    wandb_group: str = "cvae_ir"
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ("cvae", "ir", "ir_window")
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
    ir_t_mode: str | None = None
    ir_t_components: tuple[str, ...] = ()
    ir_t_dim: int | None = None
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


def normalize_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(dim) for dim in hidden_dims)
    if not normalized:
        raise ValueError("hidden_dims must contain at least one hidden layer dimension.")
    if any(dim <= 0 for dim in normalized):
        raise ValueError(f"hidden_dims must be positive integers, got {normalized}.")
    return normalized


ACTIVATION_CHOICES = ("relu", "elu", "gelu")
RECON_LOSS_CHOICES = ("mse", "l1", "huber", "mse_l1")
RECON_LOSS_SPACE_CHOICES = ("normalized", "original")
BEST_VAL_METRIC_CHOICES = ("val_loss_total", "val_loss_reconstruction", "val_value_mae", "val_value_rmse")


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation '{name}'. Expected one of {ACTIVATION_CHOICES}.")


def make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    activation: str = "relu",
    use_layer_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    hidden_dims = normalize_hidden_dims(hidden_dims)
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_make_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    return nn.Sequential(*layers)


class IRWindowEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        *,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dims = normalize_hidden_dims(hidden_dims)
        self.net = make_mlp(
            input_dim + condition_dim, hidden_dims,
            activation=activation, use_layer_norm=use_layer_norm, dropout=dropout,
        )
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(torch.cat([x, condition], dim=-1))
        return self.mu(hidden), self.logvar(hidden)


class IRWindowDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        *,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dims = normalize_hidden_dims(hidden_dims)
        decoder_hidden_dims = tuple(reversed(hidden_dims))
        self.hidden = make_mlp(
            latent_dim + condition_dim, decoder_hidden_dims,
            activation=activation, use_layer_norm=use_layer_norm, dropout=dropout,
        )
        self.output = nn.Linear(decoder_hidden_dims[-1], output_dim)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(torch.cat([z, condition], dim=-1)))


class IRWindowCVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        text_feature_dim: int,
        condition_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        *,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.text_projector = TextConditionProjector(text_feature_dim, condition_dim)
        self.encoder = IRWindowEncoder(
            input_dim, condition_dim, hidden_dims, latent_dim,
            activation=activation, use_layer_norm=use_layer_norm, dropout=dropout,
        )
        self.decoder = IRWindowDecoder(
            latent_dim, condition_dim, hidden_dims, input_dim,
            activation=activation, use_layer_norm=use_layer_norm, dropout=dropout,
        )

    def encode(self, x: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.text_projector(text_features)
        return self.encoder(x, condition)

    def decode(self, z: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        condition = self.text_projector(text_features)
        return self.decoder(z, condition)

    def forward(
        self,
        x: torch.Tensor,
        text_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, text_features)
        z = sample_latent_z(mu, logvar)
        recon = self.decode(z, text_features)
        return recon, mu, logvar


class SavedIRWindowEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        text_feature_dim: int,
        condition_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        *,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.text_projector = TextConditionProjector(text_feature_dim, condition_dim)
        self.encoder = IRWindowEncoder(
            input_dim, condition_dim, hidden_dims, latent_dim,
            activation=activation, use_layer_norm=use_layer_norm, dropout=dropout,
        )

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


def normalize_ir_window_body_source(body_source: str) -> str:
    normalized = body_source.strip().lower()
    if normalized not in U_WINDOW_BODY_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported ir_window_body_source '{body_source}'. "
            f"Expected one of {U_WINDOW_BODY_SOURCE_CHOICES}."
        )
    return normalized


def _selected_component_indices_for_body_source(feature_dim: int, body_source: str) -> tuple[int, ...]:
    body_source = normalize_ir_window_body_source(body_source)
    if body_source == "all":
        return tuple(range(feature_dim))
    if body_source == "hands":
        if feature_dim == 39:
            # all-mode ir_t layout is [left_hand(13), right_hand(13), pelvis(13)].
            return tuple(range(26))
        if feature_dim == 26:
            return tuple(range(26))
    if body_source == "pelvis":
        if feature_dim == 39:
            # all-mode ir_t layout is [left_hand(13), right_hand(13), pelvis(13)].
            return tuple(range(26, 39))
        if feature_dim == 13:
            return tuple(range(13))

    raise ValueError(
        f"Cannot select ir_window_body_source='{body_source}' from ir_window feature dim={feature_dim}. "
        "Expected dims: all=39, hands=26, pelvis=13, or all-mode 39D telemetry."
    )


def _select_ir_window_body_source(ir_window_array: np.ndarray, body_source: str) -> np.ndarray:
    if ir_window_array.ndim != 2:
        return ir_window_array
    indices = _selected_component_indices_for_body_source(int(ir_window_array.shape[1]), body_source)
    if len(indices) == int(ir_window_array.shape[1]):
        return ir_window_array
    return ir_window_array[:, indices]


def _selected_component_names_for_body_source(
    ir_t_components: tuple[str, ...],
    input_feature_dim: int,
    body_source: str,
) -> tuple[str, ...]:
    body_source = normalize_ir_window_body_source(body_source)
    if not ir_t_components:
        return ()
    if body_source == "all":
        return ir_t_components
    prefixes = ("left_hand_", "right_hand_") if body_source == "hands" else ("pelvis_",)
    selected = tuple(component for component in ir_t_components if component.startswith(prefixes))
    if selected:
        return selected
    if len(ir_t_components) == input_feature_dim:
        # The source telemetry may already be pelvis-only or hands-only and use
        # unprefixed component names.
        return ir_t_components
    return selected


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


def _extract_ir_windows_from_json_text(json_path: Path) -> list[np.ndarray]:
    text = json_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    token = '"ir_window"'
    pos = 0
    extracted: list[np.ndarray] = []

    while True:
        key_index = text.find(token, pos)
        if key_index == -1:
            break

        colon_index = text.find(":", key_index + len(token))
        if colon_index == -1:
            raise ValueError(f"Malformed JSON while searching for ir_window in: {json_path}")

        value_start = colon_index + 1
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1

        value, value_end = decoder.raw_decode(text, value_start)
        extracted.append(np.asarray(value, dtype=np.float32))
        pos = value_end

    return extracted


def _extract_telemetry_metadata_from_json_text(
    json_path: Path,
    input_shape: tuple[int, int],
    ir_window_body_source: str = "all",
) -> TelemetryMetadata:
    text = json_path.read_text(encoding="utf-8")
    ir_t_mode_value = _extract_first_json_value_by_key(text, "ir_t_mode")
    ir_t_components_value = _extract_first_json_value_by_key(text, "ir_t_components")
    ir_t_dim_value = _extract_first_json_value_by_key(text, "ir_t_dim")

    ir_t_mode = str(ir_t_mode_value) if isinstance(ir_t_mode_value, str) else None
    ir_t_components: tuple[str, ...] = ()
    if isinstance(ir_t_components_value, list):
        ir_t_components = tuple(str(value) for value in ir_t_components_value)
    ir_t_dim = int(ir_t_dim_value) if isinstance(ir_t_dim_value, int | float) else None

    ir_window_body_source = normalize_ir_window_body_source(ir_window_body_source)
    if ir_t_components and len(ir_t_components) != input_shape[1]:
        selected_components = _selected_component_names_for_body_source(
            ir_t_components,
            input_feature_dim=input_shape[1],
            body_source=ir_window_body_source,
        )
        if len(selected_components) == input_shape[1]:
            logger.info(
                f"Telemetry metadata has {len(ir_t_components)} original ir_t components; "
                f"using {ir_window_body_source}-selected metadata with {len(selected_components)} components."
            )
            ir_t_components = selected_components
            ir_t_dim = input_shape[1]
    if ir_window_body_source != "all" and ir_t_dim is not None and ir_t_dim != input_shape[1]:
        logger.info(
            f"Telemetry metadata has original ir_t_dim={ir_t_dim}; "
            f"using {ir_window_body_source}-selected ir_t_dim={input_shape[1]}."
        )
        ir_t_dim = input_shape[1]
    if ir_window_body_source != "all" and ir_t_mode is not None:
        suffix = f"_{ir_window_body_source}_only"
        if not ir_t_mode.endswith(suffix):
            ir_t_mode = f"{ir_t_mode}{suffix}"

    if ir_t_dim is not None and ir_t_dim != input_shape[1]:
        raise ValueError(
            f"Telemetry metadata ir_t_dim={ir_t_dim} does not match extracted ir_window feature dim={input_shape[1]} "
            f"for {json_path}."
        )
    if ir_t_components and len(ir_t_components) != input_shape[1]:
        raise ValueError(
            f"Telemetry metadata lists {len(ir_t_components)} ir_t components, but extracted ir_window feature dim is "
            f"{input_shape[1]} for {json_path}."
        )

    return TelemetryMetadata(
        input_shape=input_shape,
        ir_t_mode=ir_t_mode,
        ir_t_components=ir_t_components,
        ir_t_dim=ir_t_dim,
        source_episode_id=json_path.stem,
    )


def extract_episode_ir_windows(
    data_dir: Path,
    ir_window_body_source: str = "all",
) -> tuple[list[EpisodeWindows], TelemetryMetadata]:
    ir_window_body_source = normalize_ir_window_body_source(ir_window_body_source)
    json_paths = sorted(data_dir.glob("episode_env*_idx*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No episode JSON files found under: {data_dir}")

    logger.info(
        f"Scanning {len(json_paths)} telemetry JSON files under: {data_dir} "
        f"with ir_window_body_source='{ir_window_body_source}'"
    )

    episodes: list[EpisodeWindows] = []
    expected_shape: tuple[int, int] | None = None
    total_windows = 0
    metadata_source_path: Path | None = None

    for file_index, json_path in enumerate(json_paths, start=1):
        if file_index == 1 or file_index % 10 == 0 or file_index == len(json_paths):
            logger.info(f"Reading ir_window values from file {file_index}/{len(json_paths)}: {json_path.name}")

        file_windows = _extract_ir_windows_from_json_text(json_path)
        if not file_windows:
            continue

        validated_windows: list[np.ndarray] = []
        for entry_index, ir_window_array in enumerate(file_windows):
            ir_window_array = _select_ir_window_body_source(ir_window_array, ir_window_body_source)
            if ir_window_array.ndim != 2:
                raise ValueError(
                    f"Expected ir_window to have rank 2, got shape {ir_window_array.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            current_shape = (int(ir_window_array.shape[0]), int(ir_window_array.shape[1]))
            if expected_shape is None:
                expected_shape = current_shape
            elif current_shape != expected_shape:
                raise ValueError(
                    f"Inconsistent ir_window shape. Expected {expected_shape}, got {ir_window_array.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            validated_windows.append(ir_window_array)

        episode_array = np.stack(validated_windows, axis=0)
        episodes.append(EpisodeWindows(episode_id=json_path.stem, windows=episode_array))
        total_windows += int(episode_array.shape[0])
        if metadata_source_path is None:
            metadata_source_path = json_path

    if not episodes or expected_shape is None:
        raise ValueError(f"No valid ir_window entries found under: {data_dir}")

    assert metadata_source_path is not None
    telemetry_metadata = _extract_telemetry_metadata_from_json_text(
        metadata_source_path,
        expected_shape,
        ir_window_body_source=ir_window_body_source,
    )

    logger.info(
        f"Loaded {total_windows} ir_window samples from {len(episodes)} episode files with shape {expected_shape} "
        f"for ir_window_body_source='{ir_window_body_source}'"
    )
    if telemetry_metadata.ir_t_mode is not None:
        logger.info(
            f"Telemetry metadata: ir_t_mode={telemetry_metadata.ir_t_mode}, "
            f"ir_t_dim={telemetry_metadata.ir_t_dim or expected_shape[1]}, "
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


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.mean()


def sample_latent_z(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def make_reconstruction_loss_fn(loss_type: str = "mse", l1_weight: float = 0.1):
    """Return a callable (recon, target) -> scalar loss."""
    if loss_type == "mse":
        return nn.MSELoss(reduction="mean")
    if loss_type == "l1":
        return nn.L1Loss(reduction="mean")
    if loss_type == "huber":
        return nn.SmoothL1Loss(reduction="mean")
    if loss_type == "mse_l1":
        mse_fn = nn.MSELoss(reduction="mean")
        l1_fn = nn.L1Loss(reduction="mean")
        _l1w = float(l1_weight)

        def _combined(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return mse_fn(recon, target) + _l1w * l1_fn(recon, target)

        return _combined
    raise ValueError(f"Unknown recon_loss_type '{loss_type}'. Expected one of {RECON_LOSS_CHOICES}.")


def compute_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    recon_loss_fn: Any,
    *,
    loss_space: str,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> torch.Tensor:
    if loss_space == "normalized":
        return recon_loss_fn(recon, target)
    if loss_space == "original":
        return recon_loss_fn(recon * feature_std + feature_mean, target * feature_std + feature_mean)
    raise ValueError(f"Unknown recon_loss_space '{loss_space}'. Expected one of {RECON_LOSS_SPACE_CHOICES}.")


def validate_config(config: TrainConfig) -> None:
    normalize_hidden_dims(config.hidden_dims)
    if config.activation not in ACTIVATION_CHOICES:
        raise ValueError(f"activation must be one of {ACTIVATION_CHOICES}, got {config.activation!r}.")
    if config.recon_loss_type not in RECON_LOSS_CHOICES:
        raise ValueError(f"recon_loss_type must be one of {RECON_LOSS_CHOICES}, got {config.recon_loss_type!r}.")
    if config.recon_loss_space not in RECON_LOSS_SPACE_CHOICES:
        raise ValueError(f"recon_loss_space must be one of {RECON_LOSS_SPACE_CHOICES}, got {config.recon_loss_space!r}.")
    if config.best_val_metric not in BEST_VAL_METRIC_CHOICES:
        raise ValueError(f"best_val_metric must be one of {BEST_VAL_METRIC_CHOICES}, got {config.best_val_metric!r}.")
    if config.lr_scheduler not in {"none", "cosine", "cosine_warmup"}:
        raise ValueError(f"lr_scheduler must be 'none', 'cosine', or 'cosine_warmup', got {config.lr_scheduler!r}.")
    if config.min_feature_std <= 0:
        raise ValueError(f"min_feature_std must be positive, got {config.min_feature_std}.")
    if config.lr_min_factor <= 0:
        raise ValueError(f"lr_min_factor must be positive, got {config.lr_min_factor}.")
    if config.lr_warmup_epochs < 0:
        raise ValueError(f"lr_warmup_epochs must be non-negative, got {config.lr_warmup_epochs}.")
    if config.kl_warmup_epochs < 0:
        raise ValueError(f"kl_warmup_epochs must be non-negative, got {config.kl_warmup_epochs}.")
    if config.kl_weight < 0:
        raise ValueError(f"kl_weight must be non-negative, got {config.kl_weight}.")
    if config.grad_clip_norm < 0:
        raise ValueError(f"grad_clip_norm must be non-negative, got {config.grad_clip_norm}.")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}.")
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.epochs}.")
    if config.eval_latent_samples <= 0:
        raise ValueError(f"eval_latent_samples must be positive, got {config.eval_latent_samples}.")


def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_encoder_only(
    model: IRWindowCVAE,
    *,
    input_dim: int,
    text_feature_dim: int,
    condition_dim: int,
    hidden_dims: Sequence[int],
    latent_dim: int,
    activation: str = "relu",
    use_layer_norm: bool = False,
    dropout: float = 0.0,
) -> SavedIRWindowEncoder:
    encoder_only = SavedIRWindowEncoder(
        input_dim=input_dim,
        text_feature_dim=text_feature_dim,
        condition_dim=condition_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation=activation,
        use_layer_norm=use_layer_norm,
        dropout=dropout,
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
    encoder_only: SavedIRWindowEncoder,
    cvae_state_dict: dict[str, torch.Tensor],
    checkpoint_type: str,
    epoch: int,
    val_loss_total: float,
) -> dict[str, Any]:
    return {
        "model_type": "ir_window_cvae_encoder",
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
        "cvae_state_dict": cvae_state_dict,
    }


def save_encoder_checkpoint(
    checkpoint_path: Path,
    *,
    model: IRWindowCVAE,
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
        activation=config.activation,
        use_layer_norm=config.use_layer_norm,
        dropout=config.dropout,
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
        cvae_state_dict=clone_state_dict_to_cpu(model),
        checkpoint_type=checkpoint_type,
        epoch=epoch,
        val_loss_total=val_loss_total,
    )
    torch.save(payload, checkpoint_path)


@torch.no_grad()
def evaluate_model(
    model: IRWindowCVAE,
    x: torch.Tensor,
    text_features: torch.Tensor,
    *,
    batch_size: int,
    kl_weight: float,
    free_bits: float = 0.0,
    recon_loss_fn: Any = None,
    recon_loss_space: str = "normalized",
    eval_latent_samples: int = 1,
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

    if recon_loss_fn is None:
        recon_loss_fn = nn.MSELoss(reduction="mean")
    eval_latent_samples = max(int(eval_latent_samples), 1)
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

        mu, logvar = model.encode(batch_x, batch_text)
        kl_loss = kl_divergence(mu, logvar, free_bits=free_bits)

        batch_recon = 0.0
        batch_abs = 0.0
        batch_sq = 0.0
        batch_max_abs = 0.0
        for _ in range(eval_latent_samples):
            z = sample_latent_z(mu, logvar)
            recon = model.decode(z, batch_text)
            recon_loss = compute_reconstruction_loss(
                recon,
                batch_x,
                recon_loss_fn,
                loss_space=recon_loss_space,
                feature_mean=mean_device,
                feature_std=std_device,
            )

            recon_denorm = recon * std_device + mean_device
            target_denorm = batch_x * std_device + mean_device
            abs_diff = (recon_denorm - target_denorm).abs()
            sq_diff = (recon_denorm - target_denorm).pow(2)

            batch_recon += recon_loss.item()
            batch_abs += abs_diff.sum().item()
            batch_sq += sq_diff.sum().item()
            batch_max_abs = max(batch_max_abs, float(abs_diff.max().item()))

        batch_recon /= eval_latent_samples
        batch_abs /= eval_latent_samples
        batch_sq /= eval_latent_samples
        total_batch_loss = batch_recon + kl_weight * kl_loss.item()

        total_recon += batch_recon * batch_size_current
        total_kl += kl_loss.item() * batch_size_current
        total_loss += total_batch_loss * batch_size_current
        total_abs += batch_abs
        total_sq += batch_sq
        max_abs = max(max_abs, batch_max_abs)
        seen_samples += batch_size_current
        seen_values += batch_x.numel()

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
    config.ir_window_body_source = normalize_ir_window_body_source(config.ir_window_body_source)
    validate_config(config)
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
        episodes, telemetry_metadata = extract_episode_ir_windows(
            data_dir,
            ir_window_body_source=config.ir_window_body_source,
        )
        num_episodes = len(episodes)
        sample_shape = telemetry_metadata.input_shape
        split_indices = split_episode_indices(num_episodes, config.val_ratio, config.test_ratio, config.seed)

        train_windows_np, train_episode_ids = flatten_episode_split(episodes, split_indices["train"], sample_shape)
        val_windows_np, val_episode_ids = flatten_episode_split(episodes, split_indices["val"], sample_shape)
        test_windows_np, test_episode_ids = flatten_episode_split(episodes, split_indices["test"], sample_shape)
        del episodes

        logger.info(
            f"Episode split with seed={config.seed}: train={len(train_episode_ids)} episodes, "
            f"val={len(val_episode_ids)} episodes, test={len(test_episode_ids)} episodes"
        )
        logger.info(
            f"Window split: train={train_windows_np.shape[0]}, val={val_windows_np.shape[0]}, test={test_windows_np.shape[0]}"
        )

        flattened_dim = sample_shape[0] * sample_shape[1]
        x_train_raw = torch.from_numpy(train_windows_np.reshape(train_windows_np.shape[0], -1)).to(dtype=torch.float32)
        x_val_raw = torch.from_numpy(val_windows_np.reshape(val_windows_np.shape[0], -1)).to(dtype=torch.float32)
        x_test_raw = torch.from_numpy(test_windows_np.reshape(test_windows_np.shape[0], -1)).to(dtype=torch.float32)
        del train_windows_np, val_windows_np, test_windows_np

        feature_mean = x_train_raw.mean(dim=0)
        raw_feature_std = x_train_raw.std(dim=0)
        feature_std = raw_feature_std.clamp_min(config.min_feature_std)
        low_std_count = int((raw_feature_std < config.min_feature_std).sum().item())
        logger.info(
            f"Feature normalization: min_feature_std={config.min_feature_std:g}, "
            f"raw_std_min={raw_feature_std.min().item():.6g}, "
            f"raw_std_mean={raw_feature_std.mean().item():.6g}, "
            f"raw_std_max={raw_feature_std.max().item():.6g}, "
            f"clamped_dims={low_std_count}/{raw_feature_std.numel()}"
        )

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

        model = IRWindowCVAE(
            input_dim=flattened_dim,
            text_feature_dim=text_feature_dim,
            condition_dim=config.condition_dim,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
            dropout=config.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        reconstruction_loss_fn = make_reconstruction_loss_fn(config.recon_loss_type, config.l1_weight)

        # LR scheduler
        scheduler = None
        if config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs, eta_min=config.learning_rate * config.lr_min_factor
            )
        elif config.lr_scheduler == "cosine_warmup":
            if config.lr_warmup_epochs == 0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config.epochs, eta_min=config.learning_rate * config.lr_min_factor
                )
            else:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=config.lr_min_factor, total_iters=config.lr_warmup_epochs
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(config.epochs - config.lr_warmup_epochs, 1),
                    eta_min=config.learning_rate * config.lr_min_factor,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[config.lr_warmup_epochs],
                )

        feature_mean_device = feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
        feature_std_device = feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)

        logger.info(
            f"Training CVAE on train/val/test windows = {x_train.shape[0]}/{x_val.shape[0]}/{x_test.shape[0]}, "
            f"input_shape={sample_shape}, latent_dim={config.latent_dim}, device={device}, seed={config.seed}, "
            f"hidden_dims={config.hidden_dims}, clip_model={config.clip_model_id}, "
            f"ir_window_body_source={config.ir_window_body_source}, "
            f"kl_warmup_epochs={config.kl_warmup_epochs}, lr_scheduler={config.lr_scheduler}, "
            f"grad_clip_norm={config.grad_clip_norm}, recon_loss={config.recon_loss_type}/{config.recon_loss_space}, "
            f"latent=z, eval_latent_samples={config.eval_latent_samples}, best_val_metric={config.best_val_metric}"
        )
        if telemetry_metadata.ir_t_mode is not None:
            logger.info(
                f"Training on telemetry ir_t_mode={telemetry_metadata.ir_t_mode}, "
                f"ir_t_dim={telemetry_metadata.ir_t_dim or sample_shape[1]}, "
                f"ir_t_components={list(telemetry_metadata.ir_t_components) if telemetry_metadata.ir_t_components else 'n/a'}"
            )

        best_val_loss_total = float("inf")
        best_val_metric_value = float("inf")
        best_epoch = 0
        best_model_state: dict[str, torch.Tensor] | None = None
        last_model_state: dict[str, torch.Tensor] | None = None

        for epoch in range(1, config.epochs + 1):
            # KL warmup: linearly ramp kl_weight from 0 to target over warmup epochs
            if config.kl_warmup_epochs > 0 and epoch <= config.kl_warmup_epochs:
                effective_kl_weight = config.kl_weight * (epoch / config.kl_warmup_epochs)
            else:
                effective_kl_weight = config.kl_weight

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
                recon_loss = compute_reconstruction_loss(
                    recon,
                    batch_x,
                    reconstruction_loss_fn,
                    loss_space=config.recon_loss_space,
                    feature_mean=feature_mean_device,
                    feature_std=feature_std_device,
                )
                kl_loss = kl_divergence(mu, logvar, free_bits=config.free_bits)
                total_batch_loss = recon_loss + effective_kl_weight * kl_loss

                optimizer.zero_grad(set_to_none=True)
                total_batch_loss.backward()
                if config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()

                with torch.no_grad():
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

            if scheduler is not None:
                scheduler.step()

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
                free_bits=config.free_bits,
                recon_loss_fn=reconstruction_loss_fn,
                recon_loss_space=config.recon_loss_space,
                eval_latent_samples=config.eval_latent_samples,
                feature_mean=feature_mean,
                feature_std=feature_std,
                device=device,
                prefix="val",
            )

            current_val_loss_total = float(val_metrics["val_loss_total"])
            current_val_metric_value = float(val_metrics[config.best_val_metric])
            if current_val_metric_value < best_val_metric_value:
                best_val_metric_value = current_val_metric_value
                best_val_loss_total = current_val_loss_total
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
                    val_loss_total=current_val_loss_total,
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
                "best_val_loss_total": best_val_loss_total,
                "best_val_metric_value": best_val_metric_value,
                "best_epoch": best_epoch,
                "effective_kl_weight": effective_kl_weight,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "eval_latent_samples": config.eval_latent_samples,
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
                    f"best_{config.best_val_metric}={best_val_metric_value:.6f} "
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
            best_val_loss_total = float(metrics_history[-1]["val_loss_total"])
            best_val_metric_value = float(metrics_history[-1][config.best_val_metric])
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
                val_loss_total=best_val_loss_total,
            )

        model.load_state_dict(best_model_state)
        best_test_metrics = evaluate_model(
            model,
            x_test,
            test_text,
            batch_size=config.batch_size,
            kl_weight=config.kl_weight,
            free_bits=config.free_bits,
            recon_loss_fn=reconstruction_loss_fn,
            recon_loss_space=config.recon_loss_space,
            eval_latent_samples=config.eval_latent_samples,
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
            free_bits=config.free_bits,
            recon_loss_fn=reconstruction_loss_fn,
            recon_loss_space=config.recon_loss_space,
            eval_latent_samples=config.eval_latent_samples,
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
            "num_episodes": num_episodes,
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
            "best_val_loss_total": best_val_loss_total,
            "best_val_metric": config.best_val_metric,
            "best_val_metric_value": best_val_metric_value,
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
                "best_val_loss_total": best_val_loss_total,
                "best_val_metric_value": best_val_metric_value,
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


def load_encoder(checkpoint_path: str, device: str = "cpu") -> tuple[SavedIRWindowEncoder, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    config_dict = payload["config"]
    encoder = SavedIRWindowEncoder(
        input_dim=payload["flattened_dim"],
        text_feature_dim=payload["text_feature_dim"],
        condition_dim=config_dict["condition_dim"],
        hidden_dims=tuple(config_dict["hidden_dims"]),
        latent_dim=config_dict["latent_dim"],
        activation=config_dict.get("activation", "relu"),
        use_layer_norm=config_dict.get("use_layer_norm", False),
        dropout=config_dict.get("dropout", 0.0),
    )
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()
    return encoder, payload


def load_cvae(checkpoint_path: str, device: str = "cpu") -> tuple[IRWindowCVAE, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_key = "cvae_state_dict" if "cvae_state_dict" in payload else "model_state_dict"
    if state_key not in payload:
        raise ValueError(
            f"IR-CVAE checkpoint '{checkpoint_path}' does not contain a frozen decoder. "
            "Retrain or resave the IR-CVAE with the updated cvae_ir_train.py so the checkpoint includes "
            "'cvae_state_dict'. Existing encoder-only checkpoints can still provide latent targets, "
            "but they cannot decode DI latents back to ir_window values for MAE/RMSE validation."
        )

    config_dict = payload["config"]
    model = IRWindowCVAE(
        input_dim=payload["flattened_dim"],
        text_feature_dim=payload["text_feature_dim"],
        condition_dim=config_dict["condition_dim"],
        hidden_dims=tuple(config_dict["hidden_dims"]),
        latent_dim=config_dict["latent_dim"],
        activation=config_dict.get("activation", "relu"),
        use_layer_norm=config_dict.get("use_layer_norm", False),
        dropout=config_dict.get("dropout", 0.0),
    )
    model.load_state_dict(payload[state_key])
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, payload


@torch.no_grad()
def encode_ir_window_to_latent(
    checkpoint_path: str,
    ir_window: np.ndarray | list,
    condition_text: str | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    encoder, payload = load_encoder(checkpoint_path, device=device)
    ir_window_array = np.asarray(ir_window, dtype=np.float32)
    expected_shape = tuple(payload["input_shape"])
    if tuple(ir_window_array.shape) != expected_shape:
        raise ValueError(f"Expected ir_window shape {expected_shape}, got {ir_window_array.shape}")

    clip_cfg = payload["clip"]
    text_extractor = CLIPTextFeatureExtractor(
        model_id=clip_cfg["model_id"],
        device=device,
        cache_dir=clip_cfg["cache_dir"],
        local_files_only=clip_cfg["local_files_only"],
    )
    text_string = condition_text or payload["condition_text"]
    text_features = text_extractor.encode([text_string])

    x = torch.tensor(ir_window_array.reshape(1, -1), dtype=torch.float32, device=device)
    feature_mean = payload["feature_mean"].to(device=device, dtype=torch.float32)
    feature_std = payload["feature_std"].to(device=device, dtype=torch.float32)
    x = (x - feature_mean.unsqueeze(0)) / feature_std.unsqueeze(0)

    mu, logvar = encoder(x, text_features)
    latent = sample_latent_z(mu, logvar)
    return latent.squeeze(0).cpu()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a CLIP-conditioned CVAE encoder over IR telemetry ir_window sequences.")
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--condition-text", type=str, default=TrainConfig.condition_text)
    parser.add_argument(
        "--ir-window-body-source",
        type=str,
        default=TrainConfig.ir_window_body_source,
        choices=U_WINDOW_BODY_SOURCE_CHOICES,
        help="Which body subset to train on from ir_window: all, hands, or pelvis.",
    )
    parser.add_argument("--output-root", type=str, default=TrainConfig.output_root)
    parser.add_argument("--run-name", type=str, default=TrainConfig.run_name)
    parser.add_argument("--latent-dim", type=int, default=TrainConfig.latent_dim)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer dimensions for the IR-CVAE MLP, e.g. --hidden-dims 256 128 64.",
    )
    parser.add_argument(
        "--hidden-dim-1",
        type=int,
        default=None,
        help="Deprecated alias for the first hidden dim. Prefer --hidden-dims.",
    )
    parser.add_argument(
        "--hidden-dim-2",
        type=int,
        default=None,
        help="Deprecated alias for the second hidden dim. Prefer --hidden-dims.",
    )
    parser.add_argument("--condition-dim", type=int, default=TrainConfig.condition_dim)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--kl-weight", type=float, default=TrainConfig.kl_weight)
    parser.add_argument(
        "--kl-warmup-epochs", type=int, default=TrainConfig.kl_warmup_epochs,
        help="Linear KL weight warmup from 0 to kl_weight over this many epochs. 0 disables.",
    )
    parser.add_argument(
        "--lr-scheduler", type=str, default=TrainConfig.lr_scheduler,
        choices=["none", "cosine", "cosine_warmup"],
        help="LR scheduler: none, cosine, or cosine_warmup.",
    )
    parser.add_argument("--lr-warmup-epochs", type=int, default=TrainConfig.lr_warmup_epochs)
    parser.add_argument("--lr-min-factor", type=float, default=TrainConfig.lr_min_factor)
    parser.add_argument(
        "--grad-clip-norm", type=float, default=TrainConfig.grad_clip_norm,
        help="Max gradient norm for clipping. 0 disables.",
    )
    parser.add_argument(
        "--activation", type=str, default=TrainConfig.activation,
        choices=list(ACTIVATION_CHOICES),
        help="MLP activation function. GELU recommended for better gradient flow.",
    )
    parser.add_argument(
        "--use-layer-norm", type=str_to_bool, default=TrainConfig.use_layer_norm,
        help="Add LayerNorm after each hidden layer. Recommended for deeper networks.",
    )
    parser.add_argument(
        "--dropout", type=float, default=TrainConfig.dropout,
        help="Dropout rate in MLP hidden layers. 0 disables. Recommended 0.05-0.1.",
    )
    parser.add_argument(
        "--free-bits", type=float, default=TrainConfig.free_bits,
        help="Minimum KL per latent dim (nats). Prevents posterior collapse. Recommended 0.1-0.5.",
    )
    parser.add_argument(
        "--recon-loss-type", type=str, default=TrainConfig.recon_loss_type,
        choices=list(RECON_LOSS_CHOICES),
        help="Reconstruction loss type. 'mse_l1' combines MSE + weighted L1 for sharper results.",
    )
    parser.add_argument(
        "--recon-loss-space",
        type=str,
        default=TrainConfig.recon_loss_space,
        choices=list(RECON_LOSS_SPACE_CHOICES),
        help="Compute reconstruction loss in normalized space or original raw-value space.",
    )
    parser.add_argument(
        "--l1-weight", type=float, default=TrainConfig.l1_weight,
        help="Weight for L1 component when recon-loss-type='mse_l1'.",
    )
    parser.add_argument("--min-feature-std", type=float, default=TrainConfig.min_feature_std)
    parser.add_argument(
        "--eval-latent-samples",
        type=int,
        default=TrainConfig.eval_latent_samples,
        help="Number of sampled z draws used to Monte-Carlo average validation/test reconstruction metrics.",
    )
    parser.add_argument(
        "--best-val-metric",
        type=str,
        default=TrainConfig.best_val_metric,
        choices=list(BEST_VAL_METRIC_CHOICES),
        help="Validation metric used to select the best checkpoint.",
    )
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
    if args.hidden_dims is not None:
        hidden_dims = normalize_hidden_dims(args.hidden_dims)
    elif args.hidden_dim_1 is not None or args.hidden_dim_2 is not None:
        hidden_dims = normalize_hidden_dims(
            (
                args.hidden_dim_1 if args.hidden_dim_1 is not None else TrainConfig.hidden_dims[0],
                args.hidden_dim_2 if args.hidden_dim_2 is not None else TrainConfig.hidden_dims[1],
            )
        )
    else:
        hidden_dims = TrainConfig.hidden_dims

    return TrainConfig(
        data_dir=args.data_dir,
        condition_text=args.condition_text,
        ir_window_body_source=normalize_ir_window_body_source(args.ir_window_body_source),
        output_root=args.output_root,
        run_name=args.run_name,
        latent_dim=args.latent_dim,
        hidden_dims=hidden_dims,
        condition_dim=args.condition_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        kl_warmup_epochs=args.kl_warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_min_factor=args.lr_min_factor,
        grad_clip_norm=args.grad_clip_norm,
        activation=args.activation,
        use_layer_norm=bool(args.use_layer_norm),
        dropout=args.dropout,
        free_bits=args.free_bits,
        recon_loss_type=args.recon_loss_type,
        recon_loss_space=args.recon_loss_space,
        l1_weight=args.l1_weight,
        min_feature_std=args.min_feature_std,
        eval_latent_samples=args.eval_latent_samples,
        best_val_metric=args.best_val_metric,
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
