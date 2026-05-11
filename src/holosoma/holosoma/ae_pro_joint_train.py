from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from loguru import logger
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma.utils.experiment_paths import get_timestamp
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.wandb import get_wandb

DEFAULT_DATA_DIR = "/home/rllab/haechan/holosoma/logs/ir_di_pro_suitcase/20260508_ae_train/telemetry"
DEFAULT_OUTPUT_ROOT = "/home/rllab/haechan/holosoma/logs/AE/"
DEFAULT_CONDITION_TEXT = "Push the suitcase, and set it back down."
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
IR_WINDOW_BODY_SOURCE_CHOICES = ("all", "hands", "pelvis")


@dataclass
class EpisodePairedWindows:
    episode_id: str
    ir_windows: np.ndarray
    depth_windows: np.ndarray
    proprioception_windows: np.ndarray | None = None


@dataclass
class TelemetryMetadata:
    depth_input_shape: tuple[int, int, int]
    ir_window_shape: tuple[int, int]
    ir_window_body_source: str = "all"
    proprioception_window_shape: tuple[int, int] | None = None
    proprioception_term_dims: dict[str, int] | None = None
    proprioception_dof_names: tuple[str, ...] = ()
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


def iterate_batch_indices(num_samples: int, batch_size: int, *, shuffle: bool, seed: int):
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        indices = torch.randperm(num_samples, generator=generator)
    else:
        indices = torch.arange(num_samples, dtype=torch.long)

    for start in range(0, num_samples, batch_size):
        yield indices[start : start + batch_size]


def _select_ir_windows_body_source(ir_windows_array: np.ndarray, body_source: str) -> np.ndarray:
    if ir_windows_array.ndim != 3:
        return ir_windows_array
    indices = _selected_component_indices_for_body_source(int(ir_windows_array.shape[2]), body_source)
    if len(indices) == int(ir_windows_array.shape[2]):
        return ir_windows_array
    return ir_windows_array[:, :, indices]


def _build_telemetry_metadata(
    payload: dict[str, Any],
    *,
    expected_ir_shape: tuple[int, int],
    expected_depth_shape: tuple[int, int, int],
    ir_window_body_source: str,
    expected_proprioception_shape: tuple[int, int] | None,
    dataset_has_proprioception: bool | None,
    source_id: str,
) -> TelemetryMetadata:
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
            f"{expected_ir_shape[1]} for {source_id}."
        )
    if ir_t_components and len(ir_t_components) != expected_ir_shape[1]:
        raise ValueError(
            f"Telemetry metadata lists {len(ir_t_components)} ir_t components, but extracted ir_window "
            f"feature dim is {expected_ir_shape[1]} for {source_id}."
        )

    depth_resolution_value = payload.get("depth_resolution")
    depth_resolution = None
    if isinstance(depth_resolution_value, list) and len(depth_resolution_value) == 2:
        depth_resolution = (int(depth_resolution_value[0]), int(depth_resolution_value[1]))

    proprioception_window_shape = None
    proprioception_window_shape_value = payload.get("proprioception_window_shape")
    if dataset_has_proprioception:
        if isinstance(proprioception_window_shape_value, list) and len(proprioception_window_shape_value) == 2:
            proprioception_window_shape = (
                int(proprioception_window_shape_value[0]),
                int(proprioception_window_shape_value[1]),
            )
        else:
            proprioception_window_shape = expected_proprioception_shape
        if expected_proprioception_shape is not None and proprioception_window_shape != expected_proprioception_shape:
            raise ValueError(
                "Telemetry metadata proprioception_window_shape does not match extracted tensor shape. "
                f"metadata={proprioception_window_shape}, extracted={expected_proprioception_shape}, "
                f"source={source_id}"
            )

    proprioception_term_dims_value = payload.get("proprioception_term_dims")
    proprioception_term_dims = None
    if isinstance(proprioception_term_dims_value, dict):
        proprioception_term_dims = {
            str(key): int(value)
            for key, value in proprioception_term_dims_value.items()
        }

    proprioception_dof_names_value = payload.get("proprioception_dof_names")
    proprioception_dof_names: tuple[str, ...] = ()
    if isinstance(proprioception_dof_names_value, list):
        proprioception_dof_names = tuple(str(value) for value in proprioception_dof_names_value)

    return TelemetryMetadata(
        depth_input_shape=expected_depth_shape,
        ir_window_shape=expected_ir_shape,
        ir_window_body_source=ir_window_body_source,
        proprioception_window_shape=proprioception_window_shape,
        proprioception_term_dims=proprioception_term_dims,
        proprioception_dof_names=proprioception_dof_names,
        depth_resolution=depth_resolution,
        ir_t_mode=ir_t_mode,
        ir_t_components=ir_t_components,
        ir_t_dim=ir_t_dim,
        source_episode_id=source_id,
    )


def extract_episode_paired_windows(
    data_dir: Path,
    ir_window_body_source: str = "all",
) -> tuple[list[EpisodePairedWindows], TelemetryMetadata]:
    ir_window_body_source = normalize_ir_window_body_source(ir_window_body_source)
    hdf5_path = data_dir / "telemetry.h5"
    if hdf5_path.exists():
        try:
            import h5py  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"Found HDF5 telemetry at {hdf5_path}, but h5py is not installed. "
                "Install h5py to read telemetry.h5."
            ) from exc

        logger.info(
            f"Scanning HDF5 telemetry file: {hdf5_path} "
            f"with ir_window_body_source='{ir_window_body_source}'"
        )
        episodes: list[EpisodePairedWindows] = []
        expected_ir_shape: tuple[int, int] | None = None
        expected_depth_shape: tuple[int, int, int] | None = None
        expected_proprioception_shape: tuple[int, int] | None = None
        dataset_has_proprioception: bool | None = None
        total_windows = 0
        metadata: TelemetryMetadata | None = None

        with h5py.File(hdf5_path, "r") as hdf5_file:
            if "episodes" not in hdf5_file:
                raise ValueError(f"HDF5 telemetry file does not contain an 'episodes' group: {hdf5_path}")
            episodes_group = hdf5_file["episodes"]
            group_names = sorted(str(name) for name in episodes_group.keys())
            if not group_names:
                raise ValueError(f"HDF5 telemetry file contains no episode groups: {hdf5_path}")

            for file_index, group_name in enumerate(group_names, start=1):
                if file_index == 1 or file_index % 50 == 0 or file_index == len(group_names):
                    logger.info(
                        f"Reading paired ir_window/depth_window arrays from HDF5 group "
                        f"{file_index}/{len(group_names)}: {group_name}"
                    )
                group = episodes_group[group_name]
                if "ir_windows" not in group or "depth_windows" not in group:
                    raise ValueError(f"Expected ir_windows and depth_windows datasets in HDF5 group {group.name}.")
                ir_windows = np.asarray(group["ir_windows"][()], dtype=np.float32)
                ir_windows = _select_ir_windows_body_source(ir_windows, ir_window_body_source)
                depth_windows = np.asarray(group["depth_windows"][()], dtype=np.float32)
                has_proprioception_window = "proprioception_windows" in group
                proprioception_windows = (
                    np.asarray(group["proprioception_windows"][()], dtype=np.float32)
                    if has_proprioception_window
                    else None
                )
                metadata_json = group.attrs.get("metadata_json", "{}")
                if isinstance(metadata_json, bytes):
                    metadata_json = metadata_json.decode("utf-8")
                payload = json.loads(str(metadata_json))

                if dataset_has_proprioception is None:
                    dataset_has_proprioception = has_proprioception_window
                elif dataset_has_proprioception != has_proprioception_window:
                    raise ValueError(
                        "Mixed telemetry is not supported: some HDF5 groups include proprioception_windows "
                        f"but others do not. First mismatch found in {group.name}."
                    )
                if ir_windows.ndim != 3:
                    raise ValueError(
                        f"Expected ir_windows to have rank 3, got shape {ir_windows.shape} in {group.name}."
                    )
                if depth_windows.ndim != 4:
                    raise ValueError(
                        f"Expected depth_windows to have rank 4, got shape {depth_windows.shape} in {group.name}."
                    )
                if int(ir_windows.shape[0]) != int(depth_windows.shape[0]):
                    raise ValueError(
                        f"HDF5 sample count mismatch in {group.name}: "
                        f"ir={ir_windows.shape[0]}, depth={depth_windows.shape[0]}."
                    )
                if proprioception_windows is not None:
                    if proprioception_windows.ndim != 3:
                        raise ValueError(
                            "Expected proprioception_windows to have rank 3, "
                            f"got shape {proprioception_windows.shape} in {group.name}."
                        )
                    if int(proprioception_windows.shape[0]) != int(ir_windows.shape[0]):
                        raise ValueError(
                            f"HDF5 sample count mismatch in {group.name}: "
                            f"ir={ir_windows.shape[0]}, proprioception={proprioception_windows.shape[0]}."
                        )

                current_ir_shape = (int(ir_windows.shape[1]), int(ir_windows.shape[2]))
                current_depth_shape = (
                    int(depth_windows.shape[1]),
                    int(depth_windows.shape[2]),
                    int(depth_windows.shape[3]),
                )
                if expected_ir_shape is None:
                    expected_ir_shape = current_ir_shape
                elif current_ir_shape != expected_ir_shape:
                    raise ValueError(
                        f"Inconsistent ir_windows shape. Expected {expected_ir_shape}, got {current_ir_shape} "
                        f"in {group.name}."
                    )
                if expected_depth_shape is None:
                    expected_depth_shape = current_depth_shape
                elif current_depth_shape != expected_depth_shape:
                    raise ValueError(
                        f"Inconsistent depth_windows shape. Expected {expected_depth_shape}, got {current_depth_shape} "
                        f"in {group.name}."
                    )
                if proprioception_windows is not None:
                    current_proprioception_shape = (
                        int(proprioception_windows.shape[1]),
                        int(proprioception_windows.shape[2]),
                    )
                    if expected_proprioception_shape is None:
                        expected_proprioception_shape = current_proprioception_shape
                    elif current_proprioception_shape != expected_proprioception_shape:
                        raise ValueError(
                            "Inconsistent proprioception_windows shape. "
                            f"Expected {expected_proprioception_shape}, got {current_proprioception_shape} "
                            f"in {group.name}."
                        )

                episodes.append(
                    EpisodePairedWindows(
                        episode_id=group_name,
                        ir_windows=ir_windows.astype(np.float32, copy=False),
                        depth_windows=depth_windows.astype(np.float32, copy=False),
                        proprioception_windows=(
                            proprioception_windows.astype(np.float32, copy=False)
                            if proprioception_windows is not None
                            else None
                        ),
                    )
                )
                total_windows += int(ir_windows.shape[0])

                if metadata is None:
                    assert expected_ir_shape is not None
                    assert expected_depth_shape is not None
                    metadata = _build_telemetry_metadata(
                        payload,
                        expected_ir_shape=expected_ir_shape,
                        expected_depth_shape=expected_depth_shape,
                        ir_window_body_source=ir_window_body_source,
                        expected_proprioception_shape=expected_proprioception_shape,
                        dataset_has_proprioception=dataset_has_proprioception,
                        source_id=group_name,
                    )

        if not episodes or metadata is None:
            raise ValueError(f"No valid paired ir_window/depth_window arrays found in: {hdf5_path}")

        logger.info(
            f"Loaded {total_windows} paired samples from {len(episodes)} HDF5 episode groups with "
            f"ir_window_shape={metadata.ir_window_shape}, depth_window_shape={metadata.depth_input_shape}, "
            f"proprioception_window_shape={metadata.proprioception_window_shape}, "
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

    raise FileNotFoundError(
        f"Expected HDF5 telemetry file at {hdf5_path}. "
        "Re-run ir_di_pro_agent.py and collect telemetry.h5 before training."
    )


def flatten_episode_split(
    episodes: Sequence[EpisodePairedWindows],
    indices: Sequence[int],
    ir_window_shape: tuple[int, int],
    depth_window_shape: tuple[int, int, int],
    proprioception_window_shape: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    selected = [episodes[index] for index in indices]
    episode_ids = [episode.episode_id for episode in selected]
    if not selected:
        return (
            np.empty((0, ir_window_shape[0], ir_window_shape[1]), dtype=np.float32),
            np.empty((0, depth_window_shape[0], depth_window_shape[1], depth_window_shape[2]), dtype=np.float32),
            (
                np.empty((0, proprioception_window_shape[0], proprioception_window_shape[1]), dtype=np.float32)
                if proprioception_window_shape is not None
                else None
            ),
            episode_ids,
        )

    stacked_ir = np.concatenate([episode.ir_windows for episode in selected], axis=0).astype(np.float32, copy=False)
    stacked_depth = np.concatenate([episode.depth_windows for episode in selected], axis=0).astype(np.float32, copy=False)
    if proprioception_window_shape is None:
        stacked_proprioception = None
    else:
        missing_episode_ids = [
            episode.episode_id for episode in selected if episode.proprioception_windows is None
        ]
        if missing_episode_ids:
            raise ValueError(
                "Expected proprioception_windows for every selected episode, but some were missing: "
                f"{missing_episode_ids}"
            )
        stacked_proprioception = np.concatenate(
            [episode.proprioception_windows for episode in selected if episode.proprioception_windows is not None],
            axis=0,
        ).astype(np.float32, copy=False)
    return stacked_ir, stacked_depth, stacked_proprioception, episode_ids

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
    if normalized not in IR_WINDOW_BODY_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported ir_window_body_source '{body_source}'. "
            f"Expected one of {IR_WINDOW_BODY_SOURCE_CHOICES}."
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
                "CUDA was requested for AE training, but torch.cuda.is_available() is False. "
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
    logger.info(f"Saved AE config to: {run_paths.config_path}")


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
            "optim_train/*",
            "train/*",
            "val/*",
            "best_test/*",
            "last_test/*",
            "compare/*",
        ):
            wandb.define_metric(pattern, step_metric="epoch")
    logger.info(f"Initialized W&B run in: {run_paths.wandb_dir}")
    return wandb

def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

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

VALUE_LOSS_CHOICES = ("l1", "mse", "smooth_l1")
SPLIT_MODE_CHOICES = ("episode", "window")
BEST_VAL_METRIC_CHOICES = (
    "val_loss",
    "val_depth_value_mae",
    "val_depth_value_rmse",
    "val_ir_value_mae",
    "val_ir_value_rmse",
    "val_alignment_mu_mse",
    "val_alignment_mu_rmse",
)


@dataclass
class TrainConfig:
    data_dir: str = DEFAULT_DATA_DIR
    condition_text: str = DEFAULT_CONDITION_TEXT
    ir_window_body_source: str = "all"
    output_root: str = DEFAULT_OUTPUT_ROOT
    run_name: str = "ae-joint"
    latent_dim: int = 64
    condition_dim: int = 16
    ir_hidden_dims: tuple[int, int] = (256, 256)
    di_conv_channels: tuple[int, int, int] = (64, 128, 128)
    di_pool_size: tuple[int, int] = (8, 10)
    di_hidden_dim: int = 512
    use_proprioception_window: bool = True
    proprio_hidden_dim: int = 128
    batch_size: int = 512
    epochs: int = 3000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    value_loss_type: str = "l1"
    deterministic_mu_training: bool = True
    ir_value_loss_weight: float = 1.0
    di_value_loss_weight: float = 1.0
    latent_alignment_weight: float = 3.0
    decoder_train_samples: int = 1
    decoder_eval_samples: int = 1
    di_loss_updates_decoder: bool = False
    freeze_ir_after_epochs: int = 0
    best_val_metric: str = "val_alignment_mu_rmse"
    min_feature_std: float = 1e-4
    max_grad_norm: float = 1.0
    val_improvement_min_delta: float = 1e-4
    lr_plateau_patience: int = 100
    lr_plateau_factor: float = 0.5
    min_learning_rate: float = 1e-5
    early_stop_patience: int = 1000
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 10
    eval_interval: int = 100
    split_mode: str = "episode"
    depth_input_noise_std: float = 0.0
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    wandb_enabled: bool = True
    wandb_project: str = "AE"
    wandb_entity: str | None = None
    wandb_group: str = "ae_joint"
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ("ae", "joint", "depth_window", "ir_window", "end_to_end")
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID
    clip_cache_dir: str | None = None
    clip_local_files_only: bool = True
    clip_quiet_load: bool = True


def _make_value_loss(decoded_ir: torch.Tensor, target_ir: torch.Tensor, *, loss_type: str) -> torch.Tensor:
    if loss_type == "l1":
        return (decoded_ir - target_ir).abs().mean()
    if loss_type == "mse":
        return (decoded_ir - target_ir).pow(2).mean()
    if loss_type == "smooth_l1":
        return nn.functional.smooth_l1_loss(decoded_ir, target_ir)
    raise ValueError(f"Unknown value_loss_type={loss_type}.")


def sample_latent_z(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * torch.clamp(logvar, min=-20.0, max=20.0))
    return mu + std * torch.randn_like(std)


@contextmanager
def temporarily_freeze_parameters(parameters: Sequence[torch.nn.Parameter]):
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original_requires_grad):
            parameter.requires_grad_(requires_grad)


def posterior_moment_alignment(
    ir_mu: torch.Tensor,
    ir_logvar: torch.Tensor,
    depth_mu: torch.Tensor,
    depth_logvar: torch.Tensor,
) -> torch.Tensor:
    ir_std = torch.exp(0.5 * ir_logvar)
    depth_std = torch.exp(0.5 * depth_logvar)
    return (ir_mu - depth_mu).pow(2).mean() + (ir_std - depth_std).pow(2).mean()


def posterior_mu_alignment(ir_mu: torch.Tensor, depth_mu: torch.Tensor) -> torch.Tensor:
    return (ir_mu - depth_mu).pow(2).mean()


class IRWindowEncoder(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, hidden_dims: Sequence[int], latent_dim: int):
        super().__init__()
        if not hidden_dims:
            raise ValueError("ir_hidden_dims must not be empty.")
        layers: list[nn.Module] = []
        previous_dim = input_dim + condition_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(previous_dim, int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.ELU()])
            previous_dim = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(previous_dim, latent_dim)
        self.logvar = nn.Linear(previous_dim, latent_dim)

    def forward(self, ir_window_flat: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(torch.cat([ir_window_flat, condition], dim=-1))
        return self.mu(hidden), torch.clamp(self.logvar(hidden), min=-10.0, max=10.0)


class IRWindowDecoder(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        if not hidden_dims:
            raise ValueError("ir_hidden_dims must not be empty.")
        layers: list[nn.Module] = []
        previous_dim = latent_dim + condition_dim
        for hidden_dim in reversed(tuple(int(value) for value in hidden_dims)):
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU()])
            previous_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(previous_dim, output_dim)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.output(self.net(torch.cat([z, condition], dim=-1)))


class TemporalDepthEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        condition_dim: int,
        conv_channels: Sequence[int],
        pool_size: Sequence[int],
        hidden_dim: int,
        latent_dim: int,
        *,
        proprioception_input_shape: Sequence[int] | None = None,
        proprio_hidden_dim: int = 64,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"Depth input shape must be [T, H, W], got {input_shape}.")
        if len(conv_channels) != 3:
            raise ValueError(f"depth_conv_channels must have three values, got {conv_channels}.")
        if len(pool_size) != 2:
            raise ValueError(f"depth_pool_size must have two values, got {pool_size}.")
        self.window_size = int(input_shape[0])
        self.height = int(input_shape[1])
        self.width = int(input_shape[2])
        self.hidden_dim = int(hidden_dim)
        self.proprioception_input_shape = (
            tuple(int(value) for value in proprioception_input_shape)
            if proprioception_input_shape is not None
            else None
        )
        self.uses_proprioception = self.proprioception_input_shape is not None
        self.pool_height = int(pool_size[0])
        self.pool_width = int(pool_size[1])
        c1, c2, c3 = (int(value) for value in conv_channels)
        self.frame_features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((self.pool_height, self.pool_width)),
        )
        self.frame_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * self.pool_height * self.pool_width, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
        )
        self.proprio_hidden_dim = int(proprio_hidden_dim)
        if self.uses_proprioception:
            assert self.proprioception_input_shape is not None
            self.proprio_window_size = int(self.proprioception_input_shape[0])
            self.proprio_feature_dim = int(self.proprioception_input_shape[1])
            if self.proprio_window_size != self.window_size:
                raise ValueError(
                    "Depth/proprio window lengths must match for fused encoding, "
                    f"got depth={self.window_size}, proprio={self.proprio_window_size}."
                )
            self.proprio_frame_projection = nn.Sequential(
                nn.Linear(self.proprio_feature_dim, self.proprio_hidden_dim),
                nn.LayerNorm(self.proprio_hidden_dim),
                nn.ELU(),
                nn.Linear(self.proprio_hidden_dim, self.proprio_hidden_dim),
                nn.LayerNorm(self.proprio_hidden_dim),
                nn.ELU(),
            )
            self.temporal_fusion = nn.Sequential(
                nn.Linear(self.hidden_dim + self.proprio_hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ELU(),
            )
        else:
            self.proprio_window_size = 0
            self.proprio_feature_dim = 0
            self.temporal_fusion = nn.Identity()
        self.temporal_encoder = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.latent_head = nn.Sequential(
            nn.Linear(self.hidden_dim + condition_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
        )
        self.mu = nn.Linear(self.hidden_dim, latent_dim)
        self.logvar = nn.Linear(self.hidden_dim, latent_dim)

    def forward(
        self,
        depth_window: torch.Tensor,
        condition: torch.Tensor,
        proprioception_window: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if depth_window.ndim != 4:
            raise ValueError(f"Expected depth_window batch [B, T, H, W], got {tuple(depth_window.shape)}.")
        batch_size, window_size, height, width = depth_window.shape
        if (window_size, height, width) != (self.window_size, self.height, self.width):
            raise ValueError(
                f"Expected depth shape {(self.window_size, self.height, self.width)}, "
                f"got {(window_size, height, width)}."
            )
        frames = depth_window.reshape(batch_size * window_size, 1, height, width)
        frame_features = self.frame_projection(self.frame_features(frames))
        frame_features = frame_features.reshape(batch_size, window_size, self.hidden_dim)
        temporal_features = frame_features
        if self.uses_proprioception:
            if proprioception_window is None:
                raise ValueError(
                    "This depth encoder checkpoint expects proprioception_window input, but none was provided."
                )
            if proprioception_window.ndim != 3:
                raise ValueError(
                    "Expected proprioception_window batch [B, T, F], "
                    f"got {tuple(proprioception_window.shape)}."
                )
            if tuple(proprioception_window.shape[1:]) != (
                self.proprio_window_size,
                self.proprio_feature_dim,
            ):
                raise ValueError(
                    "Expected proprioception window shape "
                    f"{(self.proprio_window_size, self.proprio_feature_dim)}, "
                    f"got {tuple(proprioception_window.shape[1:])}."
                )
            proprioception_steps = proprioception_window.reshape(
                batch_size * self.proprio_window_size,
                self.proprio_feature_dim,
            )
            proprioception_steps = self.proprio_frame_projection(proprioception_steps)
            proprioception_steps = proprioception_steps.reshape(
                batch_size,
                self.proprio_window_size,
                self.proprio_hidden_dim,
            )
            temporal_features = self.temporal_fusion(torch.cat([frame_features, proprioception_steps], dim=-1))
        _, hidden = self.temporal_encoder(temporal_features)
        hidden = self.latent_head(torch.cat([hidden[-1], condition], dim=-1))
        return self.mu(hidden), torch.clamp(self.logvar(hidden), min=-10.0, max=10.0)


class JointMultimodalAE(nn.Module):
    def __init__(
        self,
        *,
        ir_input_dim: int,
        depth_input_shape: Sequence[int],
        proprioception_input_shape: Sequence[int] | None,
        text_feature_dim: int,
        condition_dim: int,
        latent_dim: int,
        ir_hidden_dims: Sequence[int],
        di_conv_channels: Sequence[int],
        di_pool_size: Sequence[int],
        di_hidden_dim: int,
        proprio_hidden_dim: int,
    ):
        super().__init__()
        self.text_projector = TextConditionProjector(text_feature_dim, condition_dim)
        self.ir_encoder = IRWindowEncoder(ir_input_dim, condition_dim, ir_hidden_dims, latent_dim)
        self.di_encoder = TemporalDepthEncoder(
            depth_input_shape,
            condition_dim,
            di_conv_channels,
            di_pool_size,
            di_hidden_dim,
            latent_dim,
            proprioception_input_shape=proprioception_input_shape,
            proprio_hidden_dim=proprio_hidden_dim,
        )
        self.decoder = IRWindowDecoder(latent_dim, condition_dim, ir_hidden_dims, ir_input_dim)

    def condition(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.text_projector(text_features)

    def encode_ir(self, ir_window_flat: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ir_encoder(ir_window_flat, self.condition(text_features))

    def encode_di(
        self,
        depth_window: torch.Tensor,
        text_features: torch.Tensor,
        proprioception_window: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.di_encoder(depth_window, self.condition(text_features), proprioception_window)

    def decode(self, z: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, self.condition(text_features))

    def decoder_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self.decoder.parameters())

    def ir_anchor_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return (
            tuple(self.text_projector.parameters())
            + tuple(self.ir_encoder.parameters())
            + tuple(self.decoder.parameters())
        )


def _decode_original_ir(
    model: JointMultimodalAE,
    z: torch.Tensor,
    text_features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> torch.Tensor:
    decoded_normalized = model.decode(z, text_features)
    return decoded_normalized * feature_std + feature_mean


def _accumulate_value_stats(
    error: torch.Tensor,
    *,
    total: dict[str, Any],
) -> None:
    value_abs = error.abs()
    total["abs"] += value_abs.sum().item()
    total["sq"] += error.square().sum().item()
    total["elements"] += int(error.numel())
    total["max_abs"] = max(float(total["max_abs"]), float(value_abs.max().item()))
    if total["abs_by_dim"] is None:
        total["abs_by_dim"] = torch.zeros(error.shape[1], device="cpu")
        total["sq_by_dim"] = torch.zeros(error.shape[1], device="cpu")
    total["abs_by_dim"] += value_abs.sum(dim=0).detach().cpu()
    total["sq_by_dim"] += error.square().sum(dim=0).detach().cpu()
    total["dim_count"] += int(error.shape[0])


def _finalize_value_stats(prefix: str, total: dict[str, Any]) -> dict[str, float]:
    if total["elements"] == 0:
        return {
            f"{prefix}_value_mae": float("nan"),
            f"{prefix}_value_rmse": float("nan"),
            f"{prefix}_value_max_abs": float("nan"),
            f"{prefix}_value_dim_mae_max": float("nan"),
            f"{prefix}_value_dim_mae_p95": float("nan"),
        }
    dim_mae = total["abs_by_dim"] / total["dim_count"]
    return {
        f"{prefix}_value_mae": total["abs"] / total["elements"],
        f"{prefix}_value_rmse": math.sqrt(total["sq"] / total["elements"]),
        f"{prefix}_value_max_abs": float(total["max_abs"]),
        f"{prefix}_value_dim_mae_max": float(dim_mae.max().item()),
        f"{prefix}_value_dim_mae_p95": float(torch.quantile(dim_mae, 0.95).item()),
    }


@torch.no_grad()
def evaluate_model(
    model: JointMultimodalAE,
    ir_windows: torch.Tensor,
    depth_windows: torch.Tensor,
    proprioception_windows: torch.Tensor | None,
    *,
    base_text_feature: torch.Tensor,
    batch_size: int,
    ir_feature_mean: torch.Tensor,
    ir_feature_std: torch.Tensor,
    di_feature_mean: torch.Tensor,
    di_feature_std: torch.Tensor,
    proprio_feature_mean: torch.Tensor | None,
    proprio_feature_std: torch.Tensor | None,
    value_loss_type: str,
    deterministic_mu_training: bool,
    ir_value_loss_weight: float,
    di_value_loss_weight: float,
    latent_alignment_weight: float,
    decoder_eval_samples: int,
    device: str,
    prefix: str,
    shuffle_batches: bool = True,
    batch_shuffle_seed: int = 0,
) -> dict[str, float | int]:
    model.eval()
    ir_feature_mean = ir_feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
    ir_feature_std = ir_feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
    di_feature_mean = di_feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
    di_feature_std = di_feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
    proprio_feature_mean_device = (
        proprio_feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
        if proprio_feature_mean is not None
        else None
    )
    proprio_feature_std_device = (
        proprio_feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
        if proprio_feature_std is not None
        else None
    )
    decoder_eval_samples = max(int(decoder_eval_samples), 1)

    total_loss = 0.0
    total_ir_loss = 0.0
    total_depth_loss = 0.0
    total_alignment = 0.0
    total_mu_sq = 0.0
    total_latent_values = 0
    seen_samples = 0
    ir_stats: dict[str, Any] = {"abs": 0.0, "sq": 0.0, "elements": 0, "max_abs": 0.0, "abs_by_dim": None, "sq_by_dim": None, "dim_count": 0}
    depth_stats: dict[str, Any] = {"abs": 0.0, "sq": 0.0, "elements": 0, "max_abs": 0.0, "abs_by_dim": None, "sq_by_dim": None, "dim_count": 0}

    for batch_indices in iterate_batch_indices(
        int(ir_windows.shape[0]),
        batch_size,
        shuffle=shuffle_batches,
        seed=batch_shuffle_seed,
    ):
        batch_ir = ir_windows.index_select(0, batch_indices).to(device=device, dtype=torch.float32, non_blocking=True)
        batch_depth = depth_windows.index_select(0, batch_indices).to(device=device, dtype=torch.float32, non_blocking=True)
        if proprioception_windows is not None:
            batch_proprioception = proprioception_windows.index_select(0, batch_indices).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        else:
            batch_proprioception = None
        batch_ir = batch_ir.reshape(batch_ir.shape[0], -1)
        batch_ir_normalized = (batch_ir - ir_feature_mean) / ir_feature_std
        batch_depth_normalized = (batch_depth - di_feature_mean) / di_feature_std
        if batch_proprioception is not None:
            if proprio_feature_mean_device is None or proprio_feature_std_device is None:
                raise RuntimeError("proprioception_windows were provided but normalization statistics are missing.")
            batch_proprioception_normalized = (
                batch_proprioception - proprio_feature_mean_device
            ) / proprio_feature_std_device
        else:
            batch_proprioception_normalized = None
        batch_text = base_text_feature.expand(batch_ir.shape[0], -1)

        ir_mu, ir_logvar = model.encode_ir(batch_ir_normalized, batch_text)
        depth_mu, depth_logvar = model.encode_di(
            batch_depth_normalized,
            batch_text,
            batch_proprioception_normalized,
        )
        target_ir_mu = ir_mu.detach()
        target_ir_logvar = ir_logvar.detach()
        if deterministic_mu_training:
            alignment = posterior_mu_alignment(target_ir_mu, depth_mu)
        else:
            alignment = posterior_moment_alignment(
                target_ir_mu,
                target_ir_logvar,
                depth_mu,
                depth_logvar,
            )
        decoded_ir_from_ir_mu = _decode_original_ir(model, ir_mu, batch_text, ir_feature_mean, ir_feature_std)
        decoded_ir_from_depth_mu = _decode_original_ir(model, depth_mu, batch_text, ir_feature_mean, ir_feature_std)

        if deterministic_mu_training:
            batch_ir_value_loss = _make_value_loss(decoded_ir_from_ir_mu, batch_ir, loss_type=value_loss_type)
            batch_depth_value_loss = _make_value_loss(decoded_ir_from_depth_mu, batch_ir, loss_type=value_loss_type)
            _accumulate_value_stats(decoded_ir_from_ir_mu - batch_ir, total=ir_stats)
            _accumulate_value_stats(decoded_ir_from_depth_mu - batch_ir, total=depth_stats)
        else:
            batch_ir_value_loss = torch.zeros((), device=device)
            batch_depth_value_loss = torch.zeros((), device=device)
            for _ in range(decoder_eval_samples):
                decoded_ir_from_ir = _decode_original_ir(
                    model,
                    sample_latent_z(ir_mu, ir_logvar),
                    batch_text,
                    ir_feature_mean,
                    ir_feature_std,
                )
                decoded_ir_from_depth = _decode_original_ir(
                    model,
                    sample_latent_z(depth_mu, depth_logvar),
                    batch_text,
                    ir_feature_mean,
                    ir_feature_std,
                )
                batch_ir_value_loss = batch_ir_value_loss + _make_value_loss(
                    decoded_ir_from_ir,
                    batch_ir,
                    loss_type=value_loss_type,
                )
                batch_depth_value_loss = batch_depth_value_loss + _make_value_loss(
                    decoded_ir_from_depth,
                    batch_ir,
                    loss_type=value_loss_type,
                )
                _accumulate_value_stats(decoded_ir_from_ir - batch_ir, total=ir_stats)
                _accumulate_value_stats(decoded_ir_from_depth - batch_ir, total=depth_stats)
            batch_ir_value_loss = batch_ir_value_loss / decoder_eval_samples
            batch_depth_value_loss = batch_depth_value_loss / decoder_eval_samples

        batch_loss = (
            ir_value_loss_weight * batch_ir_value_loss
            + di_value_loss_weight * batch_depth_value_loss
            + latent_alignment_weight * alignment
        )
        batch_size_current = int(batch_ir.shape[0])
        total_loss += batch_loss.item() * batch_size_current
        total_ir_loss += batch_ir_value_loss.item() * batch_size_current
        total_depth_loss += batch_depth_value_loss.item() * batch_size_current
        total_alignment += alignment.item() * batch_size_current
        total_mu_sq += (ir_mu - depth_mu).square().sum().item()
        total_latent_values += int(ir_mu.numel())
        seen_samples += batch_size_current

    metrics: dict[str, float | int] = {
        f"{prefix}_num_samples": int(seen_samples),
        f"{prefix}_loss": total_loss / seen_samples,
        f"{prefix}_ir_value_loss": total_ir_loss / seen_samples,
        f"{prefix}_depth_value_loss": total_depth_loss / seen_samples,
        f"{prefix}_alignment_moment": total_alignment / seen_samples,
        f"{prefix}_alignment_mu_mse": total_mu_sq / total_latent_values,
        f"{prefix}_alignment_mu_rmse": math.sqrt(total_mu_sq / total_latent_values),
    }
    metrics.update(_finalize_value_stats(f"{prefix}_ir", ir_stats))
    metrics.update(_finalize_value_stats(f"{prefix}_depth", depth_stats))
    return metrics


def make_checkpoint_payload(
    *,
    config: TrainConfig,
    model: JointMultimodalAE,
    input_shape: tuple[int, int, int],
    ir_window_shape: tuple[int, int],
    proprioception_input_shape: tuple[int, int] | None,
    num_samples: int,
    ir_feature_mean: torch.Tensor,
    ir_feature_std: torch.Tensor,
    di_feature_mean: torch.Tensor,
    di_feature_std: torch.Tensor,
    proprio_feature_mean: torch.Tensor | None,
    proprio_feature_std: torch.Tensor | None,
    text_feature_dim: int,
    telemetry_metadata: TelemetryMetadata,
    checkpoint_type: str,
    epoch: int,
    val_loss: float,
    val_selection_metric: str,
    val_selection_score: float,
) -> dict[str, Any]:
    return {
        "model_type": "joint_multimodal_ae",
        "latent_training_mode": "deterministic_mu" if config.deterministic_mu_training else "stochastic",
        "checkpoint_type": checkpoint_type,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "val_selection_metric": val_selection_metric,
        "val_selection_score": float(val_selection_score),
        "config": asdict(config),
        "input_shape": list(input_shape),
        "ir_window_shape": list(ir_window_shape),
        "proprioception_input_shape": (
            list(proprioception_input_shape) if proprioception_input_shape is not None else None
        ),
        "uses_proprioception_window": bool(proprioception_input_shape is not None),
        "num_samples": int(num_samples),
        "ir_feature_mean": ir_feature_mean.cpu(),
        "ir_feature_std": ir_feature_std.cpu(),
        "di_feature_mean": di_feature_mean.cpu(),
        "di_feature_std": di_feature_std.cpu(),
        "proprio_feature_mean": None if proprio_feature_mean is None else proprio_feature_mean.cpu(),
        "proprio_feature_std": None if proprio_feature_std is None else proprio_feature_std.cpu(),
        "text_feature_dim": int(text_feature_dim),
        "condition_text": config.condition_text,
        "telemetry": asdict(telemetry_metadata),
        "clip": {
            "model_id": config.clip_model_id,
            "cache_dir": config.clip_cache_dir,
            "local_files_only": config.clip_local_files_only,
        },
        "model_state_dict": model.state_dict(),
    }


def save_checkpoint(path: Path, **kwargs: Any) -> None:
    torch.save(make_checkpoint_payload(**kwargs), path)


def validate_config(config: TrainConfig) -> None:
    config.ir_window_body_source = normalize_ir_window_body_source(config.ir_window_body_source)
    if config.best_val_metric not in BEST_VAL_METRIC_CHOICES:
        raise ValueError(f"best_val_metric must be one of {BEST_VAL_METRIC_CHOICES}, got {config.best_val_metric}.")
    if config.value_loss_type not in VALUE_LOSS_CHOICES:
        raise ValueError(f"value_loss_type must be one of {VALUE_LOSS_CHOICES}, got {config.value_loss_type}.")
    if config.split_mode not in SPLIT_MODE_CHOICES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODE_CHOICES}, got {config.split_mode}.")
    if config.latent_dim <= 0 or config.condition_dim <= 0 or config.di_hidden_dim <= 0 or config.proprio_hidden_dim <= 0:
        raise ValueError("latent_dim, condition_dim, di_hidden_dim, and proprio_hidden_dim must be positive.")
    if not config.ir_hidden_dims or any(int(value) <= 0 for value in config.ir_hidden_dims):
        raise ValueError(f"ir_hidden_dims must contain positive values, got {config.ir_hidden_dims}.")
    if len(config.di_conv_channels) != 3 or any(int(value) <= 0 for value in config.di_conv_channels):
        raise ValueError(f"di_conv_channels must contain three positive values, got {config.di_conv_channels}.")
    if len(config.di_pool_size) != 2 or any(int(value) <= 0 for value in config.di_pool_size):
        raise ValueError(f"di_pool_size must contain two positive values, got {config.di_pool_size}.")
    if config.batch_size <= 0 or config.epochs <= 0:
        raise ValueError("batch_size and epochs must be positive.")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay must be non-negative.")
    if (
        config.ir_value_loss_weight < 0
        or config.di_value_loss_weight < 0
        or config.latent_alignment_weight < 0
    ):
        raise ValueError("Loss weights must be non-negative.")
    if config.depth_input_noise_std < 0:
        raise ValueError("depth_input_noise_std must be non-negative.")
    if config.decoder_train_samples <= 0 or config.decoder_eval_samples <= 0:
        raise ValueError("decoder_train_samples and decoder_eval_samples must be positive.")
    if config.freeze_ir_after_epochs < 0:
        raise ValueError("freeze_ir_after_epochs must be non-negative.")
    if config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")
    if config.val_improvement_min_delta < 0:
        raise ValueError("val_improvement_min_delta must be non-negative.")
    if config.lr_plateau_patience < 0 or config.early_stop_patience < 0:
        raise ValueError("lr_plateau_patience and early_stop_patience must be non-negative.")
    if not 0.0 < config.lr_plateau_factor < 1.0:
        raise ValueError("lr_plateau_factor must be in (0, 1).")


def split_paired_arrays(
    episodes: Sequence[EpisodePairedWindows],
    telemetry_metadata: TelemetryMetadata,
    *,
    split_mode: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    list[str],
]:
    if split_mode == "episode":
        split_indices = split_episode_indices(len(episodes), val_ratio, test_ratio, seed)
        train_ir_np, train_depth_np, train_proprioception_np, train_episode_ids = flatten_episode_split(
            episodes,
            split_indices["train"],
            telemetry_metadata.ir_window_shape,
            telemetry_metadata.depth_input_shape,
            telemetry_metadata.proprioception_window_shape,
        )
        val_ir_np, val_depth_np, val_proprioception_np, val_episode_ids = flatten_episode_split(
            episodes,
            split_indices["val"],
            telemetry_metadata.ir_window_shape,
            telemetry_metadata.depth_input_shape,
            telemetry_metadata.proprioception_window_shape,
        )
        test_ir_np, test_depth_np, test_proprioception_np, test_episode_ids = flatten_episode_split(
            episodes,
            split_indices["test"],
            telemetry_metadata.ir_window_shape,
            telemetry_metadata.depth_input_shape,
            telemetry_metadata.proprioception_window_shape,
        )
        return (
            train_ir_np,
            train_depth_np,
            train_proprioception_np,
            train_episode_ids,
            val_ir_np,
            val_depth_np,
            val_proprioception_np,
            val_episode_ids,
            test_ir_np,
            test_depth_np,
            test_proprioception_np,
            test_episode_ids,
        )

    if split_mode != "window":
        raise ValueError(f"split_mode must be one of {SPLIT_MODE_CHOICES}, got {split_mode}.")

    all_indices = np.arange(len(episodes))
    all_ir_np, all_depth_np, all_proprioception_np, all_episode_ids = flatten_episode_split(
        episodes,
        all_indices,
        telemetry_metadata.ir_window_shape,
        telemetry_metadata.depth_input_shape,
        telemetry_metadata.proprioception_window_shape,
    )
    num_windows = int(all_ir_np.shape[0])
    if num_windows < 3:
        raise ValueError(f"Need at least 3 windows for window split, got {num_windows}.")
    if val_ratio < 0.0 or test_ratio < 0.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"Invalid split ratios: val_ratio={val_ratio}, test_ratio={test_ratio}. Expected val+test < 1.0."
        )

    rng = np.random.default_rng(seed)
    window_indices = np.arange(num_windows)
    rng.shuffle(window_indices)
    num_test = int(round(num_windows * test_ratio))
    num_val = int(round(num_windows * val_ratio))
    if test_ratio > 0.0 and num_test == 0:
        num_test = 1
    if val_ratio > 0.0 and num_val == 0:
        num_val = 1
    num_train = num_windows - num_val - num_test
    if num_train <= 0:
        raise ValueError(
            f"Window split leaves no training data: num_windows={num_windows}, num_val={num_val}, num_test={num_test}."
        )

    train_indices = np.sort(window_indices[:num_train])
    val_indices = np.sort(window_indices[num_train : num_train + num_val])
    test_indices = np.sort(window_indices[num_train + num_val :])
    return (
        all_ir_np[train_indices],
        all_depth_np[train_indices],
        None if all_proprioception_np is None else all_proprioception_np[train_indices],
        [f"window_split_train_{len(train_indices)}_from_{len(all_episode_ids)}_episodes"],
        all_ir_np[val_indices],
        all_depth_np[val_indices],
        None if all_proprioception_np is None else all_proprioception_np[val_indices],
        [f"window_split_val_{len(val_indices)}_from_{len(all_episode_ids)}_episodes"],
        all_ir_np[test_indices],
        all_depth_np[test_indices],
        None if all_proprioception_np is None else all_proprioception_np[test_indices],
        [f"window_split_test_{len(test_indices)}_from_{len(all_episode_ids)}_episodes"],
    )


def train_joint(config: TrainConfig) -> Path:
    validate_config(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    configure_cuda_backend(device)
    logger.info(f"Using device for joint multimodal AE training: {device}")

    run_paths = create_run_paths(config)
    save_config(config, run_paths)
    wandb = None
    metrics_history: list[dict[str, float | int]] = []

    try:
        episodes, telemetry_metadata = extract_episode_paired_windows(
            Path(config.data_dir),
            ir_window_body_source=config.ir_window_body_source,
        )
        (
            train_ir_np,
            train_depth_np,
            train_proprioception_np,
            train_episode_ids,
            val_ir_np,
            val_depth_np,
            val_proprioception_np,
            val_episode_ids,
            test_ir_np,
            test_depth_np,
            test_proprioception_np,
            test_episode_ids,
        ) = split_paired_arrays(
            episodes,
            telemetry_metadata,
            split_mode=config.split_mode,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
        )
        del episodes

        train_ir = torch.from_numpy(train_ir_np)
        val_ir = torch.from_numpy(val_ir_np)
        test_ir = torch.from_numpy(test_ir_np)
        train_depth = torch.from_numpy(train_depth_np)
        val_depth = torch.from_numpy(val_depth_np)
        test_depth = torch.from_numpy(test_depth_np)
        train_proprioception = (
            torch.from_numpy(train_proprioception_np) if train_proprioception_np is not None else None
        )
        val_proprioception = torch.from_numpy(val_proprioception_np) if val_proprioception_np is not None else None
        test_proprioception = (
            torch.from_numpy(test_proprioception_np) if test_proprioception_np is not None else None
        )
        del (
            train_ir_np,
            val_ir_np,
            test_ir_np,
            train_depth_np,
            val_depth_np,
            test_depth_np,
            train_proprioception_np,
            val_proprioception_np,
            test_proprioception_np,
        )

        train_ir_flat = train_ir.reshape(train_ir.shape[0], -1)
        ir_feature_mean = train_ir_flat.mean(dim=0)
        ir_feature_std = train_ir_flat.std(dim=0).clamp_min(config.min_feature_std)
        di_feature_mean = train_depth.mean(dim=0)
        di_feature_std = train_depth.std(dim=0).clamp_min(config.min_feature_std)
        use_proprioception_window = (
            bool(config.use_proprioception_window)
            and telemetry_metadata.proprioception_window_shape is not None
        )
        if bool(config.use_proprioception_window) and telemetry_metadata.proprioception_window_shape is None:
            logger.info(
                "Telemetry does not include proprioception_window. Falling back to depth-only DI encoder training."
            )
        if not bool(config.use_proprioception_window) and telemetry_metadata.proprioception_window_shape is not None:
            logger.info("Telemetry includes proprioception_window, but use_proprioception_window=False so it will be ignored.")
        if use_proprioception_window:
            if train_proprioception is None or val_proprioception is None or test_proprioception is None:
                raise RuntimeError(
                    "Expected train/val/test proprioception tensors because proprioception_window was enabled."
                )
            proprio_feature_mean = train_proprioception.mean(dim=0)
            proprio_feature_std = train_proprioception.std(dim=0).clamp_min(config.min_feature_std)
        else:
            train_proprioception = None
            val_proprioception = None
            test_proprioception = None
            proprio_feature_mean = None
            proprio_feature_std = None

        clip_text = CLIPTextFeatureExtractor(
            model_id=config.clip_model_id,
            device=device,
            cache_dir=config.clip_cache_dir,
            local_files_only=config.clip_local_files_only,
            quiet_load=config.clip_quiet_load,
        )
        base_text_feature = clip_text.encode([config.condition_text]).to(device=device, dtype=torch.float32)
        text_feature_dim = int(base_text_feature.shape[-1])

        model = JointMultimodalAE(
            ir_input_dim=int(train_ir_flat.shape[1]),
            depth_input_shape=telemetry_metadata.depth_input_shape,
            proprioception_input_shape=(
                telemetry_metadata.proprioception_window_shape if use_proprioception_window else None
            ),
            text_feature_dim=text_feature_dim,
            condition_dim=config.condition_dim,
            latent_dim=config.latent_dim,
            ir_hidden_dims=config.ir_hidden_dims,
            di_conv_channels=config.di_conv_channels,
            di_pool_size=config.di_pool_size,
            di_hidden_dim=config.di_hidden_dim,
            proprio_hidden_dim=config.proprio_hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        decoder_parameters = model.decoder_parameters()
        ir_anchor_parameters = model.ir_anchor_parameters()

        ir_feature_mean_device = ir_feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
        ir_feature_std_device = ir_feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
        di_feature_mean_device = di_feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
        di_feature_std_device = di_feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
        proprio_feature_mean_device = (
            proprio_feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
            if proprio_feature_mean is not None
            else None
        )
        proprio_feature_std_device = (
            proprio_feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
            if proprio_feature_std is not None
            else None
        )

        logger.info(
            f"Training joint AE on train/val/test windows = "
            f"{train_ir.shape[0]}/{val_ir.shape[0]}/{test_ir.shape[0]}, "
            f"ir_window_shape={telemetry_metadata.ir_window_shape}, depth_window_shape={telemetry_metadata.depth_input_shape}, "
            f"proprioception_window_shape={telemetry_metadata.proprioception_window_shape}, "
            f"use_proprioception_window={use_proprioception_window}, "
            f"latent_dim={config.latent_dim}, condition_dim={config.condition_dim}, "
            f"di_pool_size={config.di_pool_size}, split_mode={config.split_mode}, "
            f"eval_interval={config.eval_interval}, device={device}"
        )
        objective_terms = [
            f"{config.ir_value_loss_weight}*ir_recon",
            f"{config.di_value_loss_weight}*di_recon",
            f"{config.latent_alignment_weight}*{'mu_alignment_mse' if config.deterministic_mu_training else 'moment_alignment'}",
        ]
        logger.info(
            f"Joint objective: same paired ir/depth sample -> aligned {'latent code' if config.deterministic_mu_training else 'posterior latent'}. "
            f"loss={' + '.join(objective_terms)} "
            f"best_val_metric={config.best_val_metric}, "
            f"deterministic_mu_training={config.deterministic_mu_training}, "
            f"depth_input_noise_std={config.depth_input_noise_std}, "
            f"di_loss_updates_decoder={config.di_loss_updates_decoder}, "
            f"freeze_ir_after_epochs={config.freeze_ir_after_epochs}, "
            "alignment_target=ir_mu_detached"
        )
        if config.deterministic_mu_training and (
            config.decoder_train_samples != 1 or config.decoder_eval_samples != 1
        ):
            logger.warning(
                "deterministic_mu_training=true: decoder_train_samples/decoder_eval_samples are ignored for the "
                "main objective because training/evaluation reconstruction uses mu directly."
            )

        wandb = init_wandb(config, run_paths)
        best_val_score = float("inf")
        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state: dict[str, torch.Tensor] | None = None
        last_model_state: dict[str, torch.Tensor] | None = None
        epochs_since_best = 0
        ir_anchor_frozen = False
        last_lr_reduction_epoch = 0

        for epoch in range(1, config.epochs + 1):
            if (
                config.freeze_ir_after_epochs > 0
                and not ir_anchor_frozen
                and epoch > config.freeze_ir_after_epochs
            ):
                for parameter in ir_anchor_parameters:
                    parameter.requires_grad_(False)
                ir_anchor_frozen = True
                logger.info(
                    f"Froze ir/text/decoder anchor after epoch={config.freeze_ir_after_epochs}; "
                    "depth encoder will continue adapting to the fixed latent target."
                )
            model.train()
            totals = {
                "loss": 0.0,
                "ir_value": 0.0,
                "depth_value": 0.0,
                "alignment": 0.0,
            }
            train_ir_stats: dict[str, Any] = {"abs": 0.0, "sq": 0.0, "elements": 0, "max_abs": 0.0, "abs_by_dim": None, "sq_by_dim": None, "dim_count": 0}
            train_depth_stats: dict[str, Any] = {"abs": 0.0, "sq": 0.0, "elements": 0, "max_abs": 0.0, "abs_by_dim": None, "sq_by_dim": None, "dim_count": 0}
            train_mu_sq = 0.0
            train_latent_values = 0
            seen_samples = 0
            for batch_indices in iterate_batch_indices(
                int(train_ir.shape[0]),
                config.batch_size,
                shuffle=True,
                seed=config.seed + epoch,
            ):
                batch_ir = train_ir.index_select(0, batch_indices).to(device=device, dtype=torch.float32, non_blocking=True)
                batch_depth = train_depth.index_select(0, batch_indices).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                if train_proprioception is not None:
                    batch_proprioception = train_proprioception.index_select(0, batch_indices).to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                else:
                    batch_proprioception = None
                batch_ir = batch_ir.reshape(batch_ir.shape[0], -1)
                batch_ir_normalized = (batch_ir - ir_feature_mean_device) / ir_feature_std_device
                batch_depth_normalized = (batch_depth - di_feature_mean_device) / di_feature_std_device
                if batch_proprioception is not None:
                    if proprio_feature_mean_device is None or proprio_feature_std_device is None:
                        raise RuntimeError(
                            "proprioception_window input was loaded but proprio normalization stats are missing."
                        )
                    batch_proprioception_normalized = (
                        batch_proprioception - proprio_feature_mean_device
                    ) / proprio_feature_std_device
                else:
                    batch_proprioception_normalized = None
                if config.depth_input_noise_std > 0.0:
                    batch_depth_normalized = batch_depth_normalized + torch.randn_like(
                        batch_depth_normalized,
                    ) * config.depth_input_noise_std
                batch_size_current = int(batch_ir.shape[0])
                batch_text = base_text_feature.expand(batch_size_current, -1)

                ir_mu, ir_logvar = model.encode_ir(batch_ir_normalized, batch_text)
                depth_mu, depth_logvar = model.encode_di(
                    batch_depth_normalized,
                    batch_text,
                    batch_proprioception_normalized,
                )
                target_ir_mu = ir_mu.detach()
                target_ir_logvar = ir_logvar.detach()
                if config.deterministic_mu_training:
                    alignment = posterior_mu_alignment(target_ir_mu, depth_mu)
                else:
                    alignment = posterior_moment_alignment(
                        target_ir_mu,
                        target_ir_logvar,
                        depth_mu,
                        depth_logvar,
                    )
                if config.deterministic_mu_training:
                    decoded_from_ir = _decode_original_ir(
                        model,
                        ir_mu,
                        batch_text,
                        ir_feature_mean_device,
                        ir_feature_std_device,
                    )
                    if config.di_loss_updates_decoder:
                        decoded_from_depth = _decode_original_ir(
                            model,
                            depth_mu,
                            batch_text,
                            ir_feature_mean_device,
                            ir_feature_std_device,
                        )
                    else:
                        with temporarily_freeze_parameters(decoder_parameters):
                            decoded_from_depth = _decode_original_ir(
                                model,
                                depth_mu,
                                batch_text,
                                ir_feature_mean_device,
                                ir_feature_std_device,
                            )
                    ir_value_loss = _make_value_loss(
                        decoded_from_ir,
                        batch_ir,
                        loss_type=config.value_loss_type,
                    )
                    depth_value_loss = _make_value_loss(
                        decoded_from_depth,
                        batch_ir,
                        loss_type=config.value_loss_type,
                    )
                    _accumulate_value_stats(decoded_from_ir - batch_ir, total=train_ir_stats)
                    _accumulate_value_stats(decoded_from_depth - batch_ir, total=train_depth_stats)
                else:
                    ir_value_loss = torch.zeros((), device=device)
                    depth_value_loss = torch.zeros((), device=device)
                    for _ in range(config.decoder_train_samples):
                        decoded_from_ir = _decode_original_ir(
                            model,
                            sample_latent_z(ir_mu, ir_logvar),
                            batch_text,
                            ir_feature_mean_device,
                            ir_feature_std_device,
                        )
                        depth_z = sample_latent_z(depth_mu, depth_logvar)
                        if config.di_loss_updates_decoder:
                            decoded_from_depth = _decode_original_ir(
                                model,
                                depth_z,
                                batch_text,
                                ir_feature_mean_device,
                                ir_feature_std_device,
                            )
                        else:
                            with temporarily_freeze_parameters(decoder_parameters):
                                decoded_from_depth = _decode_original_ir(
                                    model,
                                    depth_z,
                                    batch_text,
                                    ir_feature_mean_device,
                                    ir_feature_std_device,
                                )
                        ir_value_loss = ir_value_loss + _make_value_loss(
                            decoded_from_ir,
                            batch_ir,
                            loss_type=config.value_loss_type,
                        )
                        depth_value_loss = depth_value_loss + _make_value_loss(
                            decoded_from_depth,
                            batch_ir,
                            loss_type=config.value_loss_type,
                        )
                        _accumulate_value_stats(decoded_from_ir - batch_ir, total=train_ir_stats)
                        _accumulate_value_stats(decoded_from_depth - batch_ir, total=train_depth_stats)
                    ir_value_loss = ir_value_loss / config.decoder_train_samples
                    depth_value_loss = depth_value_loss / config.decoder_train_samples

                loss = (
                    config.ir_value_loss_weight * ir_value_loss
                    + config.di_value_loss_weight * depth_value_loss
                    + config.latent_alignment_weight * alignment
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Joint training loss became non-finite.")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()

                totals["loss"] += loss.item() * batch_size_current
                totals["ir_value"] += ir_value_loss.item() * batch_size_current
                totals["depth_value"] += depth_value_loss.item() * batch_size_current
                totals["alignment"] += alignment.item() * batch_size_current
                train_mu_sq += (ir_mu - depth_mu).square().sum().item()
                train_latent_values += int(ir_mu.numel())
                seen_samples += batch_size_current

            optim_train_metrics = {
                "optim_train_num_samples": int(seen_samples),
                "optim_train_loss": totals["loss"] / seen_samples,
                "optim_train_ir_value_loss": totals["ir_value"] / seen_samples,
                "optim_train_depth_value_loss": totals["depth_value"] / seen_samples,
                "optim_train_alignment_moment": totals["alignment"] / seen_samples,
                "optim_train_alignment_mu_mse": train_mu_sq / train_latent_values,
                "optim_train_alignment_mu_rmse": math.sqrt(train_mu_sq / train_latent_values),
            }
            optim_train_metrics.update(rename_metric_prefix(_finalize_value_stats("train_ir", train_ir_stats), "train_", "optim_train_"))
            optim_train_metrics.update(
                rename_metric_prefix(_finalize_value_stats("train_depth", train_depth_stats), "train_", "optim_train_")
            )
            should_evaluate = (
                epoch == 1
                or epoch % config.eval_interval == 0
                or epoch == config.epochs
            )
            if not should_evaluate and best_epoch > 0:
                epochs_since_best = epoch - best_epoch
            current_learning_rate = float(optimizer.param_groups[0]["lr"])

            epoch_metrics: dict[str, float | int] = {
                "epoch": epoch,
                **optim_train_metrics,
                "eval_performed": int(should_evaluate),
                "train/learning_rate": current_learning_rate,
                "train/ir_anchor_frozen": float(ir_anchor_frozen),
                "optim_train/loss": optim_train_metrics["optim_train_loss"],
                "optim_train/ir_value_loss": optim_train_metrics["optim_train_ir_value_loss"],
                "optim_train/depth_value_loss": optim_train_metrics["optim_train_depth_value_loss"],
                "optim_train/alignment_moment": optim_train_metrics["optim_train_alignment_moment"],
                "optim_train/alignment_mu_mse": optim_train_metrics["optim_train_alignment_mu_mse"],
                "optim_train/alignment_mu_rmse": optim_train_metrics["optim_train_alignment_mu_rmse"],
                "optim_train/ir_value_mae": optim_train_metrics["optim_train_ir_value_mae"],
                "optim_train/ir_value_rmse": optim_train_metrics["optim_train_ir_value_rmse"],
                "optim_train/depth_value_mae": optim_train_metrics["optim_train_depth_value_mae"],
                "optim_train/depth_value_rmse": optim_train_metrics["optim_train_depth_value_rmse"],
                "best_val_loss": best_val_loss,
                "best_val_score": best_val_score,
                "best_epoch": best_epoch,
                "epochs_since_best": epochs_since_best,
            }

            if should_evaluate:
                train_metrics = evaluate_model(
                    model,
                    train_ir,
                    train_depth,
                    train_proprioception,
                    base_text_feature=base_text_feature,
                    batch_size=config.batch_size,
                    ir_feature_mean=ir_feature_mean,
                    ir_feature_std=ir_feature_std,
                    di_feature_mean=di_feature_mean,
                    di_feature_std=di_feature_std,
                    proprio_feature_mean=proprio_feature_mean,
                    proprio_feature_std=proprio_feature_std,
                    value_loss_type=config.value_loss_type,
                    deterministic_mu_training=config.deterministic_mu_training,
                    ir_value_loss_weight=config.ir_value_loss_weight,
                    di_value_loss_weight=config.di_value_loss_weight,
                    latent_alignment_weight=config.latent_alignment_weight,
                    decoder_eval_samples=config.decoder_eval_samples,
                    device=device,
                    prefix="train",
                    shuffle_batches=True,
                    batch_shuffle_seed=config.seed,
                )
                val_metrics = evaluate_model(
                    model,
                    val_ir,
                    val_depth,
                    val_proprioception,
                    base_text_feature=base_text_feature,
                    batch_size=config.batch_size,
                    ir_feature_mean=ir_feature_mean,
                    ir_feature_std=ir_feature_std,
                    di_feature_mean=di_feature_mean,
                    di_feature_std=di_feature_std,
                    proprio_feature_mean=proprio_feature_mean,
                    proprio_feature_std=proprio_feature_std,
                    value_loss_type=config.value_loss_type,
                    deterministic_mu_training=config.deterministic_mu_training,
                    ir_value_loss_weight=config.ir_value_loss_weight,
                    di_value_loss_weight=config.di_value_loss_weight,
                    latent_alignment_weight=config.latent_alignment_weight,
                    decoder_eval_samples=config.decoder_eval_samples,
                    device=device,
                    prefix="val",
                    shuffle_batches=True,
                    batch_shuffle_seed=config.seed + 1,
                )
                current_val_loss = float(val_metrics["val_loss"])
                current_val_selection_score = float(val_metrics[config.best_val_metric])
                if not math.isfinite(current_val_selection_score):
                    raise RuntimeError(f"Validation metric {config.best_val_metric} is not finite.")

                improved = current_val_selection_score < best_val_score - config.val_improvement_min_delta
                if improved:
                    best_val_score = current_val_selection_score
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    epochs_since_best = 0
                    best_model_state = clone_state_dict_to_cpu(model)
                    save_checkpoint(
                        run_paths.best_checkpoint_path,
                        config=config,
                        model=model,
                        input_shape=telemetry_metadata.depth_input_shape,
                        ir_window_shape=telemetry_metadata.ir_window_shape,
                        proprioception_input_shape=(
                            telemetry_metadata.proprioception_window_shape if use_proprioception_window else None
                        ),
                        num_samples=train_ir.shape[0],
                        ir_feature_mean=ir_feature_mean,
                        ir_feature_std=ir_feature_std,
                        di_feature_mean=di_feature_mean,
                        di_feature_std=di_feature_std,
                        proprio_feature_mean=proprio_feature_mean,
                        proprio_feature_std=proprio_feature_std,
                        text_feature_dim=text_feature_dim,
                        telemetry_metadata=telemetry_metadata,
                        checkpoint_type="best",
                        epoch=epoch,
                        val_loss=current_val_loss,
                        val_selection_metric=config.best_val_metric,
                        val_selection_score=current_val_selection_score,
                    )
                else:
                    epochs_since_best = epoch - best_epoch if best_epoch > 0 else 0

                if (
                    not improved
                    and config.lr_plateau_patience > 0
                    and epochs_since_best >= config.lr_plateau_patience
                    and epoch - last_lr_reduction_epoch >= config.lr_plateau_patience
                ):
                    old_lr = float(optimizer.param_groups[0]["lr"])
                    new_lr = max(old_lr * config.lr_plateau_factor, config.min_learning_rate)
                    if new_lr < old_lr:
                        for group in optimizer.param_groups:
                            group["lr"] = new_lr
                        last_lr_reduction_epoch = epoch
                        logger.info(
                            f"Reduced learning rate after {epochs_since_best} epochs without validation improvement: "
                            f"{old_lr:.6g} -> {new_lr:.6g}"
                        )
                current_learning_rate = float(optimizer.param_groups[0]["lr"])

                epoch_metrics.update(
                    {
                        **train_metrics,
                        **val_metrics,
                        "train/loss": train_metrics["train_loss"],
                        "train/ir_value_loss": train_metrics["train_ir_value_loss"],
                        "train/depth_value_loss": train_metrics["train_depth_value_loss"],
                        "train/alignment_moment": train_metrics["train_alignment_moment"],
                        "train/alignment_mu_mse": train_metrics["train_alignment_mu_mse"],
                        "train/alignment_mu_rmse": train_metrics["train_alignment_mu_rmse"],
                        "train/ir_value_mae": train_metrics["train_ir_value_mae"],
                        "train/ir_value_rmse": train_metrics["train_ir_value_rmse"],
                        "train/depth_value_mae": train_metrics["train_depth_value_mae"],
                        "train/depth_value_rmse": train_metrics["train_depth_value_rmse"],
                        "train/learning_rate": current_learning_rate,
                        "val/loss": val_metrics["val_loss"],
                        "val/ir_value_mae": val_metrics["val_ir_value_mae"],
                        "val/ir_value_rmse": val_metrics["val_ir_value_rmse"],
                        "val/depth_value_mae": val_metrics["val_depth_value_mae"],
                        "val/depth_value_rmse": val_metrics["val_depth_value_rmse"],
                        "val/alignment_mu_mse": val_metrics["val_alignment_mu_mse"],
                        "val/alignment_mu_rmse": val_metrics["val_alignment_mu_rmse"],
                        "val/selection_score": current_val_selection_score,
                        "best_val_loss": best_val_loss,
                        "best_val_score": best_val_score,
                        "best_epoch": best_epoch,
                        "epochs_since_best": epochs_since_best,
                    }
                )

            metrics_history.append(epoch_metrics)
            if wandb is not None and wandb.run is not None:
                wandb.log(epoch_metrics, step=epoch)

            if should_evaluate and (epoch % config.log_interval == 0 or epoch == 1 or epoch == config.epochs):
                logger.info(
                    f"epoch={epoch:04d} "
                    f"train_loss={train_metrics['train_loss']:.6f} "
                    f"train_ir_mae={train_metrics['train_ir_value_mae']:.6f} "
                    f"train_depth_mae={train_metrics['train_depth_value_mae']:.6f} "
                    f"train_align_mu_mse={train_metrics['train_alignment_mu_mse']:.6f} "
                    f"train_align_mu_rmse={train_metrics['train_alignment_mu_rmse']:.6f} "
                    f"optim_train_loss={optim_train_metrics['optim_train_loss']:.6f} "
                    f"val_depth_mae={val_metrics['val_depth_value_mae']:.6f} "
                    f"val_depth_rmse={val_metrics['val_depth_value_rmse']:.6f} "
                    f"val_depth_dim_p95={val_metrics['val_depth_value_dim_mae_p95']:.6f} "
                    f"val_depth_dim_max={val_metrics['val_depth_value_dim_mae_max']:.6f} "
                    f"val_ir_mae={val_metrics['val_ir_value_mae']:.6f} "
                    f"val_align_mu_mse={val_metrics['val_alignment_mu_mse']:.6f} "
                    f"val_align_mu_rmse={val_metrics['val_alignment_mu_rmse']:.6f} "
                    f"val_select={current_val_selection_score:.6f} "
                    f"best_val_score={best_val_score:.6f} "
                    f"lr={current_learning_rate:.6g} "
                    f"ir_anchor_frozen={int(ir_anchor_frozen)} "
                    f"epochs_since_best={epochs_since_best} "
                    f"best_val_metric={config.best_val_metric}"
                )
            elif epoch % config.log_interval == 0 or epoch == config.epochs:
                next_eval_epoch = min(
                    ((epoch // config.eval_interval) + 1) * config.eval_interval,
                    config.epochs,
                )
                logger.info(
                    f"epoch={epoch:04d} "
                    f"optim_train_loss={optim_train_metrics['optim_train_loss']:.6f} "
                    f"optim_train_align_mu_rmse={optim_train_metrics['optim_train_alignment_mu_rmse']:.6f} "
                    f"lr={current_learning_rate:.6g} "
                    f"ir_anchor_frozen={int(ir_anchor_frozen)} "
                    f"eval_skipped=1 next_eval_epoch={next_eval_epoch} "
                    f"best_val_score={best_val_score:.6f} "
                    f"epochs_since_best={epochs_since_best}"
                )

            if (
                should_evaluate
                and config.early_stop_patience > 0
                and epochs_since_best >= config.early_stop_patience
            ):
                logger.info(
                    f"Early stopping at epoch={epoch} because {config.best_val_metric} did not improve for "
                    f"{epochs_since_best} epochs. best_epoch={best_epoch}, best_val_score={best_val_score:.6f}"
                )
                break

        last_model_state = clone_state_dict_to_cpu(model)
        last_epoch = int(metrics_history[-1]["epoch"])
        final_val_loss = float(metrics_history[-1]["val_loss"])
        final_val_selection_score = float(metrics_history[-1]["val/selection_score"])
        save_checkpoint(
            run_paths.last_checkpoint_path,
            config=config,
            model=model,
            input_shape=telemetry_metadata.depth_input_shape,
            ir_window_shape=telemetry_metadata.ir_window_shape,
            proprioception_input_shape=(
                telemetry_metadata.proprioception_window_shape if use_proprioception_window else None
            ),
            num_samples=train_ir.shape[0],
            ir_feature_mean=ir_feature_mean,
            ir_feature_std=ir_feature_std,
            di_feature_mean=di_feature_mean,
            di_feature_std=di_feature_std,
            proprio_feature_mean=proprio_feature_mean,
            proprio_feature_std=proprio_feature_std,
            text_feature_dim=text_feature_dim,
            telemetry_metadata=telemetry_metadata,
            checkpoint_type="last",
            epoch=last_epoch,
            val_loss=final_val_loss,
            val_selection_metric=config.best_val_metric,
            val_selection_score=final_val_selection_score,
        )
        if best_model_state is None:
            best_model_state = clone_state_dict_to_cpu(model)
            best_epoch = last_epoch
            best_val_score = final_val_selection_score
            best_val_loss = final_val_loss

        model.load_state_dict(best_model_state)
        best_test_metrics = evaluate_model(
            model,
            test_ir,
            test_depth,
            test_proprioception,
            base_text_feature=base_text_feature,
            batch_size=config.batch_size,
            ir_feature_mean=ir_feature_mean,
            ir_feature_std=ir_feature_std,
            di_feature_mean=di_feature_mean,
            di_feature_std=di_feature_std,
            proprio_feature_mean=proprio_feature_mean,
            proprio_feature_std=proprio_feature_std,
            value_loss_type=config.value_loss_type,
            deterministic_mu_training=config.deterministic_mu_training,
            ir_value_loss_weight=config.ir_value_loss_weight,
            di_value_loss_weight=config.di_value_loss_weight,
            latent_alignment_weight=config.latent_alignment_weight,
            decoder_eval_samples=config.decoder_eval_samples,
            device=device,
            prefix="test",
            shuffle_batches=True,
            batch_shuffle_seed=config.seed + 2,
        )
        model.load_state_dict(last_model_state)
        last_test_metrics = evaluate_model(
            model,
            test_ir,
            test_depth,
            test_proprioception,
            base_text_feature=base_text_feature,
            batch_size=config.batch_size,
            ir_feature_mean=ir_feature_mean,
            ir_feature_std=ir_feature_std,
            di_feature_mean=di_feature_mean,
            di_feature_std=di_feature_std,
            proprio_feature_mean=proprio_feature_mean,
            proprio_feature_std=proprio_feature_std,
            value_loss_type=config.value_loss_type,
            deterministic_mu_training=config.deterministic_mu_training,
            ir_value_loss_weight=config.ir_value_loss_weight,
            di_value_loss_weight=config.di_value_loss_weight,
            latent_alignment_weight=config.latent_alignment_weight,
            decoder_eval_samples=config.decoder_eval_samples,
            device=device,
            prefix="test",
            shuffle_batches=True,
            batch_shuffle_seed=config.seed + 2,
        )
        best_test_metrics_named = rename_metric_prefix(best_test_metrics, "test_", "best_test_")
        last_test_metrics_named = rename_metric_prefix(last_test_metrics, "test_", "last_test_")
        test_differences = compute_metric_differences(last_test_metrics, best_test_metrics)

        summary = {
            "seed": config.seed,
            "num_train_windows": int(train_ir.shape[0]),
            "num_val_windows": int(val_ir.shape[0]),
            "num_test_windows": int(test_ir.shape[0]),
            "num_train_episodes": len(train_episode_ids),
            "num_val_episodes": len(val_episode_ids),
            "num_test_episodes": len(test_episode_ids),
            "train_episode_ids": train_episode_ids,
            "val_episode_ids": val_episode_ids,
            "test_episode_ids": test_episode_ids,
            "telemetry": asdict(telemetry_metadata),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_metric": config.best_val_metric,
            "best_val_score": best_val_score,
            "best_checkpoint": str(run_paths.best_checkpoint_path),
            "last_checkpoint": str(run_paths.last_checkpoint_path),
            "best_test_metrics": best_test_metrics_named,
            "last_test_metrics": last_test_metrics_named,
            "test_metric_differences": test_differences,
            "history": metrics_history,
        }
        with run_paths.metrics_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        logger.info(f"Saved best joint AE checkpoint to: {run_paths.best_checkpoint_path}")
        logger.info(f"Saved last joint AE checkpoint to: {run_paths.last_checkpoint_path}")
        logger.info(f"Saved split and metric summary to: {run_paths.metrics_path}")
        logger.info(
            f"Test comparison: best_depth_mae={best_test_metrics_named['best_test_depth_value_mae']:.6f}, "
            f"last_depth_mae={last_test_metrics_named['last_test_depth_value_mae']:.6f}, "
            f"best_align_mu_mse={best_test_metrics_named['best_test_alignment_mu_mse']:.6f}, "
            f"best_align_mu_rmse={best_test_metrics_named['best_test_alignment_mu_rmse']:.6f}"
        )

        if wandb is not None and wandb.run is not None:
            final_log = {
                "epoch": last_epoch,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_score": best_val_score,
                "best_test/depth_value_mae": best_test_metrics_named["best_test_depth_value_mae"],
                "best_test/depth_value_rmse": best_test_metrics_named["best_test_depth_value_rmse"],
                "best_test/ir_value_mae": best_test_metrics_named["best_test_ir_value_mae"],
                "best_test/alignment_mu_mse": best_test_metrics_named["best_test_alignment_mu_mse"],
                "best_test/alignment_mu_rmse": best_test_metrics_named["best_test_alignment_mu_rmse"],
                "last_test/depth_value_mae": last_test_metrics_named["last_test_depth_value_mae"],
                "last_test/depth_value_rmse": last_test_metrics_named["last_test_depth_value_rmse"],
                "last_test/ir_value_mae": last_test_metrics_named["last_test_ir_value_mae"],
                "last_test/alignment_mu_mse": last_test_metrics_named["last_test_alignment_mu_mse"],
                "last_test/alignment_mu_rmse": last_test_metrics_named["last_test_alignment_mu_rmse"],
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


def load_joint_autoencoder(checkpoint_path: str, device: str = "cpu") -> tuple[JointMultimodalAE, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    model_type = str(payload.get("model_type", ""))
    if model_type != "joint_multimodal_ae":
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' is not a joint AE checkpoint "
            f"(model_type={model_type!r}). Re-export/train with ae_joint_train.py."
        )
    config_dict = payload["config"]
    model = JointMultimodalAE(
        ir_input_dim=int(np.prod(payload["ir_window_shape"])),
        depth_input_shape=tuple(payload["input_shape"]),
        proprioception_input_shape=(
            tuple(payload["proprioception_input_shape"])
            if payload.get("proprioception_input_shape") is not None
            else None
        ),
        text_feature_dim=int(payload["text_feature_dim"]),
        condition_dim=int(config_dict["condition_dim"]),
        latent_dim=int(config_dict["latent_dim"]),
        ir_hidden_dims=tuple(config_dict["ir_hidden_dims"]),
        di_conv_channels=tuple(config_dict["di_conv_channels"]),
        di_pool_size=tuple(config_dict["di_pool_size"]),
        di_hidden_dim=int(config_dict["di_hidden_dim"]),
        proprio_hidden_dim=int(config_dict.get("proprio_hidden_dim", TrainConfig.proprio_hidden_dim)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def load_joint_model(checkpoint_path: str, device: str = "cpu") -> tuple[JointMultimodalAE, dict[str, Any]]:
    return load_joint_autoencoder(checkpoint_path, device=device)


@torch.no_grad()
def encode_di_window_to_latent_distribution(
    checkpoint_path: str,
    depth_window: np.ndarray | list,
    condition_text: str | None = None,
    device: str = "cpu",
    proprioception_window: np.ndarray | list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model, payload = load_joint_autoencoder(checkpoint_path, device=device)
    depth_array = np.asarray(depth_window, dtype=np.float32)
    expected_shape = tuple(payload["input_shape"])
    if tuple(depth_array.shape) != expected_shape:
        raise ValueError(f"Expected depth_window shape {expected_shape}, got {depth_array.shape}.")
    depth = torch.tensor(depth_array, dtype=torch.float32, device=device).unsqueeze(0)
    depth = (depth - payload["di_feature_mean"].to(device=device, dtype=torch.float32).unsqueeze(0)) / payload[
        "di_feature_std"
    ].to(device=device, dtype=torch.float32).unsqueeze(0)
    proprioception_input_shape = payload.get("proprioception_input_shape")
    if proprioception_input_shape is not None:
        if proprioception_window is None:
            raise ValueError(
                "This DI checkpoint expects proprioception_window input. "
                "Pass proprioception_window with the saved checkpoint shape."
            )
        proprioception_array = np.asarray(proprioception_window, dtype=np.float32)
        expected_proprioception_shape = tuple(proprioception_input_shape)
        if tuple(proprioception_array.shape) != expected_proprioception_shape:
            raise ValueError(
                "Expected proprioception_window shape "
                f"{expected_proprioception_shape}, got {proprioception_array.shape}."
            )
        proprioception = torch.tensor(proprioception_array, dtype=torch.float32, device=device).unsqueeze(0)
        proprio_feature_mean = payload.get("proprio_feature_mean")
        proprio_feature_std = payload.get("proprio_feature_std")
        if proprio_feature_mean is None or proprio_feature_std is None:
            raise RuntimeError(
                "Checkpoint declares proprioception input but is missing proprio_feature_mean/std."
            )
        proprioception = (
            proprioception - proprio_feature_mean.to(device=device, dtype=torch.float32).unsqueeze(0)
        ) / proprio_feature_std.to(device=device, dtype=torch.float32).unsqueeze(0)
    else:
        proprioception = None
    clip_cfg = payload["clip"]
    text_extractor = CLIPTextFeatureExtractor(
        model_id=clip_cfg["model_id"],
        device=device,
        cache_dir=clip_cfg["cache_dir"],
        local_files_only=clip_cfg["local_files_only"],
        quiet_load=True,
    )
    text = condition_text or payload["condition_text"]
    text_feature = text_extractor.encode([text]).to(device=device, dtype=torch.float32)
    mu, logvar = model.encode_di(depth, text_feature, proprioception)
    return mu.squeeze(0).cpu(), logvar.squeeze(0).cpu()


@torch.no_grad()
def encode_di_window_to_mu_latent(
    checkpoint_path: str,
    depth_window: np.ndarray | list,
    condition_text: str | None = None,
    device: str = "cpu",
    proprioception_window: np.ndarray | list | None = None,
) -> torch.Tensor:
    mu, _ = encode_di_window_to_latent_distribution(
        checkpoint_path,
        depth_window,
        condition_text=condition_text,
        device=device,
        proprioception_window=proprioception_window,
    )
    return mu


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Joint end-to-end AE for paired ir_window and depth_window data.")
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--condition-text", type=str, default=TrainConfig.condition_text)
    parser.add_argument("--ir-window-body-source", type=str, default=TrainConfig.ir_window_body_source, choices=IR_WINDOW_BODY_SOURCE_CHOICES)
    parser.add_argument("--output-root", type=str, default=TrainConfig.output_root)
    parser.add_argument("--run-name", type=str, default=TrainConfig.run_name)
    parser.add_argument("--latent-dim", type=int, default=TrainConfig.latent_dim)
    parser.add_argument("--condition-dim", type=int, default=TrainConfig.condition_dim)
    parser.add_argument("--ir-hidden-dim-1", type=int, default=TrainConfig.ir_hidden_dims[0])
    parser.add_argument("--ir-hidden-dim-2", type=int, default=TrainConfig.ir_hidden_dims[1])
    parser.add_argument("--di-conv-channel-1", type=int, default=TrainConfig.di_conv_channels[0])
    parser.add_argument("--di-conv-channel-2", type=int, default=TrainConfig.di_conv_channels[1])
    parser.add_argument("--di-conv-channel-3", type=int, default=TrainConfig.di_conv_channels[2])
    parser.add_argument("--di-pool-height", type=int, default=TrainConfig.di_pool_size[0])
    parser.add_argument("--di-pool-width", type=int, default=TrainConfig.di_pool_size[1])
    parser.add_argument("--di-hidden-dim", type=int, default=TrainConfig.di_hidden_dim)
    parser.add_argument(
        "--use-proprioception-window",
        type=str_to_bool,
        default=TrainConfig.use_proprioception_window,
    )
    parser.add_argument("--proprio-hidden-dim", type=int, default=TrainConfig.proprio_hidden_dim)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--value-loss-type", type=str, default=TrainConfig.value_loss_type, choices=VALUE_LOSS_CHOICES)
    parser.add_argument(
        "--deterministic-mu-training",
        type=str_to_bool,
        default=TrainConfig.deterministic_mu_training,
    )
    parser.add_argument("--ir-value-loss-weight", type=float, default=TrainConfig.ir_value_loss_weight)
    parser.add_argument("--di-value-loss-weight", type=float, default=TrainConfig.di_value_loss_weight)
    parser.add_argument("--latent-alignment-weight", type=float, default=TrainConfig.latent_alignment_weight)
    parser.add_argument("--decoder-train-samples", type=int, default=TrainConfig.decoder_train_samples)
    parser.add_argument("--decoder-eval-samples", type=int, default=TrainConfig.decoder_eval_samples)
    parser.add_argument("--di-loss-updates-decoder", type=str_to_bool, default=TrainConfig.di_loss_updates_decoder)
    parser.add_argument("--freeze-ir-after-epochs", type=int, default=TrainConfig.freeze_ir_after_epochs)
    parser.add_argument("--best-val-metric", type=str, default=TrainConfig.best_val_metric, choices=BEST_VAL_METRIC_CHOICES)
    parser.add_argument("--min-feature-std", type=float, default=TrainConfig.min_feature_std)
    parser.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--val-improvement-min-delta", type=float, default=TrainConfig.val_improvement_min_delta)
    parser.add_argument("--lr-plateau-patience", type=int, default=TrainConfig.lr_plateau_patience)
    parser.add_argument("--lr-plateau-factor", type=float, default=TrainConfig.lr_plateau_factor)
    parser.add_argument("--min-learning-rate", type=float, default=TrainConfig.min_learning_rate)
    parser.add_argument("--early-stop-patience", type=int, default=TrainConfig.early_stop_patience)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--split-mode", type=str, default=TrainConfig.split_mode, choices=SPLIT_MODE_CHOICES)
    parser.add_argument("--depth-input-noise-std", type=float, default=TrainConfig.depth_input_noise_std)
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
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        ir_hidden_dims=(args.ir_hidden_dim_1, args.ir_hidden_dim_2),
        di_conv_channels=(args.di_conv_channel_1, args.di_conv_channel_2, args.di_conv_channel_3),
        di_pool_size=(args.di_pool_height, args.di_pool_width),
        di_hidden_dim=args.di_hidden_dim,
        use_proprioception_window=bool(args.use_proprioception_window),
        proprio_hidden_dim=args.proprio_hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        value_loss_type=args.value_loss_type,
        deterministic_mu_training=bool(args.deterministic_mu_training),
        ir_value_loss_weight=args.ir_value_loss_weight,
        di_value_loss_weight=args.di_value_loss_weight,
        latent_alignment_weight=args.latent_alignment_weight,
        decoder_train_samples=args.decoder_train_samples,
        decoder_eval_samples=args.decoder_eval_samples,
        di_loss_updates_decoder=bool(args.di_loss_updates_decoder),
        freeze_ir_after_epochs=args.freeze_ir_after_epochs,
        best_val_metric=args.best_val_metric,
        min_feature_std=args.min_feature_std,
        max_grad_norm=args.max_grad_norm,
        val_improvement_min_delta=args.val_improvement_min_delta,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        min_learning_rate=args.min_learning_rate,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        device=args.device,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        split_mode=args.split_mode,
        depth_input_noise_std=args.depth_input_noise_std,
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
    best_checkpoint = train_joint(config)
    logger.info(f"Finished joint AE training. Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
