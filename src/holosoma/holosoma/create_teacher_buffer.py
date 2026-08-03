"""Convert extraction telemetry into a train-ready FastSAC teacher replay buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from holosoma.ae_pro_joint_train import CLIPTextFeatureExtractor, load_joint_model
from holosoma.utils.safe_torch_import import torch


DEFAULT_DI_PRO_AE = "./logs/AE/20260730_053423-ae-pro-joint-largebox/best.pt"
REQUIRED_INPUT_DATASETS = (
    "actor_observations",
    "critic_observations",
    "next_actor_observations",
    "next_critic_observations",
    "teacher_actions",
    "sac_rewards",
    "dones",
    "truncations",
    "depth_windows",
    "proprioception_windows",
)


def _metadata(group: Any) -> dict[str, Any]:
    encoded = group.attrs.get("metadata_json", "{}")
    if isinstance(encoded, bytes):
        encoded = encoded.decode("utf-8")
    value = json.loads(str(encoded))
    if not isinstance(value, dict):
        raise ValueError(f"Expected dictionary metadata in {group.name}.")
    return value


def _validate_group(group: Any) -> None:
    missing = [name for name in REQUIRED_INPUT_DATASETS if name not in group]
    if missing:
        raise ValueError(f"Input episode {group.name} is missing datasets: {missing}")
    lengths = {name: int(group[name].shape[0]) for name in REQUIRED_INPUT_DATASETS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Input episode {group.name} has inconsistent transition counts: {lengths}")


def _next_windows(current: np.ndarray, group: Any, dataset_name: str) -> np.ndarray:
    if dataset_name in group:
        return np.asarray(group[dataset_name][()], dtype=np.float32)
    return np.concatenate((current[1:], current[-1:]), axis=0)


def _next_depth_valid(group: Any, count: int) -> np.ndarray:
    if "next_depth_valid" in group:
        return np.asarray(group["next_depth_valid"][()], dtype=np.bool_)
    valid = np.ones(count, dtype=np.bool_)
    valid[-1] = False
    return valid


class DIProEncoder:
    def __init__(self, checkpoint: Path, device: str):
        self.checkpoint = checkpoint
        self.device = device
        self.model, self.payload = load_joint_model(str(checkpoint), device=device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.depth_shape = tuple(int(value) for value in self.payload["input_shape"])
        proprio_shape = self.payload.get("proprioception_input_shape")
        if proprio_shape is None:
            raise ValueError(
                f"Checkpoint {checkpoint} is not a DI-Pro checkpoint: proprioception_input_shape is missing."
            )
        self.proprio_shape = tuple(int(value) for value in proprio_shape)
        self.latent_dim = int(self.payload["config"]["latent_dim"])
        self.depth_mean = self.payload["di_feature_mean"].to(device=device, dtype=torch.float32)
        self.depth_std = self.payload["di_feature_std"].to(device=device, dtype=torch.float32).clamp_min(1e-6)
        self.proprio_mean = self.payload["proprio_feature_mean"].to(device=device, dtype=torch.float32)
        self.proprio_std = self.payload["proprio_feature_std"].to(
            device=device, dtype=torch.float32
        ).clamp_min(1e-6)

        clip_cfg = self.payload["clip"]
        self.text_encoder = CLIPTextFeatureExtractor(
            model_id=clip_cfg["model_id"],
            device=device,
            cache_dir=clip_cfg.get("cache_dir"),
            local_files_only=bool(clip_cfg.get("local_files_only", False)),
            quiet_load=True,
        )
        self.condition_source = str(self.payload.get("condition_source", "legacy_fixed_text"))
        self.fixed_condition = str(self.payload.get("condition_text") or "").strip()
        self._text_cache: dict[str, torch.Tensor] = {}

    def _text_feature(self, motion_name: str, batch_size: int) -> torch.Tensor:
        condition = motion_name if self.condition_source == "motion_name" else self.fixed_condition
        if not condition:
            raise ValueError("DI-Pro checkpoint requires a non-empty text condition.")
        if condition not in self._text_cache:
            self._text_cache[condition] = self.text_encoder.encode([condition]).to(
                device=self.device, dtype=torch.float32
            )
        return self._text_cache[condition].expand(batch_size, -1)

    @torch.no_grad()
    def encode(
        self,
        depth_windows: np.ndarray,
        proprioception_windows: np.ndarray,
        motion_name: str,
        batch_size: int,
    ) -> np.ndarray:
        if tuple(depth_windows.shape[1:]) != self.depth_shape:
            raise ValueError(
                f"Depth window shape mismatch: checkpoint={self.depth_shape}, data={depth_windows.shape[1:]}"
            )
        if tuple(proprioception_windows.shape[1:]) != self.proprio_shape:
            raise ValueError(
                "Proprioception window shape mismatch: "
                f"checkpoint={self.proprio_shape}, data={proprioception_windows.shape[1:]}"
            )

        outputs: list[np.ndarray] = []
        for start in range(0, len(depth_windows), batch_size):
            end = min(start + batch_size, len(depth_windows))
            depth = torch.as_tensor(depth_windows[start:end], device=self.device, dtype=torch.float32)
            proprio = torch.as_tensor(
                proprioception_windows[start:end], device=self.device, dtype=torch.float32
            )
            depth = (depth - self.depth_mean.unsqueeze(0)) / self.depth_std.unsqueeze(0)
            proprio = (proprio - self.proprio_mean.unsqueeze(0)) / self.proprio_std.unsqueeze(0)
            text = self._text_feature(motion_name, end - start)
            mu, _ = self.model.encode_di(depth, text, proprio)
            outputs.append(mu.float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _append_dataset(output: Any, name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if name not in output:
        output.create_dataset(
            name,
            data=values,
            maxshape=(None, *values.shape[1:]),
            chunks=True,
            compression="lzf",
        )
        return
    dataset = output[name]
    if tuple(dataset.shape[1:]) != tuple(values.shape[1:]):
        raise ValueError(
            f"Output dataset {name!r} shape changed from {dataset.shape[1:]} to {values.shape[1:]}."
        )
    old_size = int(dataset.shape[0])
    dataset.resize(old_size + len(values), axis=0)
    dataset[old_size:] = values


def create_teacher_buffer(
    input_h5: Path,
    output_h5: Path,
    di_pro_ae: Path,
    *,
    device: str,
    batch_size: int,
    overwrite: bool,
) -> None:
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("h5py is required to create a teacher replay buffer.") from exc

    if not input_h5.is_file():
        raise FileNotFoundError(f"Input telemetry H5 does not exist: {input_h5}")
    if not di_pro_ae.is_file():
        raise FileNotFoundError(f"DI-Pro AE checkpoint does not exist: {di_pro_ae}")
    if output_h5.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_h5}. Pass --overwrite to replace it.")
    if input_h5.resolve() == output_h5.resolve():
        raise ValueError("Input and output H5 paths must be different.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    if output_h5.exists():
        output_h5.unlink()

    encoder = DIProEncoder(di_pro_ae, device=device)
    total_input = 0
    total_saved = 0
    try:
        with h5py.File(input_h5, "r") as source, h5py.File(output_h5, "w") as output:
            if "episodes" not in source:
                raise ValueError(f"Input H5 has no 'episodes' group: {input_h5}")
            episode_names = sorted(source["episodes"].keys())
            if not episode_names:
                raise ValueError(f"Input H5 contains no episodes: {input_h5}")

            for episode_index, episode_name in enumerate(episode_names, start=1):
                group = source["episodes"][episode_name]
                _validate_group(group)
                metadata = _metadata(group)
                if not bool(metadata.get("fastsac_task_object_identity_included", False)):
                    raise ValueError(
                        f"Episode {group.name} was collected before FastSAC task/object identity observations "
                        "were added. Recollect telemetry with the current data_extraction.py."
                    )
                motion_name = str(metadata.get("motion_name") or "").strip()
                if not motion_name:
                    raise ValueError(f"Episode {group.name} has no motion_name metadata.")

                depth = np.asarray(group["depth_windows"][()], dtype=np.float32)
                proprio = np.asarray(group["proprioception_windows"][()], dtype=np.float32)
                next_depth = _next_windows(depth, group, "next_depth_windows")
                next_proprio = _next_windows(proprio, group, "next_proprioception_windows")
                depth_latent = encoder.encode(depth, proprio, motion_name, batch_size)
                next_depth_latent = encoder.encode(next_depth, next_proprio, motion_name, batch_size)

                actor_obs = np.asarray(group["actor_observations"][()], dtype=np.float32)
                next_actor_obs = np.asarray(group["next_actor_observations"][()], dtype=np.float32)
                critic_obs = np.asarray(group["critic_observations"][()], dtype=np.float32)
                next_critic_obs = np.asarray(group["next_critic_observations"][()], dtype=np.float32)
                observations = np.concatenate((actor_obs, depth_latent), axis=1)
                next_observations = np.concatenate((next_actor_obs, next_depth_latent), axis=1)
                dones = np.asarray(group["dones"][()], dtype=np.bool_)
                truncations = np.asarray(group["truncations"][()], dtype=np.bool_)
                next_valid = _next_depth_valid(group, len(depth))
                if next_valid.shape != dones.shape:
                    raise ValueError(
                        f"next_depth_valid shape {next_valid.shape} does not match dones shape "
                        f"{dones.shape} in {group.name}."
                    )

                # A terminal transition does not bootstrap, so its repeated next depth is harmless.
                # Timeout/run-end transitions require an exact next latent and are excluded when unavailable.
                keep = next_valid | (dones & ~truncations)
                total_input += len(keep)
                total_saved += int(keep.sum())
                data = {
                    "observations": observations,
                    "critic_observations": critic_obs,
                    "actions": np.asarray(group["teacher_actions"][()], dtype=np.float32),
                    "rewards": np.asarray(group["sac_rewards"][()], dtype=np.float32),
                    "dones": dones,
                    "truncations": truncations,
                    "next_observations": next_observations,
                    "next_critic_observations": next_critic_obs,
                    "depth_latents": depth_latent,
                    "next_depth_latents": next_depth_latent,
                }
                if np.any(keep):
                    for name, values in data.items():
                        _append_dataset(output, name, values[keep])

                logger.info(
                    f"Encoded episode {episode_index}/{len(episode_names)} {episode_name}: "
                    f"saved={int(keep.sum())}/{len(keep)}"
                )

            if total_saved == 0:
                raise ValueError("No trainable transitions remained after next-depth validity filtering.")

            output.attrs["format"] = "holosoma_fastsac_teacher_buffer"
            output.attrs["format_version"] = 1
            output.attrs["source_h5"] = str(input_h5.resolve())
            output.attrs["di_pro_ae"] = str(di_pro_ae.resolve())
            output.attrs["latent_mode"] = "mu"
            output.attrs["latent_dim"] = encoder.latent_dim
            output.attrs["num_transitions"] = total_saved
            output.attrs["num_dropped_transitions"] = total_input - total_saved
            output.attrs["observation_layout"] = "actor_observations_then_depth_latent"
    except Exception:
        if output_h5.exists():
            output_h5.unlink()
        raise

    logger.info(
        f"Created FastSAC teacher buffer: {output_h5} "
        f"({total_saved} transitions, dropped {total_input - total_saved})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode extraction depth windows and create a train-ready FastSAC teacher H5 buffer."
    )
    parser.add_argument("--input-h5", type=Path, required=True, help="data_extraction.py telemetry.h5")
    parser.add_argument("--output-h5", type=Path, required=True, help="Output FastSAC teacher-buffer H5")
    parser.add_argument("--di-pro-ae", type=Path, default=Path(DEFAULT_DI_PRO_AE))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_teacher_buffer(
        input_h5=args.input_h5,
        output_h5=args.output_h5,
        di_pro_ae=args.di_pro_ae,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
