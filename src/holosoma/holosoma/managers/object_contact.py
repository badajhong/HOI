"""Shared utilities for robot-object contact rewards and observations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Mapping, Sequence

import numpy as np
import torch

from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import quaternion_to_matrix

if TYPE_CHECKING:
    from holosoma.managers.command.terms.wbt import MotionCommand


def resolve_sample_points_root(env, sample_points_root: str | None) -> Path:
    """Resolve object sample-point root, defaulting to robot.object object-URDF folder."""
    if sample_points_root:
        root = Path(resolve_data_file_path(sample_points_root))
        if not root.exists():
            raise FileNotFoundError(f"sample_points_root does not exist: {root}")
        return root

    object_cfg = getattr(getattr(env, "robot_config", None), "object", None)
    candidates: list[str] = []
    for attr_name in ("object_urdf_asset", "object_urdf_folder"):
        candidate = getattr(object_cfg, attr_name, None)
        if candidate:
            candidates.append(candidate)

    object_urdf_path = getattr(object_cfg, "object_urdf_path", None)
    if object_urdf_path:
        candidates.append(str(Path(resolve_data_file_path(object_urdf_path)).parent))

    for candidate in candidates:
        root = Path(resolve_data_file_path(candidate))
        if root.exists():
            return root

    raise FileNotFoundError(
        "Could not infer sample_points_root. Set robot.object.object_urdf_asset/"
        "object_urdf_folder, or explicitly pass sample_points_root."
    )


def object_keys_for_envs(motion_command: MotionCommand, num_envs: int) -> list[str | None]:
    """Return active object key for each env."""
    object_key_to_id = getattr(motion_command, "object_key_to_id", None) or {}
    if not object_key_to_id:
        key = motion_command.motion.clip_object_keys[0] if motion_command.motion.clip_object_keys else None
        return [key] * num_envs

    id_to_key = {int(idx): key for key, idx in object_key_to_id.items()}
    object_type_ids = motion_command.object_type_ids.detach().cpu().tolist()
    return [id_to_key.get(int(type_id)) for type_id in object_type_ids]


def object_key_masks_for_envs(
    motion_command: MotionCommand,
    num_envs: int,
    device: torch.device | str,
) -> Iterator[tuple[str | None, torch.Tensor]]:
    """Yield active object keys with GPU masks for the envs using each object."""
    object_key_to_id = getattr(motion_command, "object_key_to_id", None) or {}
    object_type_ids = getattr(motion_command, "object_type_ids", None)
    if object_key_to_id and object_type_ids is not None:
        object_type_ids = object_type_ids.to(device=device)
        for object_key, object_type_id in sorted(object_key_to_id.items(), key=lambda item: str(item[0])):
            yield object_key, object_type_ids == int(object_type_id)
        return

    key = motion_command.motion.clip_object_keys[0] if motion_command.motion.clip_object_keys else None
    yield key, torch.ones(num_envs, dtype=torch.bool, device=device)


def resolve_sample_points_path(root: Path, object_key: str | None) -> Path:
    """Resolve sample_points.npy for a specific object key."""
    candidates: list[Path] = []
    if object_key is not None:
        candidates.append(root / object_key / "sample_points.npy")
        candidates.append(root / f"{object_key}_sample_points.npy")
    candidates.append(root / "sample_points.npy")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find sample_points.npy for object_key={object_key!r}. "
        f"Searched: {[str(path) for path in candidates]}"
    )


def load_sample_points_by_key(
    *,
    env,
    motion_command: MotionCommand,
    sample_points_root: str | None,
) -> tuple[Path, dict[str | None, torch.Tensor]]:
    """Load object surface sample points for every object used by the active motion set."""
    root = resolve_sample_points_root(env, sample_points_root)
    object_keys = sorted({key for key in motion_command.motion.clip_object_keys if key is not None})
    if not object_keys:
        object_keys = [None]

    sample_points_by_key: dict[str | None, torch.Tensor] = {}
    for object_key in object_keys:
        sample_path = resolve_sample_points_path(root, object_key)
        sample_points_by_key[object_key] = torch.tensor(
            np.load(sample_path),
            dtype=torch.float32,
            device=env.device,
        )
    return root, sample_points_by_key


def resolve_body_indices(env, body_names: Sequence[str]) -> torch.Tensor:
    """Resolve robot body names into simulator body indices."""
    available_body_names = list(getattr(env.simulator, "body_names", []))
    missing_names = [name for name in body_names if name not in available_body_names]
    if missing_names:
        raise ValueError(f"Body names not found in simulator body_names: {missing_names}")
    return torch.tensor(
        [available_body_names.index(name) for name in body_names],
        dtype=torch.long,
        device=env.device,
    )


def get_cached_object_surface_distances(
    *,
    env,
    motion_command: MotionCommand,
    body_names: Sequence[str],
    body_indices: torch.Tensor,
    sample_points_by_key: Mapping[str | None, torch.Tensor],
) -> torch.Tensor:
    """Current min distance from each selected robot body to the active object surface.

    Distances are cached at env level so reward and observation terms can share the
    same robot-object distance computation within a simulation step.
    """
    body_names_key = tuple(str(name) for name in body_names)
    object_keys_key = tuple(sorted("__none__" if key is None else str(key) for key in sample_points_by_key))
    cache_key = (body_names_key, object_keys_key)
    cache_generation = getattr(env, "_object_contact_surface_cache_generation", 0)
    sim_step = getattr(env.simulator, "_sim_step_counter", None)

    cache = getattr(env, "_object_contact_surface_cache", None)
    if cache is None:
        cache = {}
        env._object_contact_surface_cache = cache

    cached = cache.get(cache_key)
    if cached is not None:
        cached_step, cached_generation, cached_distances = cached
        if cached_step == sim_step and cached_generation == cache_generation:
            return cached_distances

    body_pos_w = env.simulator._rigid_body_pos[:, body_indices, :]
    object_pos_w = motion_command.simulator_object_pos_w
    object_quat_w = motion_command.simulator_object_quat_w

    rot_w_from_obj = quaternion_to_matrix(object_quat_w, w_last=True)
    body_pos_obj = torch.bmm(
        rot_w_from_obj.transpose(1, 2),
        (body_pos_w - object_pos_w[:, None, :]).transpose(1, 2),
    ).transpose(1, 2)

    distances = torch.full(
        (env.num_envs, len(body_names_key)),
        float("inf"),
        dtype=torch.float32,
        device=env.device,
    )

    object_scales = getattr(env, "object_scale_factors", None)
    for object_key, mask in object_key_masks_for_envs(motion_command, env.num_envs, env.device):
        sample_points = sample_points_by_key.get(object_key)
        if sample_points is None:
            sample_points = sample_points_by_key.get(None)
        if sample_points is None:
            raise RuntimeError(f"No object sample points loaded for object key: {object_key}")

        local_points = sample_points.to(device=env.device, dtype=torch.float32)
        if object_scales is not None:
            scaled_points = local_points.unsqueeze(0) * object_scales[mask].to(
                device=env.device, dtype=torch.float32
            ).unsqueeze(1)
            diff = body_pos_obj[mask, :, None, :] - scaled_points[:, None, :, :]
        else:
            diff = body_pos_obj[mask, :, None, :] - local_points[None, None, :, :]
        distances[mask] = torch.linalg.norm(diff, dim=-1).amin(dim=-1)

    cache[cache_key] = (sim_step, cache_generation, distances)
    return distances
