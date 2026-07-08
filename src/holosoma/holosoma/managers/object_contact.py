"""Shared utilities for robot-object contact rewards and observations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Mapping, Sequence

import numpy as np
import torch

from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import quat_apply, quaternion_to_matrix

if TYPE_CHECKING:
    from holosoma.managers.command.terms.wbt import MotionCommand


VIRTUAL_CONTACT_BODY_SPECS: dict[str, tuple[str, tuple[float, float, float]]] = {
    "left_hand_contact_link": ("left_wrist_roll_link", (0.07, 0.0, 0.0)),
    "right_hand_contact_link": ("right_wrist_roll_link", (0.07, 0.0, 0.0)),
}


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


def select_contact_body_columns(
    available_contact_body_names: Sequence[str],
    *,
    body_names: Sequence[str] | str | None = None,
    body_names_regex: str = ".*",
) -> tuple[list[str], list[int]]:
    """Select contact-label body columns by explicit names or regex."""
    available = [str(name) for name in available_contact_body_names]
    if body_names is not None:
        requested = [str(body_names)] if isinstance(body_names, str) else [str(name) for name in body_names]
        missing = [name for name in requested if name not in available]
        if missing:
            raise RuntimeError(
                f"Contact target body names not found in motion labels: {missing}. Available: {available}"
            )
        return requested, [available.index(name) for name in requested]

    body_regex = re.compile(body_names_regex)
    selected = [(idx, name) for idx, name in enumerate(available) if body_regex.search(name)]
    if not selected:
        raise RuntimeError(
            f"No contact target body names matched regex '{body_names_regex}'. Available: {available}"
        )
    return [name for _, name in selected], [idx for idx, _ in selected]


def is_resolvable_contact_body_name(available_body_names: Sequence[str], body_name: str) -> bool:
    """Return whether a real or supported virtual contact body can be resolved."""
    if body_name in available_body_names:
        return True
    virtual_spec = VIRTUAL_CONTACT_BODY_SPECS.get(body_name)
    if virtual_spec is None:
        return False
    parent_body_name, _ = virtual_spec
    return parent_body_name in available_body_names


def resolve_contact_body_indices_and_offsets_from_names(
    available_body_names: Sequence[str],
    body_names: Sequence[str],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve contact labels into simulator body indices plus local offsets.

    Real simulator bodies use a zero local offset.  Supported virtual contact
    bodies, such as R1's fixed hand-contact markers, are represented as a parent
    body and a fixed local offset.  This keeps rewards/observations aligned with
    retargeting labels without adding fake actions or controllable links.
    """
    available_body_names = list(available_body_names)
    missing_names: list[str] = []
    body_indices: list[int] = []
    local_offsets: list[tuple[float, float, float]] = []

    for body_name in body_names:
        if body_name in available_body_names:
            body_indices.append(available_body_names.index(body_name))
            local_offsets.append((0.0, 0.0, 0.0))
            continue

        virtual_spec = VIRTUAL_CONTACT_BODY_SPECS.get(body_name)
        if virtual_spec is not None:
            parent_body_name, local_offset = virtual_spec
            if parent_body_name in available_body_names:
                body_indices.append(available_body_names.index(parent_body_name))
                local_offsets.append(local_offset)
                continue

        missing_names.append(body_name)

    if missing_names:
        raise ValueError(f"Contact body names not found in simulator body_names or virtual specs: {missing_names}")

    return (
        torch.tensor(body_indices, dtype=torch.long, device=device),
        torch.tensor(local_offsets, dtype=torch.float32, device=device),
    )


def resolve_contact_body_indices_and_offsets(
    env,
    body_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve contact labels against an environment's simulator body names."""
    return resolve_contact_body_indices_and_offsets_from_names(
        getattr(env.simulator, "body_names", []),
        body_names,
        device=env.device,
    )


def get_contact_body_positions_w(
    *,
    env,
    body_indices: torch.Tensor,
    body_local_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """World positions for real body centers or virtual contact points."""
    body_pos_w = env.simulator._rigid_body_pos[:, body_indices, :]
    if body_local_offsets is None or body_local_offsets.numel() == 0:
        return body_pos_w

    body_local_offsets = body_local_offsets.to(device=env.device, dtype=torch.float32)
    if torch.all(body_local_offsets == 0.0):
        return body_pos_w

    body_quat_w = env.simulator._rigid_body_rot[:, body_indices, :]
    offsets = body_local_offsets.unsqueeze(0).expand(env.num_envs, -1, -1)
    offsets_w = quat_apply(
        body_quat_w.reshape(-1, 4),
        offsets.reshape(-1, 3),
        w_last=True,
    ).view(env.num_envs, body_indices.numel(), 3)
    return body_pos_w + offsets_w


def get_object_local_points_w(
    *,
    env,
    motion_command: MotionCommand,
    object_local_points: torch.Tensor,
) -> torch.Tensor:
    """Transform object-local points to world frame using the live simulator object pose."""
    if object_local_points.ndim != 4 or object_local_points.shape[-1] != 3:
        raise ValueError(
            "object_local_points must have shape [num_envs, num_bodies, num_points, 3], "
            f"got {tuple(object_local_points.shape)}"
        )

    local_points = object_local_points.to(device=env.device, dtype=torch.float32)
    object_scales = getattr(env, "object_scale_factors", None)
    if object_scales is not None:
        local_points = local_points * object_scales.to(device=env.device, dtype=torch.float32)[:, None, None, :]

    num_envs, num_bodies, num_points, _ = local_points.shape
    flat_points = local_points.reshape(num_envs, num_bodies * num_points, 3)
    rot_w_from_obj = quaternion_to_matrix(motion_command.simulator_object_quat_w, w_last=True)
    points_w = (
        torch.bmm(flat_points, rot_w_from_obj.transpose(1, 2))
        + motion_command.simulator_object_pos_w[:, None, :]
    )
    return points_w.reshape(num_envs, num_bodies, num_points, 3)


def get_contact_target_point_distances(
    *,
    env,
    motion_command: MotionCommand,
    body_indices: torch.Tensor,
    body_local_offsets: torch.Tensor | None,
    target_points_obj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distances from contact bodies to their nearest labeled object target point.

    Returns:
        ``(distances, nearest_target_points_w, body_pos_w)`` with shapes
        ``[num_envs, num_bodies]``, ``[num_envs, num_bodies, 3]``, and
        ``[num_envs, num_bodies, 3]``.
    """
    body_pos_w = get_contact_body_positions_w(
        env=env,
        body_indices=body_indices,
        body_local_offsets=body_local_offsets,
    )
    if target_points_obj.shape[2] == 0:
        distances = torch.full(
            body_pos_w.shape[:2],
            float("inf"),
            dtype=torch.float32,
            device=env.device,
        )
        return distances, torch.zeros_like(body_pos_w), body_pos_w

    target_points_w = get_object_local_points_w(
        env=env,
        motion_command=motion_command,
        object_local_points=target_points_obj,
    )
    per_target_distances = torch.linalg.norm(body_pos_w[:, :, None, :] - target_points_w, dim=-1)
    distances, nearest_indices = torch.min(per_target_distances, dim=-1)
    nearest_points = torch.gather(
        target_points_w,
        dim=2,
        index=nearest_indices[:, :, None, None].expand(-1, -1, 1, 3),
    ).squeeze(2)
    return distances, nearest_points, body_pos_w


def get_cached_object_surface_distances(
    *,
    env,
    motion_command: MotionCommand,
    body_names: Sequence[str],
    body_indices: torch.Tensor,
    body_local_offsets: torch.Tensor | None = None,
    sample_points_by_key: Mapping[str | None, torch.Tensor],
) -> torch.Tensor:
    """Current min distance from each selected robot body to the active object surface.

    Distances are cached at env level so reward and observation terms can share the
    same robot-object distance computation within a simulation step.
    """
    body_names_key = tuple(str(name) for name in body_names)
    object_keys_key = tuple(sorted("__none__" if key is None else str(key) for key in sample_points_by_key))
    offsets_key = ()
    if body_local_offsets is not None and body_local_offsets.numel() > 0:
        offsets_key = tuple(round(float(value), 6) for value in body_local_offsets.detach().cpu().flatten().tolist())
    cache_key = (body_names_key, object_keys_key, offsets_key)
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

    body_pos_w = get_contact_body_positions_w(
        env=env,
        body_indices=body_indices,
        body_local_offsets=body_local_offsets,
    )
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
