"""Deployment-facing command observations for R1 FastSAC."""

from __future__ import annotations

from holosoma.utils.rotations import (
    quat_rotate_inverse,
    quaternion_to_matrix,
    subtract_frame_transforms,
)
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.task_phase import DEFAULT_TASK_PHASE_ANNOTATIONS, TwoPhaseSchedule


def _motion_command(env):
    command = env.command_manager.get_state("motion_command")
    if command is None:
        raise RuntimeError("R1 FastSAC observations require motion_command.")
    return command


def _phase_schedule(env, command, annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS) -> TwoPhaseSchedule:
    schedules = getattr(env, "_r1_fastsac_observation_phase_schedules", None)
    if schedules is None:
        schedules = {}
        env._r1_fastsac_observation_phase_schedules = schedules
    if annotation_path not in schedules:
        schedules[annotation_path] = TwoPhaseSchedule(command, annotation_path=annotation_path)
    return schedules[annotation_path]


def reference_velocity_command(env) -> torch.Tensor:
    command = _motion_command(env)
    return _phase_schedule(env, command).velocity_command(command)


def task_phase_one_hot(env, annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS) -> torch.Tensor:
    command = _motion_command(env)
    schedule = _phase_schedule(env, command, annotation_path)
    return torch.nn.functional.one_hot(schedule.phase(command), num_classes=2).float()


def interaction_progress(env, annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS) -> torch.Tensor:
    """Normalized Phase-1 clock; zero throughout variable-length Phase 0."""
    command = _motion_command(env)
    schedule = _phase_schedule(env, command, annotation_path)
    phase_one = schedule.phase(command).bool()
    starts = schedule.phase_1_start_steps_for_envs(command)
    ends = schedule.phase_1_end_steps_for_envs(command)
    # ``ends`` is exclusive.  Normalize the first and last actual reference
    # frames to exactly 0 and 1; synthetic append frames then stay clamped at 1.
    duration = (ends - 1 - starts).clamp_min(1)
    progress = (command.time_steps - starts).float() / duration.float()
    progress = torch.where(phase_one, progress.clamp(0.0, 1.0), torch.zeros_like(progress))
    return progress.unsqueeze(-1)


def object_position_robot_frame(env) -> torch.Tensor:
    command = _motion_command(env)
    position, _ = subtract_frame_transforms(
        command.robot_root_pos_w,
        command.robot_root_quat_w,
        command.simulator_object_pos_w,
        command.simulator_object_quat_w,
    )
    return position


def object_orientation_robot_frame(env) -> torch.Tensor:
    command = _motion_command(env)
    _, orientation = subtract_frame_transforms(
        command.robot_root_pos_w,
        command.robot_root_quat_w,
        command.simulator_object_pos_w,
        command.simulator_object_quat_w,
    )
    return quaternion_to_matrix(orientation, w_last=True)[..., :2].reshape(env.num_envs, 6)


def object_linear_velocity_robot_frame(env) -> torch.Tensor:
    command = _motion_command(env)
    return quat_rotate_inverse(
        command.robot_root_quat_w, command.simulator_object_lin_vel_w, w_last=True
    )


def object_angular_velocity_robot_frame(env) -> torch.Tensor:
    command = _motion_command(env)
    return quat_rotate_inverse(
        command.robot_root_quat_w, command.simulator_object_ang_vel_w, w_last=True
    )


def object_world_position(env) -> torch.Tensor:
    return _motion_command(env).simulator_object_pos_w


def object_world_orientation(env) -> torch.Tensor:
    quat = _motion_command(env).simulator_object_quat_w
    return quaternion_to_matrix(quat, w_last=True)[..., :2].reshape(env.num_envs, 6)


def robot_world_position(env) -> torch.Tensor:
    return _motion_command(env).robot_root_pos_w


def robot_world_orientation(env) -> torch.Tensor:
    quat = _motion_command(env).robot_root_quat_w
    return quaternion_to_matrix(quat, w_last=True)[..., :2].reshape(env.num_envs, 6)


def object_scale(env) -> torch.Tensor:
    scales = getattr(env, "object_scale_factors", None)
    if scales is None:
        return torch.ones(env.num_envs, 3, device=env.device, dtype=torch.float32)
    return scales.to(device=env.device, dtype=torch.float32)


def _select_active_object_values(env, values_by_key: dict[str, torch.Tensor], width: int) -> torch.Tensor:
    command = _motion_command(env)
    result = torch.zeros(env.num_envs, width, device=env.device, dtype=torch.float32)
    key_to_id = getattr(command, "object_key_to_id", None) or {}
    type_ids = getattr(command, "object_type_ids", None)
    for key, values in values_by_key.items():
        if type_ids is None or key not in key_to_id:
            result.copy_(values.to(device=env.device, dtype=torch.float32))
        else:
            mask = type_ids == int(key_to_id[key])
            result[mask] = values.to(device=env.device, dtype=torch.float32)[mask]
    return result


def _object_physics_values(env) -> dict[str, dict[str, torch.Tensor]]:
    """Cache actual mass, local COM and friction after startup randomization."""
    cached = getattr(env, "_r1_fastsac_object_physics", None)
    if cached is not None:
        return cached

    command = _motion_command(env)
    cached = {"mass": {}, "com": {}, "friction": {}}
    for key in getattr(command, "object_key_to_id", {}) or {}:
        asset_name = f"object_{key}"
        try:
            asset = env.simulator.scene[asset_name]
            masses = asset.root_physx_view.get_masses().to(env.device)
            coms = asset.root_physx_view.get_coms().to(env.device)
            materials = asset.root_physx_view.get_material_properties().to(env.device)
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                f"FastSAC privileged object physics requires IsaacSim scene asset {asset_name!r}."
            ) from exc

        if masses.ndim == 1:
            masses = masses.unsqueeze(-1)
        cached["mass"][key] = masses.sum(dim=1, keepdim=True).float()
        if coms.ndim == 2:
            coms = coms.unsqueeze(1)
        cached["com"][key] = coms[:, 0, :3].float()
        if materials.ndim == 2:
            materials = materials.unsqueeze(1)
        cached["friction"][key] = materials[..., :2].mean(dim=1).float()

    env._r1_fastsac_object_physics = cached
    return cached


def object_mass(env) -> torch.Tensor:
    return _select_active_object_values(env, _object_physics_values(env)["mass"], 1)


def object_friction(env) -> torch.Tensor:
    """Actual static and dynamic friction; restitution is excluded."""
    return _select_active_object_values(env, _object_physics_values(env)["friction"], 2)


def object_center_of_mass(env) -> torch.Tensor:
    """Actual local center-of-mass offset of the active object."""
    return _select_active_object_values(env, _object_physics_values(env)["com"], 3)


def object_initial_position_robot_frame(env) -> torch.Tensor:
    command = _motion_command(env)
    starts = command.clip_start_steps
    object_position = command.motion.object_pos_w[starts]
    spawn_offset = getattr(command, "object_pos_reward_offset", None)
    if spawn_offset is not None:
        object_position = object_position + spawn_offset
    robot_position = command.motion.body_pos_w[starts, 0]
    robot_orientation = command.motion.body_quat_w[starts, 0]
    position, _ = subtract_frame_transforms(
        robot_position, robot_orientation, object_position, command.motion.object_quat_w[starts]
    )
    return position


def object_initial_orientation_robot_frame(env) -> torch.Tensor:
    command = _motion_command(env)
    starts = command.clip_start_steps
    _, orientation = subtract_frame_transforms(
        command.motion.body_pos_w[starts, 0],
        command.motion.body_quat_w[starts, 0],
        command.motion.object_pos_w[starts],
        command.motion.object_quat_w[starts],
    )
    return quaternion_to_matrix(orientation, w_last=True)[..., :2].reshape(env.num_envs, 6)
