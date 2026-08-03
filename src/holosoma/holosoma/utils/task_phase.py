"""Two-phase task annotations shared by R1 FastSAC and data extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.rotations import get_euler_xyz, quat_mul, quat_inverse, quat_rotate_inverse


DEFAULT_TASK_PHASE_ANNOTATIONS = "train_r1/annotations/task_phase_labels.yaml"


class TwoPhaseSchedule:
    """Resolve per-motion approach/interaction boundaries from YAML."""

    def __init__(self, motion_command: Any, annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS):
        resolved = Path(resolve_data_file_path(annotation_path))
        payload = yaml.safe_load(resolved.read_text()) or {}
        motions = payload.get("motions", {})
        if not isinstance(motions, dict):
            raise ValueError(f"Expected a 'motions' mapping in {resolved}.")

        clip_files = list(getattr(motion_command.motion, "clip_files", ()))
        clip_ranges = list(getattr(motion_command.motion, "clip_ranges", ()))
        if len(clip_files) != len(clip_ranges):
            raise ValueError("Motion clip files and clip ranges have different lengths.")

        # ``clip_ranges`` includes any synthetic default-pose prepend/append
        # frames inserted by MotionCommand.  Phase annotations, however, are
        # local to the original motion files, so anchor them to the preserved
        # real-motion ranges when those ranges are available.
        real_clip_ranges_value = getattr(motion_command.motion, "real_motion_clip_ranges", None)
        if real_clip_ranges_value is None:
            phase_clip_ranges = clip_ranges
        else:
            phase_clip_ranges = list(real_clip_ranges_value)
            if len(clip_files) != len(phase_clip_ranges):
                raise ValueError("Motion clip files and real motion clip ranges have different lengths.")

        phase_starts: list[int] = []
        phase_ends: list[int] = []
        for clip_file, (clip_start, clip_end), (real_start, real_end) in zip(
            clip_files, clip_ranges, phase_clip_ranges
        ):
            key = self._match_key(str(clip_file), motions)
            entry = motions[key]
            local_start = entry.get("phase_1_start_timestep") if isinstance(entry, dict) else None
            if local_start is None:
                raise ValueError(f"Missing phase_1_start_timestep for motion '{key}' in {resolved}.")
            local_start = int(local_start)
            clip_start = int(clip_start)
            clip_end = int(clip_end)
            real_start = int(real_start)
            real_end = int(real_end)
            if not clip_start <= real_start < real_end <= clip_end:
                raise ValueError(
                    f"Invalid real motion range ({real_start}, {real_end}) for '{key}' "
                    f"inside clip range ({clip_start}, {clip_end})."
                )
            real_motion_length = real_end - real_start
            if not 0 < local_start < real_motion_length:
                raise ValueError(
                    f"Invalid phase_1_start_timestep={local_start} for '{key}' "
                    f"with {real_motion_length} real motion frames."
                )
            phase_starts.append(real_start + local_start)
            phase_ends.append(real_end)

        self.annotation_path = str(resolved)
        self.phase_1_start_steps = torch.tensor(phase_starts, dtype=torch.long, device=motion_command.device)
        self.phase_1_end_steps = torch.tensor(phase_ends, dtype=torch.long, device=motion_command.device)

    @staticmethod
    def _match_key(clip_file: str, motions: dict[str, Any]) -> str:
        normalized = clip_file.replace("\\", "/")
        suffix_matches = [str(key) for key in motions if normalized.endswith(str(key).replace("\\", "/"))]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        stem_matches = [str(key) for key in motions if Path(str(key)).stem == Path(clip_file).stem]
        if len(stem_matches) == 1:
            return stem_matches[0]
        raise KeyError(f"Could not uniquely match motion annotation for '{clip_file}'.")

    def phase(self, motion_command: Any) -> torch.Tensor:
        adaptive_phase = getattr(motion_command, "_fastsac_phase_one", None)
        if bool(getattr(getattr(motion_command, "motion_cfg", None), "adaptive_phase_zero", False)) and adaptive_phase is not None:
            return adaptive_phase.to(dtype=torch.long)
        starts = self.phase_1_start_steps[motion_command.clip_ids]
        return (motion_command.time_steps >= starts).to(dtype=torch.long)

    def phase_1_start_steps_for_envs(self, motion_command: Any) -> torch.Tensor:
        return self.phase_1_start_steps[motion_command.clip_ids]

    def phase_1_end_steps_for_envs(self, motion_command: Any) -> torch.Tensor:
        """Return the exclusive original-motion end, excluding synthetic append frames."""
        return self.phase_1_end_steps[motion_command.clip_ids]

    def _phase_one_robot_target(self, motion_command: Any) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.phase_1_start_steps_for_envs(motion_command)
        reference_object_pos = motion_command.motion.object_pos_w[starts]
        reference_robot_pos = motion_command.motion.body_pos_w[starts, 0]
        reference_robot_quat = motion_command.motion.body_quat_w[starts, 0]
        # Preserve the reference robot-object displacement at the live object's
        # shifted position. Spawn randomization currently changes XY translation only.
        target_robot_pos = reference_robot_pos + (motion_command.simulator_object_pos_w - reference_object_pos)
        return target_robot_pos, reference_robot_quat

    def approach_errors(self, motion_command: Any) -> tuple[torch.Tensor, torch.Tensor]:
        target_pos, target_quat = self._phase_one_robot_target(motion_command)
        position_error_world = target_pos - motion_command.robot_root_pos_w
        position_error_body = quat_rotate_inverse(
            motion_command.robot_root_quat_w, position_error_world, w_last=True
        )
        delta_quat = quat_mul(
            target_quat, quat_inverse(motion_command.robot_root_quat_w, w_last=True), w_last=True
        )
        yaw_error = get_euler_xyz(delta_quat, w_last=True)[2]
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        return position_error_body, yaw_error

    def update_adaptive_phase(self, motion_command: Any) -> torch.Tensor:
        """Update readiness and return which reference clocks may advance."""
        starts = self.phase_1_start_steps_for_envs(motion_command)
        phase_one = motion_command._fastsac_phase_one
        hold_counts = motion_command._fastsac_phase_zero_ready_counts
        at_boundary = (motion_command.time_steps >= starts) & ~phase_one
        position_error, yaw_error = self.approach_errors(motion_command)
        cfg = motion_command.motion_cfg
        ready = (
            torch.linalg.norm(position_error[:, :2], dim=-1) <= float(cfg.phase_zero_position_threshold_m)
        ) & (torch.abs(yaw_error) <= float(cfg.phase_zero_yaw_threshold_rad))
        hold_counts[at_boundary & ready] += 1
        hold_counts[at_boundary & ~ready] = 0
        required = max(int(cfg.phase_zero_ready_hold_steps), 1)
        phase_one |= at_boundary & (hold_counts >= required)
        return (motion_command.time_steps < starts) | phase_one

    def velocity_command(self, motion_command: Any) -> torch.Tensor:
        """Phase-0 pose-goal command and unmodified reference command in Phase 1."""
        lin_ref = quat_rotate_inverse(motion_command.root_quat_w, motion_command.root_lin_vel_w, w_last=True)
        ang_ref = quat_rotate_inverse(motion_command.root_quat_w, motion_command.root_ang_vel_w, w_last=True)
        result = torch.stack((lin_ref[:, 0], lin_ref[:, 1], ang_ref[:, 2]), dim=-1)
        if not bool(getattr(getattr(motion_command, "motion_cfg", None), "adaptive_phase_zero", False)):
            return result
        position_error, yaw_error = self.approach_errors(motion_command)
        cfg = motion_command.motion_cfg
        approach = torch.zeros_like(result)
        approach[:, :2] = position_error[:, :2] * float(cfg.phase_zero_linear_velocity_gain)
        speed = torch.linalg.norm(approach[:, :2], dim=-1, keepdim=True).clamp_min(1e-6)
        max_speed = float(cfg.phase_zero_max_linear_velocity)
        approach[:, :2] *= torch.clamp(max_speed / speed, max=1.0)
        approach[:, 2] = torch.clamp(
            yaw_error * float(cfg.phase_zero_angular_velocity_gain),
            min=-float(cfg.phase_zero_max_angular_velocity),
            max=float(cfg.phase_zero_max_angular_velocity),
        )
        return torch.where(self.phase(motion_command).bool().unsqueeze(1), result, approach)
