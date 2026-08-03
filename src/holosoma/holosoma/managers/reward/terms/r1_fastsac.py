"""Reference-guided, two-phase rewards for R1 FastSAC."""

from __future__ import annotations

from typing import Any

from holosoma.managers.object_contact import (
    get_cached_object_surface_distances,
    get_contact_target_point_distances,
    limit_contact_target_topk,
)
from holosoma.managers.reward.base import RewardTermBase
from holosoma.managers.reward.terms.wbt import (
    MotionJointPositionErrorExp,
    MotionJointVelocityErrorExp,
    ObjectContactLabelDistance,
    ObjectContactTargetPointDistance,
    _get_motion_command_and_assert_type,
    motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp,
)
from holosoma.utils.rotations import quat_error_magnitude, quat_rotate_inverse
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.task_phase import DEFAULT_TASK_PHASE_ANNOTATIONS, TwoPhaseSchedule


def _schedule(env: Any, annotation_path: str) -> TwoPhaseSchedule:
    schedules = getattr(env, "_r1_fastsac_phase_schedules", None)
    if schedules is None:
        schedules = {}
        env._r1_fastsac_phase_schedules = schedules
    if annotation_path not in schedules:
        schedules[annotation_path] = TwoPhaseSchedule(
            _get_motion_command_and_assert_type(env), annotation_path=annotation_path
        )
    return schedules[annotation_path]


def task_phase(env: Any, annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return _schedule(env, annotation_path).phase(motion_command)


def _phase_mask(env: Any, phase: int, annotation_path: str) -> torch.Tensor:
    return (task_phase(env, annotation_path) == int(phase)).float()


def _phase_one_blend(
    env: Any,
    annotation_path: str,
    blend_steps: int = 20,
) -> torch.Tensor:
    """Ramp reference priors in after the variable-length approach phase."""
    command = _get_motion_command_and_assert_type(env)
    schedule = _schedule(env, annotation_path)
    starts = schedule.phase_1_start_steps_for_envs(command)
    elapsed = (command.time_steps - starts + 1).clamp_min(0).float()
    ramp = torch.clamp(elapsed / max(float(blend_steps), 1.0), max=1.0)
    return ramp * (schedule.phase(command) == 1).float()


def _object_scale(env: Any) -> torch.Tensor:
    scales = getattr(env, "object_scale_factors", None)
    if scales is None:
        return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
    return scales.to(device=env.device, dtype=torch.float32).mean(dim=-1).clamp_min(1e-6)


def _spawn_adjusted_object_position(command: Any) -> torch.Tensor:
    """Translate the reference object path onto the episode's live spawn anchor.

    ``object_pos_reward_offset`` is recorded by ``MotionCommand`` after all reset
    adjustments have been applied.  It therefore contains the full XYZ shift:
    sector sampling in XY as well as the grounding/support-height correction in
    Z caused by object scaling.  Adding the same offset at every reference frame
    preserves the demonstrated object displacement while allowing a different
    episode spawn position.
    """
    reference_position = command.object_pos_w
    spawn_offset = getattr(command, "object_pos_reward_offset", None)
    if spawn_offset is not None:
        reference_position = reference_position + spawn_offset.to(
            device=reference_position.device,
            dtype=reference_position.dtype,
        )
    return reference_position


def tracking_lin_vel(
    env: Any,
    tracking_sigma: float = 0.25,
    annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
) -> torch.Tensor:
    """Track reference-root XY velocity in the reference/current root body frames."""
    command = _get_motion_command_and_assert_type(env)
    target = _schedule(env, annotation_path).velocity_command(command)
    actual = quat_rotate_inverse(command.robot_root_quat_w, command.robot_root_lin_vel_w, w_last=True)
    error = torch.sum(torch.square(target[:, :2] - actual[:, :2]), dim=-1)
    return torch.exp(-error / float(tracking_sigma))


def tracking_ang_vel(
    env: Any,
    tracking_sigma: float = 0.25,
    annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
) -> torch.Tensor:
    command = _get_motion_command_and_assert_type(env)
    target = _schedule(env, annotation_path).velocity_command(command)
    actual = quat_rotate_inverse(command.robot_root_quat_w, command.robot_root_ang_vel_w, w_last=True)
    error = torch.square(target[:, 2] - actual[:, 2])
    return torch.exp(-error / float(tracking_sigma))


class ScaledSurfaceDistanceProgress(ObjectContactLabelDistance):
    """Reward progress of annotated interaction bodies toward the scaled object surface."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous_distance = torch.full((env.num_envs,), float("nan"), device=env.device)

    def __call__(self, env: Any, annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS, **kwargs):
        command = _get_motion_command_and_assert_type(env)
        super().__call__(env, **kwargs)
        distances = get_cached_object_surface_distances(
            env=env,
            motion_command=command,
            body_names=self.contact_body_names,
            body_indices=self.contact_body_indices,
            body_local_offsets=self.contact_body_local_offsets,
            sample_points_by_key=self.sample_points_by_key,
        )
        starts = _schedule(env, annotation_path).phase_1_start_steps_for_envs(command)
        expected = command.motion.contact_object_label[starts][:, self.contact_label_columns].float()
        denom = expected.sum(dim=1).clamp_min(1.0)
        distance = (distances * expected).sum(dim=1) / denom / _object_scale(env)
        previous = torch.where(torch.isfinite(self.previous_distance), self.previous_distance, distance)
        progress = (previous - distance).clamp(min=-1.0, max=1.0)
        self.previous_distance.copy_(distance.detach())
        return progress * _phase_mask(env, 0, annotation_path)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.previous_distance.fill_(float("nan"))
        else:
            self.previous_distance[env_ids] = float("nan")


class _PhaseZeroContactTarget(ObjectContactTargetPointDistance):
    def _values(self, env: Any, annotation_path: str, target_topk: int | None, **kwargs):
        command = _get_motion_command_and_assert_type(env)
        if not self.initialized:
            self._initialize(
                command,
                body_names=kwargs.get("body_names"),
                contact_body_names_regex=kwargs.get("contact_body_names_regex", ".*"),
                fail_on_missing_targets=kwargs.get("fail_on_missing_targets", True),
            )
        starts = _schedule(env, annotation_path).phase_1_start_steps_for_envs(command)
        expected = command.motion.contact_object_label[starts][:, self.contact_label_columns]
        valid = command.motion.contact_object_target_valid[starts][:, self.contact_label_columns]
        active = expected & valid
        targets = command.motion.contact_object_target_points_obj[starts][:, self.contact_label_columns]
        targets = limit_contact_target_topk(targets, target_topk)
        distances, _, _ = get_contact_target_point_distances(
            env=env,
            motion_command=command,
            body_indices=self.contact_body_indices,
            body_local_offsets=self.contact_body_local_offsets,
            target_points_obj=targets,
        )
        return distances / _object_scale(env).unsqueeze(1), active


class ScaledContactTargetDistance(_PhaseZeroContactTarget):
    def __call__(
        self,
        env: Any,
        annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
        distance_scale: float = 0.12,
        target_topk: int | None = None,
        **kwargs,
    ):
        distances, active = self._values(env, annotation_path, target_topk, **kwargs)
        reward = torch.exp(-distances / max(float(distance_scale), 1e-6)) * active.float()
        reward = reward.sum(1) / active.float().sum(1).clamp_min(1.0)
        return reward * _phase_mask(env, 0, annotation_path)


class ScaledContactTargetCoverage(_PhaseZeroContactTarget):
    def __call__(
        self,
        env: Any,
        annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
        distance_threshold: float = 0.10,
        temperature: float = 0.02,
        target_topk: int | None = None,
        **kwargs,
    ):
        distances, active = self._values(env, annotation_path, target_topk, **kwargs)
        coverage = torch.sigmoid((float(distance_threshold) - distances) / max(float(temperature), 1e-6))
        reward = (coverage * active.float()).sum(1) / active.float().sum(1).clamp_min(1.0)
        return reward * _phase_mask(env, 0, annotation_path)


class PhaseTransitionBonus(RewardTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env: Any, annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS, **kwargs):
        phase = task_phase(env, annotation_path)
        bonus = (self.initialized & (self.previous_phase == 0) & (phase == 1)).float()
        self.previous_phase.copy_(phase.detach())
        self.initialized.fill_(True)
        return bonus

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.previous_phase.zero_()
            self.initialized.zero_()
        else:
            self.previous_phase[env_ids] = 0
            self.initialized[env_ids] = False


class PhaseOneReferenceContact(ObjectContactLabelDistance):
    def __call__(self, env: Any, annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS, **kwargs):
        return super().__call__(env, **kwargs) * _phase_mask(env, 1, annotation_path)


def weak_anchored_body_position_tracking(
    env: Any,
    sigma: float = 0.5,
    blend_steps: int = 20,
    annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
):
    reward = motion_relative_body_position_error_exp(env, sigma=float(sigma))
    return reward * _phase_one_blend(env, annotation_path, blend_steps)


def weak_anchored_body_orientation_tracking(
    env: Any,
    sigma: float = 0.7,
    blend_steps: int = 20,
    annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
):
    reward = motion_relative_body_orientation_error_exp(env, sigma=float(sigma))
    return reward * _phase_one_blend(env, annotation_path, blend_steps)


class PhaseOneMotionJointPositionErrorExp(MotionJointPositionErrorExp):
    """Relaxed joint-position prior enabled only after interaction begins."""

    def __call__(
        self,
        env: Any,
        *,
        sigma: float = 0.5,
        blend_steps: int = 20,
        annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
        **kwargs,
    ) -> torch.Tensor:
        reward = super().__call__(env, sigma=sigma, **kwargs)
        return reward * _phase_one_blend(env, annotation_path, blend_steps)


class PhaseOneMotionJointVelocityErrorExp(MotionJointVelocityErrorExp):
    """Relaxed joint-velocity prior enabled only after interaction begins."""

    def __call__(
        self,
        env: Any,
        *,
        sigma: float = 2.0,
        blend_steps: int = 20,
        annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
        **kwargs,
    ) -> torch.Tensor:
        reward = super().__call__(env, sigma=sigma, **kwargs)
        return reward * _phase_one_blend(env, annotation_path, blend_steps)


def delta_object_position_tracking(
    env: Any, sigma: float = 0.4, annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS
):
    command = _get_motion_command_and_assert_type(env)
    target_position = _spawn_adjusted_object_position(command)
    error = torch.linalg.norm(target_position - command.simulator_object_pos_w, dim=-1) / _object_scale(env)
    return torch.exp(-torch.square(error) / float(sigma) ** 2) * _phase_mask(env, 1, annotation_path)


def delta_object_orientation_tracking(
    env: Any, sigma: float = 0.7, annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS
):
    command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(command.object_quat_w, command.simulator_object_quat_w, w_last=True)
    return torch.exp(-torch.square(error) / float(sigma) ** 2) * _phase_mask(env, 1, annotation_path)


def delta_object_velocity_tracking(
    env: Any, linear_sigma: float = 1.0, angular_sigma: float = 2.0,
    annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
):
    command = _get_motion_command_and_assert_type(env)
    lin_error = torch.sum(torch.square(command.object_lin_vel_w - command.simulator_object_lin_vel_w), dim=-1)
    ang_error = torch.sum(torch.square(command.object_ang_vel_w - command.simulator_object_ang_vel_w), dim=-1)
    reward = torch.exp(-lin_error / float(linear_sigma) ** 2 - ang_error / float(angular_sigma) ** 2)
    return reward * _phase_mask(env, 1, annotation_path)


class FinishSuccessBonus(RewardTermBase):
    """Pay the terminal success bonus at most once per environment episode."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: Any,
        position_threshold: float = 0.20,
        orientation_threshold: float = 0.50,
        velocity_threshold: float = 0.30,
        annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
        **kwargs,
    ):
        command = _get_motion_command_and_assert_type(env)
        # Success belongs to the demonstrated task completion, not to the end
        # of the synthetic default-pose append used only for stabilization.
        phase_one_ends = _schedule(env, annotation_path).phase_1_end_steps_for_envs(command)
        near_end = command.time_steps >= (phase_one_ends - 3)
        target_position = _spawn_adjusted_object_position(command)
        pos_error = (
            torch.linalg.norm(target_position - command.simulator_object_pos_w, dim=-1)
            / _object_scale(env)
        )
        ori_error = quat_error_magnitude(
            command.object_quat_w, command.simulator_object_quat_w, w_last=True
        )
        vel_error = torch.linalg.norm(
            command.object_lin_vel_w - command.simulator_object_lin_vel_w, dim=-1
        )
        success = near_end & (pos_error < position_threshold) & (ori_error < orientation_threshold)
        success &= vel_error < velocity_threshold
        success &= task_phase(env, annotation_path) == 1

        bonus = success & ~self.paid
        self.paid |= success
        return bonus.float()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.paid.zero_()
        else:
            self.paid[env_ids] = False
