"""Whole Body Tracking-specific termination terms."""

from __future__ import annotations

from typing import Any, List

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import (
    quat_error_magnitude,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.task_phase import DEFAULT_TASK_PHASE_ANNOTATIONS, TwoPhaseSchedule


#########################################################################################################
## Termination terms
#########################################################################################################
def motion_ends(env, **_) -> torch.Tensor:
    """Terminate if the motion ends."""
    motion_command = env.command_manager.get_state("motion_command")
    clip_end_steps = getattr(motion_command, "clip_end_steps", None)
    if clip_end_steps is not None:
        return motion_command.time_steps >= (clip_end_steps - 2)
    return motion_command.time_steps >= motion_command.motion.time_step_total - 2


class ObjectRobotDistanceXY(TerminationTermBase):
    """Terminate when the object stays beyond a command-tightened XY threshold."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.exceeded_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.commanded_closing_distance = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self.last_dynamic_threshold = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._phase_schedule: TwoPhaseSchedule | None = None

    def __call__(
        self,
        env: Any,
        *,
        initial_distance_xy: float = 0.8,
        final_distance_margin_xy: float = 0.1,
        consecutive_steps: int = 10,
        annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
        **kwargs,
    ) -> torch.Tensor:
        motion_command = env.command_manager.get_state("motion_command")
        if motion_command is None or not motion_command.motion.has_object:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        if self._phase_schedule is None:
            self._phase_schedule = TwoPhaseSchedule(motion_command, annotation_path=annotation_path)

        delta_xy = motion_command.simulator_object_pos_w[:, :2] - motion_command.robot_root_pos_w[:, :2]
        distance_xy = torch.linalg.norm(delta_xy, dim=-1)

        # The adaptive Phase-0 command is expressed in the live robot-root frame.
        # Only its component pointing toward the object tightens the threshold.
        delta_xyz = motion_command.simulator_object_pos_w - motion_command.robot_root_pos_w
        delta_robot = quat_rotate_inverse(
            motion_command.robot_root_quat_w, delta_xyz, w_last=True
        )
        direction_xy = delta_robot[:, :2] / torch.linalg.norm(
            delta_robot[:, :2], dim=-1, keepdim=True
        ).clamp_min(1e-6)
        command_velocity = self._phase_schedule.velocity_command(motion_command)
        closing_velocity = torch.sum(command_velocity[:, :2] * direction_xy, dim=-1).clamp_min(0.0)
        phase_one = self._phase_schedule.phase(motion_command).bool()
        self.commanded_closing_distance += torch.where(
            phase_one,
            torch.zeros_like(closing_velocity),
            closing_velocity * float(env.dt),
        )

        starts = self._phase_schedule.phase_1_start_steps_for_envs(motion_command)
        reference_object_xy = motion_command.motion.object_pos_w[starts, :2]
        reference_robot_xy = motion_command.motion.body_pos_w[starts, 0, :2]
        final_threshold = (
            torch.linalg.norm(reference_object_xy - reference_robot_xy, dim=-1)
            + float(final_distance_margin_xy)
        )
        phase_zero_threshold = torch.maximum(
            final_threshold,
            torch.full_like(final_threshold, float(initial_distance_xy)) - self.commanded_closing_distance,
        )
        dynamic_threshold = torch.where(phase_one, final_threshold, phase_zero_threshold)
        self.last_dynamic_threshold.copy_(dynamic_threshold.detach())

        exceeded = distance_xy > dynamic_threshold
        self.exceeded_steps = torch.where(exceeded, self.exceeded_steps + 1, torch.zeros_like(self.exceeded_steps))
        return self.exceeded_steps >= max(int(consecutive_steps), 1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.exceeded_steps.zero_()
            self.commanded_closing_distance.zero_()
            self.last_dynamic_threshold.zero_()
        else:
            self.exceeded_steps[env_ids] = 0
            self.commanded_closing_distance[env_ids] = 0.0
            self.last_dynamic_threshold[env_ids] = 0.0


class ReferencePelvisHeightFallen(TerminationTermBase):
    """Terminate after the pelvis remains far below a phase-aware reference.

    Phase 0 uses the minimum reference pelvis height over the annotated
    approach segment, so bending/squatting demonstrations remain valid even
    when adaptive approach extends the phase. Phase 1 follows the live
    reference clock (already clamped by MotionCommand).
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.exceeded_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._phase_schedule: TwoPhaseSchedule | None = None
        self._pelvis_body_index: int | None = None
        self._phase_zero_min_height_by_clip: torch.Tensor | None = None

    def _initialize(self, motion_command: MotionCommand, pelvis_body_name: str, annotation_path: str) -> None:
        if pelvis_body_name not in self.env.body_names:
            raise ValueError(f"Fallen termination pelvis body '{pelvis_body_name}' was not found.")
        self._pelvis_body_index = self.env.body_names.index(pelvis_body_name)
        self._phase_schedule = TwoPhaseSchedule(motion_command, annotation_path=annotation_path)
        minima: list[torch.Tensor] = []
        for clip_id, (clip_start, _clip_end) in enumerate(motion_command.motion.clip_ranges):
            phase_one_start = int(self._phase_schedule.phase_1_start_steps[clip_id].item())
            heights = motion_command.motion.body_pos_w[int(clip_start):phase_one_start, self._pelvis_body_index, 2]
            if heights.numel() == 0:
                raise ValueError(f"Motion clip {clip_id} has an empty Phase-0 range.")
            minima.append(heights.min())
        self._phase_zero_min_height_by_clip = torch.stack(minima).to(self.env.device)

    def __call__(
        self,
        env: Any,
        *,
        pelvis_body_name: str = "pelvis_link",
        height_margin: float = 0.30,
        consecutive_steps: int = 5,
        annotation_path: str = DEFAULT_TASK_PHASE_ANNOTATIONS,
        **kwargs,
    ) -> torch.Tensor:
        motion_command = env.command_manager.get_state("motion_command")
        if self._phase_schedule is None:
            self._initialize(motion_command, pelvis_body_name, annotation_path)
        assert self._pelvis_body_index is not None
        assert self._phase_schedule is not None
        assert self._phase_zero_min_height_by_clip is not None

        env_origin_z = env.simulator.scene.env_origins[:, 2]
        phase_zero_height = self._phase_zero_min_height_by_clip[motion_command.clip_ids] + env_origin_z
        phase_one_height = motion_command.motion.body_pos_w[
            motion_command.time_steps, self._pelvis_body_index, 2
        ] + env_origin_z
        reference_height = torch.where(
            self._phase_schedule.phase(motion_command).bool(), phase_one_height, phase_zero_height
        )
        current_height = env.simulator._rigid_body_pos[:, self._pelvis_body_index, 2]
        exceeded = current_height < (reference_height - float(height_margin))
        self.exceeded_steps = torch.where(exceeded, self.exceeded_steps + 1, torch.zeros_like(self.exceeded_steps))
        return self.exceeded_steps >= max(int(consecutive_steps), 1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.exceeded_steps.zero_()
        else:
            self.exceeded_steps[env_ids] = 0


class BadTracking(TerminationTermBase):
    """Terminate if the tracking is bad.

    - bad ref pos
    - bad ref ori
    - bad motion body pos
    if has object:
        - bad object pos
        - bad object ori

    When bad tracking is detected, the motion_commmand.AdaptiveTimestepsSampler will be updated.
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self.bad_ref_pos_threshold = cfg.params["bad_ref_pos_threshold"]
        self.bad_ref_ori_threshold = cfg.params["bad_ref_ori_threshold"]

        self.bad_motion_body_pos_body_names = cfg.params["bad_motion_body_pos_body_names"]

        # NOTE: body_names_to_track is shared with command_manager
        self.body_names_to_track = cfg.params["body_names_to_track"]
        self.bad_motion_body_pos_threshold = cfg.params["bad_motion_body_pos_threshold"]
        self.bad_motion_body_pos_body_indexes = self._get_index_of_a_in_b(
            self.bad_motion_body_pos_body_names, self.body_names_to_track, self.env.device
        )

        self.bad_object_pos_threshold = cfg.params["bad_object_pos_threshold"]
        self.bad_object_ori_threshold = cfg.params["bad_object_ori_threshold"]
        self.last_reason_results: dict[str, torch.Tensor] = {}

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        # return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        bad_ref_pos = self.bad_ref_pos(motion_command)
        bad_ref_ori = self.bad_ref_ori(motion_command)
        bad_motion_body_pos = self.bad_motion_body_pos(motion_command)
        bad_tracking = bad_ref_pos | bad_ref_ori | bad_motion_body_pos
        bad_object_pos = torch.zeros_like(bad_tracking)
        bad_object_ori = torch.zeros_like(bad_tracking)

        if motion_command.motion.has_object:
            bad_object_pos = self.bad_object_pos(motion_command)
            bad_object_ori = self.bad_object_ori(motion_command)
            bad_tracking |= bad_object_pos | bad_object_ori

        self.last_reason_results = {
            "bad_ref_pos": bad_ref_pos.detach().clone(),
            "bad_ref_ori": bad_ref_ori.detach().clone(),
            "bad_motion_body_pos": bad_motion_body_pos.detach().clone(),
            "bad_object_pos": bad_object_pos.detach().clone(),
            "bad_object_ori": bad_object_ori.detach().clone(),
        }

        if motion_command.motion_cfg.use_adaptive_timesteps_sampler and torch.any(bad_tracking):
            failed_at_time_step = motion_command.time_steps[bad_tracking]
            motion_command.adaptive_timesteps_sampler.update_current_bin_failed_count(failed_at_time_step)

        return bad_tracking

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference position is too far from the robot's position."""
        return torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1) > self.bad_ref_pos_threshold

    def bad_ref_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference orientation is too far from the robot's orientation."""
        motion_projected_gravity_b = quat_rotate_inverse(
            motion_command.ref_quat_w, gravity_vector(self.env), w_last=True
        )
        robot_projected_gravity_b = quat_rotate_inverse(
            motion_command.robot_ref_quat_w, gravity_vector(self.env), w_last=True
        )
        return (
            torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > self.bad_ref_ori_threshold
        )

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the motion body position is too far from the robot's body position."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.norm(
            motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx], dim=-1
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)

    def bad_object_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object position is too far from the simulator's object position."""
        ref_pos = motion_command.object_pos_w
        if hasattr(motion_command, "object_pos_reward_offset"):
            ref_pos = ref_pos + motion_command.object_pos_reward_offset
        return (
            torch.norm(ref_pos - motion_command.simulator_object_pos_w, dim=-1)
            > self.bad_object_pos_threshold
        )

    def bad_object_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object orientation is too far from the simulator's object orientation."""
        return (
            quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)
            > self.bad_object_ori_threshold
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
