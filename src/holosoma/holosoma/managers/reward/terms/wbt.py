"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch
from loguru import logger

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.object_contact import (
    get_contact_target_point_distances,
    get_cached_object_surface_distances,
    load_sample_points_by_key,
    object_key_masks_for_envs,
    select_contact_body_columns,
)
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude, quaternion_to_matrix

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def penalty_residual_action_l2(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize residual corrections away from the frozen student's base action."""
    residual_actions = env.action_manager.action - env.student_base_actions
    return torch.sum(torch.square(residual_actions), dim=1)


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    ref_pos = motion_command.object_pos_w
    if hasattr(motion_command, "object_pos_reward_offset"):
        ref_pos = ref_pos + motion_command.object_pos_reward_offset
    error = torch.sum(torch.square(ref_pos - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


class ObjectPointCloudDistanceExp(RewardTermBase):
    """Reward object pose tracking by comparing sampled object surface points."""

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.initialized = False
        self.sample_points_by_key: dict[str | None, torch.Tensor] = {}

    def __call__(
        self,
        env: WholeBodyTrackingManager,
        *,
        sample_points_root: str | None = None,
        distance_scale: float = 10.0,
        max_points: int = 1024,
        **kwargs,
    ) -> torch.Tensor:
        motion_command = _get_motion_command_and_assert_type(env)
        if not self.initialized:
            self._initialize(motion_command, sample_points_root=sample_points_root, max_points=max_points)

        distances = self._object_point_cloud_distance(env, motion_command)
        return torch.exp(-float(distance_scale) * distances)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _initialize(
        self,
        motion_command: MotionCommand,
        *,
        sample_points_root: str | None,
        max_points: int,
    ) -> None:
        root, sample_points_by_key = load_sample_points_by_key(
            env=self.env,
            motion_command=motion_command,
            sample_points_root=sample_points_root,
        )

        if max_points > 0:
            for object_key, sample_points in list(sample_points_by_key.items()):
                if sample_points.shape[0] <= max_points:
                    continue
                indices = torch.linspace(
                    0,
                    sample_points.shape[0] - 1,
                    max_points,
                    dtype=torch.long,
                    device=sample_points.device,
                )
                sample_points_by_key[object_key] = sample_points.index_select(0, indices)

        self.sample_points_by_key = sample_points_by_key
        logger.info(
            "Initialized ObjectPointCloudDistanceExp: "
            f"objects={list(self.sample_points_by_key.keys())}, sample_points_root={root}, max_points={max_points}"
        )
        self.initialized = True

    def _object_point_cloud_distance(
        self,
        env: WholeBodyTrackingManager,
        motion_command: MotionCommand,
    ) -> torch.Tensor:
        ref_pos_w = motion_command.object_pos_w
        if hasattr(motion_command, "object_pos_reward_offset"):
            ref_pos_w = ref_pos_w + motion_command.object_pos_reward_offset
        ref_quat_w = motion_command.object_quat_w

        current_pos_w = motion_command.simulator_object_pos_w
        current_quat_w = motion_command.simulator_object_quat_w
        ref_rot_w_from_obj = quaternion_to_matrix(ref_quat_w, w_last=True)
        current_rot_w_from_obj = quaternion_to_matrix(current_quat_w, w_last=True)

        distances = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        object_scales = getattr(env, "object_scale_factors", None)

        for object_key, mask in object_key_masks_for_envs(motion_command, env.num_envs, env.device):
            local_points = self.sample_points_by_key.get(object_key)
            if local_points is None:
                local_points = self.sample_points_by_key.get(None)
            if local_points is None:
                raise RuntimeError(f"No object sample points loaded for object key: {object_key}")

            local_points = local_points.to(device=env.device, dtype=torch.float32)
            selected_ref_pos_w = ref_pos_w[mask]
            selected_ref_rot_w_from_obj = ref_rot_w_from_obj[mask]
            selected_current_pos_w = current_pos_w[mask]
            selected_current_rot_w_from_obj = current_rot_w_from_obj[mask]
            if object_scales is not None:
                points = local_points.unsqueeze(0) * object_scales[mask].to(
                    device=env.device, dtype=torch.float32
                ).unsqueeze(1)
            else:
                points = local_points.unsqueeze(0).expand(selected_ref_pos_w.shape[0], -1, -1)

            ref_points_w = (
                torch.bmm(points, selected_ref_rot_w_from_obj.transpose(1, 2))
                + selected_ref_pos_w.unsqueeze(1)
            )
            current_points_w = (
                torch.bmm(points, selected_current_rot_w_from_obj.transpose(1, 2))
                + selected_current_pos_w.unsqueeze(1)
            )
            distances[mask] = torch.linalg.norm(current_points_w - ref_points_w, dim=-1).mean(dim=-1)

        return distances


# ================================================================================================
# Labeled Object Contact Rewards
# ================================================================================================


class ObjectContactLabelDistance(RewardTermBase):
    """Reward expected contact bodies for being near object surface sample points.

    Contact labels are expected to come from SMPLH/object annotations loaded by
    ``MotionCommand``. The live robot body positions are compared against object
    sample points in the active object's local frame.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.initialized = False
        self.warned_missing_labels = False
        self.sample_points_by_key: dict[str | None, torch.Tensor] = {}
        self.contact_label_columns = torch.zeros(0, dtype=torch.long, device=env.device)
        self.contact_body_names: list[str] = []
        self.contact_body_indices = torch.zeros(0, dtype=torch.long, device=env.device)
        self.contact_body_local_offsets = torch.zeros(0, 3, dtype=torch.float32, device=env.device)

    def __call__(
        self,
        env: WholeBodyTrackingManager,
        *,
        sample_points_root: str | None = None,
        threshold: float = 0.08,
        distance_scale: float | None = None,
        contact_body_names_regex: str = ".*",
        fail_on_missing_labels: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        motion_command = _get_motion_command_and_assert_type(env)
        if not self.initialized:
            self._initialize(
                motion_command,
                sample_points_root=sample_points_root,
                contact_body_names_regex=contact_body_names_regex,
                fail_on_missing_labels=fail_on_missing_labels,
            )

        if not motion_command.has_contact_labels or self.contact_label_columns.numel() == 0:
            return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

        expected = motion_command.contact_object_label[:, self.contact_label_columns]
        has_expected = expected.any(dim=1)
        if not torch.any(has_expected):
            return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

        active_env_ids = has_expected.nonzero(as_tuple=False).flatten()
        distances = torch.full(
            (env.num_envs, self.contact_label_columns.numel()),
            float("inf"),
            dtype=torch.float32,
            device=env.device,
        )
        distances[active_env_ids] = get_cached_object_surface_distances(
            env=env,
            motion_command=motion_command,
            body_names=self.contact_body_names,
            body_indices=self.contact_body_indices,
            body_local_offsets=self.contact_body_local_offsets,
            sample_points_by_key=self.sample_points_by_key,
            env_ids=active_env_ids,
        )

        if distance_scale is None:
            distance_scale = threshold
        scale_tensor = torch.tensor(max(float(distance_scale), 1e-6), dtype=torch.float32, device=env.device)
        per_body_reward = torch.exp(-distances / scale_tensor)
        per_body_reward = per_body_reward * expected.float()
        denom = expected.float().sum(dim=1).clamp(min=1.0)
        reward = per_body_reward.sum(dim=1) / denom
        return torch.where(has_expected, reward, torch.zeros_like(reward))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _initialize(
        self,
        motion_command: MotionCommand,
        *,
        sample_points_root: str | None,
        contact_body_names_regex: str,
        fail_on_missing_labels: bool,
    ) -> None:
        if not motion_command.has_contact_labels:
            message = (
                "ObjectContactLabelDistance is enabled but motion_command has no contact labels. "
                "Embed contact labels in the motion files or set motion_config.contact_file for a single motion."
            )
            if fail_on_missing_labels:
                raise RuntimeError(message)
            if not self.warned_missing_labels:
                logger.warning(message)
                self.warned_missing_labels = True
            self.initialized = True
            return

        body_regex = re.compile(contact_body_names_regex)
        selected_cols = [
            idx for idx, name in enumerate(motion_command.motion.contact_body_names) if body_regex.search(name)
        ]
        if not selected_cols:
            raise RuntimeError(
                f"No contact label body names matched regex '{contact_body_names_regex}'. "
                f"Available: {motion_command.motion.contact_body_names}"
            )
        self.contact_label_columns = torch.tensor(selected_cols, dtype=torch.long, device=self.env.device)
        self.contact_body_names = [motion_command.motion.contact_body_names[i] for i in selected_cols]
        contact_body_indices = torch.as_tensor(
            motion_command.motion.contact_body_indices,
            dtype=torch.long,
            device=self.env.device,
        )
        self.contact_body_indices = contact_body_indices[self.contact_label_columns]
        contact_body_local_offsets = getattr(motion_command.motion, "contact_body_local_offsets", None)
        if contact_body_local_offsets is None:
            contact_body_local_offsets = torch.zeros(
                len(motion_command.motion.contact_body_names),
                3,
                dtype=torch.float32,
                device=self.env.device,
            )
        contact_body_local_offsets = torch.as_tensor(
            contact_body_local_offsets,
            dtype=torch.float32,
            device=self.env.device,
        )
        self.contact_body_local_offsets = contact_body_local_offsets[self.contact_label_columns]

        root, self.sample_points_by_key = load_sample_points_by_key(
            env=self.env,
            motion_command=motion_command,
            sample_points_root=sample_points_root,
        )

        logger.info(
            "Initialized ObjectContactLabelDistance: "
            f"bodies={[motion_command.motion.contact_body_names[i] for i in selected_cols]}, "
            f"objects={list(self.sample_points_by_key.keys())}, sample_points_root={root}"
        )
        self.initialized = True


class ObjectContactTargetPointDistance(RewardTermBase):
    """Reward labeled contact bodies for reaching their labeled object-surface target points."""

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.initialized = False
        self.warned_missing_targets = False
        self.contact_label_columns = torch.zeros(0, dtype=torch.long, device=env.device)
        self.contact_body_names: list[str] = []
        self.contact_body_indices = torch.zeros(0, dtype=torch.long, device=env.device)
        self.contact_body_local_offsets = torch.zeros(0, 3, dtype=torch.float32, device=env.device)

    def __call__(
        self,
        env: WholeBodyTrackingManager,
        *,
        distance_scale: float | None = None,
        margin: float | None = None,
        body_names: tuple[str, ...] | list[str] | str | None = None,
        contact_body_names_regex: str = ".*",
        fail_on_missing_targets: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        motion_command = _get_motion_command_and_assert_type(env)
        if not self.initialized:
            self._initialize(
                motion_command,
                body_names=body_names,
                contact_body_names_regex=contact_body_names_regex,
                fail_on_missing_targets=fail_on_missing_targets,
            )

        if self.contact_label_columns.numel() == 0:
            return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

        expected = motion_command.contact_object_label[:, self.contact_label_columns]
        target_valid = motion_command.contact_object_target_valid[:, self.contact_label_columns]
        active = expected & target_valid
        has_active = active.any(dim=1)
        if not torch.any(has_active):
            return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

        active_env_ids = has_active.nonzero(as_tuple=False).flatten()
        target_points_obj = motion_command.contact_object_target_points_obj[:, self.contact_label_columns]
        distances = torch.full(
            (env.num_envs, self.contact_label_columns.numel()),
            float("inf"),
            dtype=torch.float32,
            device=env.device,
        )
        active_distances, _, _ = get_contact_target_point_distances(
            env=env,
            motion_command=motion_command,
            body_indices=self.contact_body_indices,
            body_local_offsets=self.contact_body_local_offsets,
            target_points_obj=target_points_obj[active_env_ids],
            env_ids=active_env_ids,
        )
        distances[active_env_ids] = active_distances

        if distance_scale is None:
            distance_scale = 0.12 if margin is None else margin
        scale_tensor = torch.tensor(max(float(distance_scale), 1e-6), dtype=torch.float32, device=env.device)
        per_body_reward = torch.exp(-distances / scale_tensor)
        per_body_reward = per_body_reward * active.float()
        denom = active.float().sum(dim=1).clamp(min=1.0)
        reward = per_body_reward.sum(dim=1) / denom
        return torch.where(has_active, reward, torch.zeros_like(reward))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _initialize(
        self,
        motion_command: MotionCommand,
        *,
        body_names: tuple[str, ...] | list[str] | str | None,
        contact_body_names_regex: str,
        fail_on_missing_targets: bool,
    ) -> None:
        if not motion_command.has_contact_labels or not motion_command.has_contact_target_points:
            message = (
                "ObjectContactTargetPointDistance is enabled but motion files have no contact target points. "
                "Regenerate motions with contact_object_target_points_obj labels."
            )
            if fail_on_missing_targets:
                raise RuntimeError(message)
            if not self.warned_missing_targets:
                logger.warning(message)
                self.warned_missing_targets = True
            self.initialized = True
            return

        selected_names, selected_cols = select_contact_body_columns(
            motion_command.motion.contact_body_names,
            body_names=body_names,
            body_names_regex=contact_body_names_regex,
        )
        self.contact_label_columns = torch.tensor(selected_cols, dtype=torch.long, device=self.env.device)
        self.contact_body_names = selected_names

        contact_body_indices = torch.as_tensor(
            motion_command.motion.contact_body_indices,
            dtype=torch.long,
            device=self.env.device,
        )
        contact_body_local_offsets = torch.as_tensor(
            motion_command.motion.contact_body_local_offsets,
            dtype=torch.float32,
            device=self.env.device,
        )
        self.contact_body_indices = contact_body_indices[self.contact_label_columns]
        self.contact_body_local_offsets = contact_body_local_offsets[self.contact_label_columns]

        logger.info(
            "Initialized ObjectContactTargetPointDistance: "
            f"bodies={self.contact_body_names}, topk={motion_command.motion.contact_object_target_points_obj.shape[2]}"
        )
        self.initialized = True


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
