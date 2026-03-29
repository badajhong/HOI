"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
import torch

from holosoma.cvae_ir_train import CLIPTextFeatureExtractor, load_encoder
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.base import ObservationTermBase
from holosoma.utils.rotations import quat_rotate_inverse, quaternion_to_matrix, subtract_frame_transforms
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


IR_FEATURE_EPS = 1e-6


#########################################################################################################
## terms same to managers/observation/terms/locomotion.py
#########################################################################################################
def _base_quat(env: WholeBodyTrackingManager) -> torch.Tensor:
    return env.base_quat


def gravity_vector(env: WholeBodyTrackingManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: WholeBodyTrackingManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def motion_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.command


def motion_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    _, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )
    mat = quaternion_to_matrix(ori_b, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    return pos.view(env.num_envs, -1)


def obj_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    unit_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    vel_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w.clone(),
        motion_command.robot_ref_quat_w.clone(),
        motion_command.simulator_object_lin_vel_w,
        unit_quat,
    )
    return vel_b.view(env.num_envs, -1)


def obj_type_one_hot(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    num_classes = max(int(getattr(motion_command, "num_object_types", 1)), 1)
    return torch.nn.functional.one_hot(motion_command.object_type_ids, num_classes=num_classes).float()


def _rotation_matrix_to_6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    return rotation_matrix[..., :2].reshape(rotation_matrix.shape[0], -1)


def _safe_normalize_vectors(vectors: torch.Tensor, eps: float = IR_FEATURE_EPS) -> tuple[torch.Tensor, torch.Tensor]:
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    normalized = torch.where(norms > eps, vectors / norms.clamp_min(eps), torch.zeros_like(vectors))
    return normalized, norms


def _compute_object_centric_ir_u_t(
    pelvis_pos_w: torch.Tensor,
    pelvis_lin_vel_w: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    object_lin_vel_w: torch.Tensor,
) -> torch.Tensor:
    object_rotation = quaternion_to_matrix(object_quat_w, w_last=True)
    object_rotation_inv = object_rotation.transpose(1, 2)

    rel_pos_w = pelvis_pos_w - object_pos_w
    rel_vel_w = pelvis_lin_vel_w - object_lin_vel_w

    r_obj = torch.bmm(object_rotation_inv, rel_pos_w.unsqueeze(-1)).squeeze(-1)
    v_rel_obj = torch.bmm(object_rotation_inv, rel_vel_w.unsqueeze(-1)).squeeze(-1)

    n_obj, _ = _safe_normalize_vectors(r_obj)
    radial_velocity_obj = torch.sum(v_rel_obj * n_obj, dim=-1, keepdim=True)
    v_close = -radial_velocity_obj
    v_orbit_obj = v_rel_obj - radial_velocity_obj * n_obj
    obj_ori_6d = _rotation_matrix_to_6d(object_rotation)
    return torch.cat([r_obj, v_close, v_orbit_obj, obj_ori_6d], dim=-1)


class IRCVAELatent(ObservationTermBase):
    """Frozen IR-CVAE encoder that turns a live u_window into a latent vector."""

    def __init__(self, cfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        checkpoint_path = cfg.params.get("checkpoint_path") or getattr(env, "ir_cvae", None)
        if not checkpoint_path:
            raise ValueError(
                "IR-CVAE latent observation requires a checkpoint. "
                "Pass `--ir_cvae=/path/to/best.pt` or set observation term param `checkpoint_path`."
            )

        pelvis_body_name = str(cfg.params.get("pelvis_body_name", "pelvis"))
        if pelvis_body_name not in env.body_names:
            raise ValueError(f"Pelvis body '{pelvis_body_name}' was not found in environment body names.")

        self.device = env.device
        self.pelvis_body_name = pelvis_body_name
        self.pelvis_body_index = env.body_names.index(pelvis_body_name)
        self.encoder, payload = load_encoder(str(checkpoint_path), device=self.device)

        input_shape = tuple(int(v) for v in payload["input_shape"])
        if len(input_shape) != 2:
            raise ValueError(f"IR-CVAE checkpoint input_shape must have length 2, got {input_shape}")

        self.window_size, self.u_t_dim = input_shape
        self.latent_dim = int(payload["config"]["latent_dim"])
        self.feature_mean = payload["feature_mean"].to(device=self.device, dtype=torch.float32)
        self.feature_std = payload["feature_std"].to(device=self.device, dtype=torch.float32).clamp_min(1e-6)

        condition_text = str(cfg.params.get("condition_text") or payload["condition_text"])
        clip_cfg = payload["clip"]
        text_extractor = CLIPTextFeatureExtractor(
            model_id=clip_cfg["model_id"],
            device=self.device,
            cache_dir=clip_cfg["cache_dir"],
            local_files_only=clip_cfg["local_files_only"],
        )
        self.text_features = text_extractor.encode([condition_text]).to(device=self.device, dtype=torch.float32)
        self.u_window_history = torch.zeros(env.num_envs, self.window_size, self.u_t_dim, device=self.device)
        self.is_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

        logger.info(
            f"Loaded frozen IR-CVAE latent encoder from {checkpoint_path} with "
            f"window_size={self.window_size}, u_t_dim={self.u_t_dim}, latent_dim={self.latent_dim}"
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.u_window_history.zero_()
            self.is_initialized.zero_()
            return

        env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return
        self.u_window_history[env_ids_tensor] = 0.0
        self.is_initialized[env_ids_tensor] = False

    @torch.no_grad()
    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        motion_command = _get_motion_command_and_assert_type(env)
        if not getattr(motion_command.motion, "has_object", False):
            return torch.zeros(env.num_envs, self.latent_dim, device=self.device)

        pelvis_pos_w = env.simulator._rigid_body_pos[:, self.pelvis_body_index, :]
        pelvis_lin_vel_w = env.simulator._rigid_body_vel[:, self.pelvis_body_index, :]
        current_u_t = _compute_object_centric_ir_u_t(
            pelvis_pos_w=pelvis_pos_w,
            pelvis_lin_vel_w=pelvis_lin_vel_w,
            object_pos_w=motion_command.simulator_object_pos_w,
            object_quat_w=motion_command.simulator_object_quat_w,
            object_lin_vel_w=motion_command.simulator_object_lin_vel_w,
        )
        if current_u_t.shape[1] != self.u_t_dim:
            raise ValueError(
                f"IR feature dim mismatch: checkpoint expects {self.u_t_dim}, got {current_u_t.shape[1]}"
            )

        new_mask = ~self.is_initialized
        if torch.any(new_mask):
            repeated = current_u_t[new_mask].unsqueeze(1).repeat(1, self.window_size, 1)
            self.u_window_history[new_mask] = repeated

        existing_mask = self.is_initialized
        if torch.any(existing_mask):
            existing_history = self.u_window_history[existing_mask].clone()
            existing_history[:, :-1] = existing_history[:, 1:].clone()
            existing_history[:, -1] = current_u_t[existing_mask]
            self.u_window_history[existing_mask] = existing_history

        self.is_initialized[:] = True

        flat_window = self.u_window_history.reshape(env.num_envs, -1)
        normalized_window = (flat_window - self.feature_mean.unsqueeze(0)) / self.feature_std.unsqueeze(0)
        text_features = self.text_features.expand(env.num_envs, -1)
        mu, _ = self.encoder(normalized_window, text_features)
        return mu
