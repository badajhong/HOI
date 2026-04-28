"""Whole body tracking observation terms."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
import torch

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.ae_joint_train import CLIPTextFeatureExtractor, load_joint_model
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.base import ObservationTermBase
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.rotations import quat_rotate_inverse, quaternion_to_matrix, subtract_frame_transforms
from holosoma.utils.surface_features import SurfaceFeatureComputer
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


IR_SURFACE_FEATURE_BODY_SOURCE_CHOICES = ("pelvis", "hands", "all")
IR_SURFACE_FEATURE_BODY_SOURCE_ALL_RESOLVED = ("hands", "pelvis")
IR_HAND_BODY_LABELS = ("left_hand", "right_hand")
IR_LEFT_HAND_BODY_NAME_CANDIDATES = (
    "left_hand_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
)
IR_RIGHT_HAND_BODY_NAME_CANDIDATES = (
    "right_hand_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
)


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


def student_actions(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Last frozen-student base actions used as the student's action history."""
    return env.student_prev_actions


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

def motion_command_joint_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command_joint_pos = _get_motion_command_and_assert_type(env)
    return motion_command_joint_pos.joint_pos

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


def _normalize_ir_surface_feature_body_source(body_source: str) -> str:
    normalized = body_source.strip().lower()
    if normalized not in IR_SURFACE_FEATURE_BODY_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported IR latent body source '{body_source}'. "
            f"Expected one of {IR_SURFACE_FEATURE_BODY_SOURCE_CHOICES}."
        )
    return normalized


def _parse_ir_surface_feature_body_sources(body_source: str) -> tuple[str, ...]:
    normalized = _normalize_ir_surface_feature_body_source(body_source)
    if normalized == "all":
        return IR_SURFACE_FEATURE_BODY_SOURCE_ALL_RESOLVED
    return (normalized,)


def _canonical_ir_surface_feature_body_source(body_sources: tuple[str, ...]) -> str:
    if body_sources == ("pelvis",):
        return "pelvis"
    if body_sources == ("hands",):
        return "hands"
    if body_sources == IR_SURFACE_FEATURE_BODY_SOURCE_ALL_RESOLVED:
        return "all"
    return ",".join(body_sources)


def _infer_ir_surface_feature_body_source(payload: dict, ir_t_dim: int) -> str:
    config = payload.get("config", {})
    if isinstance(config, dict):
        body_source = config.get("ir_window_body_source")
        if isinstance(body_source, str):
            return _normalize_ir_surface_feature_body_source(body_source)

    telemetry = payload.get("telemetry", {})
    if isinstance(telemetry, dict):
        body_source = telemetry.get("ir_window_body_source") or telemetry.get("surface_feature_body_source")
        if isinstance(body_source, str):
            return _normalize_ir_surface_feature_body_source(body_source)

    if ir_t_dim == 13:
        return "pelvis"
    if ir_t_dim == 26:
        return "hands"
    if ir_t_dim == 39:
        return "all"
    raise ValueError(
        f"Cannot infer IR latent body source from checkpoint ir_t_dim={ir_t_dim}. "
        "Pass --ir_ae_body_source as one of: pelvis, hands, all."
    )


def _resolve_ir_surface_feature_body_name(
    available_body_names: set[str],
    *,
    explicit_name: str | None,
    candidates: tuple[str, ...],
    label: str,
) -> str:
    if explicit_name:
        if explicit_name not in available_body_names:
            raise ValueError(f"Configured {label} body '{explicit_name}' was not found in environment body names.")
        return explicit_name

    for candidate in candidates:
        if candidate in available_body_names:
            return candidate
    raise ValueError(
        f"Could not resolve {label} body. Tried candidates={candidates}; "
        "set observation param or top-level override for the body name."
    )


def _object_keys_for_envs(motion_command, num_envs: int) -> list[str | None]:
    object_key_to_id = getattr(motion_command, "object_key_to_id", None) or {}
    if not object_key_to_id:
        return [None] * num_envs
    id_to_key = {int(idx): key for key, idx in object_key_to_id.items()}
    object_type_ids = motion_command.object_type_ids.detach().cpu().tolist()
    return [id_to_key.get(int(type_id)) for type_id in object_type_ids]


def _load_ir_latent_model(checkpoint_path: str, device: str):
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_type = str(payload.get("model_type", ""))
    if model_type != "joint_multimodal_ae":
        raise ValueError(
            f"IR latent checkpoint '{checkpoint_path}' is not a joint AE checkpoint "
            f"(model_type={model_type!r}). Export it again with ae_joint_train.py."
        )
    model, payload = load_joint_model(checkpoint_path, device=device)
    return model, payload, "joint_multimodal_ae"


def _load_di_latent_model(checkpoint_path: str, device: str):
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_type = str(payload.get("model_type", ""))
    if model_type != "joint_multimodal_ae":
        raise ValueError(
            f"Depth latent checkpoint '{checkpoint_path}' is not a joint AE checkpoint "
            f"(model_type={model_type!r}). Export it again with ae_joint_train.py."
        )
    model, payload = load_joint_model(checkpoint_path, device=device)
    return model, payload, "joint_multimodal_ae"


class IRAELatent(ObservationTermBase):
    """Frozen latent encoder that turns a live ir_window into a latent vector.

    Supports ``body_source`` parameter (preferred) or legacy
    ``--ir_ae_body_source`` on the command line to select which rigid bodies
    contribute to the surface-feature
    vector ``ir_t``.  Accepted values:

    * ``"pelvis"`` – pelvis body only (13-D ``ir_t``)
    * ``"hands"``  – left + right hand bodies (26-D ``ir_t``)
    * ``"all"``    – hands + pelvis (39-D ``ir_t``)

    When ``body_source`` is not explicitly provided the value is inferred from
    the AE checkpoint metadata (``ir_window_body_source`` in the config or
    telemetry block, or from the ``ir_t_dim``).
    """

    def __init__(self, cfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        checkpoint_path = cfg.params.get("checkpoint_path") or getattr(env, "ir_ae", None)
        if not checkpoint_path:
            raise ValueError(
                "Latent observation requires a checkpoint. "
                "Set observation term param `checkpoint_path` "
                "(preferred) or pass legacy `--ir_ae=/path/to/best.pt`."
            )

        self.device = env.device
        self.encoder, payload, self.checkpoint_model_type = _load_ir_latent_model(str(checkpoint_path), device=self.device)
        input_shape = tuple(int(v) for v in payload["ir_window_shape"])
        if len(input_shape) != 2:
            raise ValueError(f"Joint AE checkpoint ir_window_shape must have length 2, got {input_shape}")
        self.feature_mean = payload["ir_feature_mean"].to(device=self.device, dtype=torch.float32)
        self.feature_std = payload["ir_feature_std"].to(device=self.device, dtype=torch.float32).clamp_min(1e-6)

        self.window_size, self.ir_t_dim = input_shape
        self.latent_dim = int(payload["config"]["latent_dim"])

        # --- body source resolution -------------------------------------------
        explicit_body_source = (
            cfg.params.get("body_source")
            or getattr(env, "ir_ae_body_source", None)
        )
        if explicit_body_source:
            self.body_source = _normalize_ir_surface_feature_body_source(explicit_body_source)
        else:
            self.body_source = _infer_ir_surface_feature_body_source(payload, self.ir_t_dim)
        self.body_sources = _parse_ir_surface_feature_body_sources(self.body_source)

        # --- resolve body names & indices ------------------------------------
        available = set(env.body_names)
        self._body_labels: list[str] = []
        self._body_indices: list[int] = []

        for src in self.body_sources:
            if src == "pelvis":
                pelvis_name = _resolve_ir_surface_feature_body_name(
                    available,
                    explicit_name=cfg.params.get("pelvis_body_name"),
                    candidates=("pelvis",),
                    label="pelvis",
                )
                self._body_labels.append("pelvis")
                self._body_indices.append(env.body_names.index(pelvis_name))
            elif src == "hands":
                left_name = _resolve_ir_surface_feature_body_name(
                    available,
                    explicit_name=cfg.params.get("left_hand_body_name"),
                    candidates=IR_LEFT_HAND_BODY_NAME_CANDIDATES,
                    label="left hand",
                )
                right_name = _resolve_ir_surface_feature_body_name(
                    available,
                    explicit_name=cfg.params.get("right_hand_body_name"),
                    candidates=IR_RIGHT_HAND_BODY_NAME_CANDIDATES,
                    label="right hand",
                )
                self._body_labels.extend(["left_hand", "right_hand"])
                self._body_indices.extend([
                    env.body_names.index(left_name),
                    env.body_names.index(right_name),
                ])

        # Validate expected ir_t dim vs body source
        expected_ir_t_dim = 13 * len(self._body_labels)
        if expected_ir_t_dim != self.ir_t_dim:
            raise ValueError(
                f"body_source='{self.body_source}' produces {expected_ir_t_dim}-D ir_t "
                f"but checkpoint expects {self.ir_t_dim}-D. "
                "Check observation term param `body_source` "
                "(or legacy `--ir_ae_body_source`) matches the latent model training body source."
            )

        # --- surface feature computer ----------------------------------------
        object_cfg = env.robot_config.object
        self._surface_feature_computer = SurfaceFeatureComputer.from_object_config(object_cfg)

        # --- CLIP text features ----------------------------------------------
        condition_text = str(cfg.params.get("condition_text") or payload["condition_text"])
        clip_cfg = payload["clip"]
        text_extractor = CLIPTextFeatureExtractor(
            model_id=clip_cfg["model_id"],
            device=self.device,
            cache_dir=clip_cfg["cache_dir"],
            local_files_only=clip_cfg["local_files_only"],
        )
        self.text_features = text_extractor.encode([condition_text]).to(device=self.device, dtype=torch.float32)
        self.ir_window_history = torch.zeros(env.num_envs, self.window_size, self.ir_t_dim, device=self.device)
        self.is_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

        logger.info(
            f"Loaded frozen latent encoder from {checkpoint_path} with "
            f"model_type={self.checkpoint_model_type}, latent_mode=mu, "
            f"body_source='{self.body_source}', "
            f"window_size={self.window_size}, ir_t_dim={self.ir_t_dim}, latent_dim={self.latent_dim}"
        )

    def _compute_current_ir_t(
        self,
        env: WholeBodyTrackingManager,
        motion_command: MotionCommand,
    ) -> torch.Tensor:
        """Compute the current surface-feature vector for all envs.

        Returns a tensor of shape ``[num_envs, ir_t_dim]`` whose layout matches
        the body-source ordering used during latent training.
        """
        object_keys = _object_keys_for_envs(motion_command, env.num_envs)
        ir_t_parts: list[torch.Tensor] = []

        for label, body_idx in zip(self._body_labels, self._body_indices):
            body_pos_w = env.simulator._rigid_body_pos[:, body_idx, :]
            body_lin_vel_w = env.simulator._rigid_body_vel[:, body_idx, :]
            features = self._surface_feature_computer.compute_batch(
                body_pos_w=body_pos_w,
                body_lin_vel_w=body_lin_vel_w,
                object_pos_w=motion_command.simulator_object_pos_w,
                object_quat_w=motion_command.simulator_object_quat_w,
                object_keys=object_keys,
            )
            ir_t_parts.append(features["ir_t"])

        return torch.cat(ir_t_parts, dim=-1) if len(ir_t_parts) > 1 else ir_t_parts[0]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.ir_window_history.zero_()
            self.is_initialized.zero_()
            return

        env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return
        self.ir_window_history[env_ids_tensor] = 0.0
        self.is_initialized[env_ids_tensor] = False

    @torch.no_grad()
    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        modify_history = bool(kwargs.pop("modify_history", True))
        motion_command = _get_motion_command_and_assert_type(env)
        if not getattr(motion_command.motion, "has_object", False):
            return torch.zeros(env.num_envs, self.latent_dim, device=self.device)

        current_ir_t = self._compute_current_ir_t(env, motion_command)
        if current_ir_t.shape[1] != self.ir_t_dim:
            raise ValueError(
                f"IR feature dim mismatch: checkpoint expects {self.ir_t_dim}, got {current_ir_t.shape[1]}"
            )

        if modify_history:
            new_mask = ~self.is_initialized
            if torch.any(new_mask):
                repeated = current_ir_t[new_mask].unsqueeze(1).repeat(1, self.window_size, 1)
                self.ir_window_history[new_mask] = repeated

            existing_mask = self.is_initialized
            if torch.any(existing_mask):
                existing_history = self.ir_window_history[existing_mask].clone()
                existing_history[:, :-1] = existing_history[:, 1:].clone()
                existing_history[:, -1] = current_ir_t[existing_mask]
                self.ir_window_history[existing_mask] = existing_history

            self.is_initialized[:] = True
            effective_window = self.ir_window_history
        else:
            effective_window = torch.cat([self.ir_window_history[:, 1:], current_ir_t.unsqueeze(1)], dim=1)
            new_mask = ~self.is_initialized
            if torch.any(new_mask):
                effective_window[new_mask] = current_ir_t[new_mask].unsqueeze(1).repeat(1, self.window_size, 1)

        flat_window = effective_window.reshape(env.num_envs, -1)
        normalized_window = (flat_window - self.feature_mean.unsqueeze(0)) / self.feature_std.unsqueeze(0)
        text_features = self.text_features.expand(env.num_envs, -1)
        mu, _ = self.encoder.encode_ir(normalized_window, text_features)
        return mu

def _get_observation_compute_token(env: WholeBodyTrackingManager) -> int:
    observation_manager = getattr(env, "observation_manager", None)
    if observation_manager is None:
        return -1
    return int(getattr(observation_manager, "_compute_invocation_id", -1))


class _DepthLatentObservationTermBase(ObservationTermBase):
    """Shared depth-latent encoder runtime for residual observations."""

    def __init__(self, cfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        di_checkpoint = cfg.params.get("checkpoint_path") or getattr(env, "di_ae", None)
        if not di_checkpoint:
            raise ValueError(
                "Depth-latent observation requires a depth latent checkpoint. "
                "Set observation term param `checkpoint_path` "
                "(preferred) or pass legacy `--di_ae=/path/to/best.pt`."
            )

        self.device = env.device
        self.di_checkpoint = str(di_checkpoint)
        self.debug_save_depth_images = bool(cfg.params.get("debug_save_depth_images", False))
        self.debug_depth_save_interval = max(int(cfg.params.get("debug_depth_save_interval", 200)), 1)
        debug_depth_env_ids = cfg.params.get("debug_depth_env_ids", (0,))
        if debug_depth_env_ids is None:
            self.debug_depth_env_ids: tuple[int, ...] = (0,)
        elif isinstance(debug_depth_env_ids, int):
            self.debug_depth_env_ids = (int(debug_depth_env_ids),)
        else:
            self.debug_depth_env_ids = tuple(int(env_id) for env_id in debug_depth_env_ids)
        self.debug_depth_env_id_set = set(self.debug_depth_env_ids)
        self.debug_depth_dir = self._resolve_debug_depth_dir(env) if self.debug_save_depth_images else None
        self._debug_episode_indices = torch.full((env.num_envs,), -1, device=self.device, dtype=torch.long)
        self._debug_last_saved_episode_steps = torch.full((env.num_envs,), -1, device=self.device, dtype=torch.long)
        self._debug_invalid_env_ids_logged = False
        self._debug_save_disabled_logged = False

        self.di_encoder, payload, self.di_checkpoint_model_type = _load_di_latent_model(
            self.di_checkpoint,
            device=self.device,
        )
        for parameter in self.di_encoder.parameters():
            parameter.requires_grad_(False)

        input_shape = tuple(int(v) for v in payload["input_shape"])
        if len(input_shape) != 3:
            raise ValueError(f"Depth latent checkpoint input_shape must have length 3, got {input_shape}")
        self.window_size, self.depth_height, self.depth_width = input_shape
        self.latent_dim = int(payload["config"]["latent_dim"])
        self.di_feature_mean = payload["di_feature_mean"].to(device=self.device, dtype=torch.float32)
        self.di_feature_std = payload["di_feature_std"].to(device=self.device, dtype=torch.float32).clamp_min(1e-6)
        self.depth_latent_mode = "mu"

        condition_text = str(cfg.params.get("condition_text") or payload["condition_text"])
        clip_cfg = payload["clip"]
        text_extractor = CLIPTextFeatureExtractor(
            model_id=clip_cfg["model_id"],
            device=self.device,
            cache_dir=clip_cfg["cache_dir"],
            local_files_only=clip_cfg["local_files_only"],
            quiet_load=True,
        )
        self.di_text_features = text_extractor.encode([condition_text]).to(device=self.device, dtype=torch.float32)

        self.depth_window_history = torch.zeros(
            env.num_envs,
            self.window_size,
            self.depth_height,
            self.depth_width,
            device=self.device,
            dtype=torch.float32,
        )
        self.depth_is_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self._cached_compute_token = -1
        self._cached_modify_history: bool | None = None
        self._cached_latent: torch.Tensor | None = None

        logger.info(
            f"Loaded frozen di latent encoder from {self.di_checkpoint} with "
            f"model_type={self.di_checkpoint_model_type}, latent_mode={self.depth_latent_mode}, "
            f"window_size={self.window_size}, depth_shape=({self.depth_height}, {self.depth_width}), latent_dim={self.latent_dim}"
        )
        if self.debug_depth_dir is not None:
            self.debug_depth_dir.mkdir(parents=True, exist_ok=True)
            debug_targets = "all envs" if not self.debug_depth_env_ids else f"env_ids={list(self.debug_depth_env_ids)}"
            logger.info(
                f"Depth debug image saving enabled: dir={self.debug_depth_dir}, "
                f"interval={self.debug_depth_save_interval}, targets={debug_targets}"
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.depth_window_history.zero_()
            self.depth_is_initialized.zero_()
            self._debug_episode_indices += 1
            self._debug_last_saved_episode_steps.fill_(-1)
            self._cached_compute_token = -1
            self._cached_modify_history = None
            self._cached_latent = None
            return

        env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return
        self.depth_window_history[env_ids_tensor] = 0.0
        self.depth_is_initialized[env_ids_tensor] = False
        self._debug_episode_indices[env_ids_tensor] += 1
        self._debug_last_saved_episode_steps[env_ids_tensor] = -1
        self._cached_compute_token = -1
        self._cached_modify_history = None
        self._cached_latent = None

    def _resolve_debug_depth_dir(self, env: WholeBodyTrackingManager) -> Path | None:
        save_dir = getattr(getattr(env.simulator, "video_config", None), "save_dir", None)
        if save_dir is None:
            logger.warning("Depth debug image saving requested, but simulator video_config.save_dir is not available.")
            return None
        return Path(save_dir).resolve().parent / "depth_debug"

    def _save_depth_preview(self, depth_frame: torch.Tensor, output_path: Path) -> None:
        from PIL import Image  # noqa: PLC0415

        output_path.parent.mkdir(parents=True, exist_ok=True)
        depth_array = depth_frame.detach().cpu()
        finite_mask = torch.isfinite(depth_array) & (depth_array > 0.0)
        preview = torch.zeros_like(depth_array, dtype=torch.uint8)
        if torch.any(finite_mask):
            valid = depth_array[finite_mask]
            lo = float(valid.min().item())
            hi = float(torch.quantile(valid, 0.99).item())
            if hi <= lo:
                hi = lo + 1e-6
            normalized = torch.clamp((depth_array - lo) / (hi - lo), 0.0, 1.0)
            preview = (normalized * 255.0).to(torch.uint8)
        Image.fromarray(preview.numpy(), mode="L").save(output_path)

    def _maybe_save_debug_depth_frames(self, env: WholeBodyTrackingManager, depth_frames: torch.Tensor) -> None:
        if not self.debug_save_depth_images or self.debug_depth_dir is None:
            return

        episode_steps = env.episode_length_buf.detach()
        for env_id in range(env.num_envs):
            if self.debug_depth_env_id_set and env_id not in self.debug_depth_env_id_set:
                continue

            episode_step = int(episode_steps[env_id].item())
            if episode_step <= 0:
                continue
            if episode_step != 1 and episode_step % self.debug_depth_save_interval != 0:
                continue
            if int(self._debug_last_saved_episode_steps[env_id].item()) == episode_step:
                continue

            episode_index = int(self._debug_episode_indices[env_id].item())
            if episode_index < 0:
                episode_index = 0
            output_path = (
                self.debug_depth_dir
                / f"env_{env_id:03d}"
                / f"episode_{episode_index:03d}"
                / f"step_{episode_step:06d}_depth.png"
            )
            try:
                self._save_depth_preview(depth_frames[env_id], output_path)
            except Exception as exc:  # pragma: no cover - debug path best effort
                if not self._debug_save_disabled_logged:
                    logger.warning(
                        f"Disabling depth debug image saving after failure writing {output_path}: {type(exc).__name__}: {exc}"
                    )
                    self._debug_save_disabled_logged = True
                self.debug_save_depth_images = False
                return
            self._debug_last_saved_episode_steps[env_id] = episode_step

        if not self._debug_invalid_env_ids_logged and self.debug_depth_env_ids:
            invalid_env_ids = [env_id for env_id in self.debug_depth_env_ids if env_id < 0 or env_id >= env.num_envs]
            if invalid_env_ids:
                logger.warning(
                    f"Ignoring out-of-range debug_depth_env_ids={invalid_env_ids}; available env ids are [0, {env.num_envs - 1}]."
                )
                self._debug_invalid_env_ids_logged = True

    def _read_depth_frames(self, env: WholeBodyTrackingManager) -> torch.Tensor:
        depth_camera = getattr(env.simulator, "robot_depth_camera", None)
        if depth_camera is None:
            raise RuntimeError(
                "Simulator did not create a robot-mounted depth camera. "
                "Expected IsaacSim to register 'robot_depth_camera' from the URDF optical frame."
            )

        depth_output = depth_camera.data.output.get("distance_to_image_plane")
        if depth_output is None:
            raise RuntimeError("Robot depth camera has no 'distance_to_image_plane' output.")

        depth_frames = depth_output.to(device=self.device, dtype=torch.float32)
        if depth_frames.ndim == 4 and depth_frames.shape[-1] == 1:
            depth_frames = depth_frames[..., 0]
        if depth_frames.ndim != 3:
            raise RuntimeError(
                f"Unexpected robot depth batch shape {tuple(depth_frames.shape)}; expected [num_envs, H, W] or [num_envs, H, W, 1]."
            )
        if depth_frames.shape[1:] == (self.depth_width, self.depth_height):
            depth_frames = depth_frames.transpose(1, 2)
        if depth_frames.shape[1:] != (self.depth_height, self.depth_width):
            raise RuntimeError(
                f"Unexpected robot depth spatial shape {tuple(depth_frames.shape[1:])}; "
                f"expected {(self.depth_height, self.depth_width)}."
            )

        return torch.nan_to_num(depth_frames, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_depth_window(self, depth_frames: torch.Tensor, *, modify_history: bool) -> torch.Tensor:
        if modify_history:
            new_mask = ~self.depth_is_initialized
            if torch.any(new_mask):
                repeated = depth_frames[new_mask].unsqueeze(1).repeat(1, self.window_size, 1, 1)
                self.depth_window_history[new_mask] = repeated

            existing_mask = self.depth_is_initialized
            if torch.any(existing_mask):
                existing_history = self.depth_window_history[existing_mask].clone()
                existing_history[:, :-1] = existing_history[:, 1:].clone()
                existing_history[:, -1] = depth_frames[existing_mask]
                self.depth_window_history[existing_mask] = existing_history

            self.depth_is_initialized[:] = True
            return self.depth_window_history

        effective_window = torch.cat([self.depth_window_history[:, 1:], depth_frames.unsqueeze(1)], dim=1)
        new_mask = ~self.depth_is_initialized
        if torch.any(new_mask):
            effective_window[new_mask] = depth_frames[new_mask].unsqueeze(1).repeat(1, self.window_size, 1, 1)
        return effective_window

    def _encode_di_latent(self, depth_window: torch.Tensor) -> torch.Tensor:
        normalized_window = (depth_window - self.di_feature_mean.unsqueeze(0)) / self.di_feature_std.unsqueeze(0)
        text_features = self.di_text_features.expand(depth_window.shape[0], -1)
        mu, _ = self.di_encoder.encode_di(normalized_window, text_features)
        return mu

    def _compute_depth_latent(self, env: WholeBodyTrackingManager, *, modify_history: bool) -> torch.Tensor:
        compute_token = _get_observation_compute_token(env)
        if (
            self._cached_latent is not None
            and self._cached_compute_token == compute_token
            and self._cached_modify_history == modify_history
        ):
            return self._cached_latent

        motion_command = _get_motion_command_and_assert_type(env)
        if not getattr(motion_command.motion, "has_object", False):
            zero_latent = torch.zeros(env.num_envs, self.latent_dim, device=self.device)
            self._cached_compute_token = compute_token
            self._cached_modify_history = modify_history
            self._cached_latent = zero_latent
            return zero_latent

        try:
            depth_frames = self._read_depth_frames(env)
        except RuntimeError:
            if not modify_history:
                zero_latent = torch.zeros(env.num_envs, self.latent_dim, device=self.device)
                self._cached_compute_token = compute_token
                self._cached_modify_history = modify_history
                self._cached_latent = zero_latent
                return zero_latent
            raise
        if modify_history:
            self._maybe_save_debug_depth_frames(env, depth_frames)
        depth_window = self._build_depth_window(depth_frames, modify_history=modify_history)
        depth_latent = self._encode_di_latent(depth_window)
        self._cached_compute_token = compute_token
        self._cached_modify_history = modify_history
        self._cached_latent = depth_latent
        return depth_latent


class DIAELatent(_DepthLatentObservationTermBase):
    """Frozen depth latent encoder that exposes a live depth latent vector."""

    @torch.no_grad()
    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        modify_history = bool(kwargs.pop("modify_history", True))
        return self._compute_depth_latent(env, modify_history=modify_history)


class FrozenStudentBaseAction(ObservationTermBase):
    """Frozen student actor that consumes a shared depth-latent observation."""

    def __init__(self, cfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        student_checkpoint = cfg.params.get("student_checkpoint") or getattr(env, "student", None)
        if not student_checkpoint:
            raise ValueError(
                "Frozen student base-action observation requires a student checkpoint. "
                "Set observation term param `student_checkpoint` "
                "(preferred) or pass legacy `--student=/path/to/model.pt`."
            )

        self.device = env.device
        self.student_obs_group = str(cfg.params.get("student_obs_group", "student_actor_obs"))
        self.latent_obs_group = str(cfg.params.get("latent_obs_group", "di_ae_latent"))
        self.student_checkpoint = str(student_checkpoint)
        self.student_actor = None
        self.student_input_keys: list[str] = []
        self.student_obs_dim: int | None = None
        self.student_latent_dim: int | None = None
        self.num_actions = env.robot_config.actions_dim

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        return

    def _get_student_actor_obs(self, env: WholeBodyTrackingManager) -> torch.Tensor:
        student_obs = env.observation_manager.compute_group(self.student_obs_group, modify_history=False)
        if not isinstance(student_obs, torch.Tensor):
            raise ValueError(f"Observation group '{self.student_obs_group}' must be concatenated for frozen student input.")
        return student_obs

    def _setup_student_actor(self, student_obs_dim: int, latent_dim: int) -> None:
        if self.student_actor is not None:
            if self.student_obs_dim != student_obs_dim:
                raise ValueError(
                    f"Frozen student obs dim changed from {self.student_obs_dim} to {student_obs_dim}; "
                    "this likely means the observation preset no longer matches the student checkpoint."
                )
            if self.student_latent_dim != latent_dim:
                raise ValueError(
                    f"Frozen student latent dim changed from {self.student_latent_dim} to {latent_dim}; "
                    "this likely means the depth latent preset no longer matches the student checkpoint."
                )
            return

        student_config, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=self.student_checkpoint))
        student_payload = torch.load(self.student_checkpoint, map_location=self.device)
        student_algo_config = getattr(student_config.algo, "config", None)
        if student_algo_config is None or not hasattr(student_algo_config, "module_dict"):
            raise ValueError("Student checkpoint must contain a PPO-style actor module configuration.")

        student_actor_cfg = student_algo_config.module_dict.actor
        self.student_input_keys = list(student_actor_cfg.input_dim)
        expected_keys = {"actor_obs", "ir_ae_latent"}
        actual_keys = set(self.student_input_keys)
        if actual_keys != expected_keys:
            raise ValueError(
                f"Frozen student actor expects inputs {self.student_input_keys}; supported inputs are {sorted(expected_keys)}."
            )

        self.student_actor = setup_ppo_actor_module(
            obs_dim_dict={"actor_obs": student_obs_dim, "ir_ae_latent": latent_dim},
            module_config=student_actor_cfg,
            num_actions=self.num_actions,
            init_noise_std=getattr(student_algo_config, "init_noise_std", 1.0),
            device=self.device,
            history_length={"actor_obs": 1, "ir_ae_latent": 1},
        )
        self.student_actor.load_state_dict(student_payload["actor_model_state_dict"])
        self.student_actor.eval()
        for parameter in self.student_actor.parameters():
            parameter.requires_grad_(False)
        if hasattr(self.student_actor, "std"):
            self.student_actor.std.requires_grad_(False)

        self.student_obs_dim = student_obs_dim
        self.student_latent_dim = latent_dim
        logger.info(
            f"Loaded frozen student actor from {self.student_checkpoint} with "
            f"student_obs_dim={student_obs_dim}, latent_dim={latent_dim}"
        )

    def _build_student_input(self, student_obs: torch.Tensor, depth_latent: torch.Tensor) -> torch.Tensor:
        inputs = []
        for key in self.student_input_keys:
            if key == "actor_obs":
                inputs.append(student_obs)
            elif key == "ir_ae_latent":
                inputs.append(depth_latent)
            else:
                raise ValueError(f"Unsupported frozen student input key '{key}'.")
        return torch.cat(inputs, dim=1)

    def _get_depth_latent(self, env: WholeBodyTrackingManager, *, modify_history: bool) -> torch.Tensor:
        observation_manager = getattr(env, "observation_manager", None)
        if observation_manager is None:
            raise RuntimeError("Environment is missing an observation manager.")
        if self.latent_obs_group not in observation_manager.cfg.groups:
            raise ValueError(
                "Frozen student base-action observation requires a depth-latent observation group. "
                f"Missing group '{self.latent_obs_group}'."
            )
        depth_latent = observation_manager.compute_group(self.latent_obs_group, modify_history=modify_history)
        if not isinstance(depth_latent, torch.Tensor):
            raise ValueError(f"Observation group '{self.latent_obs_group}' must be concatenated for latent input.")
        return depth_latent

    @torch.no_grad()
    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        modify_history = bool(kwargs.pop("modify_history", True))
        motion_command = _get_motion_command_and_assert_type(env)
        if not getattr(motion_command.motion, "has_object", False):
            zero_actions = torch.zeros(env.num_envs, self.num_actions, device=self.device)
            if modify_history:
                env.student_prev_actions.zero_()
                env.student_base_actions.zero_()
            return zero_actions

        student_obs = self._get_student_actor_obs(env)
        depth_latent = self._get_depth_latent(env, modify_history=modify_history)
        self._setup_student_actor(student_obs.shape[1], depth_latent.shape[1])
        student_input = self._build_student_input(student_obs, depth_latent)
        base_action = self.student_actor.act_inference({"actor_obs": student_input})

        if modify_history:
            env.student_prev_actions[:] = base_action
            env.student_base_actions[:] = base_action

        return base_action
