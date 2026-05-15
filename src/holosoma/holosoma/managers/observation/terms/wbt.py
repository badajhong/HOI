"""Whole body tracking observation terms."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

from holosoma.ae_joint_train import CLIPTextFeatureExtractor, load_joint_model
from holosoma.ae_pro_joint_train import load_joint_model as load_pro_joint_model
from holosoma.agents.modules.module_utils import setup_ppo_actor_module
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


class ObjectScaleSequenceProbe(torch.nn.Module):
    """Trainable temporal head for object-scale classification from frozen DI-pro features."""

    def __init__(
        self,
        *,
        input_dim: int,
        gru_hidden_dim: int,
        gru_num_layers: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
    ):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(gru_hidden_dim),
            num_layers=int(gru_num_layers),
            batch_first=True,
        )
        layers: list[torch.nn.Module] = []
        previous_dim = int(gru_hidden_dim)
        for hidden_dim in hidden_dims:
            layers.extend([torch.nn.Linear(previous_dim, int(hidden_dim)), torch.nn.ELU()])
            previous_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(previous_dim, int(output_dim)))
        self.head = torch.nn.Sequential(*layers)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError(f"ObjectScaleSequenceProbe expects [B, T, F], got {tuple(sequence.shape)}.")
        _, hidden = self.gru(sequence)
        return self.head(hidden[-1])


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


def _load_di_latent_model(checkpoint_path: str, device: str, *, use_pro_checkpoint: bool = False):
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_type = str(payload.get("model_type", ""))
    if model_type != "joint_multimodal_ae":
        raise ValueError(
            f"Depth latent checkpoint '{checkpoint_path}' is not a joint AE checkpoint "
            f"(model_type={model_type!r}). Export it again with "
            f"{'ae_pro_joint_train.py' if use_pro_checkpoint else 'ae_joint_train.py'}."
        )
    loader = load_pro_joint_model if use_pro_checkpoint else load_joint_model
    model, payload = loader(checkpoint_path, device=device)
    checkpoint_kind = "joint_multimodal_ae_pro" if use_pro_checkpoint else "joint_multimodal_ae"
    return model, payload, checkpoint_kind


class IRAELatent(ObservationTermBase):
    """Frozen latent encoder that turns a live ir_window into a latent vector.

    Supports ``body_source`` parameter (preferred) or legacy
    ``--ir_ae_body_source`` on the command line to select which rigid bodies
    contribute to the surface-feature
    vector ``ir_t``.  Accepted values:

    * ``"pelvis"`` - pelvis body only (13-D ``ir_t``)
    * ``"hands"``  - left + right hand bodies (26-D ``ir_t``)
    * ``"all"``    - hands + pelvis (39-D ``ir_t``)

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
        self.encoder, payload, self.checkpoint_model_type = _load_ir_latent_model(
            str(checkpoint_path),
            device=self.device,
        )
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

        for _label, body_idx in zip(self._body_labels, self._body_indices):
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


class AELatent(ObservationTermBase):
    """AE latent observation backed by either IR or depth latent encoders.

    This term provides a source-agnostic latent observation for student
    policies while allowing student training to swap between:

    * IR latent (`--ir_ae`, optional `--ir_ae_body_source`)
    * DI latent (`--di_ae`)
    * DI+proprioception latent (`--di_pro_ae`)
    """

    def __init__(self, cfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        ir_checkpoint = cfg.params.get("checkpoint_path") or getattr(env, "ir_ae", None)
        di_checkpoint = cfg.params.get("di_checkpoint_path") or getattr(env, "di_ae", None)
        di_pro_checkpoint = cfg.params.get("di_pro_checkpoint_path") or getattr(env, "di_pro_ae", None)
        requested_source = str(cfg.params.get("source", "")).strip().lower()

        if requested_source and requested_source not in {"ir", "di", "di_pro"}:
            raise ValueError(
                f"Unsupported student latent source '{requested_source}'. Expected one of ('', 'ir', 'di', 'di_pro')."
            )

        if not requested_source:
            provided_sources = [
                source
                for source, checkpoint in (
                    ("ir", ir_checkpoint),
                    ("di", di_checkpoint),
                    ("di_pro", di_pro_checkpoint),
                )
                if checkpoint
            ]
            if len(provided_sources) > 1:
                raise ValueError(
                    "AE latent observation received multiple latent checkpoints "
                    f"({provided_sources}). Pass only one of `--ir_ae` / `--di_ae` / `--di_pro_ae`, "
                    "or set observation param `source` to disambiguate."
                )
            if di_pro_checkpoint:
                requested_source = "di_pro"
            elif di_checkpoint:
                requested_source = "di"
            elif ir_checkpoint:
                requested_source = "ir"
            else:
                raise ValueError(
                    "AE latent observation requires either an IR latent checkpoint or a DI latent checkpoint. "
                    "Pass `--ir_ae=/path/to/best.pt` (optionally with `--ir_ae_body_source`) "
                    "or `--di_ae=/path/to/best.pt` or `--di_pro_ae=/path/to/best.pt`."
                )

        if requested_source == "ir":
            if not ir_checkpoint:
                raise ValueError(
                    "AE latent source is set to 'ir', but no IR latent checkpoint was provided. "
                    "Pass `--ir_ae=/path/to/best.pt`."
                )
            inner_params = dict(cfg.params)
            inner_params["checkpoint_path"] = ir_checkpoint
            inner_cfg = dataclasses.replace(cfg, params=inner_params)
            self.impl = IRAELatent(inner_cfg, env)
        elif requested_source == "di":
            if not di_checkpoint:
                raise ValueError(
                    "AE latent source is set to 'di', but no DI latent checkpoint was provided. "
                    "Pass `--di_ae=/path/to/best.pt`."
                )
            inner_params = dict(cfg.params)
            inner_params["checkpoint_path"] = di_checkpoint
            inner_params["ignore_di_pro_ae_fallback"] = True
            inner_cfg = dataclasses.replace(cfg, params=inner_params)
            self.impl = DIAELatent(inner_cfg, env)
        else:
            if not di_pro_checkpoint:
                raise ValueError(
                    "AE latent source is set to 'di_pro', but no DI+proprioception checkpoint was provided. "
                    "Pass `--di_pro_ae=/path/to/best.pt`."
                )
            inner_params = dict(cfg.params)
            inner_params["checkpoint_path"] = ""
            inner_params["di_pro_checkpoint_path"] = di_pro_checkpoint
            inner_params["ignore_di_ae_fallback"] = True
            inner_cfg = dataclasses.replace(cfg, params=inner_params)
            self.impl = DIAELatent(inner_cfg, env)

        self.source = requested_source
        logger.info(f"AE latent observation is using source='{self.source}'.")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.impl.reset(env_ids)

    @torch.no_grad()
    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        return self.impl(env, **kwargs)


class StudentLatent(AELatent):
    """Backward-compatible alias for older student_latent observation configs."""

def _get_observation_compute_token(env: WholeBodyTrackingManager) -> int:
    observation_manager = getattr(env, "observation_manager", None)
    if observation_manager is None:
        return -1
    return int(getattr(observation_manager, "_compute_invocation_id", -1))


def _parse_debug_depth_env_ids(raw_env_ids: object) -> tuple[int, ...]:
    """Parse debug env ids from int, iterable, or a compact CLI-friendly string."""
    if raw_env_ids is None:
        return (0,)

    if isinstance(raw_env_ids, int):
        return (int(raw_env_ids),)

    if isinstance(raw_env_ids, str):
        text = raw_env_ids.strip()
        if text.lower() in {"", "all", "*"}:
            return ()

        if (text.startswith("(") and text.endswith(")")) or (text.startswith("[") and text.endswith("]")):
            text = text[1:-1].strip()
        if text.endswith(","):
            text = text[:-1].strip()
        if text.lower() in {"", "all", "*"}:
            return ()

        parts = [part.strip() for part in text.split(",") if part.strip()]
        if not parts:
            return ()
        return tuple(int(part) for part in parts)

    return tuple(int(env_id) for env_id in raw_env_ids)


class _DepthLatentObservationTermBase(ObservationTermBase):
    """Shared depth-latent encoder runtime for residual observations."""

    def __init__(self, cfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        di_checkpoint = cfg.params.get("checkpoint_path")
        if not di_checkpoint and not bool(cfg.params.get("ignore_di_ae_fallback", False)):
            di_checkpoint = getattr(env, "di_ae", None)
        di_pro_checkpoint = cfg.params.get("di_pro_checkpoint_path")
        if not di_pro_checkpoint and not bool(cfg.params.get("ignore_di_pro_ae_fallback", False)):
            di_pro_checkpoint = getattr(env, "di_pro_ae", None)
        if di_checkpoint and di_pro_checkpoint:
            raise ValueError(
                "Depth-latent observation received both `di_ae` and `di_pro_ae` checkpoints. "
                "Pass only one, or configure only one observation-term checkpoint path."
            )
        if not di_checkpoint and not di_pro_checkpoint:
            raise ValueError(
                "Depth-latent observation requires a depth latent checkpoint. "
                "Set observation term param `checkpoint_path` "
                "(preferred), pass legacy `--di_ae=/path/to/best.pt`, "
                "or pass `--di_pro_ae=/path/to/best.pt` for ae_pro_joint_train.py checkpoints."
            )

        self.device = env.device
        self.uses_di_pro_checkpoint = bool(di_pro_checkpoint)
        self.di_checkpoint = str(di_pro_checkpoint or di_checkpoint)
        self.debug_save_depth_images = bool(cfg.params.get("debug_save_depth_images", False))
        self.debug_depth_save_interval = max(int(cfg.params.get("debug_depth_save_interval", 200)), 1)
        debug_depth_env_ids = cfg.params.get("debug_depth_env_ids", "0")
        self.debug_depth_env_ids = _parse_debug_depth_env_ids(debug_depth_env_ids)
        self.debug_depth_env_id_set = set(self.debug_depth_env_ids)
        self.debug_depth_dir = self._resolve_debug_depth_dir(env) if self.debug_save_depth_images else None
        self._debug_episode_indices = torch.full((env.num_envs,), -1, device=self.device, dtype=torch.long)
        self._debug_last_saved_episode_steps = torch.full((env.num_envs,), -1, device=self.device, dtype=torch.long)
        self._debug_invalid_env_ids_logged = False
        self._debug_save_disabled_logged = False
        self.di_encoder, payload, self.di_checkpoint_model_type = _load_di_latent_model(
            self.di_checkpoint,
            device=self.device,
            use_pro_checkpoint=self.uses_di_pro_checkpoint,
        )
        for parameter in self.di_encoder.parameters():
            parameter.requires_grad_(False)

        input_shape = tuple(int(v) for v in payload["input_shape"])
        if len(input_shape) != 3:
            raise ValueError(f"Depth latent checkpoint input_shape must have length 3, got {input_shape}")
        self.window_size, self.depth_height, self.depth_width = input_shape
        self.latent_dim = int(payload["config"]["latent_dim"])
        self.depth_projection_dim = int(getattr(self.di_encoder.di_encoder, "hidden_dim", self.latent_dim))
        self.di_feature_mean = payload["di_feature_mean"].to(device=self.device, dtype=torch.float32)
        self.di_feature_std = payload["di_feature_std"].to(device=self.device, dtype=torch.float32).clamp_min(1e-6)
        self.proprioception_input_shape = (
            tuple(int(v) for v in payload["proprioception_input_shape"])
            if payload.get("proprioception_input_shape") is not None
            else None
        )
        self.uses_proprioception_window = self.proprioception_input_shape is not None
        self.proprio_feature_mean = None
        self.proprio_feature_std = None
        if self.uses_proprioception_window:
            if len(self.proprioception_input_shape) != 2:
                raise ValueError(
                    "DI+proprioception checkpoint proprioception_input_shape must have length 2, "
                    f"got {self.proprioception_input_shape}."
                )
            self.proprio_window_size, self.proprio_feature_dim = self.proprioception_input_shape
            if self.proprio_window_size != self.window_size:
                raise ValueError(
                    "DI+proprioception checkpoint window mismatch: "
                    f"depth window={self.window_size}, proprioception window={self.proprio_window_size}."
                )
            expected_proprio_dim = 3 + 2 * len(env.dof_names)
            if self.proprio_feature_dim != expected_proprio_dim:
                raise ValueError(
                    "DI+proprioception checkpoint feature dim mismatch: "
                    f"checkpoint expects {self.proprio_feature_dim}, live env produces {expected_proprio_dim} "
                    "(base_ang_vel + dof_pos + dof_vel)."
                )
            proprio_feature_mean = payload.get("proprio_feature_mean")
            proprio_feature_std = payload.get("proprio_feature_std")
            if proprio_feature_mean is None or proprio_feature_std is None:
                raise RuntimeError(
                    "DI+proprioception checkpoint declares proprioception input but is missing "
                    "proprio_feature_mean/proprio_feature_std."
                )
            self.proprio_feature_mean = proprio_feature_mean.to(device=self.device, dtype=torch.float32)
            self.proprio_feature_std = proprio_feature_std.to(device=self.device, dtype=torch.float32).clamp_min(1e-6)
        else:
            self.proprio_window_size = 0
            self.proprio_feature_dim = 0
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
        self.proprioception_window_history = torch.zeros(
            env.num_envs,
            self.window_size,
            self.proprio_feature_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.proprioception_is_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self._cached_compute_token = -1
        self._cached_modify_history: bool | None = None
        self._cached_latent: torch.Tensor | None = None
        self._cached_scale_probe_feature: torch.Tensor | None = None
        self._cached_scale_probe_sequence: torch.Tensor | None = None

        logger.info(
            f"Loaded frozen di latent encoder from {self.di_checkpoint} with "
            f"model_type={self.di_checkpoint_model_type}, latent_mode={self.depth_latent_mode}, "
            f"window_size={self.window_size}, depth_shape=({self.depth_height}, {self.depth_width}), "
            f"proprioception_dim={self.proprio_feature_dim}, latent_dim={self.latent_dim}, "
            f"depth_projection_dim={self.depth_projection_dim}"
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
            self.proprioception_window_history.zero_()
            self.proprioception_is_initialized.zero_()
            self._debug_episode_indices += 1
            self._debug_last_saved_episode_steps.fill_(-1)
            self._cached_compute_token = -1
            self._cached_modify_history = None
            self._cached_latent = None
            self._cached_scale_probe_feature = None
            self._cached_scale_probe_sequence = None
            return

        env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return
        self.depth_window_history[env_ids_tensor] = 0.0
        self.depth_is_initialized[env_ids_tensor] = False
        self.proprioception_window_history[env_ids_tensor] = 0.0
        self.proprioception_is_initialized[env_ids_tensor] = False
        self._debug_episode_indices[env_ids_tensor] += 1
        self._debug_last_saved_episode_steps[env_ids_tensor] = -1
        self._cached_compute_token = -1
        self._cached_modify_history = None
        self._cached_latent = None
        self._cached_scale_probe_feature = None
        self._cached_scale_probe_sequence = None

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

            episode_index = max(int(self._debug_episode_indices[env_id].item()), 0)
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
                        "Disabling depth debug image saving after failure writing "
                        f"{output_path}: {type(exc).__name__}: {exc}"
                    )
                    self._debug_save_disabled_logged = True
                self.debug_save_depth_images = False
                return
            self._debug_last_saved_episode_steps[env_id] = episode_step

        if not self._debug_invalid_env_ids_logged and self.debug_depth_env_ids:
            invalid_env_ids = [env_id for env_id in self.debug_depth_env_ids if env_id < 0 or env_id >= env.num_envs]
            if invalid_env_ids:
                logger.warning(
                    f"Ignoring out-of-range debug_depth_env_ids={invalid_env_ids}; "
                    f"available env ids are [0, {env.num_envs - 1}]."
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
                f"Unexpected robot depth batch shape {tuple(depth_frames.shape)}; "
                "expected [num_envs, H, W] or [num_envs, H, W, 1]."
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

    def _compute_current_proprioception(self, env: WholeBodyTrackingManager) -> torch.Tensor:
        base_ang_vel = quat_rotate_inverse(
            env.base_quat,
            env.simulator.robot_root_states[:, 10:13],
            w_last=True,
        )
        dof_pos = env.simulator.dof_pos - env.default_dof_pos
        dof_vel = env.simulator.dof_vel
        proprioception = torch.cat((base_ang_vel, dof_pos, dof_vel), dim=-1)
        if proprioception.shape[1] != self.proprio_feature_dim:
            raise RuntimeError(
                "Live proprioception feature dim changed unexpectedly: "
                f"expected {self.proprio_feature_dim}, got {proprioception.shape[1]}."
            )
        return proprioception

    def _build_proprioception_window(self, proprioception: torch.Tensor, *, modify_history: bool) -> torch.Tensor:
        if modify_history:
            new_mask = ~self.proprioception_is_initialized
            if torch.any(new_mask):
                repeated = proprioception[new_mask].unsqueeze(1).repeat(1, self.window_size, 1)
                self.proprioception_window_history[new_mask] = repeated

            existing_mask = self.proprioception_is_initialized
            if torch.any(existing_mask):
                existing_history = self.proprioception_window_history[existing_mask].clone()
                existing_history[:, :-1] = existing_history[:, 1:].clone()
                existing_history[:, -1] = proprioception[existing_mask]
                self.proprioception_window_history[existing_mask] = existing_history

            self.proprioception_is_initialized[:] = True
            return self.proprioception_window_history

        effective_window = torch.cat([self.proprioception_window_history[:, 1:], proprioception.unsqueeze(1)], dim=1)
        new_mask = ~self.proprioception_is_initialized
        if torch.any(new_mask):
            effective_window[new_mask] = proprioception[new_mask].unsqueeze(1).repeat(1, self.window_size, 1)
        return effective_window

    def _encode_di_latent_and_projection_feature(
        self,
        depth_window: torch.Tensor,
        proprioception_window: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_window = (depth_window - self.di_feature_mean.unsqueeze(0)) / self.di_feature_std.unsqueeze(0)
        text_features = self.di_text_features.expand(depth_window.shape[0], -1)
        condition = self.di_encoder.condition(text_features)
        depth_encoder = self.di_encoder.di_encoder

        if normalized_window.ndim != 4:
            raise ValueError(f"Expected depth_window batch [B, T, H, W], got {tuple(normalized_window.shape)}.")
        batch_size, window_size, height, width = normalized_window.shape
        if (window_size, height, width) != (self.window_size, self.depth_height, self.depth_width):
            raise ValueError(
                f"Expected depth shape {(self.window_size, self.depth_height, self.depth_width)}, "
                f"got {(window_size, height, width)}."
            )

        frames = normalized_window.reshape(batch_size * window_size, 1, height, width)
        frame_features = depth_encoder.frame_projection(depth_encoder.frame_features(frames))
        frame_features = frame_features.reshape(batch_size, window_size, self.depth_projection_dim)
        projection_feature = frame_features.mean(dim=1)
        temporal_features = frame_features

        if self.uses_proprioception_window:
            if proprioception_window is None:
                raise RuntimeError("DI+proprioception checkpoint requires a live proprioception window.")
            assert self.proprio_feature_mean is not None
            assert self.proprio_feature_std is not None
            normalized_proprioception = (
                proprioception_window - self.proprio_feature_mean.unsqueeze(0)
            ) / self.proprio_feature_std.unsqueeze(0)
            proprioception_steps = normalized_proprioception.reshape(
                batch_size * self.proprio_window_size,
                self.proprio_feature_dim,
            )
            proprioception_steps = depth_encoder.proprio_frame_projection(proprioception_steps)
            proprioception_steps = proprioception_steps.reshape(
                batch_size,
                self.proprio_window_size,
                depth_encoder.proprio_hidden_dim,
            )
            temporal_features = depth_encoder.temporal_fusion(torch.cat([frame_features, proprioception_steps], dim=-1))

        _, hidden = depth_encoder.temporal_encoder(temporal_features)
        latent_hidden = depth_encoder.latent_head(torch.cat([hidden[-1], condition], dim=-1))
        mu = depth_encoder.mu(latent_hidden)
        return mu, projection_feature, temporal_features

    def _cache_depth_outputs(
        self,
        *,
        compute_token: int,
        modify_history: bool,
        latent: torch.Tensor,
        scale_probe_feature: torch.Tensor,
        scale_probe_sequence: torch.Tensor,
    ) -> None:
        self._cached_compute_token = compute_token
        self._cached_modify_history = modify_history
        self._cached_latent = latent
        self._cached_scale_probe_feature = scale_probe_feature
        self._cached_scale_probe_sequence = scale_probe_sequence

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
            zero_feature = torch.zeros(env.num_envs, self.depth_projection_dim, device=self.device)
            zero_sequence = torch.zeros(
                env.num_envs,
                self.window_size,
                self.depth_projection_dim,
                device=self.device,
            )
            self._cache_depth_outputs(
                compute_token=compute_token,
                modify_history=modify_history,
                latent=zero_latent,
                scale_probe_feature=zero_feature,
                scale_probe_sequence=zero_sequence,
            )
            return zero_latent

        try:
            depth_frames = self._read_depth_frames(env)
        except RuntimeError:
            if not modify_history:
                zero_latent = torch.zeros(env.num_envs, self.latent_dim, device=self.device)
                zero_feature = torch.zeros(env.num_envs, self.depth_projection_dim, device=self.device)
                zero_sequence = torch.zeros(
                    env.num_envs,
                    self.window_size,
                    self.depth_projection_dim,
                    device=self.device,
                )
                self._cache_depth_outputs(
                    compute_token=compute_token,
                    modify_history=modify_history,
                    latent=zero_latent,
                    scale_probe_feature=zero_feature,
                    scale_probe_sequence=zero_sequence,
                )
                return zero_latent
            raise
        if modify_history:
            self._maybe_save_debug_depth_frames(env, depth_frames)
        depth_window = self._build_depth_window(depth_frames, modify_history=modify_history)
        proprioception_window = None
        if self.uses_proprioception_window:
            proprioception = self._compute_current_proprioception(env)
            proprioception_window = self._build_proprioception_window(
                proprioception,
                modify_history=modify_history,
            )
        depth_latent, scale_probe_feature, scale_probe_sequence = self._encode_di_latent_and_projection_feature(
            depth_window,
            proprioception_window,
        )
        self._cache_depth_outputs(
            compute_token=compute_token,
            modify_history=modify_history,
            latent=depth_latent,
            scale_probe_feature=scale_probe_feature,
            scale_probe_sequence=scale_probe_sequence,
        )
        return depth_latent

    def get_cached_depth_projection_feature(
        self,
        env: WholeBodyTrackingManager,
        *,
        modify_history: bool,
    ) -> torch.Tensor:
        self._compute_depth_latent(env, modify_history=modify_history)
        if self._cached_scale_probe_feature is None:
            raise RuntimeError("Depth projection feature cache was not populated by the DI encoder.")
        return self._cached_scale_probe_feature

    def get_cached_di_fused_sequence_feature(
        self,
        env: WholeBodyTrackingManager,
        *,
        modify_history: bool,
    ) -> torch.Tensor:
        self._compute_depth_latent(env, modify_history=modify_history)
        if self._cached_scale_probe_sequence is None:
            raise RuntimeError("DI fused sequence feature cache was not populated by the DI encoder.")
        return self._cached_scale_probe_sequence


class DIAELatent(_DepthLatentObservationTermBase):
    """Frozen depth latent encoder that exposes a live depth latent vector."""

    @torch.no_grad()
    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        modify_history = bool(kwargs.pop("modify_history", True))
        return self._compute_depth_latent(env, modify_history=modify_history)


class ObjectScaleBinInput(ObservationTermBase):
    """Object-scale bin observation.

    `source="predicted"` returns classifier output as soft probabilities or predicted one-hot bins.
    `source="real"` returns the debug-only hard one-hot bin from env.object_scale_factors.
    """

    def __init__(self, cfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.device = env.device
        self.source = str(cfg.params.get("source", "predicted"))
        self.latent_obs_group = str(cfg.params.get("latent_obs_group", "di_ae_latent"))
        self.feature_source = str(cfg.params.get("feature_source", "latent"))
        self.target_mode = str(cfg.params.get("target", "uniform"))
        default_output_mode = "one_hot" if self.source == "real" else "soft"
        self.output_mode = self._parse_output_mode(cfg.params.get("output_mode", default_output_mode))
        scale_values_param = cfg.params.get("scale_values", "auto")
        bin_min_param = cfg.params.get("bin_min", "auto")
        bin_max_param = cfg.params.get("bin_max", "auto")
        num_bins_param = cfg.params.get("num_bins", "auto")
        desired_bin_size = float(cfg.params.get("bin_size", 0.1))
        self.hidden_dims = self._parse_hidden_dims(cfg.params.get("hidden_dims", ""))
        self.gru_hidden_dim = int(cfg.params.get("gru_hidden_dim", 256))
        self.gru_num_layers = max(int(cfg.params.get("gru_num_layers", 1)), 1)
        self.train_online = bool(cfg.params.get("train_online", self.source == "predicted"))
        self.learning_rate = float(cfg.params.get("learning_rate", 1e-3))
        self.weight_decay = float(cfg.params.get("weight_decay", 1e-4))
        self.max_grad_norm = float(cfg.params.get("max_grad_norm", 1.0))
        self.train_batch_size = int(cfg.params.get("train_batch_size", 4096))
        self.train_every = max(int(cfg.params.get("train_every", 1)), 1)
        self.log_metrics = bool(cfg.params.get("log_metrics", True))
        self.log_target_summary = bool(cfg.params.get("log_target_summary", True))
        self.log_pred_summary = bool(cfg.params.get("log_pred_summary", True))
        self.log_distribution = bool(cfg.params.get("log_distribution", False))
        self.log_prefix = str(cfg.params.get("log_prefix", "ScaleBinProbe"))
        self._online_train_calls = 0

        if self.source not in {"predicted", "real"}:
            raise ValueError(f"ObjectScaleBinInput source must be 'predicted' or 'real', got {self.source!r}.")
        if self.feature_source not in {
            "latent",
            "di_projection",
            "depth_projection",
            "di_fused_sequence",
            "di_temporal",
            "di_pro_sequence",
        }:
            raise ValueError(
                "ObjectScaleBinInput feature_source must be one of 'latent', 'di_projection', "
                "or 'di_fused_sequence', "
                f"got {self.feature_source!r}."
            )
        if self.feature_source == "depth_projection":
            self.feature_source = "di_projection"
        if self.feature_source in {"di_temporal", "di_pro_sequence"}:
            self.feature_source = "di_fused_sequence"
        if self.gru_hidden_dim <= 0:
            raise ValueError(f"ObjectScaleBinInput gru_hidden_dim must be positive, got {self.gru_hidden_dim}.")
        if self.target_mode not in {"uniform", "z"}:
            raise ValueError(f"ObjectScaleBinInput target must be 'uniform' or 'z', got {self.target_mode!r}.")
        if desired_bin_size <= 0.0:
            raise ValueError(f"ObjectScaleBinInput bin_size must be positive, got {desired_bin_size}.")

        self.scale_values = self._resolve_scale_values(env, scale_values_param)
        if self.scale_values is not None:
            if self.scale_values.numel() == 0:
                raise ValueError("ObjectScaleBinInput scale_values must contain at least one value.")
            self.num_bins = int(self.scale_values.numel())
            self.bin_min = float(self.scale_values.min().item())
            self.bin_max = float(self.scale_values.max().item())
            self.bin_size = self._infer_discrete_bin_size(self.scale_values)
        else:
            auto_bin_min = self._is_auto(bin_min_param)
            auto_bin_max = self._is_auto(bin_max_param)
            auto_num_bins = self._is_auto(num_bins_param)
            randomization_bin_min = None
            randomization_bin_max = None
            if auto_bin_min or auto_bin_max or auto_num_bins:
                randomization_bin_min, randomization_bin_max = self._resolve_randomized_scale_range(env)
            self.bin_min = float(randomization_bin_min if auto_bin_min else bin_min_param)
            self.bin_max = float(randomization_bin_max if auto_bin_max else bin_max_param)
            if self.bin_max <= self.bin_min:
                raise ValueError(
                    f"ObjectScaleBinInput requires bin_max > bin_min, got {self.bin_min} and {self.bin_max}."
                )
            if auto_num_bins:
                self.num_bins = self._num_bins_from_size(self.bin_min, self.bin_max, desired_bin_size)
                self.bin_size = desired_bin_size
            else:
                self.num_bins = int(num_bins_param)
                self.bin_size = (self.bin_max - self.bin_min) / float(self.num_bins)
        if self.num_bins <= 0:
            raise ValueError(f"ObjectScaleBinInput num_bins must be positive, got {self.num_bins}.")

        self.probe: torch.nn.Module | None = None
        self.probe_optimizer: torch.optim.Optimizer | None = None
        self.probe_payload: dict | None = None
        if self.source == "real":
            self.train_online = False
            logger.info(
                "Object scale GT input will return hard one-hot bins from env.object_scale_factors "
                f"with target={self.target_mode!r}, values={self._scale_values_for_log()}, "
                f"range=[{self.bin_min}, {self.bin_max}], bin_size={self.bin_size}, num_bins={self.num_bins}."
            )
        else:
            logger.info(
                "Object scale actor input will return object-scale bins from an online classifier "
                f"with output_mode={self.output_mode!r}, feature_source={self.feature_source!r}, "
                f"target={self.target_mode!r}, "
                f"values={self._scale_values_for_log()}, "
                f"range=[{self.bin_min}, {self.bin_max}], bin_size={self.bin_size}, "
                f"num_bins={self.num_bins}, hidden_dims={self.hidden_dims}, "
                f"gru_hidden_dim={self.gru_hidden_dim if self.feature_source == 'di_fused_sequence' else None}."
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        return

    def _parse_scale_values(self, raw_value) -> torch.Tensor | None:
        if raw_value is None or self._is_auto(raw_value):
            return None
        scale_values = torch.as_tensor(raw_value, dtype=torch.float32).detach().cpu()
        if scale_values.ndim == 0:
            scale_values = scale_values.reshape(1)
        elif scale_values.ndim == 1:
            scale_values = scale_values.flatten()
        else:
            raise ValueError(
                "ObjectScaleBinInput scale_values must be a scalar or a 1-D sequence of object volume ratios, "
                f"got shape {tuple(scale_values.shape)}."
            )
        if torch.any(scale_values <= 0.0):
            raise ValueError(f"ObjectScaleBinInput object volume ratios must be positive, got {scale_values.tolist()}.")
        return scale_values.to(device=self.device, dtype=torch.float32)

    def _resolve_scale_values(self, env: WholeBodyTrackingManager, raw_value) -> torch.Tensor | None:
        if not self._is_auto(raw_value):
            return self._parse_scale_values(raw_value)
        params = self._get_object_scale_randomization_params(env)
        if params is None:
            return None
        scale_value = params.get("scale_value")
        if scale_value is not None:
            return self._parse_scale_values(scale_value)
        scale_values = params.get("scale_values")
        if scale_values is not None:
            return self._parse_scale_values(scale_values)
        return None

    @staticmethod
    def _infer_discrete_bin_size(scale_values: torch.Tensor) -> float:
        if scale_values.numel() <= 1:
            return 0.0
        sorted_values = torch.sort(scale_values.detach().cpu().float()).values
        diffs = torch.diff(sorted_values)
        return float(diffs.min().item())

    def _scale_values_for_log(self) -> list[float] | None:
        if self.scale_values is None:
            return None
        return [float(value) for value in self.scale_values.detach().cpu().tolist()]

    @staticmethod
    def _parse_output_mode(raw_value) -> str:
        output_mode = str(raw_value).lower()
        if output_mode in {"soft", "prob", "probs", "probability", "probabilities"}:
            return "soft"
        if output_mode in {"one_hot", "onehot", "hard", "hard_one_hot"}:
            return "one_hot"
        raise ValueError(
            "ObjectScaleBinInput output_mode must be one of 'soft' or 'one_hot', "
            f"got {raw_value!r}."
        )

    @staticmethod
    def _is_auto(value) -> bool:
        return value is None or (isinstance(value, str) and value.lower() == "auto")

    @staticmethod
    def _num_bins_from_size(bin_min: float, bin_max: float, bin_size: float) -> int:
        raw_bins = (bin_max - bin_min) / bin_size
        rounded_bins = round(raw_bins)
        if abs(raw_bins - rounded_bins) < 1e-6:
            return max(int(rounded_bins), 1)
        return max(int(torch.ceil(torch.tensor(raw_bins)).item()), 1)

    @staticmethod
    def _get_object_scale_randomization_params(env: WholeBodyTrackingManager) -> dict | None:
        randomization_cfg = getattr(env, "domain_rand_cfg", None)
        if randomization_cfg is None and getattr(env, "randomization_manager", None) is not None:
            randomization_cfg = getattr(env.randomization_manager, "cfg", None)
        setup_terms = getattr(randomization_cfg, "setup_terms", {}) or {}

        term_cfg = setup_terms.get("randomize_object_scale_startup")
        if term_cfg is None:
            term_cfg = next(
                (
                    candidate
                    for candidate in setup_terms.values()
                    if str(getattr(candidate, "func", "")).endswith(":randomize_object_scale_startup")
                ),
                None,
            )
        if term_cfg is None:
            return None
        params = dict(getattr(term_cfg, "params", {}) or {})
        if not bool(params.get("enabled", True)):
            return None
        return params

    def _resolve_randomized_scale_range(self, env: WholeBodyTrackingManager) -> tuple[float, float]:
        params = self._get_object_scale_randomization_params(env)
        if params is None:
            raise ValueError(
                "ObjectScaleBinInput bin range is set to 'auto', but enabled "
                "randomize_object_scale_startup was not found in the randomization config."
            )

        scale_range = params.get("scale_range")
        if isinstance(scale_range, dict):
            raise ValueError("ObjectScaleBinInput scale_range must be a 2-value object volume ratio range.")
        if scale_range is not None:
            if len(scale_range) != 2:
                raise ValueError(f"scale_range must have two values, got {scale_range!r}.")
            return float(scale_range[0]), float(scale_range[1])

        scale_value = params.get("scale_value")
        if scale_value is None:
            raise ValueError(
                "ObjectScaleBinInput could not infer scale bins because randomize_object_scale_startup "
                "has neither scale_range nor scale_value."
            )
        if isinstance(scale_value, (int, float)):
            value = float(scale_value)
        else:
            raise ValueError(f"scale_value must be a scalar object volume ratio, got {scale_value!r}.")
        half_width = 0.5 * float(self.cfg.params.get("bin_size", 0.1))
        return value - half_width, value + half_width

    def get_checkpoint_state(self) -> dict:
        if self.source == "real" or self.probe is None:
            return {}
        state = {
            "source": self.source,
            "feature_source": self.feature_source,
            "output_mode": self.output_mode,
            "target_mode": self.target_mode,
            "bin_min": self.bin_min,
            "bin_max": self.bin_max,
            "bin_size": self.bin_size,
            "scale_values": self._scale_values_for_log(),
            "num_bins": self.num_bins,
            "hidden_dims": self.hidden_dims,
            "gru_hidden_dim": self.gru_hidden_dim,
            "gru_num_layers": self.gru_num_layers,
            "train_online": self.train_online,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "train_batch_size": self.train_batch_size,
            "train_every": self.train_every,
            "online_train_calls": self._online_train_calls,
            "probe_payload": self.probe_payload,
            "probe_model_state_dict": self.probe.state_dict(),
        }
        if self.probe_optimizer is not None:
            state["probe_optimizer_state_dict"] = self.probe_optimizer.state_dict()
        return state

    def load_checkpoint_state(self, state: dict | None) -> None:
        if not state or self.source == "real":
            return
        payload = state.get("probe_payload")
        if not payload:
            return
        if not self._checkpoint_scale_config_matches(state, payload):
            logger.warning(
                "Skipping object-scale bin classifier checkpoint state because its scale-class config does not "
                "match the current observation config."
            )
            return
        self._online_train_calls = int(state.get("online_train_calls", 0))
        self._setup_online_probe(int(payload["input_dim"]))
        assert self.probe is not None
        self.probe.load_state_dict(state["probe_model_state_dict"])
        if self.probe_optimizer is not None and state.get("probe_optimizer_state_dict") is not None:
            self.probe_optimizer.load_state_dict(state["probe_optimizer_state_dict"])
        self.probe.eval()

    def _checkpoint_scale_config_matches(self, state: dict, payload: dict) -> bool:
        checkpoint_target = str(state.get("target_mode", payload.get("target", self.target_mode)))
        if checkpoint_target != self.target_mode:
            return False

        checkpoint_feature_source = str(state.get("feature_source", payload.get("feature_source", "latent")))
        if checkpoint_feature_source != self.feature_source:
            return False

        checkpoint_hidden_dims = tuple(int(dim) for dim in state.get("hidden_dims", self.hidden_dims))
        if checkpoint_hidden_dims != self.hidden_dims:
            return False
        if int(state.get("gru_hidden_dim", payload.get("gru_hidden_dim", self.gru_hidden_dim))) != self.gru_hidden_dim:
            return False
        if int(state.get("gru_num_layers", payload.get("gru_num_layers", self.gru_num_layers))) != self.gru_num_layers:
            return False

        checkpoint_num_bins = int(state.get("num_bins", payload.get("num_bins", self.num_bins)))
        if checkpoint_num_bins != self.num_bins:
            return False

        checkpoint_scale_values = self._parse_scale_values(state.get("scale_values", payload.get("scale_values")))
        if self.scale_values is None or checkpoint_scale_values is None:
            return (
                self.scale_values is None
                and checkpoint_scale_values is None
                and abs(float(state.get("bin_min", payload.get("bin_min", self.bin_min))) - self.bin_min) < 1e-6
                and abs(float(state.get("bin_max", payload.get("bin_max", self.bin_max))) - self.bin_max) < 1e-6
                and abs(float(state.get("bin_size", payload.get("bin_size", self.bin_size))) - self.bin_size) < 1e-6
            )
        return torch.allclose(
            checkpoint_scale_values.to(device=self.device),
            self.scale_values.to(device=self.device),
            atol=1e-6,
            rtol=0.0,
        )

    @staticmethod
    def _parse_hidden_dims(raw_value) -> tuple[int, ...]:
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if raw_value == "":
                return ()
            return tuple(int(part) for part in raw_value.split(",") if part.strip())
        if raw_value is None:
            return ()
        return tuple(int(value) for value in raw_value)

    def _build_probe(self, input_dim: int) -> torch.nn.Module:
        if self.feature_source == "di_fused_sequence":
            return ObjectScaleSequenceProbe(
                input_dim=int(input_dim),
                gru_hidden_dim=self.gru_hidden_dim,
                gru_num_layers=self.gru_num_layers,
                hidden_dims=self.hidden_dims,
                output_dim=self.num_bins,
            )
        layers: list[torch.nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in self.hidden_dims:
            layers.extend([torch.nn.Linear(previous_dim, int(hidden_dim)), torch.nn.ELU()])
            previous_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(previous_dim, self.num_bins))
        return torch.nn.Sequential(*layers)

    def _setup_online_probe(self, input_dim: int) -> None:
        if self.probe is not None:
            expected_dim = int(self.probe_payload["input_dim"])
            if input_dim != expected_dim:
                raise ValueError(
                    f"Object scale bin classifier expected input dim {expected_dim}, got {input_dim} "
                    f"from feature_source={self.feature_source!r}."
                )
            return

        with torch.inference_mode(False):
            self.probe = self._build_probe(input_dim).to(self.device)
            final_linear = next(
                (module for module in reversed(list(self.probe.modules())) if isinstance(module, torch.nn.Linear)),
                None,
            )
            if final_linear is not None:
                torch.nn.init.zeros_(final_linear.weight)
                if final_linear.bias is not None:
                    torch.nn.init.zeros_(final_linear.bias)
            self.probe_payload = {
                "model_type": "object_scale_bin_classifier",
                "input_dim": int(input_dim),
                "output_dim": self.num_bins,
                "feature_source": self.feature_source,
                "output_mode": self.output_mode,
                "hidden_dims": self.hidden_dims,
                "gru_hidden_dim": self.gru_hidden_dim,
                "gru_num_layers": self.gru_num_layers,
                "target": self.target_mode,
                "bin_min": self.bin_min,
                "bin_max": self.bin_max,
                "bin_size": self.bin_size,
                "scale_values": self._scale_values_for_log(),
                "num_bins": self.num_bins,
            }
            self.probe_optimizer = torch.optim.AdamW(
                self.probe.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            self.probe.train()
        logger.info(
            f"Initialized online object-scale bin classifier with input_dim={input_dim}, "
            f"feature_source={self.feature_source!r}, num_bins={self.num_bins}, hidden_dims={self.hidden_dims}, "
            f"gru_hidden_dim={self.gru_hidden_dim if self.feature_source == 'di_fused_sequence' else None}."
        )

    def _get_scale_scalar(self, env: WholeBodyTrackingManager) -> torch.Tensor:
        if hasattr(env, "object_scale_factors"):
            object_scale = env.object_scale_factors.detach().to(device=self.device, dtype=torch.float32)
        else:
            object_scale = torch.ones(env.num_envs, 3, device=self.device, dtype=torch.float32)

        if self.target_mode == "uniform":
            return object_scale.prod(dim=1)
        if self.target_mode == "z":
            return torch.pow(object_scale[:, 2], 3.0)
        raise ValueError(f"Unsupported object scale target mode {self.target_mode!r}.")

    def _target_bin_indices(self, env: WholeBodyTrackingManager) -> torch.Tensor:
        scale = self._get_scale_scalar(env)
        if self.scale_values is not None:
            distances = torch.abs(scale.unsqueeze(1) - self.scale_values.to(device=scale.device).unsqueeze(0))
            return torch.argmin(distances, dim=1)
        return torch.floor((scale - self.bin_min) / self.bin_size).long().clamp(0, self.num_bins - 1)

    def _target_one_hot(self, env: WholeBodyTrackingManager) -> torch.Tensor:
        target_bin = self._target_bin_indices(env)
        return torch.nn.functional.one_hot(target_bin, num_classes=self.num_bins).to(dtype=torch.float32)

    def _train_online_probe(self, env: WholeBodyTrackingManager, probe_input: torch.Tensor) -> None:
        if not self.train_online or self.probe_optimizer is None:
            return
        assert self.probe is not None
        target_bin = self._target_bin_indices(env)

        self._online_train_calls += 1
        if self._online_train_calls % self.train_every != 0:
            return

        with torch.inference_mode(False), torch.enable_grad():
            train_probe_input = probe_input.detach().clone()
            train_target = target_bin.detach().clone()
            if self.train_batch_size > 0 and train_probe_input.shape[0] > self.train_batch_size:
                batch_ids = torch.randperm(train_probe_input.shape[0], device=self.device)[: self.train_batch_size]
                train_probe_input = train_probe_input[batch_ids]
                train_target = train_target[batch_ids]

            self.probe.train()
            logits = self.probe(train_probe_input)
            loss = torch.nn.functional.cross_entropy(logits, train_target)
            self.probe_optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.probe.parameters(), self.max_grad_norm)
            self.probe_optimizer.step()
            self.probe.eval()

        env.log_dict[f"{self.log_prefix}/train_ce"] = loss.detach()

    def _log_target_metrics(self, env: WholeBodyTrackingManager, target_bin: torch.Tensor) -> None:
        if self.log_target_summary:
            scale = self._get_scale_scalar(env)
            env.log_dict[f"{self.log_prefix}/target_scale_mean"] = scale.mean()
            env.log_dict[f"{self.log_prefix}/target_bin_mean"] = target_bin.float().mean()
        if not self.log_distribution:
            return
        for bin_index in range(self.num_bins):
            env.log_dict[f"{self.log_prefix}/target_frac_bin_{bin_index}"] = (target_bin == bin_index).float().mean()

    def _log_prediction_metrics(self, env: WholeBodyTrackingManager, probs: torch.Tensor) -> None:
        if not self.log_metrics:
            return
        target_bin = self._target_bin_indices(env)
        pred_bin = probs.argmax(dim=1)
        confidence = probs.max(dim=1).values

        self._log_target_metrics(env, target_bin)
        env.log_dict[f"{self.log_prefix}/accuracy"] = (pred_bin == target_bin).float().mean()
        env.log_dict[f"{self.log_prefix}/bin_mae"] = (pred_bin - target_bin).abs().float().mean()
        if self.log_pred_summary:
            env.log_dict[f"{self.log_prefix}/pred_bin_mean"] = pred_bin.float().mean()
            env.log_dict[f"{self.log_prefix}/confidence"] = confidence.mean()
        log_probs = probs.clamp_min(1e-8).log()
        env.log_dict[f"{self.log_prefix}/ce"] = torch.nn.functional.nll_loss(log_probs, target_bin)
        if not self.log_distribution:
            return
        for bin_index in range(self.num_bins):
            env.log_dict[f"{self.log_prefix}/pred_frac_bin_{bin_index}"] = (pred_bin == bin_index).float().mean()
            env.log_dict[f"{self.log_prefix}/prob_mean_bin_{bin_index}"] = probs[:, bin_index].mean()

    def _format_prediction_output(self, probs: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "soft":
            return probs
        pred_bin = probs.argmax(dim=1)
        return torch.nn.functional.one_hot(pred_bin, num_classes=self.num_bins).to(dtype=torch.float32)

    def _get_depth_latent_term(self, env: WholeBodyTrackingManager):
        group_instances = getattr(env.observation_manager, "_term_instances", {}).get(self.latent_obs_group, {})
        if not group_instances:
            raise ValueError(
                f"ObjectScaleBinInput feature_source={self.feature_source!r} requires stateful observation group "
                f"'{self.latent_obs_group}'."
            )
        return group_instances.get(self.latent_obs_group) or next(iter(group_instances.values()))

    def _get_probe_input(self, env: WholeBodyTrackingManager, *, modify_history: bool) -> torch.Tensor:
        if self.feature_source == "latent":
            probe_input = env.observation_manager.compute_group(self.latent_obs_group, modify_history=modify_history)
        elif self.feature_source == "di_projection":
            depth_latent_term = self._get_depth_latent_term(env)
            get_projection_feature = getattr(depth_latent_term, "get_cached_depth_projection_feature", None)
            if not callable(get_projection_feature):
                raise ValueError(
                    f"Observation group '{self.latent_obs_group}' does not expose cached depth projection features."
                )
            probe_input = get_projection_feature(env, modify_history=modify_history)
        else:
            depth_latent_term = self._get_depth_latent_term(env)
            get_fused_sequence = getattr(depth_latent_term, "get_cached_di_fused_sequence_feature", None)
            if not callable(get_fused_sequence):
                raise ValueError(
                    f"Observation group '{self.latent_obs_group}' does not expose cached DI fused sequence features."
                )
            probe_input = get_fused_sequence(env, modify_history=modify_history)
        if not isinstance(probe_input, torch.Tensor):
            raise ValueError(
                f"ObjectScaleBinInput feature_source={self.feature_source!r} produced a non-tensor input."
            )
        return probe_input

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        modify_history = bool(kwargs.pop("modify_history", True))
        if self.source == "real":
            target_bin = self._target_bin_indices(env)
            if modify_history and self.log_metrics:
                self._log_target_metrics(env, target_bin)
            return torch.nn.functional.one_hot(target_bin, num_classes=self.num_bins).to(dtype=torch.float32)

        probe_input = self._get_probe_input(env, modify_history=modify_history)
        self._setup_online_probe(probe_input.shape[-1])
        assert self.probe is not None
        with torch.no_grad():
            probs = torch.softmax(self.probe(probe_input), dim=1).detach()
        if modify_history:
            self._log_prediction_metrics(env, probs)
            self._train_online_probe(env, probe_input)
        return self._format_prediction_output(probs)


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
        self.student_latent_input_key: str | None = None
        self.student_obs_dim: int | None = None
        self.student_latent_dim: int | None = None
        self.num_actions = env.robot_config.actions_dim

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        return

    def _get_student_actor_obs(self, env: WholeBodyTrackingManager) -> torch.Tensor:
        student_obs = env.observation_manager.compute_group(self.student_obs_group, modify_history=False)
        if not isinstance(student_obs, torch.Tensor):
            raise ValueError(
                f"Observation group '{self.student_obs_group}' must be concatenated for frozen student input."
            )
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
        actual_keys = set(self.student_input_keys)
        supported_latent_keys = {"ae_latent", "student_latent", "ir_ae_latent"}
        latent_input_keys = [key for key in self.student_input_keys if key != "actor_obs"]
        if (
            "actor_obs" not in actual_keys
            or len(latent_input_keys) != 1
            or latent_input_keys[0] not in supported_latent_keys
        ):
            raise ValueError(
                "Frozen student actor expects input keys ['actor_obs', '<latent_key>'] where "
                f"<latent_key> is one of {sorted(supported_latent_keys)}. Got {self.student_input_keys}."
            )
        self.student_latent_input_key = latent_input_keys[0]

        self.student_actor = setup_ppo_actor_module(
            obs_dim_dict={"actor_obs": student_obs_dim, self.student_latent_input_key: latent_dim},
            module_config=student_actor_cfg,
            num_actions=self.num_actions,
            init_noise_std=getattr(student_algo_config, "init_noise_std", 1.0),
            device=self.device,
            history_length={"actor_obs": 1, self.student_latent_input_key: 1},
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
            elif key in {"ae_latent", "student_latent", "ir_ae_latent"}:
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
