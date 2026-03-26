from __future__ import annotations

from typing import Sequence

from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.utils.pelvis_surface_features import PelvisSurfaceFeatureComputer


class PelvisSurfaceFeatureEvalCallback(RLEvalCallback):
    """Compute pelvis-based surface features live during evaluation playback."""

    def __init__(
        self,
        training_loop,
        log_env_ids: Sequence[int] | None = None,
        log_every_n_steps: int = 1,
        pelvis_body_name: str = "pelvis",
        mesh_mode: str = "box",
    ):
        config = {
            "log_env_ids": list(log_env_ids or [0]),
            "log_every_n_steps": max(int(log_every_n_steps), 1),
            "pelvis_body_name": pelvis_body_name,
            "mesh_mode": mesh_mode,
        }
        super().__init__(config=config, training_loop=training_loop)
        self.log_env_ids = list(log_env_ids or [0])
        self.log_every_n_steps = max(int(log_every_n_steps), 1)
        self.pelvis_body_name = pelvis_body_name
        self.mesh_mode = mesh_mode
        self._enabled = True
        self._feature_computer: PelvisSurfaceFeatureComputer | None = None
        self._pelvis_body_index: int | None = None

    def _unwrap_env(self):
        return self.training_loop._unwrap_env()

    def _object_keys_for_envs(self, motion_command, num_envs: int) -> list[str | None]:
        object_key_to_id = getattr(motion_command, "object_key_to_id", None) or {}
        if not object_key_to_id:
            return [None] * num_envs
        id_to_key = {int(idx): key for key, idx in object_key_to_id.items()}
        object_type_ids = motion_command.object_type_ids.detach().cpu().tolist()
        return [id_to_key.get(int(type_id)) for type_id in object_type_ids]

    def on_pre_evaluate_policy(self):
        env = self._unwrap_env()
        motion_command = env.command_manager.get_state("motion_command")
        if motion_command is None:
            logger.warning("Live pelvis surface features were requested, but no motion_command is registered.")
            self._enabled = False
            return
        if not getattr(motion_command.motion, "has_object", False):
            logger.warning("Live pelvis surface features require an object in the motion data. Disabling callback.")
            self._enabled = False
            return
        if self.pelvis_body_name not in env.body_names:
            logger.warning(
                f"Pelvis body '{self.pelvis_body_name}' was not found in simulator body names: {env.body_names}."
            )
            self._enabled = False
            return

        self._pelvis_body_index = env.body_names.index(self.pelvis_body_name)

        try:
            self._feature_computer = PelvisSurfaceFeatureComputer.from_object_config(
                env.robot_config.object, mesh_mode=self.mesh_mode
            )
        except Exception as exc:
            logger.warning(f"Failed to initialize live pelvis surface feature callback: {exc}")
            self._enabled = False
            return

        logger.info(
            f"Live pelvis surface features enabled for body '{self.pelvis_body_name}' and env ids {self.log_env_ids} "
            f"(every {self.log_every_n_steps} step(s), mesh_mode={self.mesh_mode})."
        )

    def on_post_eval_env_step(self, actor_state):
        if not self._enabled or self._feature_computer is None or self._pelvis_body_index is None:
            return actor_state

        env = self._unwrap_env()
        motion_command = env.command_manager.get_state("motion_command")
        if motion_command is None:
            return actor_state

        pelvis_pos_w = env.simulator._rigid_body_pos[:, self._pelvis_body_index, :]
        pelvis_lin_vel_w = env.simulator._rigid_body_vel[:, self._pelvis_body_index, :]
        object_keys = self._object_keys_for_envs(motion_command, env.num_envs)
        features = self._feature_computer.compute_batch(
            pelvis_pos_w=pelvis_pos_w,
            pelvis_lin_vel_w=pelvis_lin_vel_w,
            object_pos_w=motion_command.simulator_object_pos_w,
            object_quat_w=motion_command.simulator_object_quat_w,
            object_keys=object_keys,
        )

        actor_state["pelvis_surface_features"] = features
        env.live_pelvis_surface_features = features
        env.live_pelvis_surface_feature_object_keys = object_keys

        step = int(actor_state.get("step", -1))
        if step >= 0 and step % self.log_every_n_steps == 0:
            valid_env_ids = [env_id for env_id in self.log_env_ids if 0 <= env_id < env.num_envs]
            for env_id in valid_env_ids:
                phi = float(features["phi"][env_id, 0].item())
                grad_phi = [round(float(v), 6) for v in features["grad_phi"][env_id].tolist()]
                v_norm = [round(float(v), 6) for v in features["v_norm"][env_id].tolist()]
                v_tan = [round(float(v), 6) for v in features["v_tan"][env_id].tolist()]
                object_key = object_keys[env_id]
                logger.info(
                    f"[pelvis_u_t] step={step} env={env_id} body={self.pelvis_body_name} object={object_key} "
                    f"phi={phi:.6f} grad_phi={grad_phi} v_norm={v_norm} v_tan={v_tan}"
                )

        return actor_state
