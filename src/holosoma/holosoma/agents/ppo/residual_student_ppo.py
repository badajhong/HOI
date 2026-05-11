from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.agents.ppo.ppo import PPO
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


class _FusedStudentResidualActorWrapper(nn.Module):
    def __init__(
        self,
        residual_actor: nn.Module,
        student_actor: nn.Module,
        student_input_keys: list[str],
        shared_obs_dim: int,
        di_encoder: nn.Module,
        depth_shape: tuple[int, int, int],
        di_feature_mean: torch.Tensor,
        di_feature_std: torch.Tensor,
        di_text_features: torch.Tensor,
        proprioception_shape: tuple[int, int] | None = None,
        proprio_feature_mean: torch.Tensor | None = None,
        proprio_feature_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.actor = residual_actor
        self.student_actor = student_actor
        self.di_encoder = di_encoder
        self.student_input_keys = list(student_input_keys)
        self.shared_obs_dim = int(shared_obs_dim)
        self.depth_shape = tuple(int(value) for value in depth_shape)
        self.depth_flat_dim = int(self.depth_shape[0] * self.depth_shape[1] * self.depth_shape[2])
        self.proprioception_shape = (
            tuple(int(value) for value in proprioception_shape) if proprioception_shape is not None else None
        )
        self.proprioception_flat_dim = (
            int(self.proprioception_shape[0] * self.proprioception_shape[1])
            if self.proprioception_shape is not None
            else 0
        )
        self.onnx_input_dim = self.shared_obs_dim + self.depth_flat_dim + self.proprioception_flat_dim
        self.register_buffer("di_feature_mean", di_feature_mean)
        self.register_buffer("di_feature_std", di_feature_std)
        self.register_buffer("di_text_features", di_text_features)
        if self.proprioception_shape is not None:
            if proprio_feature_mean is None or proprio_feature_std is None:
                raise ValueError("DI+proprioception fused export requires proprioception normalization statistics.")
            self.register_buffer("proprio_feature_mean", proprio_feature_mean)
            self.register_buffer("proprio_feature_std", proprio_feature_std)

    def _build_student_input(self, shared_obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        inputs = []
        for key in self.student_input_keys:
            if key == "actor_obs":
                inputs.append(shared_obs)
            else:
                inputs.append(latent)
        return torch.cat(inputs, dim=1)

    def _extract_depth_window(self, actor_obs: torch.Tensor) -> torch.Tensor:
        depth_flat = actor_obs[:, self.shared_obs_dim : self.shared_obs_dim + self.depth_flat_dim]
        return depth_flat.reshape(actor_obs.shape[0], *self.depth_shape)

    def _extract_proprioception_window(self, actor_obs: torch.Tensor) -> torch.Tensor | None:
        if self.proprioception_shape is None:
            return None
        start = self.shared_obs_dim + self.depth_flat_dim
        end = start + self.proprioception_flat_dim
        proprioception_flat = actor_obs[:, start:end]
        return proprioception_flat.reshape(actor_obs.shape[0], *self.proprioception_shape)

    def _encode_di_latent(
        self,
        depth_window: torch.Tensor,
        proprioception_window: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = (depth_window - self.di_feature_mean.unsqueeze(0)) / self.di_feature_std.unsqueeze(0)
        text_features = self.di_text_features.expand(depth_window.shape[0], -1)
        if self.proprioception_shape is not None:
            if proprioception_window is None:
                raise RuntimeError("DI+proprioception fused export requires a proprioception window.")
            normalized_proprioception = (
                proprioception_window - self.proprio_feature_mean.unsqueeze(0)
            ) / self.proprio_feature_std.unsqueeze(0)
            mu, _ = self.di_encoder.encode_di(normalized, text_features, normalized_proprioception)
        else:
            mu, _ = self.di_encoder.encode_di(normalized, text_features)
        return mu

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        shared_obs = actor_obs[:, : self.shared_obs_dim]
        depth_window = self._extract_depth_window(actor_obs)
        proprioception_window = self._extract_proprioception_window(actor_obs)
        latent = self._encode_di_latent(depth_window, proprioception_window)
        student_input = self._build_student_input(shared_obs, latent)
        base_action = self.student_actor.act_inference({"actor_obs": student_input})
        residual_input = torch.cat([shared_obs, base_action, latent], dim=1)
        residual_action = self.actor.act_inference({"actor_obs": residual_input})
        return residual_action + base_action


class ResidualStudentPPO(PPO):
    """PPO variant that predicts residual actions on top of a frozen student base action."""

    def _init_config(self) -> None:
        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()

        assert self.env.observation_manager is not None
        self._init_obs_keys()
        self.algo_history_length_dict = {}
        for key in set(self.actor_obs_keys + self.critic_obs_keys):
            if key not in self.env.observation_manager.cfg.groups:
                raise KeyError(
                    f"ResidualStudentPPO expects observation group '{key}', "
                    f"available groups are {sorted(self.env.observation_manager.cfg.groups.keys())}."
                )
            self.algo_history_length_dict[key] = self.env.observation_manager.cfg.groups[key].history_length

        self.num_act = self.env.robot_config.actions_dim

        self.actor_learning_rate = self.config.actor_learning_rate
        self.max_actor_learning_rate = self.config.max_actor_learning_rate or max(self.actor_learning_rate, 1e-2)
        self.min_actor_learning_rate = self.config.min_actor_learning_rate or min(self.actor_learning_rate, 1e-5)
        self.critic_learning_rate = self.config.critic_learning_rate
        self.max_critic_learning_rate = self.config.max_critic_learning_rate or max(self.critic_learning_rate, 1e-2)
        self.min_critic_learning_rate = self.config.min_critic_learning_rate or min(self.critic_learning_rate, 1e-5)

        self.use_symmetry = self.config.use_symmetry
        self.base_action_key = "student_base_action"
        if self.base_action_key not in self.actor_obs_keys:
            raise ValueError(
                f"ResidualStudentPPO requires actor input key '{self.base_action_key}', got {self.actor_obs_keys}."
            )

        base_action_index = self.actor_obs_keys.index(self.base_action_key)
        self.base_action_dim = self._get_obs_dim([self.base_action_key])
        self.base_action_start = self._get_obs_dim(self.actor_obs_keys[:base_action_index])
        self.base_action_end = self.base_action_start + self.base_action_dim

        if self.base_action_dim != self.num_act:
            raise ValueError(
                f"Base-action observation dim must equal action dim ({self.num_act}), got {self.base_action_dim}."
            )

    def _extract_base_action(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.base_action_key not in obs_dict:
            raise KeyError(
                f"ResidualStudentPPO expected observation key '{self.base_action_key}', "
                f"available keys are {sorted(obs_dict.keys())}."
            )
        return obs_dict[self.base_action_key]

    def _compose_final_action(self, residual_action: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        return residual_action + base_action

    def _sync_student_prev_action(self, final_action: torch.Tensor) -> None:
        """Expose the executed final action as the frozen student's previous-action input.

        During residual training, the frozen student should autoregress on the
        actual executed action from the previous step (student base + residual),
        not on the frozen student's standalone base action.
        """
        self.env.student_prev_actions[:] = final_action.detach()

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                actor_obs = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
                base_action = self._extract_base_action(obs_dict)

                residual_actions = self.actor.act({"actor_obs": actor_obs})
                final_actions = self._compose_final_action(residual_actions, base_action)
                values = self.critic.evaluate({"critic_obs": critic_obs}).detach()

                self._sync_student_prev_action(final_actions)
                obs_dict, rewards, dones, infos = self.env.step({"actions": final_actions})

                for obs_key in obs_dict:
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                final_rewards = torch.zeros_like(rewards)
                if infos["time_outs"].any():
                    final_critic_obs = torch.cat([infos["final_observations"][k] for k in self.critic_obs_keys], dim=1)
                    final_values = self.critic.evaluate({"critic_obs": final_critic_obs}).detach()
                    final_rewards += self.config.gamma * torch.squeeze(
                        final_values * infos["time_outs"].unsqueeze(1).to(self.device), 1
                    )

                self.storage.add(
                    actor_obs=actor_obs,
                    critic_obs=critic_obs,
                    actions=residual_actions,
                    values=values,
                    actions_log_prob=self.actor.get_actions_log_prob(residual_actions).detach().unsqueeze(1),
                    action_mean=self.actor.action_mean.detach(),
                    action_sigma=self.actor.action_std.detach(),
                    rewards=(rewards + final_rewards).view(-1, 1),
                    dones=dones.view(-1, 1),
                )

                self.actor.reset(dones)
                self.critic.reset(dones)

                if self.log_dir is not None:
                    self.logging_helper.update_episode_stats(rewards, dones, infos)

            last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
            last_values = self.critic.evaluate({"critic_obs": last_critic_obs}).detach().to(self.device)
            returns, advantages = self._compute_returns_and_advantages(
                last_values,
                self.storage["values"].to(self.device),
                self.storage["dones"].to(self.device),
                self.storage["rewards"].to(self.device),
            )

            self.storage["returns"] = returns
            self.storage["advantages"] = advantages

        return obs_dict

    def env_step(self, actor_state):
        self._sync_student_prev_action(actor_state["actions"])
        return super().env_step(actor_state)

    def _load_frozen_student_export_bundle(self) -> dict[str, Any]:
        if hasattr(self, "_student_export_bundle") and self._student_export_bundle is not None:
            return self._student_export_bundle

        student_checkpoint = getattr(self.env, "student", None)
        if not student_checkpoint:
            raise ValueError("Residual fused ONNX export requires a frozen student checkpoint on the environment.")

        if "student_actor_obs" not in self.algo_obs_dim_dict:
            raise KeyError("Residual fused ONNX export requires observation group 'student_actor_obs'.")
        if "residual_actor_obs" not in self.algo_obs_dim_dict:
            raise KeyError("Residual fused ONNX export requires observation group 'residual_actor_obs'.")
        if "di_ae_latent" not in self.algo_obs_dim_dict:
            raise KeyError("Residual fused ONNX export requires observation group 'di_ae_latent'.")

        shared_obs_dim = int(self.algo_obs_dim_dict["student_actor_obs"])
        residual_shared_obs_dim = int(self.algo_obs_dim_dict["residual_actor_obs"])
        if shared_obs_dim != residual_shared_obs_dim:
            raise ValueError(
                "Residual fused ONNX export expects student_actor_obs and residual_actor_obs to share the same dim, "
                f"got {shared_obs_dim} and {residual_shared_obs_dim}."
            )

        latent_dim = int(self.algo_obs_dim_dict["di_ae_latent"])
        student_config, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(student_checkpoint)))
        student_payload = torch.load(student_checkpoint, map_location=self.device)
        student_algo_config = getattr(student_config.algo, "config", None)
        if student_algo_config is None or not hasattr(student_algo_config, "module_dict"):
            raise ValueError("Student checkpoint must contain a PPO-style actor module configuration.")

        student_actor_cfg = student_algo_config.module_dict.actor
        student_input_keys = list(student_actor_cfg.input_dim)
        latent_input_keys = [key for key in student_input_keys if key != "actor_obs"]
        supported_latent_keys = {"ae_latent", "student_latent", "ir_ae_latent"}
        if "actor_obs" not in student_input_keys or len(latent_input_keys) != 1:
            raise ValueError(
                f"Frozen student export expects input keys ['actor_obs', '<latent_key>'], got {student_input_keys}."
            )
        latent_key = latent_input_keys[0]
        if latent_key not in supported_latent_keys:
            raise ValueError(
                f"Unsupported frozen student latent key '{latent_key}'. "
                f"Expected one of {sorted(supported_latent_keys)}."
            )

        student_actor = setup_ppo_actor_module(
            obs_dim_dict={"actor_obs": shared_obs_dim, latent_key: latent_dim},
            module_config=student_actor_cfg,
            num_actions=self.num_act,
            init_noise_std=getattr(student_algo_config, "init_noise_std", 1.0),
            device=self.device,
            history_length={"actor_obs": 1, latent_key: 1},
        )
        student_actor.load_state_dict(student_payload["actor_model_state_dict"])
        student_actor.eval()
        for parameter in student_actor.parameters():
            parameter.requires_grad_(False)
        if hasattr(student_actor, "std"):
            student_actor.std.requires_grad_(False)

        self._student_export_bundle = {
            "actor": student_actor,
            "input_keys": student_input_keys,
            "latent_key": latent_key,
            "shared_obs_dim": shared_obs_dim,
            "latent_dim": latent_dim,
        }
        return self._student_export_bundle

    def _load_di_export_bundle(self) -> dict[str, Any]:
        if hasattr(self, "_di_export_bundle") and self._di_export_bundle is not None:
            return self._di_export_bundle

        di_checkpoint = getattr(self.env, "di_ae", None)
        di_pro_checkpoint = getattr(self.env, "di_pro_ae", None)
        if di_checkpoint and di_pro_checkpoint:
            raise ValueError(
                "Residual fused ONNX export received both DI and DI+proprioception latent checkpoints. "
                "Pass only one of `--di_ae` or `--di_pro_ae`."
            )
        if not di_checkpoint and not di_pro_checkpoint:
            raise ValueError(
                "Residual fused ONNX export requires a DI latent checkpoint on the environment. "
                "Pass `--di_ae=/path/to/best.pt` or `--di_pro_ae=/path/to/best.pt`."
            )

        from holosoma.ae_joint_train import CLIPTextFeatureExtractor, load_joint_model  # noqa: PLC0415
        from holosoma.ae_pro_joint_train import load_joint_model as load_pro_joint_model  # noqa: PLC0415

        uses_di_pro = bool(di_pro_checkpoint)
        checkpoint_path = str(di_pro_checkpoint or di_checkpoint)
        loader = load_pro_joint_model if uses_di_pro else load_joint_model
        di_encoder, payload = loader(checkpoint_path, device=self.device)
        di_encoder.eval()
        for parameter in di_encoder.parameters():
            parameter.requires_grad_(False)

        feature_mean = payload["di_feature_mean"].to(device=self.device, dtype=torch.float32)
        feature_std = payload["di_feature_std"].to(device=self.device, dtype=torch.float32).clamp_min(1e-6)
        depth_shape = tuple(int(value) for value in payload["input_shape"])
        latent_dim = int(payload["config"]["latent_dim"])
        proprioception_shape = (
            tuple(int(value) for value in payload["proprioception_input_shape"])
            if payload.get("proprioception_input_shape") is not None
            else None
        )
        proprio_feature_mean = None
        proprio_feature_std = None
        if proprioception_shape is not None:
            if len(proprioception_shape) != 2:
                raise ValueError(
                    "DI+proprioception checkpoint proprioception_input_shape must have length 2, "
                    f"got {proprioception_shape}."
                )
            if int(proprioception_shape[0]) != int(depth_shape[0]):
                raise ValueError(
                    "DI+proprioception checkpoint window mismatch: "
                    f"depth window={depth_shape[0]}, proprioception window={proprioception_shape[0]}."
                )
            expected_proprio_dim = 3 + 2 * len(self.env.dof_names)
            if int(proprioception_shape[1]) != expected_proprio_dim:
                raise ValueError(
                    "DI+proprioception checkpoint feature dim mismatch: "
                    f"checkpoint expects {proprioception_shape[1]}, live env produces {expected_proprio_dim} "
                    "(base_ang_vel + dof_pos + dof_vel)."
                )
            payload_proprio_mean = payload.get("proprio_feature_mean")
            payload_proprio_std = payload.get("proprio_feature_std")
            if payload_proprio_mean is None or payload_proprio_std is None:
                raise RuntimeError(
                    "DI+proprioception checkpoint declares proprioception input but is missing "
                    "proprio_feature_mean/proprio_feature_std."
                )
            proprio_feature_mean = payload_proprio_mean.to(device=self.device, dtype=torch.float32)
            proprio_feature_std = payload_proprio_std.to(device=self.device, dtype=torch.float32).clamp_min(1e-6)

        clip_cfg = payload["clip"]
        text_extractor = CLIPTextFeatureExtractor(
            model_id=clip_cfg["model_id"],
            device=self.device,
            cache_dir=clip_cfg["cache_dir"],
            local_files_only=clip_cfg["local_files_only"],
            quiet_load=True,
        )
        text_features = text_extractor.encode([payload["condition_text"]]).to(device=self.device, dtype=torch.float32)

        self._di_export_bundle = {
            "encoder": di_encoder,
            "payload": payload,
            "checkpoint_path": checkpoint_path,
            "checkpoint_kind": "di_pro" if uses_di_pro else "di",
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "depth_shape": depth_shape,
            "latent_dim": latent_dim,
            "text_features": text_features,
            "proprioception_shape": proprioception_shape,
            "proprio_feature_mean": proprio_feature_mean,
            "proprio_feature_std": proprio_feature_std,
        }
        return self._di_export_bundle

    def _build_fused_export_spec(self) -> dict[str, Any]:
        observation_manager = self.env.observation_manager
        assert observation_manager is not None
        group_name = "student_actor_obs"
        if group_name not in observation_manager.cfg.groups:
            raise KeyError(f"Residual fused ONNX export requires observation group '{group_name}'.")
        group_cfg = observation_manager.cfg.groups[group_name]
        # Preserve the exact training-time concatenation order for shared observations.
        term_names = list(group_cfg.terms.keys())
        term_dims: dict[str, int] = {}
        term_scales: dict[str, Any] = {}
        for term_name in term_names:
            term_cfg = group_cfg.terms[term_name]
            term_obs = observation_manager._compute_term(group_name, term_name, term_cfg, modify_history=False)
            term_dims[term_name] = int(term_obs.shape[1])
            term_scales[term_name] = _jsonable_value(term_cfg.scale)

        bundle = self._load_frozen_student_export_bundle()
        di_bundle = self._load_di_export_bundle()
        proprioception_shape = di_bundle["proprioception_shape"]
        proprioception_flat_dim = (
            int(proprioception_shape[0] * proprioception_shape[1]) if proprioception_shape is not None else 0
        )
        return {
            "kind": "wbt_student_residual_fused",
            "shared_obs_terms": term_names,
            "shared_obs_dims": term_dims,
            "shared_obs_scales": term_scales,
            "shared_obs_history_length": int(group_cfg.history_length),
            "latent_dim": int(bundle["latent_dim"]),
            "num_actions": int(self.num_act),
            "di_encoder_mode": "fused",
            "di_checkpoint_kind": di_bundle["checkpoint_kind"],
            "depth_window_shape": list(di_bundle["depth_shape"]),
            "depth_window_flat_dim": int(
                di_bundle["depth_shape"][0] * di_bundle["depth_shape"][1] * di_bundle["depth_shape"][2]
            ),
            "uses_proprioception_window": proprioception_shape is not None,
            "proprioception_window_shape": list(proprioception_shape) if proprioception_shape is not None else None,
            "proprioception_window_flat_dim": proprioception_flat_dim,
        }

    def _checkpoint_metadata(self, iteration: int | None = None) -> dict[str, Any]:
        metadata = super()._checkpoint_metadata(iteration=iteration)
        metadata["policy_export_spec"] = self._build_fused_export_spec()
        return metadata

    @property
    def actor_onnx_wrapper(self):
        bundle = self._load_frozen_student_export_bundle()
        di_bundle = self._load_di_export_bundle()
        if int(bundle["latent_dim"]) != int(di_bundle["latent_dim"]):
            raise ValueError(
                "Residual fused ONNX export requires student latent dim to match DI encoder latent dim, "
                f"got {bundle['latent_dim']} and {di_bundle['latent_dim']}."
            )
        return _FusedStudentResidualActorWrapper(
            residual_actor=self.actor,
            student_actor=bundle["actor"],
            student_input_keys=bundle["input_keys"],
            shared_obs_dim=bundle["shared_obs_dim"],
            di_encoder=di_bundle["encoder"],
            depth_shape=di_bundle["depth_shape"],
            di_feature_mean=di_bundle["feature_mean"],
            di_feature_std=di_bundle["feature_std"],
            di_text_features=di_bundle["text_features"],
            proprioception_shape=di_bundle["proprioception_shape"],
            proprio_feature_mean=di_bundle["proprio_feature_mean"],
            proprio_feature_std=di_bundle["proprio_feature_std"],
        )

    def get_inference_policy(self, device=None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        self.actor.eval()
        if device is not None:
            self.actor.to(device)

        def inference_fn(policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            actor_obs = policy_state_dict["actor_obs"]
            residual_action = self.actor.act_inference({"actor_obs": actor_obs})
            base_action = actor_obs[:, self.base_action_start : self.base_action_end]
            return residual_action + base_action

        return inference_fn
