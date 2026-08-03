from __future__ import annotations

import torch
from loguru import logger

from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization
from holosoma.agents.ppo.ppo import PPO
from holosoma.utils.eval_utils import CheckpointConfig, load_checkpoint, load_saved_experiment_config


class StudentInitializedPPO(PPO):
    """PPO whose actor mean network starts from a trained DAgger student."""

    ACTION_CLIP = 20.0

    def _init_config(self) -> None:
        super()._init_config()
        assert self.env.observation_manager is not None
        for key in set(self.actor_obs_keys + self.critic_obs_keys):
            group = self.env.observation_manager.cfg.groups.get(key)
            if group is None:
                raise KeyError(
                    f"StudentInitializedPPO requires observation group '{key}'; "
                    f"available={sorted(self.env.observation_manager.cfg.groups)}."
                )
            self.algo_history_length_dict[key] = group.history_length

    def _setup_models_and_optimizer(self) -> None:
        super()._setup_models_and_optimizer()
        self._load_student_actor_initialization()

    def _load_student_actor_initialization(self) -> None:
        student_reference = getattr(self.env, "student", None)
        if not student_reference:
            raise ValueError(
                "StudentInitializedPPO requires `--student /path/to/student_checkpoint.pt`."
            )

        student_path = load_checkpoint(str(student_reference), self.log_dir)
        student_config, _ = load_saved_experiment_config(
            CheckpointConfig(checkpoint=str(student_path))
        )
        student_algo_config = getattr(student_config.algo, "config", None)
        if student_algo_config is None or not hasattr(student_algo_config, "module_dict"):
            raise ValueError("The student checkpoint has no actor module configuration.")

        student_actor_config = student_algo_config.module_dict.actor
        student_input_keys = list(student_actor_config.input_dim)
        ppo_input_keys = list(self.config.module_dict.actor.input_dim)
        if student_input_keys != ppo_input_keys:
            raise ValueError(
                "Student and final-PPO actor input groups must match exactly: "
                f"student={student_input_keys}, final_ppo={ppo_input_keys}."
            )

        # New DAgger checkpoints can contain a multi-gigabyte replay snapshot
        # for exact resume.  Stage the payload on CPU so PPO initialization does
        # not temporarily duplicate that buffer in scarce GPU memory.
        payload = torch.load(student_path, map_location="cpu", weights_only=False)
        if "actor_model_state_dict" not in payload:
            raise KeyError(f"Student checkpoint has no actor_model_state_dict: {student_path}")

        # Load the complete actor so all learned feature and policy layers are
        # preserved. DAgger does not train its action std, so reset only that
        # parameter to the deliberately small PPO exploration value.
        self.actor.load_state_dict(payload["actor_model_state_dict"], strict=True)
        if "critic_model_state_dict" not in payload:
            raise KeyError(
                "Student checkpoint has no critic_model_state_dict. Train a new r1-student "
                f"checkpoint with joint V/Q critic learning: {student_path}"
            )
        self.critic.load_state_dict(payload["critic_model_state_dict"], strict=True)
        normalization_info = payload.get("critic_normalization")
        if not isinstance(normalization_info, dict) or not bool(
            normalization_info.get("value_critic_obs_normalized", False)
        ):
            raise ValueError(
                "Student V critic was not trained with the normalized-observation schema. "
                "Train a new r1-student checkpoint before initializing final PPO."
            )
        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        if int(normalization_info.get("critic_obs_dim", -1)) != critic_obs_dim:
            raise ValueError(
                "Student/final-PPO critic observation dimension mismatch: "
                f"student={normalization_info.get('critic_obs_dim')}, final_ppo={critic_obs_dim}."
            )
        self.critic_obs_normalizer = EmpiricalNormalization(
            shape=critic_obs_dim,
            device=self.device,
            eps=float(normalization_info["eps"]),
        )
        normalizer_state = payload.get("critic_obs_normalizer_state")
        if normalizer_state is None:
            raise KeyError(
                f"Student checkpoint has no critic_obs_normalizer_state: {student_path}"
            )
        self.critic_obs_normalizer.load_state_dict(normalizer_state, strict=True)
        with torch.no_grad():
            self.actor.std.fill_(float(self.config.init_noise_std))

        if self.is_multi_gpu:
            self._synchronize_model_weights()

        logger.info(
            f"Initialized final PPO actor from DAgger student {student_path}; "
            f"input_keys={ppo_input_keys}, exploration_std={self.config.init_noise_std}. "
            "The PPO value critic was initialized from the student's V(s) critic; "
            f"PPO optimizers start from scratch; executed actions are clipped to "
            f"[-{self.ACTION_CLIP:g}, {self.ACTION_CLIP:g}]."
        )

    def _normalize_critic_obs(
        self,
        critic_obs: torch.Tensor,
        *,
        update: bool,
    ) -> torch.Tensor:
        normalizer = getattr(self, "critic_obs_normalizer", None)
        if normalizer is None:
            return critic_obs
        return normalizer(critic_obs, update=update)

    def _train_mode(self) -> None:
        super()._train_mode()
        if hasattr(self, "critic_obs_normalizer"):
            self.critic_obs_normalizer.train()

    def _eval_mode(self) -> None:
        super()._eval_mode()
        if hasattr(self, "critic_obs_normalizer"):
            self.critic_obs_normalizer.eval()

    def _extra_checkpoint_state(self) -> dict:
        if not hasattr(self, "critic_obs_normalizer"):
            return {}
        return {
            "critic_obs_normalizer_state": self.critic_obs_normalizer.state_dict(),
            "critic_normalization": {
                "schema": 2,
                "enabled": True,
                "value_critic_obs_normalized": True,
                "eps": float(self.critic_obs_normalizer.eps),
                "critic_obs_dim": self._get_obs_dim(self.critic_obs_keys),
            },
        }

    def _load_extra_checkpoint_state(self, loaded_dict: dict) -> None:
        normalization_info = loaded_dict.get("critic_normalization")
        if not isinstance(normalization_info, dict) or not bool(
            normalization_info.get("value_critic_obs_normalized", False)
        ):
            raise ValueError(
                "Final-PPO checkpoint predates critic normalization and cannot be resumed "
                "with normalized student V semantics."
            )
        normalizer_state = loaded_dict.get("critic_obs_normalizer_state")
        if normalizer_state is None:
            raise KeyError("Final-PPO checkpoint is missing critic_obs_normalizer_state.")
        checkpoint_eps = float(normalization_info.get("eps", float("nan")))
        if checkpoint_eps != float(self.critic_obs_normalizer.eps):
            raise ValueError(
                "Final-PPO critic normalizer epsilon mismatch: "
                f"checkpoint={checkpoint_eps}, configured={self.critic_obs_normalizer.eps}."
            )
        self.critic_obs_normalizer.load_state_dict(normalizer_state, strict=True)

    def _actions_for_env(self, actions: torch.Tensor) -> torch.Tensor:
        """Bound executed actions while leaving PPO's sampled actions unchanged."""
        return actions.clamp(-self.ACTION_CLIP, self.ACTION_CLIP)

    def env_step(self, actor_state: dict):
        """Apply the same bound after evaluation callbacks and before simulation."""
        actor_state["actions"] = self._actions_for_env(actor_state["actions"])
        return super().env_step(actor_state)

    def get_inference_policy(self, device=None):
        self.actor.eval()
        if device is not None:
            self.actor.to(device)

        def inference_policy(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            return self._actions_for_env(self.actor.act_inference(obs_dict))

        return inference_policy

    @property
    def actor_onnx_wrapper(self):
        class ActorWrapper(torch.nn.Module):
            def __init__(self, actor: torch.nn.Module, action_clip: float):
                super().__init__()
                self.actor = actor
                self.action_clip = float(action_clip)

            def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
                actions = self.actor.act_inference({"actor_obs": actor_obs})
                return actions.clamp(-self.action_clip, self.action_clip)

        return ActorWrapper(self.actor, self.ACTION_CLIP)


__all__ = ["StudentInitializedPPO"]
