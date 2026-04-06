from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from holosoma.agents.ppo.ppo import PPO


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

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                actor_obs = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
                base_action = self._extract_base_action(obs_dict)

                residual_actions = self.actor.act({"actor_obs": actor_obs})
                final_actions = self._compose_final_action(residual_actions, base_action)
                values = self.critic.evaluate({"critic_obs": critic_obs}).detach()

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

    @property
    def actor_onnx_wrapper(self):
        class ActorWrapper(nn.Module):
            def __init__(self, actor, base_action_start: int, base_action_end: int):
                super().__init__()
                self.actor = actor
                self.base_action_start = base_action_start
                self.base_action_end = base_action_end

            def forward(self, actor_obs):
                residual_action = self.actor.act_inference({"actor_obs": actor_obs})
                base_action = actor_obs[:, self.base_action_start : self.base_action_end]
                return residual_action + base_action

        return ActorWrapper(self.actor, self.base_action_start, self.base_action_end)

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
