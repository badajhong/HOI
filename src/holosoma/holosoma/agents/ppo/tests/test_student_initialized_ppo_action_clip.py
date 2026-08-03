from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from holosoma.agents.ppo.student_initialized_ppo import StudentInitializedPPO


class _FakeActor(nn.Module):
    def __init__(self, actions: torch.Tensor):
        super().__init__()
        self.actions = actions
        self.action_mean = actions.clone()
        self.action_std = torch.ones_like(actions)
        self.log_prob_actions: torch.Tensor | None = None

    def act(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        del obs_dict
        return self.actions.clone()

    def act_inference(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        del obs_dict
        return self.actions.clone()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        self.log_prob_actions = actions.clone()
        return actions.sum(dim=-1)

    def reset(self, dones: torch.Tensor) -> None:
        del dones


class _FakeCritic:
    def __init__(self) -> None:
        self.seen_obs: list[torch.Tensor] = []

    def evaluate(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        critic_obs = obs_dict["critic_obs"]
        self.seen_obs.append(critic_obs.clone())
        return torch.zeros((critic_obs.shape[0], 1), device=critic_obs.device)

    def reset(self, dones: torch.Tensor) -> None:
        del dones


class _FakeEnv:
    def __init__(self, obs_dict: dict[str, torch.Tensor]):
        self.obs_dict = obs_dict
        self.executed_actions: torch.Tensor | None = None

    def step(self, action_dict: dict[str, torch.Tensor]):
        self.executed_actions = action_dict["actions"].clone()
        num_envs = self.executed_actions.shape[0]
        rewards = torch.zeros(num_envs)
        dones = torch.zeros(num_envs, dtype=torch.bool)
        infos = {
            "time_outs": torch.zeros(num_envs, dtype=torch.bool),
            "final_observations": self.obs_dict,
        }
        return self.obs_dict, rewards, dones, infos


class _FakeStorage:
    def __init__(self):
        self.transition: dict[str, torch.Tensor] = {}
        self.computed: dict[str, torch.Tensor] = {}

    def add(self, **transition: torch.Tensor) -> None:
        self.transition = {key: value.clone() for key, value in transition.items()}

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self.computed:
            return self.computed[key]
        return self.transition[key].unsqueeze(0)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self.computed[key] = value


class _FakeNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.update_flags: list[bool] = []

    def forward(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        self.update_flags.append(update)
        return obs + 5.0


def _make_uninitialized_algo(actions: torch.Tensor) -> StudentInitializedPPO:
    algo = object.__new__(StudentInitializedPPO)
    algo.config = SimpleNamespace(num_steps_per_env=1, gamma=0.99, lam=0.95)
    algo.actor_obs_keys = ["actor_obs"]
    algo.critic_obs_keys = ["critic_obs"]
    algo.device = "cpu"
    algo.actor = _FakeActor(actions)
    algo.critic = _FakeCritic()
    algo.storage = _FakeStorage()
    algo.log_dir = None
    algo.is_multi_gpu = False
    return algo


def test_rollout_executes_clipped_action_but_stores_and_scores_raw_sample() -> None:
    raw_actions = torch.tensor([[25.0, -30.0, 3.0], [-4.0, 7.0, 24.0]])
    obs_dict = {
        "actor_obs": torch.zeros((2, 2)),
        "critic_obs": torch.zeros((2, 2)),
    }
    algo = _make_uninitialized_algo(raw_actions)
    algo.env = _FakeEnv(obs_dict)

    algo._rollout_step(obs_dict)

    expected_executed = torch.tensor([[20.0, -20.0, 3.0], [-4.0, 7.0, 20.0]])
    assert torch.equal(algo.env.executed_actions, expected_executed)
    assert torch.equal(algo.storage.transition["actions"], raw_actions)
    assert torch.equal(algo.actor.log_prob_actions, raw_actions)


def test_inference_and_onnx_wrapper_apply_same_action_bound() -> None:
    raw_actions = torch.tensor([[21.0, -21.0, 2.0]])
    algo = _make_uninitialized_algo(raw_actions)
    obs_dict = {"actor_obs": torch.zeros((1, 2))}

    inference_actions = algo.get_inference_policy()(obs_dict)
    exported_actions = algo.actor_onnx_wrapper(obs_dict["actor_obs"])

    expected = torch.tensor([[20.0, -20.0, 2.0]])
    assert torch.equal(inference_actions, expected)
    assert torch.equal(exported_actions, expected)


def test_eval_env_step_reapplies_bound_after_callback_action_changes() -> None:
    obs_dict = {
        "actor_obs": torch.zeros((1, 2)),
        "critic_obs": torch.zeros((1, 2)),
    }
    algo = _make_uninitialized_algo(torch.zeros((1, 3)))
    algo.env = _FakeEnv(obs_dict)
    actor_state = {"actions": torch.tensor([[40.0, -22.0, 1.0]])}

    result = algo.env_step(actor_state)

    expected = torch.tensor([[20.0, -20.0, 1.0]])
    assert torch.equal(algo.env.executed_actions, expected)
    assert torch.equal(result["actions"], expected)


def test_final_ppo_updates_critic_normalizer_only_on_current_rollout_obs() -> None:
    obs_dict = {
        "actor_obs": torch.zeros((2, 2)),
        "critic_obs": torch.ones((2, 2)),
    }
    algo = _make_uninitialized_algo(torch.zeros((2, 3)))
    algo.env = _FakeEnv(obs_dict)
    algo.critic_obs_normalizer = _FakeNormalizer()

    algo._rollout_step(obs_dict)

    assert algo.critic_obs_normalizer.update_flags == [True, False]
    assert len(algo.critic.seen_obs) == 2
    torch.testing.assert_close(algo.critic.seen_obs[0], torch.full((2, 2), 6.0))
    torch.testing.assert_close(algo.critic.seen_obs[1], torch.full((2, 2), 6.0))
