from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from holosoma.agents.dagger_student.dagger_student import DaggerStudent


class _Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def act_inference(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.linear(obs_dict["actor_obs"])


class _Value(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.last_obs: torch.Tensor | None = None

    def evaluate(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        self.last_obs = obs_dict["critic_obs"].detach().clone()
        return self.linear(obs_dict["critic_obs"])


class _DistributionalQ(nn.Module):
    def __init__(self, num_q: int = 2, num_atoms: int = 3) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_q, num_atoms))

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        del actions
        return self.logits[:, None, :].expand(-1, obs.shape[0], -1)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        del actions, rewards, bootstrap, discount
        target = torch.zeros(
            self.logits.shape[0], obs.shape[0], self.logits.shape[1], device=obs.device
        )
        target[..., 0] = 1.0
        return target


class _IdentityNormalizer(nn.Module):
    def forward(self, obs: torch.Tensor, update: bool = False) -> torch.Tensor:
        del update
        return obs


class _TrackingNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.update_flags: list[bool] = []

    def forward(self, obs: torch.Tensor, update: bool = False) -> torch.Tensor:
        self.update_flags.append(update)
        return obs + 10.0


class _FixedBuffer:
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch
        self.capacity = batch["obs"].shape[0]

    def __len__(self) -> int:
        return self.batch["obs"].shape[0]

    def sample(self, batch_size: int, device: str | torch.device) -> dict[str, torch.Tensor]:
        assert batch_size == len(self)
        return {key: value.clone().to(device) for key, value in self.batch.items()}

    def valid_actor_indices(self) -> torch.Tensor:
        return torch.nonzero(
            self.batch["teacher_action_valid"][:, 0] > 0.5,
            as_tuple=False,
        ).squeeze(-1)

    def sample_actor(
        self,
        batch_size: int,
        valid_indices: torch.Tensor,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        repeats = (batch_size + valid_indices.numel() - 1) // valid_indices.numel()
        indices = valid_indices.repeat(repeats)[:batch_size]
        return {
            "obs": self.batch["obs"].index_select(0, indices).to(device),
            "teacher_actions": self.batch["teacher_actions"].index_select(0, indices).to(device),
        }


def _make_agent(is_student_action: torch.Tensor) -> DaggerStudent:
    torch.manual_seed(7)
    agent = DaggerStudent.__new__(DaggerStudent)
    agent.device = "cpu"
    agent.is_multi_gpu = False
    agent.gpu_world_size = 1
    agent.config = SimpleNamespace(
        num_updates_per_iteration=1,
        batch_size=4,
        student_action_clip=20.0,
        gamma=0.99,
        value_loss_coef=1.0,
        q_loss_coef=1.0,
        max_grad_norm=0.0,
        critic_obs_normalization=False,
        teacher_anchor_sampling_ratio=0.5,
        actor_huber_delta=1.0,
    )
    agent.value_target_tau = 0.005
    agent.q_target_tau = 0.05
    agent.actor = _Actor()
    agent.value_critic = _Value()
    agent.target_value_critic = copy.deepcopy(agent.value_critic)
    agent.qnet = _DistributionalQ()
    agent.qnet_target = copy.deepcopy(agent.qnet)
    agent.critic_obs_normalizer = _IdentityNormalizer()
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=1e-3)
    agent.value_optimizer = torch.optim.Adam(agent.value_critic.parameters(), lr=1e-3)
    agent.q_optimizer = torch.optim.Adam(agent.qnet.parameters(), lr=1e-3)

    obs = torch.tensor([[0.0, 0.5], [1.0, -0.5], [0.25, 0.75], [-1.0, 0.25]])
    agent.buffer = _FixedBuffer(
        {
            "obs": obs,
            "teacher_actions": torch.tensor([[0.2], [-0.3], [0.4], [0.1]]),
            "critic_obs": obs + 0.1,
            "actions": torch.tensor([[0.1], [-0.2], [0.3], [0.0]]),
            "is_student_action": is_student_action.float().view(-1, 1),
            "teacher_action_valid": torch.ones(4, 1),
            "rewards": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            "next_obs": obs + 0.2,
            "next_critic_obs": obs + 0.3,
            "terminals": torch.tensor([[0.0], [0.0], [1.0], [0.0]]),
        }
    )
    return agent


def test_value_update_is_skipped_for_pure_teacher_batch_while_q_updates() -> None:
    agent = _make_agent(torch.zeros(4, dtype=torch.bool))
    value_before = [parameter.detach().clone() for parameter in agent.value_critic.parameters()]
    target_value_before = [parameter.detach().clone() for parameter in agent.target_value_critic.parameters()]
    q_before = [parameter.detach().clone() for parameter in agent.qnet.parameters()]

    metrics = agent._training_step()

    assert metrics["student_transition_ratio"] == 0.0
    assert metrics["value_loss"] == 0.0
    for before, after in zip(value_before, agent.value_critic.parameters()):
        torch.testing.assert_close(after, before)
    for before, after in zip(target_value_before, agent.target_value_critic.parameters()):
        torch.testing.assert_close(after, before)
    assert any(not torch.equal(before, after) for before, after in zip(q_before, agent.qnet.parameters()))


def test_value_update_uses_only_student_executed_rows_and_logs_td_stats() -> None:
    agent = _make_agent(torch.tensor([True, False, True, False]))
    value_before = [parameter.detach().clone() for parameter in agent.value_critic.parameters()]

    metrics = agent._training_step()

    assert metrics["student_transition_ratio"] == pytest.approx(0.5)
    assert metrics["value_loss"] > 0.0
    assert metrics["value_target_std"] >= 0.0
    assert metrics["value_td_abs_mean"] > 0.0
    assert all(torch.isfinite(torch.tensor(metrics[key])) for key in (
        "value_mean",
        "value_target_mean",
        "value_target_std",
        "value_td_abs_mean",
    ))
    assert any(not torch.equal(before, after) for before, after in zip(value_before, agent.value_critic.parameters()))


def test_value_and_target_value_use_frozen_critic_normalization() -> None:
    agent = _make_agent(torch.tensor([True, False, True, False]))
    agent.config.critic_obs_normalization = True
    normalizer = _TrackingNormalizer()
    agent.critic_obs_normalizer = normalizer

    agent._training_step()

    assert len(normalizer.update_flags) == 4
    assert normalizer.update_flags == [False, False, False, False]
    expected_current = agent.buffer.batch["critic_obs"][[0, 2]] + 10.0
    expected_next = agent.buffer.batch["next_critic_obs"][[0, 2]] + 10.0
    torch.testing.assert_close(agent.value_critic.last_obs, expected_current)
    torch.testing.assert_close(agent.target_value_critic.last_obs, expected_next)
