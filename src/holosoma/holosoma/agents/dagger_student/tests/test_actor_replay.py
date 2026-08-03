from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.agents.dagger_student.dagger_student import (
    DaggerStudent,
    FixedActorAnchorBuffer,
    StackDaggerBuffer,
    _valid_teacher_action_rows,
)


def _add_recent(buffer: StackDaggerBuffer) -> None:
    obs = torch.tensor([[1.0, 1.0], [9.0, 9.0]])
    buffer.add(
        obs=obs,
        teacher_actions=torch.tensor([[2.0], [20.0]]),
        critic_obs=obs,
        executed_actions=torch.zeros(2, 1),
        rewards=torch.zeros(2),
        next_obs=obs,
        next_critic_obs=obs,
        terminals=torch.zeros(2, dtype=torch.bool),
        is_student_action=torch.ones(2, dtype=torch.bool),
        teacher_action_valid=torch.tensor([True, False]),
    )


def _make_agent() -> DaggerStudent:
    agent = DaggerStudent.__new__(DaggerStudent)
    agent.device = "cpu"
    agent.config = SimpleNamespace(teacher_anchor_sampling_ratio=0.5)
    agent.buffer = StackDaggerBuffer(
        8,
        obs_dim=2,
        critic_obs_dim=2,
        action_dim=1,
        storage_device="cpu",
    )
    _add_recent(agent.buffer)
    agent.teacher_anchor_buffer = FixedActorAnchorBuffer(
        4,
        obs_dim=2,
        action_dim=1,
        storage_device="cpu",
    )
    agent.teacher_anchor_buffer.add(
        torch.full((2, 2), -1.0),
        torch.full((2, 1), -2.0),
    )
    return agent


def test_actor_sampling_uses_half_anchor_half_valid_recent() -> None:
    agent = _make_agent()

    batch, anchor_ratio = agent._sample_actor_supervision_batch(
        10,
        agent.buffer.valid_actor_indices(),
    )

    assert batch is not None
    assert anchor_ratio == 0.5
    assert torch.all(batch["obs"][:5] == -1.0)
    assert torch.all(batch["teacher_actions"][:5] == -2.0)
    assert torch.all(batch["obs"][5:] == 1.0)
    assert torch.all(batch["teacher_actions"][5:] == 2.0)


def test_actor_sampling_falls_back_to_valid_recent_when_anchor_empty() -> None:
    agent = _make_agent()
    agent.teacher_anchor_buffer = FixedActorAnchorBuffer(
        4,
        obs_dim=2,
        action_dim=1,
        storage_device="cpu",
    )

    batch, anchor_ratio = agent._sample_actor_supervision_batch(
        7,
        agent.buffer.valid_actor_indices(),
    )

    assert batch is not None
    assert anchor_ratio == 0.0
    assert torch.all(batch["obs"] == 1.0)
    assert torch.all(batch["teacher_actions"] == 2.0)


def test_actor_sampling_falls_back_to_anchor_without_valid_recent_labels() -> None:
    agent = _make_agent()
    agent.buffer.teacher_action_valid[: len(agent.buffer)].zero_()

    batch, anchor_ratio = agent._sample_actor_supervision_batch(
        7,
        agent.buffer.valid_actor_indices(),
    )

    assert batch is not None
    assert anchor_ratio == 1.0
    assert torch.all(batch["obs"] == -1.0)
    assert torch.all(batch["teacher_actions"] == -2.0)


def test_teacher_action_row_validity_rejects_outliers_and_nonfinite_values() -> None:
    actions = torch.tensor(
        [
            [20.0, -20.0],
            [20.0001, 0.0],
            [float("nan"), 0.0],
            [float("inf"), 0.0],
        ]
    )

    torch.testing.assert_close(
        _valid_teacher_action_rows(actions, 20.0),
        torch.tensor([True, False, False, False]),
    )
