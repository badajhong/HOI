from __future__ import annotations

import io
import random

import numpy as np
import torch

from holosoma.agents.dagger_student.dagger_student import (
    FixedActorAnchorBuffer,
    StackDaggerBuffer,
    _capture_rng_state,
    _restore_rng_state,
)


def _add_range(buffer: StackDaggerBuffer, start: int, count: int) -> None:
    value = torch.arange(start, start + count, dtype=torch.float32).unsqueeze(1)
    buffer.add(
        obs=value.repeat(1, 2),
        teacher_actions=value.repeat(1, 2),
        critic_obs=value.repeat(1, 3),
        executed_actions=-value.repeat(1, 2),
        rewards=value.squeeze(1),
        next_obs=(value + 1).repeat(1, 2),
        next_critic_obs=(value + 1).repeat(1, 3),
        terminals=(value.squeeze(1) % 2).bool(),
        is_student_action=(value.squeeze(1) % 2 == 0),
        teacher_action_valid=torch.ones(count, dtype=torch.bool),
    )


def test_dagger_buffer_checkpoint_round_trip_preserves_ring_layout() -> None:
    source = StackDaggerBuffer(5, obs_dim=2, critic_obs_dim=3, action_dim=2, storage_device="cpu")
    _add_range(source, 0, 3)
    _add_range(source, 3, 4)
    assert source.write_idx == 2
    assert len(source) == 5

    serialized = io.BytesIO()
    torch.save(source.get_checkpoint_state(), serialized)
    serialized.seek(0)
    saved_state = torch.load(serialized, map_location="cpu", weights_only=False)

    restored = StackDaggerBuffer(5, obs_dim=2, critic_obs_dim=3, action_dim=2, storage_device="cpu")
    restored.load_checkpoint_state(saved_state)

    assert restored.size == source.size
    assert restored.write_idx == source.write_idx
    for name in StackDaggerBuffer._TENSOR_NAMES:
        torch.testing.assert_close(getattr(restored, name), getattr(source, name))

    # The restored cursor must also produce the same next overwrite.
    _add_range(source, 7, 1)
    _add_range(restored, 7, 1)
    assert restored.write_idx == source.write_idx
    for name in StackDaggerBuffer._TENSOR_NAMES:
        torch.testing.assert_close(getattr(restored, name), getattr(source, name))


def test_dagger_buffer_loads_v1_checkpoint_without_execution_source() -> None:
    source = StackDaggerBuffer(5, obs_dim=2, critic_obs_dim=3, action_dim=2, storage_device="cpu")
    _add_range(source, 0, 3)
    legacy_state = source.get_checkpoint_state()
    legacy_state["version"] = 1
    del legacy_state["tensors"]["is_student_action"]

    restored = StackDaggerBuffer(5, obs_dim=2, critic_obs_dim=3, action_dim=2, storage_device="cpu")
    restored.load_checkpoint_state(legacy_state)

    assert restored.size == 3
    torch.testing.assert_close(restored.is_student_action[:3], torch.zeros(3, 1))
    for name in StackDaggerBuffer._TENSOR_NAMES:
        if name != "is_student_action":
            torch.testing.assert_close(getattr(restored, name)[:3], getattr(source, name)[:3])


def test_dagger_buffer_loads_v2_checkpoint_without_raw_teacher_validity() -> None:
    source = StackDaggerBuffer(5, obs_dim=2, critic_obs_dim=3, action_dim=2, storage_device="cpu")
    _add_range(source, 0, 3)
    legacy_state = source.get_checkpoint_state()
    legacy_state["version"] = 2
    del legacy_state["tensors"]["teacher_action_valid"]

    restored = StackDaggerBuffer(5, obs_dim=2, critic_obs_dim=3, action_dim=2, storage_device="cpu")
    restored.load_checkpoint_state(legacy_state)

    assert restored.size == 3
    torch.testing.assert_close(restored.teacher_action_valid[:3], torch.zeros(3, 1))


def test_fixed_actor_anchor_never_overwrites_first_capacity() -> None:
    anchor = FixedActorAnchorBuffer(3, obs_dim=2, action_dim=1, storage_device="cpu")

    first = torch.tensor([[0.0], [1.0]])
    second = torch.tensor([[2.0], [3.0]])
    assert anchor.add(first.repeat(1, 2), first) == 2
    assert anchor.add(second.repeat(1, 2), second) == 1
    assert anchor.add(torch.full((1, 2), 9.0), torch.full((1, 1), 9.0)) == 0

    torch.testing.assert_close(anchor.obs, torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    torch.testing.assert_close(anchor.actions, torch.tensor([[0.0], [1.0], [2.0]]))


def test_fixed_actor_anchor_checkpoint_round_trip() -> None:
    source = FixedActorAnchorBuffer(4, obs_dim=2, action_dim=1, storage_device="cpu")
    source.add(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0], [6.0]]))

    restored = FixedActorAnchorBuffer(4, obs_dim=2, action_dim=1, storage_device="cpu")
    restored.load_checkpoint_state(source.get_checkpoint_state())

    assert len(restored) == 2
    torch.testing.assert_close(restored.obs[:2], source.obs[:2])
    torch.testing.assert_close(restored.actions[:2], source.actions[:2])


def test_rng_checkpoint_round_trip_restores_python_numpy_and_torch() -> None:
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    serialized = io.BytesIO()
    torch.save(_capture_rng_state(include_cuda=False), serialized)
    serialized.seek(0)
    state = torch.load(serialized, map_location="cpu", weights_only=False)

    expected_python = [random.random() for _ in range(4)]
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)

    # Advance each generator before restoring the captured state.
    _ = [random.random() for _ in range(7)]
    _ = np.random.random(7)
    _ = torch.rand(7)
    _restore_rng_state(state)

    assert [random.random() for _ in range(4)] == expected_python
    np.testing.assert_array_equal(np.random.random(4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0.0, atol=0.0)
