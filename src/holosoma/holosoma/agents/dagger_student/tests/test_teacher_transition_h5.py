from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from holosoma.agents.dagger_student.teacher_transition_h5 import (
    TEACHER_TRANSITION_FIELDS,
    TeacherTransitionH5Writer,
)
from holosoma.agents.dagger_student.dagger_student import DaggerStudent
from holosoma.agents.fast_sac.fast_sac_agent import TeacherReplayBuffer


def _batch(start: int, count: int) -> dict[str, np.ndarray]:
    ids = np.arange(start, start + count, dtype=np.float32)
    return {
        "observations": np.stack((ids, ids + 0.1), axis=1),
        "critic_observations": np.stack((ids, ids + 0.2, ids + 0.3), axis=1),
        "actions": np.stack((ids + 10.0, ids + 20.0), axis=1),
        "rewards": ids + 30.0,
        "dones": (ids.astype(np.int64) % 3) == 0,
        "truncations": (ids.astype(np.int64) % 5) == 0,
        "next_observations": np.stack((ids + 1.0, ids + 1.1), axis=1),
        "next_critic_observations": np.stack((ids + 1.0, ids + 1.2, ids + 1.3), axis=1),
    }


def _writer(path, *, mode: str = "direct_ir", capacity: int = 5) -> TeacherTransitionH5Writer:
    return TeacherTransitionH5Writer(
        path,
        max_transitions=capacity,
        actor_obs_dim=2,
        critic_obs_dim=3,
        action_dim=2,
        actor_obs_keys=["actor_obs"],
        critic_obs_keys=["critic_obs"],
        observation_mode=mode,
        seed=17,
    )


def test_writer_is_bounded_and_directly_loadable_by_fastsac(tmp_path) -> None:
    path = tmp_path / "teacher_buffer.h5"
    writer = _writer(path)
    first = writer.append(_batch(0, 3))
    second = writer.append(_batch(3, 7))

    assert first.accepted == 3
    assert first.seen == 3
    assert second.seen == 10
    assert second.saved == 5

    with h5py.File(path, "r") as source:
        assert source.attrs["format"] == "holosoma_fastsac_teacher_buffer"
        assert int(source.attrs["format_version"]) == 2
        assert json.loads(source.attrs["actor_obs_keys"]) == ["actor_obs"]
        assert int(source.attrs["num_seen_transitions"]) == 10
        assert int(source.attrs["num_transitions"]) == 5
        for name in TEACHER_TRANSITION_FIELDS:
            assert source[name].shape[0] == 5

        # Every reservoir row must remain a coherent transition even when two
        # incoming rows selected the same replacement slot.
        ids = source["observations"][:, 0]
        np.testing.assert_allclose(source["critic_observations"][:, 0], ids)
        np.testing.assert_allclose(source["actions"][:, 0], ids + 10.0)
        np.testing.assert_allclose(source["rewards"][:], ids + 30.0)
        np.testing.assert_allclose(source["next_observations"][:, 0], ids + 1.0)

    replay = TeacherReplayBuffer(
        str(path),
        actor_obs_dim=2,
        critic_obs_dim=3,
        action_dim=2,
        actor_obs_keys=["actor_obs"],
        critic_obs_keys=["critic_obs"],
        observation_mode="direct_ir",
    )
    assert replay.count == 5
    sample = replay.sample(3, "cpu")
    assert tuple(sample["observations"].shape) == (3, 2)
    assert tuple(sample["critic_observations"].shape) == (3, 3)


def test_loader_rejects_same_shape_but_different_observation_mode(tmp_path) -> None:
    path = tmp_path / "teacher_buffer.h5"
    writer = _writer(path, mode="direct_ir")
    writer.append(_batch(0, 2))

    with pytest.raises(ValueError, match="observation mode"):
        TeacherReplayBuffer(
            str(path),
            actor_obs_dim=2,
            critic_obs_dim=3,
            action_dim=2,
            actor_obs_keys=["actor_obs"],
            critic_obs_keys=["critic_obs"],
            observation_mode="di_pro_latent",
        )


def test_pending_journal_is_replayed_on_reopen(tmp_path) -> None:
    path = tmp_path / "teacher_buffer.h5"
    writer = _writer(path)
    writer.append(_batch(0, 2))
    pending_row = _batch(9, 1)

    # Model a process interruption after the durable journal was written but
    # before its row/count update reached the root datasets.
    with h5py.File(path, "a") as output:
        pending = output.create_group("__pending_reservoir_update__")
        pending.attrs["new_seen"] = 3
        pending.attrs["new_saved"] = 3
        pending.attrs["rng_state_after"] = output.attrs["reservoir_rng_state"]
        pending.create_dataset("slots", data=np.asarray([2], dtype=np.int64))
        for name in TEACHER_TRANSITION_FIELDS:
            pending.create_dataset(name, data=pending_row[name])
        output.flush()

    recovered = _writer(path)
    assert recovered.seen == 3
    assert recovered.saved == 3
    with h5py.File(path, "r") as source:
        assert "__pending_reservoir_update__" not in source
        assert source["observations"][2, 0] == pytest.approx(9.0)
        assert source["actions"][2, 0] == pytest.approx(19.0)


def test_auto_pre_sampling_targets_one_bounded_buffer_over_mixture_schedule() -> None:
    agent = DaggerStudent.__new__(DaggerStudent)
    agent.env = SimpleNamespace(num_envs=256)
    agent.config = SimpleNamespace(
        teacher_buffer_sampling_probability=None,
        teacher_buffer_max_transitions=524_288,
        num_learning_iterations=50_000,
        num_steps_per_env=32,
        teacher_mixture_decay_iterations=1_000,
        teacher_mixture_start=1.0,
        teacher_mixture_end=0.0,
    )

    probability = agent._resolve_teacher_buffer_sampling_probability()

    expected_teacher_rows = 256 * 32 * 500.5
    assert probability == pytest.approx(524_288 / expected_teacher_rows)


def test_reopen_rejects_changed_float_sampling_metadata(tmp_path) -> None:
    path = tmp_path / "teacher_buffer.h5"
    common = dict(
        max_transitions=5,
        actor_obs_dim=2,
        critic_obs_dim=3,
        action_dim=2,
        actor_obs_keys=["actor_obs"],
        critic_obs_keys=["critic_obs"],
        observation_mode="direct_ir",
        seed=17,
    )
    TeacherTransitionH5Writer(
        path,
        metadata={"teacher_row_sampling_probability": 0.1},
        **common,
    )

    with pytest.raises(ValueError, match="source_teacher_row_sampling_probability"):
        TeacherTransitionH5Writer(
            path,
            metadata={"teacher_row_sampling_probability": 0.2},
            **common,
        )
