from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("trimesh")

import holosoma.managers.observation.terms.wbt as wbt_obs
from holosoma.config_types.env import resolve_observation_term_overrides
from holosoma.config_values.wbt.r1.experiment import r1_fastsac, r1_student
from holosoma.config_values.wbt.r1.observation import (
    r1_26dof_fastsac_observation,
    r1_student_direct_ir_actor_obs,
    r1_student_privileged_critic_obs,
)


def test_object_randomization_privileged_selects_active_object_and_spawn_offset(monkeypatch):
    motion_command = SimpleNamespace(
        object_key_to_id={"box": 0, "chair": 1},
        object_type_ids=torch.tensor([0, 1, 0], dtype=torch.long),
        object_pos_reward_offset=torch.tensor(
            [
                [0.01, 0.02, 0.0],
                [-0.03, 0.04, 0.0],
                [0.05, -0.01, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        object_randomization_privileged_by_key={
            "box": torch.tensor(
                [
                    [0.3, 0.2, 0.1, 1.0, 1.1, 1.2, 1.3, 0.9, 0.8, 0.7],
                    [9.0] * 10,
                    [0.4, 0.25, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            "chair": torch.tensor(
                [
                    [8.0] * 10,
                    [0.8, 0.6, 0.2, 3.0, 0.5, 1.0, 1.5, 1.0, 1.0, 1.0],
                    [7.0] * 10,
                ],
                dtype=torch.float32,
            ),
        },
    )
    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    privileged = wbt_obs.object_randomization_privileged(env)

    assert privileged.shape == (3, 13)
    torch.testing.assert_close(privileged[0, :10], env.object_randomization_privileged_by_key["box"][0])
    torch.testing.assert_close(privileged[1, :10], env.object_randomization_privileged_by_key["chair"][1])
    torch.testing.assert_close(privileged[2, :10], env.object_randomization_privileged_by_key["box"][2])
    torch.testing.assert_close(privileged[:, 10:13], motion_command.object_pos_reward_offset)


def test_task_index_one_hot_uses_configured_motion_stem_order(tmp_path, monkeypatch):
    object_parm = tmp_path / "objects_parm.yaml"
    object_parm.write_text(
        "task_index:\n"
        "  - task_b\n"
        "  - task_a\n"
        "  - unused_task\n",
        encoding="utf-8",
    )
    motion_command = SimpleNamespace(
        motion=SimpleNamespace(clip_files=["motions/task_a.npz", "motions/task_b.npz"]),
        clip_ids=torch.tensor([0, 1, 0], dtype=torch.long),
    )
    env = SimpleNamespace(
        device=torch.device("cpu"),
        robot_config=SimpleNamespace(object=SimpleNamespace(object_parm=str(object_parm))),
    )
    monkeypatch.setattr(wbt_obs, "_get_motion_command_and_assert_type", lambda _: motion_command)

    actual = wbt_obs.task_index_one_hot(env)

    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_r1_student_observations_include_task_and_object_identity_once():
    identity_terms = {"task_index_one_hot", "obj_type_one_hot"}

    assert identity_terms <= set(r1_student_direct_ir_actor_obs.terms)
    assert identity_terms <= set(r1_student_privileged_critic_obs.terms)

    latent_actor_terms = r1_26dof_fastsac_observation.groups["actor_obs"].terms
    assert identity_terms <= set(latent_actor_terms)
    assert list(latent_actor_terms).count("task_index_one_hot") == 1
    assert list(latent_actor_terms).count("obj_type_one_hot") == 1


def test_r1_student_uses_slow_teacher_floor_and_robust_actor_replay():
    config = r1_student.algo.config

    assert config.teacher_mixture_start == 1.0
    assert config.teacher_mixture_end == 0.2
    assert config.teacher_mixture_decay_iterations == 5000
    assert config.num_updates_per_iteration == 32
    assert config.save_interval == 100
    assert config.teacher_anchor_capacity == 262144
    assert config.teacher_anchor_sampling_ratio == 0.5
    assert config.teacher_action_outlier_threshold == 20.0
    assert config.actor_huber_delta == 1.0
    assert config.critic_obs_normalization


def test_r1_fastsac_without_ae_uses_same_direct_ir_actor_layout_as_student():
    student_resolved = resolve_observation_term_overrides(r1_student)
    resolved = resolve_observation_term_overrides(r1_fastsac)

    assert resolved.algo.config.actor_obs_keys == ["actor_obs"]
    assert resolved.observation.groups["actor_obs"] == student_resolved.observation.groups["actor_obs"]
    assert resolved.observation.groups["critic_obs"] == student_resolved.observation.groups["critic_obs"]
    assert "ae_latent" not in resolved.observation.groups
    assert not resolved.simulator.config.enable_robot_depth_camera
