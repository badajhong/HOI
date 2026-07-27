from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("trimesh")

import holosoma.managers.observation.terms.wbt as wbt_obs


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
