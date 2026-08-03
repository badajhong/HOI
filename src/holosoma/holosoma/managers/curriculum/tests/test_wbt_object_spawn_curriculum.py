from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
from holosoma.managers.curriculum.terms.wbt import ObjectSpawnSuccessCurriculum


class _CommandManager:
    def __init__(self):
        self.command = SimpleNamespace(
            motion_cfg=MotionConfig(
                body_name_ref=["pelvis"],
                body_names_to_track=["pelvis"],
                noise_to_initial_pose=NoiseToInitialPoseConfig(object_sector_radius=[0.0, 0.5]),
            )
        )

    def get_state(self, name):
        return self.command if name == "motion_command" else None


def _make_term(*, promote_threshold=0.75, demote_threshold=0.4):
    env = SimpleNamespace(
        device="cpu",
        command_manager=_CommandManager(),
        termination_manager=SimpleNamespace(last_term_results={}),
        log_dict={},
    )
    cfg = SimpleNamespace(
        params={
            "radius_steps": (0.0, 0.1, 0.25),
            "initial_level": 0,
            "ema_alpha": 1.0,
            "promote_threshold": promote_threshold,
            "demote_threshold": demote_threshold,
            "promote_windows": 2,
            "demote_windows": 2,
            "window_episodes": 4,
        }
    )
    term = ObjectSpawnSuccessCurriculum(cfg, env)
    term.setup()
    return term, env


def _reset_window(term, env, successes: int):
    env.termination_manager.last_term_results = {
        "motion_ends": torch.tensor([i < successes for i in range(4)]),
        "object_robot_distance_xy": torch.tensor([i >= successes for i in range(4)]),
        "timeout": torch.zeros(4, dtype=torch.bool),
    }
    term.reset(torch.arange(4))


def test_promotes_and_demotes_spawn_radius_after_consecutive_windows():
    term, env = _make_term()
    assert env.command_manager.command.motion_cfg.noise_to_initial_pose.object_sector_radius == [0.0, 0.0]

    _reset_window(term, env, successes=4)
    _reset_window(term, env, successes=4)
    assert term.level == 1
    assert env.command_manager.command.motion_cfg.noise_to_initial_pose.object_sector_radius == [0.0, 0.1]

    _reset_window(term, env, successes=0)
    _reset_window(term, env, successes=0)
    assert term.level == 0


def test_bootstrap_reset_is_not_counted_as_failure_and_state_restores():
    term, env = _make_term()
    env.termination_manager.last_term_results = {
        "motion_ends": torch.zeros(4, dtype=torch.bool),
        "object_robot_distance_xy": torch.zeros(4, dtype=torch.bool),
    }
    term.reset(torch.arange(4))
    assert term._window_episodes == 0

    _reset_window(term, env, successes=4)
    state = term.state_dict()
    restored, restored_env = _make_term()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored_env.command_manager.command.motion_cfg.noise_to_initial_pose.object_sector_radius == [0.0, 0.0]
