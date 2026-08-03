"""Whole-body interaction curricula."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from loguru import logger

from holosoma.managers.curriculum.base import CurriculumTermBase


class ObjectSpawnSuccessCurriculum(CurriculumTermBase):
    """Expand object spawn radius when motion-end success remains high."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.radius_steps = tuple(float(value) for value in params.get("radius_steps", (0.0, 0.1, 0.25, 0.4, 0.5)))
        if not self.radius_steps or any(value < 0.0 for value in self.radius_steps):
            raise ValueError(f"radius_steps must contain non-negative values, got {self.radius_steps}.")
        if any(right < left for left, right in zip(self.radius_steps, self.radius_steps[1:])):
            raise ValueError(f"radius_steps must be non-decreasing, got {self.radius_steps}.")

        self.ema_alpha = float(params.get("ema_alpha", 0.05))
        self.promote_threshold = float(params.get("promote_threshold", 0.75))
        self.demote_threshold = float(params.get("demote_threshold", 0.40))
        self.promote_windows = max(int(params.get("promote_windows", 5)), 1)
        self.demote_windows = max(int(params.get("demote_windows", 3)), 1)
        self.window_episodes = max(int(params.get("window_episodes", 1024)), 1)
        self.level = min(max(int(params.get("initial_level", 0)), 0), len(self.radius_steps) - 1)

        self.success_ema = 0.0
        self._ema_initialized = False
        self._window_successes = 0
        self._window_episodes = 0
        self._promote_count = 0
        self._demote_count = 0

    def setup(self) -> None:
        self._apply_level()
        self._publish_metrics()

    def reset(self, env_ids) -> None:
        if env_ids is None or self.env.termination_manager is None:
            return
        env_ids = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long).view(-1)
        if env_ids.numel() == 0:
            return

        # Ignore reset_all/bootstrap resets which did not satisfy any
        # termination condition. This does not depend on the episode-length
        # tracker, which may already have consumed its pending values.
        term_results = self.env.termination_manager.last_term_results
        valid = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.env.device)
        for result in term_results.values():
            valid |= result[env_ids]
        if not valid.any():
            return
        valid_ids = env_ids[valid]
        motion_ends = term_results.get("motion_ends")
        if motion_ends is None:
            raise RuntimeError("Object spawn curriculum requires the 'motion_ends' termination term.")

        self._window_successes += int(motion_ends[valid_ids].sum().item())
        self._window_episodes += int(valid_ids.numel())
        if self._window_episodes < self.window_episodes:
            self._publish_metrics()
            return

        window_rate = self._window_successes / max(self._window_episodes, 1)
        if self._ema_initialized:
            self.success_ema = (1.0 - self.ema_alpha) * self.success_ema + self.ema_alpha * window_rate
        else:
            self.success_ema = window_rate
            self._ema_initialized = True
        self._window_successes = 0
        self._window_episodes = 0

        if self.success_ema >= self.promote_threshold and self.level < len(self.radius_steps) - 1:
            self._promote_count += 1
            self._demote_count = 0
        elif self.success_ema <= self.demote_threshold and self.level > 0:
            self._demote_count += 1
            self._promote_count = 0
        else:
            self._promote_count = 0
            self._demote_count = 0

        previous_level = self.level
        if self._promote_count >= self.promote_windows:
            self.level += 1
            self._promote_count = 0
        elif self._demote_count >= self.demote_windows:
            self.level -= 1
            self._demote_count = 0

        if self.level != previous_level:
            self._apply_level()
            logger.info(
                f"Object spawn curriculum changed level {previous_level} -> {self.level}: "
                f"radius_max={self.radius_steps[self.level]:.2f} m, success_ema={self.success_ema:.3f}."
            )
        self._publish_metrics(window_rate=window_rate)

    def step(self) -> None:
        return

    def _apply_level(self) -> None:
        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is None:
            raise RuntimeError("Object spawn curriculum requires motion_command.")
        noise = replace(
            motion_command.motion_cfg.noise_to_initial_pose,
            object_sector_radius=[0.0, self.radius_steps[self.level]],
        )
        motion_command.motion_cfg = replace(motion_command.motion_cfg, noise_to_initial_pose=noise)

    def _publish_metrics(self, *, window_rate: float | None = None) -> None:
        if not hasattr(self.env, "log_dict"):
            return
        device = self.env.device
        self.env.log_dict["Curriculum/object_spawn_level"] = torch.tensor(float(self.level), device=device)
        self.env.log_dict["Curriculum/object_spawn_radius_max_m"] = torch.tensor(
            self.radius_steps[self.level], device=device
        )
        self.env.log_dict["Curriculum/success_ema"] = torch.tensor(self.success_ema, device=device)
        if window_rate is not None:
            self.env.log_dict["Curriculum/window_success_rate"] = torch.tensor(window_rate, device=device)

    def state_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "success_ema": self.success_ema,
            "ema_initialized": self._ema_initialized,
            "window_successes": self._window_successes,
            "window_episodes": self._window_episodes,
            "promote_count": self._promote_count,
            "demote_count": self._demote_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.level = min(max(int(state.get("level", self.level)), 0), len(self.radius_steps) - 1)
        self.success_ema = float(state.get("success_ema", self.success_ema))
        self._ema_initialized = bool(state.get("ema_initialized", self._ema_initialized))
        self._window_successes = int(state.get("window_successes", 0))
        self._window_episodes = int(state.get("window_episodes", 0))
        self._promote_count = int(state.get("promote_count", 0))
        self._demote_count = int(state.get("demote_count", 0))
        self._apply_level()
        self._publish_metrics()


__all__ = ["ObjectSpawnSuccessCurriculum"]
