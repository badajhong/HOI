"""Whole Body Tracking curriculum presets for the R1 robot."""

from dataclasses import replace

from holosoma.config_types.curriculum import CurriculumTermCfg
from holosoma.config_values.wbt.g1.curriculum import (
    g1_29dof_wbt_curriculum as r1_26dof_wbt_curriculum,
)

r1_26dof_final_ppo_curriculum = replace(
    r1_26dof_wbt_curriculum,
    setup_terms={
        **r1_26dof_wbt_curriculum.setup_terms,
        "object_spawn_success_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.wbt:ObjectSpawnSuccessCurriculum",
            params={
                "radius_steps": (0.0, 0.1, 0.25, 0.4, 0.5),
                "initial_level": 0,
                "ema_alpha": 0.05,
                "promote_threshold": 0.75,
                "demote_threshold": 0.40,
                "promote_windows": 5,
                "demote_windows": 3,
                "window_episodes": 1024,
            },
        ),
    },
)

__all__ = ["r1_26dof_final_ppo_curriculum", "r1_26dof_wbt_curriculum"]
