"""Whole Body Tracking randomization presets for the R1 robot."""

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.config_values.wbt.g1.randomization import (
    g1_29dof_wbt_randomization_w_object as r1_26dof_wbt_randomization_w_object,
)

r1_26dof_fastsac_randomization = RandomizationManagerCfg(
    setup_terms={
        **r1_26dof_wbt_randomization_w_object.setup_terms,
        "randomize_object_scale_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_object_scale_startup",
            params={
                # Volume ratios; uniform XYZ scale is their cube root.
                "scale_values": (0.6, 0.8, 1.0, 1.2, 1.4),
                "object_height": 0.0,
                "enabled": True,
            },
        ),
    },
    reset_terms={**r1_26dof_wbt_randomization_w_object.reset_terms},
    step_terms={**r1_26dof_wbt_randomization_w_object.step_terms},
)

__all__ = ["r1_26dof_fastsac_randomization", "r1_26dof_wbt_randomization_w_object"]
