"""Whole Body Tracking observation presets for the R1 robot."""

from dataclasses import replace

from holosoma.config_types.observation import ObsTermCfg
from holosoma.config_values.wbt.g1.observation import (
    g1_29dof_wbt_observation_w_object_multi_teacher,
)
from holosoma.config_values.wbt.r1.contact import (
    R1_OBJECT_CONTACT_BODY_NAMES,
    R1_OBJECT_CONTACT_DISTANCE_CLIP,
    R1_OBJECT_CONTACT_THRESHOLD,
)

object_randomization_privileged_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:object_randomization_privileged",
    scale=1.0,
    noise=0.0,
)

object_contact_current_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:ObjectContactCurrent",
    params={
        "body_names": R1_OBJECT_CONTACT_BODY_NAMES,
        "threshold": R1_OBJECT_CONTACT_THRESHOLD,
        "distance_clip": R1_OBJECT_CONTACT_DISTANCE_CLIP,
        "include_soft_contact": True,
        "include_distance": True,
    },
    scale=1.0,
    noise=0.0,
    clip=(0.0, 1.0),
)

r1_26dof_wbt_observation_w_object_multi_teacher = replace(
    g1_29dof_wbt_observation_w_object_multi_teacher,
    groups={
        **g1_29dof_wbt_observation_w_object_multi_teacher.groups,
        "actor_obs": replace(
            g1_29dof_wbt_observation_w_object_multi_teacher.groups["actor_obs"],
            terms={
                **g1_29dof_wbt_observation_w_object_multi_teacher.groups["actor_obs"].terms,
                "object_contact_current": object_contact_current_term,
                "object_randomization_privileged": object_randomization_privileged_term,
            },
        ),
        "critic_obs": replace(
            g1_29dof_wbt_observation_w_object_multi_teacher.groups["critic_obs"],
            terms={
                **g1_29dof_wbt_observation_w_object_multi_teacher.groups["critic_obs"].terms,
                "object_contact_current": object_contact_current_term,
                "object_randomization_privileged": object_randomization_privileged_term,
            },
        ),
    },
)

__all__ = ["r1_26dof_wbt_observation_w_object_multi_teacher"]
