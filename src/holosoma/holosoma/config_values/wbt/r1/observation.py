"""Whole Body Tracking observation presets for the R1 robot."""

from dataclasses import replace

from holosoma.config_types.observation import ObsTermCfg
from holosoma.config_values.wbt.g1.observation import (
    g1_29dof_wbt_observation_w_object_multi_teacher,
)
from holosoma.config_values.wbt.r1.contact import (
    R1_OBJECT_CONTACT_BODY_NAMES,
    R1_OBJECT_CONTACT_DISTANCE_CLIP,
)

object_randomization_privileged_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:object_randomization_privileged",
    scale=1.0,
    noise=0.0,
)

motion_command_future_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:motion_command_future",
    params={
        "offsets": [1, 2, 3],
    },
    scale=1.0,
    noise=0.0,
)

object_distance_current_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:ObjectDistanceCurrent",
    params={
        "body_names": R1_OBJECT_CONTACT_BODY_NAMES,
        "distance_clip": R1_OBJECT_CONTACT_DISTANCE_CLIP,
    },
    scale=1.0,
    noise=0.0,
    clip=(0.0, 1.0),
)

R1_TEACHER_HISTORY_LENGTH = 3
R1_TEACHER_HISTORY_TERMS = {
    "actions",
    "base_ang_vel",
    "base_lin_vel",
    "dof_pos",
    "dof_vel",
    "obj_lin_vel_b",
    "obj_ori_b",
    "obj_pos_b",
}


def _with_selective_history(terms: dict[str, ObsTermCfg]) -> dict[str, ObsTermCfg]:
    return {
        name: replace(term, history_length=R1_TEACHER_HISTORY_LENGTH) if name in R1_TEACHER_HISTORY_TERMS else term
        for name, term in terms.items()
    }


actor_terms = _with_selective_history(
    {
        **g1_29dof_wbt_observation_w_object_multi_teacher.groups["actor_obs"].terms,
        "motion_command_future": motion_command_future_term,
        "object_distance_current": object_distance_current_term,
        "object_randomization_privileged": object_randomization_privileged_term,
    }
)

critic_terms = _with_selective_history(
    {
        **g1_29dof_wbt_observation_w_object_multi_teacher.groups["critic_obs"].terms,
        "motion_command_future": motion_command_future_term,
        "object_distance_current": object_distance_current_term,
        "object_randomization_privileged": object_randomization_privileged_term,
    }
)

r1_26dof_wbt_observation_w_object_multi_teacher = replace(
    g1_29dof_wbt_observation_w_object_multi_teacher,
    groups={
        **g1_29dof_wbt_observation_w_object_multi_teacher.groups,
        "actor_obs": replace(
            g1_29dof_wbt_observation_w_object_multi_teacher.groups["actor_obs"],
            history_length=1,
            terms=actor_terms,
        ),
        "critic_obs": replace(
            g1_29dof_wbt_observation_w_object_multi_teacher.groups["critic_obs"],
            history_length=1,
            terms=critic_terms,
        ),
    },
)

__all__ = ["r1_26dof_wbt_observation_w_object_multi_teacher"]
