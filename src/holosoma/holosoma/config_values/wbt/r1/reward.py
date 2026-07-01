"""Whole Body Tracking reward presets for the R1 robot."""

from dataclasses import replace

from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_w_object_multi_teacher

r1_26dof_wbt_reward_w_object_multi_teacher = replace(
    g1_29dof_wbt_reward_w_object_multi_teacher,
    terms={
        **g1_29dof_wbt_reward_w_object_multi_teacher.terms,
        "object_global_ref_position_error_exp": replace(
            g1_29dof_wbt_reward_w_object_multi_teacher.terms["object_global_ref_position_error_exp"],
            params={
                **g1_29dof_wbt_reward_w_object_multi_teacher.terms[
                    "object_global_ref_position_error_exp"
                ].params,
                "sigma": 0.6,
            },
            weight=3.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    "(?!left_wrist_roll_link$)(?!right_wrist_roll_link$)"
                    "(?!left_foot_.*_link$)(?!right_foot_.*_link$).+$"
                ),
            },
            weight=-0.5,
        ),
        "object_point_cloud_distance_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectPointCloudDistanceExp",
            params={
                "distance_scale": 10.0,
                "max_points": 1024,
            },
            weight=2.0,
        ),
    },
)

__all__ = ["r1_26dof_wbt_reward_w_object_multi_teacher"]
