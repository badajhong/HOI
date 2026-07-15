"""Whole Body Tracking reward presets for the R1 robot."""

from dataclasses import replace

from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_w_object_multi_teacher
from holosoma.config_values.wbt.r1.contact import (
    R1_OBJECT_CONTACT_BODY_NAMES,
    R1_OBJECT_CONTACT_LABEL_DISTANCE_SCALE,
    R1_OBJECT_CONTACT_TARGET_DISTANCE_SCALE,
    R1_OBJECT_CONTACT_THRESHOLD,
    R1_OBJECT_HEAD_PROXIMITY_BODY_NAMES,
    R1_OBJECT_HEAD_PROXIMITY_DISTANCE_CUTOFF,
    R1_OBJECT_HEAD_PROXIMITY_DISTANCE_SCALE,
)

r1_26dof_wbt_reward_w_object_multi_teacher = replace(
    g1_29dof_wbt_reward_w_object_multi_teacher,
    terms={
        **g1_29dof_wbt_reward_w_object_multi_teacher.terms,
        "action_rate_l2": replace(
            g1_29dof_wbt_reward_w_object_multi_teacher.terms["action_rate_l2"],
            weight=-0.5,
        ),
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
        "object_contact_label_distance": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectContactLabelDistance",
            params={
                "threshold": R1_OBJECT_CONTACT_THRESHOLD,
                "distance_scale": R1_OBJECT_CONTACT_LABEL_DISTANCE_SCALE,
                "contact_body_names_regex": r"^(?!head_yaw_link$).*",
                "fail_on_missing_labels": True,
            },
            weight=2.0,
        ),
        "object_contact_target_point_distance": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectContactTargetPointDistance",
            params={
                "distance_scale": R1_OBJECT_CONTACT_TARGET_DISTANCE_SCALE,
                "body_names": R1_OBJECT_CONTACT_BODY_NAMES,
                "fail_on_missing_targets": True,
            },
            weight=3.0,
        ),
        "object_head_proximity_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectBodyProximityPenalty",
            params={
                "body_names": R1_OBJECT_HEAD_PROXIMITY_BODY_NAMES,
                "distance_scale": R1_OBJECT_HEAD_PROXIMITY_DISTANCE_SCALE,
                "distance_cutoff": R1_OBJECT_HEAD_PROXIMITY_DISTANCE_CUTOFF,
            },
            weight=-1.0,
        ),
    },
)

__all__ = ["r1_26dof_wbt_reward_w_object_multi_teacher"]
