"""Whole Body Tracking reward presets for the R1 robot."""

from dataclasses import replace

from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_values.action import R1_24DOF_ACTION_DOF_NAMES
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_w_object_multi_teacher
from holosoma.config_values.wbt.r1.contact import (
    R1_OBJECT_CONTACT_LABEL_DISTANCE_SCALE,
    R1_OBJECT_CONTACT_REWARD_BODY_NAMES,
    R1_OBJECT_CONTACT_REWARD_BODY_NAMES_REGEX,
    R1_OBJECT_CONTACT_TARGET_COVERAGE_TEMPERATURE,
    R1_OBJECT_CONTACT_TARGET_COVERAGE_THRESHOLD,
    R1_OBJECT_CONTACT_TARGET_DISTANCE_SCALE,
    R1_OBJECT_CONTACT_TARGET_TOPK,
    R1_OBJECT_CONTACT_THRESHOLD,
    R1_OBJECT_HEAD_PROXIMITY_BODY_NAMES,
    R1_OBJECT_HEAD_PROXIMITY_BODY_POINTS,
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
        "motion_relative_body_position_error_exp": replace(
            g1_29dof_wbt_reward_w_object_multi_teacher.terms["motion_relative_body_position_error_exp"],
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": replace(
            g1_29dof_wbt_reward_w_object_multi_teacher.terms["motion_relative_body_orientation_error_exp"],
            weight=1.0,
        ),
        "motion_joint_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:MotionJointPositionErrorExp",
            params={"sigma": 0.5, "joint_names": R1_24DOF_ACTION_DOF_NAMES},
            weight=1.0,
        ),
        "motion_joint_velocity_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:MotionJointVelocityErrorExp",
            params={"sigma": 2.0, "joint_names": R1_24DOF_ACTION_DOF_NAMES},
            weight=0.1,
        ),
        "object_global_ref_position_error_exp": replace(
            g1_29dof_wbt_reward_w_object_multi_teacher.terms["object_global_ref_position_error_exp"],
            params={
                **g1_29dof_wbt_reward_w_object_multi_teacher.terms[
                    "object_global_ref_position_error_exp"
                ].params,
                "sigma": 0.6,
            },
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": replace(
            g1_29dof_wbt_reward_w_object_multi_teacher.terms["object_global_ref_orientation_error_exp"],
            weight=1.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "respect_motion_contact_labels": True,
                "undesired_contacts_body_names": (
                    "^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
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
            weight=1.0,
        ),
        "object_contact_label_distance": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectContactLabelDistance",
            params={
                "threshold": R1_OBJECT_CONTACT_THRESHOLD,
                "distance_scale": R1_OBJECT_CONTACT_LABEL_DISTANCE_SCALE,
                "contact_body_names_regex": R1_OBJECT_CONTACT_REWARD_BODY_NAMES_REGEX,
                "fail_on_missing_labels": True,
            },
            weight=2.0,
        ),
        "object_contact_target_point_distance": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectContactTargetPointDistance",
            params={
                "distance_scale": R1_OBJECT_CONTACT_TARGET_DISTANCE_SCALE,
                "body_names": R1_OBJECT_CONTACT_REWARD_BODY_NAMES,
                "target_topk": R1_OBJECT_CONTACT_TARGET_TOPK,
                "fail_on_missing_targets": True,
            },
            weight=0.5,
        ),
        "object_contact_target_point_coverage": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectContactTargetPointCoverage",
            params={
                "distance_threshold": R1_OBJECT_CONTACT_TARGET_COVERAGE_THRESHOLD,
                "temperature": R1_OBJECT_CONTACT_TARGET_COVERAGE_TEMPERATURE,
                "body_names": R1_OBJECT_CONTACT_REWARD_BODY_NAMES,
                "target_topk": R1_OBJECT_CONTACT_TARGET_TOPK,
                "fail_on_missing_targets": True,
            },
            weight=0.25,
        ),
        "object_head_proximity_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectBodyProximityPenalty",
            params={
                "body_names": R1_OBJECT_HEAD_PROXIMITY_BODY_NAMES,
                "body_points": R1_OBJECT_HEAD_PROXIMITY_BODY_POINTS,
                "body_distance_reduce": "min",
                "distance_scale": R1_OBJECT_HEAD_PROXIMITY_DISTANCE_SCALE,
                "distance_cutoff": R1_OBJECT_HEAD_PROXIMITY_DISTANCE_CUTOFF,
            },
            weight=-1.0,
        ),
    },
)

__all__ = ["r1_26dof_wbt_reward_w_object_multi_teacher"]
