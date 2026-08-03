"""Whole Body Tracking reward presets for the R1 robot."""

from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
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

r1_26dof_fastsac_reward = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:tracking_lin_vel",
            params={"tracking_sigma": 0.25},
            weight=1.5,
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:tracking_ang_vel",
            params={"tracking_sigma": 0.25},
            weight=0.75,
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.005,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.95},
            weight=-1.0,
        ),
        "scaled_surface_distance_progress": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:ScaledSurfaceDistanceProgress",
            params={
                "contact_body_names_regex": R1_OBJECT_CONTACT_REWARD_BODY_NAMES_REGEX,
                "fail_on_missing_labels": True,
            },
            weight=1.0,
        ),
        "scaled_contact_target_distance": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:ScaledContactTargetDistance",
            params={
                "distance_scale": 0.35,
                "body_names": R1_OBJECT_CONTACT_REWARD_BODY_NAMES,
                "target_topk": R1_OBJECT_CONTACT_TARGET_TOPK,
                "fail_on_missing_targets": True,
            },
            weight=0.5,
        ),
        "scaled_contact_target_coverage": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:ScaledContactTargetCoverage",
            params={
                "distance_threshold": R1_OBJECT_CONTACT_TARGET_COVERAGE_THRESHOLD,
                "temperature": R1_OBJECT_CONTACT_TARGET_COVERAGE_TEMPERATURE,
                "body_names": R1_OBJECT_CONTACT_REWARD_BODY_NAMES,
                "target_topk": R1_OBJECT_CONTACT_TARGET_TOPK,
                "fail_on_missing_targets": True,
            },
            weight=0.5,
        ),
        "phase_transition_bonus": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:PhaseTransitionBonus",
            weight=0.5,
        ),
        "reference_contact_tracking": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:PhaseOneReferenceContact",
            params={
                "threshold": R1_OBJECT_CONTACT_THRESHOLD,
                "distance_scale": R1_OBJECT_CONTACT_LABEL_DISTANCE_SCALE,
                "contact_body_names_regex": R1_OBJECT_CONTACT_REWARD_BODY_NAMES_REGEX,
                "fail_on_missing_labels": True,
            },
            weight=1.5,
        ),
        "weak_anchored_body_position_tracking": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:weak_anchored_body_position_tracking",
            params={"sigma": 0.5, "blend_steps": 20},
            weight=0.5,
        ),
        "weak_anchored_body_orientation_tracking": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:weak_anchored_body_orientation_tracking",
            params={"sigma": 0.7, "blend_steps": 20},
            weight=0.3,
        ),
        "phase_one_joint_position_tracking": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:PhaseOneMotionJointPositionErrorExp",
            params={
                "sigma": 0.5,
                "blend_steps": 20,
                "joint_names": R1_24DOF_ACTION_DOF_NAMES,
            },
            weight=0.5,
        ),
        "phase_one_joint_velocity_tracking": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:PhaseOneMotionJointVelocityErrorExp",
            params={
                "sigma": 2.0,
                "blend_steps": 20,
                "joint_names": R1_24DOF_ACTION_DOF_NAMES,
            },
            weight=0.2,
        ),
        "delta_object_position_tracking": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:delta_object_position_tracking",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        "delta_object_orientation_tracking": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:delta_object_orientation_tracking",
            params={"sigma": 0.7},
            weight=0.5,
        ),
        "delta_object_velocity_tracking": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:delta_object_velocity_tracking",
            params={"linear_sigma": 1.0, "angular_sigma": 2.0},
            weight=0.5,
        ),
        "finish_success_bonus": RewardTermCfg(
            func="holosoma.managers.reward.terms.r1_fastsac:FinishSuccessBonus",
            params={
                "position_threshold": 0.20,
                "orientation_threshold": 0.50,
                "velocity_threshold": 0.30,
            },
            weight=3.0,
        ),
    },
)

__all__ = ["r1_26dof_fastsac_reward", "r1_26dof_wbt_reward_w_object_multi_teacher"]
