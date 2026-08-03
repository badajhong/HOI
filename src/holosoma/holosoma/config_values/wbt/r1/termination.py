"""Whole Body Tracking termination presets for the R1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values.wbt.r1.command import R1_WBT_BODY_NAMES_TO_TRACK

r1_26dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "motion_ends": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:motion_ends",
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTracking",
            params={
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                "body_names_to_track": R1_WBT_BODY_NAMES_TO_TRACK,
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_roll_link",
                    "right_wrist_roll_link",
                ],
                "bad_object_pos_threshold": 0.2,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

r1_26dof_fastsac_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "motion_ends": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:motion_ends",
        ),
        "object_robot_distance_xy": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:ObjectRobotDistanceXY",
            params={
                "initial_distance_xy": 0.8,
                "final_distance_margin_xy": 0.3,
                "consecutive_steps": 25,
            },
        ),
        "fallen": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:ReferencePelvisHeightFallen",
            params={
                "pelvis_body_name": "pelvis_link",
                "height_margin": 0.30,
                "consecutive_steps": 5,
            },
        ),
    }
)

__all__ = ["r1_26dof_fastsac_termination", "r1_26dof_wbt_termination"]
