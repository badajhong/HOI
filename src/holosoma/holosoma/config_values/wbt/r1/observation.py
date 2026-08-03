"""Whole Body Tracking observation presets for the R1 robot."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_values.wbt.g1.observation import (
    g1_29dof_wbt_observation_w_object_multi_teacher,
    g1_29dof_wbt_observation_w_object_multi_student,
)
from holosoma.config_values.wbt.r1.contact import (
    R1_OBJECT_CONTACT_BODY_NAMES,
    R1_OBJECT_CONTACT_DISTANCE_CLIP,
    R1_OBJECT_CONTACT_TARGET_RELATIVE_CLIP,
    R1_OBJECT_CONTACT_TARGET_TOPK,
)

object_randomization_privileged_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:object_randomization_privileged",
    scale=1.0,
    noise=0.0,
)

task_index_one_hot_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:task_index_one_hot",
    scale=1.0,
    noise=0.0,
)

object_type_one_hot_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:obj_type_one_hot",
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

object_contact_target_current_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:ObjectContactTargetCurrent",
    params={
        "body_names": R1_OBJECT_CONTACT_BODY_NAMES,
        "relative_clip": R1_OBJECT_CONTACT_TARGET_RELATIVE_CLIP,
        "distance_clip": R1_OBJECT_CONTACT_DISTANCE_CLIP,
        "include_active": True,
        "include_distance": True,
        "target_topk": R1_OBJECT_CONTACT_TARGET_TOPK,
        "fail_on_missing_targets": True,
    },
    scale=1.0,
    noise=0.0,
    clip=(-1.0, 1.0),
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
        "object_contact_target_current": object_contact_target_current_term,
        "object_randomization_privileged": object_randomization_privileged_term,
        "task_index_one_hot": task_index_one_hot_term,
    }
)

critic_terms = _with_selective_history(
    {
        **g1_29dof_wbt_observation_w_object_multi_teacher.groups["critic_obs"].terms,
        "motion_command_future": motion_command_future_term,
        "object_distance_current": object_distance_current_term,
        "object_contact_target_current": object_contact_target_current_term,
        "object_randomization_privileged": object_randomization_privileged_term,
        "task_index_one_hot": task_index_one_hot_term,
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

r1_26dof_wbt_observation_w_object_multi_student = ObservationManagerCfg(
    groups={
        # The R1 student is reference-free: it must infer object-relative
        # interaction state from the live depth/proprioception latent rather
        # than consuming the demonstration's current target joint pose.
        "actor_obs": replace(
            g1_29dof_wbt_observation_w_object_multi_student.groups["actor_obs"],
            terms={
                name: term
                for name, term in g1_29dof_wbt_observation_w_object_multi_student.groups["actor_obs"].terms.items()
                if name != "motion_command_joint_pos"
            },
        ),
        "ae_latent": replace(
            g1_29dof_wbt_observation_w_object_multi_student.groups["ae_latent"],
            terms={
                "ae_latent": replace(
                    g1_29dof_wbt_observation_w_object_multi_student.groups["ae_latent"].terms["ae_latent"],
                    params={
                        **g1_29dof_wbt_observation_w_object_multi_student.groups["ae_latent"]
                        .terms["ae_latent"]
                        .params,
                        "depth_pixel_noise_max_std_m": 0.005,
                        "depth_dropout_probability": 0.001,
                    },
                )
            },
        ),
        # Keep this group identical to the R1 teacher actor observation. The
        # frozen teacher checkpoint is reconstructed against this exact input.
        "teacher_obs": replace(
            r1_26dof_wbt_observation_w_object_multi_teacher.groups["actor_obs"],
        ),
    },
)

# Checkpoint-free student actor interface.  This group is selected at config
# resolution time only when none of --ir-ae/--di-ae/--di-pro-ae is supplied.
# Keeping it as a separate fallback group preserves compatibility with older
# latent-based student checkpoints.
r1_student_direct_ir_actor_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "projected_gravity": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:projected_gravity",
            scale=1.0,
            noise=0.0,
        ),
        "base_angular_velocity": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.0,
        ),
        "joint_position": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.0,
        ),
        "joint_velocity": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.0,
        ),
        "previous_action": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:student_actions",
            history_length=3,
            scale=1.0,
            noise=0.0,
        ),
        "command_velocity": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:reference_velocity_command",
            scale=1.0,
            noise=0.0,
        ),
        "task_phase_one_hot": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:task_phase_one_hot",
            scale=1.0,
            noise=0.0,
        ),
        "interaction_progress": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:interaction_progress",
            scale=1.0,
            noise=0.0,
            clip=(0.0, 1.0),
        ),
        "task_index_one_hot": task_index_one_hot_term,
        "obj_type_one_hot": object_type_one_hot_term,
        "interaction_representation": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:ObjectInteractionRepresentation",
            scale=1.0,
            noise=0.0,
        ),
    },
)

r1_student_privileged_critic_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "projected_gravity": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:projected_gravity"),
        "base_linear_velocity": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_lin_vel"),
        "base_angular_velocity": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel"),
        "joint_position": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos"),
        "joint_velocity": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel"),
        "previous_action": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:student_actions",
            history_length=3,
        ),
        "command_velocity": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:reference_velocity_command"
        ),
        "task_phase_one_hot": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:task_phase_one_hot"
        ),
        "interaction_progress": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:interaction_progress",
            clip=(0.0, 1.0),
        ),
        "task_index_one_hot": task_index_one_hot_term,
        "obj_type_one_hot": object_type_one_hot_term,
        "object_position_robot_frame": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:object_position_robot_frame"
        ),
        "object_orientation_robot_frame": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:object_orientation_robot_frame"
        ),
        "object_linear_velocity_robot_frame": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:object_linear_velocity_robot_frame"
        ),
        "object_angular_velocity_robot_frame": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:object_angular_velocity_robot_frame"
        ),
        "object_world_position": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:object_world_position"
        ),
        "object_world_orientation": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:object_world_orientation"
        ),
        "robot_world_position": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:robot_world_position"
        ),
        "robot_world_orientation": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:robot_world_orientation"
        ),
        "measured_contact_state": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:ObjectContactCurrent",
            params={
                "body_names": R1_OBJECT_CONTACT_BODY_NAMES,
                "include_soft_contact": True,
                "include_distance": False,
            },
        ),
        "interaction_representation": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:ObjectInteractionRepresentation"
        ),
        "object_scale": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_scale"),
        "object_mass": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_mass"),
        "object_center_of_mass": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:object_center_of_mass"
        ),
        "object_friction": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_friction"),
        "object_initial_position_robot_frame": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:object_initial_position_robot_frame"
        ),
        "object_initial_orientation_robot_frame": ObsTermCfg(
            func="holosoma.managers.observation.terms.r1_fastsac:object_initial_orientation_robot_frame"
        ),
    },
)

r1_student_direct_ir_observation = ObservationManagerCfg(
    groups={
        "actor_obs": r1_student_direct_ir_actor_obs,
        "critic_obs": r1_student_privileged_critic_obs,
        "teacher_obs": r1_26dof_wbt_observation_w_object_multi_student.groups["teacher_obs"],
    }
)

r1_26dof_fastsac_observation = replace(
    r1_26dof_wbt_observation_w_object_multi_student,
    groups={
        **r1_26dof_wbt_observation_w_object_multi_student.groups,
        "actor_obs": replace(
            r1_26dof_wbt_observation_w_object_multi_student.groups["actor_obs"],
            terms={
                **r1_26dof_wbt_observation_w_object_multi_student.groups["actor_obs"].terms,
                "velocity_command": ObsTermCfg(
                    func="holosoma.managers.observation.terms.r1_fastsac:reference_velocity_command",
                    scale=1.0,
                    noise=0.0,
                ),
                "task_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.r1_fastsac:task_phase_one_hot",
                    scale=1.0,
                    noise=0.0,
                ),
                "interaction_progress": ObsTermCfg(
                    func="holosoma.managers.observation.terms.r1_fastsac:interaction_progress",
                    scale=1.0,
                    noise=0.0,
                    clip=(0.0, 1.0),
                ),
                "task_index_one_hot": task_index_one_hot_term,
            },
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "projected_gravity": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:projected_gravity"),
                "base_linear_velocity": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_lin_vel"),
                "base_angular_velocity": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel"),
                "joint_position": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos"),
                "joint_velocity": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel"),
                "previous_action": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions"),
                "command_velocity": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:reference_velocity_command"),
                "task_phase_one_hot": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:task_phase_one_hot"),
                "interaction_progress": ObsTermCfg(
                    func="holosoma.managers.observation.terms.r1_fastsac:interaction_progress",
                    clip=(0.0, 1.0),
                ),
                "object_position_robot_frame": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_position_robot_frame"),
                "object_orientation_robot_frame": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_orientation_robot_frame"),
                "object_linear_velocity_robot_frame": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_linear_velocity_robot_frame"),
                "object_angular_velocity_robot_frame": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_angular_velocity_robot_frame"),
                "object_world_position": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_world_position"),
                "object_world_orientation": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_world_orientation"),
                "robot_world_position": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:robot_world_position"),
                "robot_world_orientation": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:robot_world_orientation"),
                "measured_contact_state": ObsTermCfg(
                    func="holosoma.managers.observation.terms.wbt:ObjectContactCurrent",
                    params={
                        "body_names": R1_OBJECT_CONTACT_BODY_NAMES,
                        "include_soft_contact": True,
                        "include_distance": False,
                    },
                ),
                "object_scale": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_scale"),
                "object_mass": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_mass"),
                "object_center_of_mass": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_center_of_mass"),
                "object_friction": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_friction"),
                "object_initial_position_robot_frame": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_initial_position_robot_frame"),
                "object_initial_orientation_robot_frame": ObsTermCfg(func="holosoma.managers.observation.terms.r1_fastsac:object_initial_orientation_robot_frame"),
                "task_index_one_hot": task_index_one_hot_term,
                "obj_type_one_hot": object_type_one_hot_term,
            },
        ),
    },
)

__all__ = [
    "r1_26dof_fastsac_observation",
    "r1_26dof_wbt_observation_w_object_multi_student",
    "r1_26dof_wbt_observation_w_object_multi_teacher",
    "r1_student_direct_ir_actor_obs",
    "r1_student_direct_ir_observation",
    "r1_student_privileged_critic_obs",
]
