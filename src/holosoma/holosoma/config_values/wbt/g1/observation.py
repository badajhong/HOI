"""Whole Body Tracking observation presets for the G1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.5,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
    },
)

critic_obs_shared_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.25,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

critic_obs_w_object_terms = critic_obs_shared_terms.copy()
critic_obs_w_object_terms.update(
    {
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_type_one_hot": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_type_one_hot",
            scale=1.0,
            noise=0.0,
        ),
    }
)

g1_29dof_wbt_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

actor_obs_w_object_terms = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.5,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
        # Object-related observation terms
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_type_one_hot": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_type_one_hot",
            scale=1.0,
            noise=0.0,
        ),
    },
)

g1_29dof_wbt_observation_w_object_multi = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_w_object_terms,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

"""The teacher observation configuration includes all the terms from the multi-object observation, but with reduced noise for the actor observation terms to provide a clearer learning signal during training."""

obs_teacher = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.25,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
    # Object-related observation terms
    "obj_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "obj_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "obj_lin_vel_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
        scale=1.0,
        noise=0.0,
    ),
    "obj_type_one_hot": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_type_one_hot",
        scale=1.0,
        noise=0.0,
    ),
}

g1_29dof_wbt_observation_w_object_multi_teacher = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=3,
            terms=obs_teacher,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=3,
            terms=obs_teacher,
        ),
    },
)

student_obs_w_object_terms = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "motion_command_joint_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command_joint_pos",
            scale=1.0,
            noise=0.0,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.0,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.0,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.0,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:student_actions",
            scale=1.0,
            noise=0.0,
        ),
        "obj_type_one_hot": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_type_one_hot",
            scale=1.0,
            noise=0.0,
        ),
    },
)

# Residual actor should condition on the actually executed previous action
# (student base action + residual correction), not the frozen student's
# previous base-action rollout.
residual_actor_obs_w_object_terms = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "motion_command_joint_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command_joint_pos",
            scale=1.0,
            noise=0.0,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.0,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.0,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.0,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
        "obj_type_one_hot": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_type_one_hot",
            scale=1.0,
            noise=0.0,
        ),
    },
)

g1_29dof_wbt_observation_w_object_multi_student = ObservationManagerCfg(
    groups={
        "actor_obs": student_obs_w_object_terms,
        "teacher_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=3,
            terms=obs_teacher,
        ),
        "ir_ae_latent": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "ir_ae_latent": ObsTermCfg(
                    func="holosoma.managers.observation.terms.wbt:IRAELatent",
                    params={
                        # Preferred override path:
                        # --observation.groups.ir_ae_latent.terms.ir_ae_latent.params.checkpoint_path=/path/to/best.pt
                        "checkpoint_path": "",
                        # Preferred override path:
                        # --observation.groups.ir_ae_latent.terms.ir_ae_latent.params.body_source=all
                        "body_source": "",
                    },
                    scale=1.0,
                    noise=0.0,
                )
            },
        ),
    },
)

g1_29dof_wbt_observation_w_object_multi_res = ObservationManagerCfg(
    groups={
        "residual_actor_obs": residual_actor_obs_w_object_terms,
        "student_actor_obs": student_obs_w_object_terms,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
        "di_ae_latent": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "di_ae_latent": ObsTermCfg(
                    func="holosoma.managers.observation.terms.wbt:DIAELatent",
                    params={
                        # Preferred override path:
                        # --observation.groups.di_ae_latent.terms.di_ae_latent.params.checkpoint_path=/path/to/di_ae.pt
                        "checkpoint_path": "",
                        "debug_save_depth_images": False,
                        "debug_depth_save_interval": 200,
                        "debug_depth_env_ids": (0,),
                    },
                    scale=1.0,
                    noise=0.0,
                )
            },
        ),
        "student_base_action": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "student_base_action": ObsTermCfg(
                    func="holosoma.managers.observation.terms.wbt:FrozenStudentBaseAction",
                    params={
                        # Preferred override path:
                        # --observation.groups.student_base_action.terms.student_base_action.params.student_checkpoint=/path/to/student.pt
                        "student_checkpoint": "",
                        "student_obs_group": "student_actor_obs",
                        "latent_obs_group": "di_ae_latent",
                    },
                    scale=1.0,
                    noise=0.0,
                )
            },
        ),
    },
)

__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_w_object_multi",
    "g1_29dof_wbt_observation_w_object_multi_teacher",
    "g1_29dof_wbt_observation_w_object_multi_student",
    "g1_29dof_wbt_observation_w_object_multi_res",
]
