"""Whole Body Tracking observation presets for the G1 robot."""

from dataclasses import replace

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

"""Teacher observations use multi-object terms with reduced actor-observation noise."""

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
        "ae_latent": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "ae_latent": ObsTermCfg(
                    func="holosoma.managers.observation.terms.wbt:AELatent",
                    params={
                        # Preferred IR override path:
                        # --observation.groups.ae_latent.terms.ae_latent.params.checkpoint_path=/path/to/ir_ae.pt
                        "checkpoint_path": "",
                        # Preferred DI override path:
                        # --observation.groups.ae_latent.terms.ae_latent.params.di_checkpoint_path=/path/to/di_ae.pt
                        "di_checkpoint_path": "",
                        # Preferred DI+proprioception override path:
                        # --observation.groups.ae_latent.terms.ae_latent.params.di_pro_checkpoint_path=...
                        "di_pro_checkpoint_path": "",
                        # Optional explicit source: "", "ir", "di", or "di_pro"
                        "source": "",
                        # Keep the original robot asset and let IsaacSim attach
                        # the depth camera using the built-in fallback torso mount.
                        "robot_depth_asset_mode": "original",
                        # Preferred IR-only override path:
                        # --observation.groups.ae_latent.terms.ae_latent.params.body_source=all
                        "body_source": "",
                        # Optional DI debug image saving when source resolves to DIAELatent.
                        "debug_save_depth_images": False,
                        "debug_depth_save_interval": 200,
                        # Accept compact CLI values like "0" or "0,3".
                        "debug_depth_env_ids": "0",
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
                        # Preferred DI+proprioception override path:
                        # --observation.groups.di_ae_latent.terms.di_ae_latent.params.di_pro_checkpoint_path=...
                        "di_pro_checkpoint_path": "",
                        # Keep the original robot asset and let IsaacSim attach
                        # the depth camera using the built-in fallback torso mount.
                        "robot_depth_asset_mode": "original",
                        "debug_save_depth_images": False,
                        "debug_depth_save_interval": 200,
                        # Accept compact CLI values like "0" or "0,3".
                        "debug_depth_env_ids": "0",
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
                        # --observation.groups.student_base_action.terms.student_base_action.params.student_checkpoint
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

object_scale_bin_common_params = {
    "target": "uniform",
    # Synced from randomize_object_scale_startup.scale_values at runtime.
    "scale_values": "auto",
    "log_metrics": True,
    "log_target_summary": False,
    "log_pred_summary": False,
    "log_distribution": False,
}

object_scale_input_group = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "object_scale_input": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:ObjectScaleBinInput",
            params={
                **object_scale_bin_common_params,
                # Actor-side input: predicted hard one-hot object-scale bin.
                "source": "predicted",
                "output_mode": "one_hot",
                "latent_obs_group": "di_ae_latent",
                # Reuse the frozen DI-pro depth CNN + frame projection feature computed for di_ae_latent.
                "feature_source": "di_projection",
                "train_online": True,
                "hidden_dims": "256,128",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "train_batch_size": 4096,
                "train_every": 1,
                "max_grad_norm": 1.0,
                "log_prefix": "ScaleBinProbe",
            },
            scale=1.0,
            noise=0.0,
        )
    },
)

object_scale_gt_input_group = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "object_scale_gt_input": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:ObjectScaleBinInput",
            params={
                **object_scale_bin_common_params,
                # Critic-side privileged input: debug GT hard one-hot scale bin.
                "source": "real",
                "log_metrics": False,
                "log_prefix": "ScaleBinGT",
            },
            scale=1.0,
            noise=0.0,
        )
    },
)

g1_29dof_wbt_observation_w_object_multi_res = replace(
    g1_29dof_wbt_observation_w_object_multi_res,
    groups={
        **g1_29dof_wbt_observation_w_object_multi_res.groups,
        "object_scale_gt_input": object_scale_gt_input_group,
    },
)

g1_29dof_wbt_observation_w_object_multi_res_scale_probe = replace(
    g1_29dof_wbt_observation_w_object_multi_res,
    groups={
        **g1_29dof_wbt_observation_w_object_multi_res.groups,
        "object_scale_input": object_scale_input_group,
        "object_scale_gt_input": object_scale_gt_input_group,
    },
)

__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_w_object_multi",
    "g1_29dof_wbt_observation_w_object_multi_res",
    "g1_29dof_wbt_observation_w_object_multi_res_scale_probe",
    "g1_29dof_wbt_observation_w_object_multi_student",
    "g1_29dof_wbt_observation_w_object_multi_teacher",
]
