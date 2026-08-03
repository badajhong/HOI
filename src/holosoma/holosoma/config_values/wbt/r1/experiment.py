"""Whole Body Tracking experiment presets for the R1 robot."""

from dataclasses import replace

from holosoma.config_types.video import CartesianCameraConfig
from holosoma.config_values import action, algo, logger as logger_values, robot, simulator
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object_multi_teacher
from holosoma.config_values.wbt.r1 import command, curriculum, observation, randomization, reward, termination

R1_TEACHER_HIDDEN_DIMS = [1024, 512, 256, 128]

r1_teacher_algo = replace(
    g1_29dof_wbt_w_object_multi_teacher.algo,
    config=replace(
        g1_29dof_wbt_w_object_multi_teacher.algo.config,
        init_noise_std=0.5,
        save_interval=1000,
        module_dict=replace(
            g1_29dof_wbt_w_object_multi_teacher.algo.config.module_dict,
            actor=replace(
                g1_29dof_wbt_w_object_multi_teacher.algo.config.module_dict.actor,
                layer_config=replace(
                    g1_29dof_wbt_w_object_multi_teacher.algo.config.module_dict.actor.layer_config,
                    hidden_dims=R1_TEACHER_HIDDEN_DIMS,
                ),
            ),
            critic=replace(
                g1_29dof_wbt_w_object_multi_teacher.algo.config.module_dict.critic,
                layer_config=replace(
                    g1_29dof_wbt_w_object_multi_teacher.algo.config.module_dict.critic.layer_config,
                    hidden_dims=R1_TEACHER_HIDDEN_DIMS,
                ),
            ),
        ),
    ),
)

r1_teacher_logger = replace(
    logger_values.wandb,
    video=replace(
        logger_values.wandb.video,
        interval=20,
        camera=CartesianCameraConfig(offset=[3.0, -3.0, 1.5]),
    ),
)

r1_teacher_simulator = replace(
    simulator.isaacsim,
    config=replace(
        simulator.isaacsim.config,
        scene=replace(simulator.isaacsim.config.scene, env_spacing=5.0),
        sim=replace(
            simulator.isaacsim.config.sim,
            physx=replace(
                simulator.isaacsim.config.sim.physx,
                gpu_max_rigid_patch_count=524288,
            ),
        ),
    ),
)

r1_teacher = replace(
    g1_29dof_wbt_w_object_multi_teacher,
    training=replace(
        g1_29dof_wbt_w_object_multi_teacher.training,
        project="teacher",
        name="r1_teacher",
        num_envs=4096,
    ),
    robot=replace(
        robot.r1_26dof_w_object_multi_teacher,
        asset=replace(
            robot.r1_26dof_w_object_multi_teacher.asset,
            enable_self_collisions=True,
        ),
        object=replace(
            robot.r1_26dof_w_object_multi_teacher.object,
            object_urdf_asset="train_r1/objects",
            object_urdf_folder="train_r1/objects",
            object_urdf_path="train_r1/objects/largebox/largebox.urdf",
            object_parm="train_r1/objects/objects_parm.yaml",
        ),
        control=replace(robot.r1_26dof_w_object_multi_teacher.control, action_scale=0.25),
        init_state=replace(robot.r1_26dof_w_object_multi_teacher.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    algo=r1_teacher_algo,
    logger=r1_teacher_logger,
    simulator=r1_teacher_simulator,
    action=action.r1_24dof_joint_pos,
    command=command.r1_26dof_wbt_command_w_object_multi_teacher,
    termination=termination.r1_26dof_wbt_termination,
    randomization=randomization.r1_26dof_wbt_randomization_w_object,
    observation=observation.r1_26dof_wbt_observation_w_object_multi_teacher,
    reward=reward.r1_26dof_wbt_reward_w_object_multi_teacher,
    curriculum=curriculum.r1_26dof_wbt_curriculum,
)

r1_student_motion_term = replace(
    r1_teacher.command.setup_terms["motion_command"],
    params={
        **r1_teacher.command.setup_terms["motion_command"].params,
        "motion_config": replace(
            r1_teacher.command.setup_terms["motion_command"].params["motion_config"],
            noise_to_initial_pose=replace(
                r1_teacher.command.setup_terms["motion_command"].params[
                    "motion_config"
                ].noise_to_initial_pose,
                object_pos=[0.0, 0.0, 0.0],
                root_pos=[0.0, 0.0, 0.01],
            ),
        ),
    },
)

r1_student_command = replace(
    r1_teacher.command,
    setup_terms={
        **r1_teacher.command.setup_terms,
        "motion_command": r1_student_motion_term,
    },
)

r1_student = replace(
    r1_teacher,
    training=replace(
        r1_teacher.training,
        project="student",
        name="r1_student",
        num_envs=64,
    ),
    algo=replace(
        algo.dagger_student,
        config=replace(
            algo.dagger_student.config,
            num_learning_iterations=50000,
            num_steps_per_env=32,
            num_updates_per_iteration=32,
            batch_size=4096,
            actor_learning_rate=3e-4,
            value_learning_rate=1e-4,
            q_learning_rate=3e-4,
            teacher_mixture_start=1.0,
            teacher_mixture_end=0.2,
            teacher_mixture_decay_iterations=5000,
            teacher_action_outlier_threshold=20.0,
            actor_huber_delta=1.0,
            student_action_clip=20.0,
            value_target_tau=0.005,
            q_target_tau=0.05,
            save_interval=100,
            stack_buffer=524288,
            teacher_anchor_capacity=262144,
            teacher_anchor_sampling_ratio=0.5,
            buffer_device="gpu",
            teacher_buffer_output="teacher_buffer.h5",
            teacher_buffer_max_transitions=524288,
            module_dict=replace(
                algo.dagger_student.config.module_dict,
                actor=replace(
                    algo.dagger_student.config.module_dict.actor,
                    layer_config=replace(
                        algo.dagger_student.config.module_dict.actor.layer_config,
                        hidden_dims=[512, 256, 128],
                    ),
                ),
            ),
        ),
    ),
    # Match the DAgger student's deployable observation interface to the
    # FastSAC actor: proprioception + object/task identity + velocity command
    # + task phase, followed by the same frozen depth/proprioception latent.
    # The privileged teacher_obs group remains unchanged for supervision.
    observation=replace(
        observation.r1_26dof_fastsac_observation,
        groups={
            **observation.r1_26dof_fastsac_observation.groups,
            "critic_obs": replace(
                observation.r1_student_privileged_critic_obs,
                terms={
                    **observation.r1_student_privileged_critic_obs.terms,
                    "previous_action": replace(
                        observation.r1_student_privileged_critic_obs.terms["previous_action"],
                        func="holosoma.managers.observation.terms.wbt:actions",
                    ),
                },
            ),
        },
    ),
    command=r1_student_command,
    reward=reward.r1_26dof_fastsac_reward,
    termination=termination.r1_26dof_fastsac_termination,
    simulator=replace(
        r1_teacher.simulator,
        config=replace(
            r1_teacher.simulator.config,
            robot_depth_camera_position_noise_m=0.02,
        ),
    ),
)

# Retain the direct-IR fallback alongside the legacy latent observation.  The
# config resolver activates exactly one of them based on whether an AE
# checkpoint was supplied.
r1_student = replace(
    r1_student,
    observation=replace(
        r1_student.observation,
        groups={
            **r1_student.observation.groups,
            "critic_obs": replace(
                observation.r1_student_privileged_critic_obs,
                terms={
                    **observation.r1_student_privileged_critic_obs.terms,
                    "previous_action": replace(
                        observation.r1_student_privileged_critic_obs.terms["previous_action"],
                        func="holosoma.managers.observation.terms.wbt:actions",
                    ),
                },
            ),
            "direct_ir_actor_obs": observation.r1_student_direct_ir_actor_obs,
        },
    ),
)

r1_fastsac = replace(
    r1_student,
    simulator=replace(
        r1_student.simulator,
        config=replace(
            r1_student.simulator.config,
            scene=replace(r1_student.simulator.config.scene, replicate_physics=False),
        ),
    ),
    command=replace(
        r1_student.command,
        setup_terms={
            **r1_student.command.setup_terms,
            "motion_command": replace(
                r1_student.command.setup_terms["motion_command"],
                params={
                    **r1_student.command.setup_terms["motion_command"].params,
                    "motion_config": replace(
                        r1_student.command.setup_terms["motion_command"].params["motion_config"],
                        start_at_timestep_zero_prob=1.0,
                        adaptive_phase_zero=True,
                        phase_zero_position_threshold_m=0.15,
                        phase_zero_yaw_threshold_rad=0.35,
                        phase_zero_ready_hold_steps=8,
                        phase_zero_linear_velocity_gain=1.5,
                        phase_zero_angular_velocity_gain=1.5,
                        phase_zero_max_linear_velocity=0.5,
                        phase_zero_max_angular_velocity=0.6,
                        noise_to_initial_pose=replace(
                            r1_student.command.setup_terms["motion_command"]
                            .params["motion_config"]
                            .noise_to_initial_pose,
                            object_sector_radius=[0.0, 0.50],
                            object_sector_half_angle_deg=30.0,
                            object_sector_min_front_clearance=0.05,
                        ),
                    ),
                },
            ),
        },
    ),
    training=replace(
        r1_student.training,
        project="fastsac",
        name="r1_fastsac",
        num_envs=64,
    ),
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            gamma=0.99,
            num_steps=1,
            num_updates=4,
            policy_frequency=2,
            alpha_init=0.001,
            alpha_learning_rate=3e-4,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
            actor_obs_keys=["actor_obs", "ae_latent"],
            critic_obs_keys=["critic_obs"],
        ),
    ),
    reward=reward.r1_26dof_fastsac_reward,
    observation=replace(
        observation.r1_26dof_fastsac_observation,
        groups={
            **observation.r1_26dof_fastsac_observation.groups,
            # Without an AE checkpoint, resolve_observation_term_overrides
            # swaps this private marker into actor_obs and switches FastSAC to
            # actor_obs_keys=["actor_obs"].  This keeps no-AE student replay
            # tensor-compatible with a no-AE r1-fastsac run.
            "direct_ir_actor_obs": observation.r1_student_direct_ir_actor_obs,
            "critic_obs": replace(
                observation.r1_student_privileged_critic_obs,
                terms={
                    **observation.r1_student_privileged_critic_obs.terms,
                    "previous_action": replace(
                        observation.r1_student_privileged_critic_obs.terms["previous_action"],
                        func="holosoma.managers.observation.terms.wbt:actions",
                    ),
                },
            ),
        },
    ),
    termination=termination.r1_26dof_fastsac_termination,
    randomization=randomization.r1_26dof_fastsac_randomization,
)

r1_final_ppo = replace(
    r1_fastsac,
    object_scale_curriculum_level=0,
    simulator=replace(
        r1_fastsac.simulator,
        config=replace(r1_fastsac.simulator.config, enable_robot_depth_camera=False),
    ),
    training=replace(
        r1_fastsac.training,
        project="final-ppo",
        name="r1_final_ppo",
        num_envs=256,
    ),
    algo=replace(
        algo.ppo,
        _target_="holosoma.agents.ppo.student_initialized_ppo.StudentInitializedPPO",
        config=replace(
            algo.ppo.config,
            num_learning_iterations=100000,
            num_steps_per_env=32,
            num_learning_epochs=3,
            num_mini_batches=8,
            actor_learning_rate=1e-5,
            critic_learning_rate=3e-4,
            entropy_coef=0.001,
            init_noise_std=0.1,
            init_at_random_ep_len=False,
            use_symmetry=False,
            save_interval=1000,
            module_dict=replace(
                algo.ppo.config.module_dict,
                actor=replace(
                    algo.ppo.config.module_dict.actor,
                    input_dim=["actor_obs"],
                    layer_config=replace(
                        algo.ppo.config.module_dict.actor.layer_config,
                        hidden_dims=[512, 256, 128],
                    ),
                ),
                critic=replace(
                    algo.ppo.config.module_dict.critic,
                    input_dim=["critic_obs"],
                    layer_config=replace(
                        algo.ppo.config.module_dict.critic.layer_config,
                        hidden_dims=[512, 256, 128],
                    ),
                ),
            ),
        ),
    ),
    observation=replace(
        observation.r1_student_direct_ir_observation,
        groups={
            "actor_obs": replace(
                observation.r1_student_direct_ir_observation.groups["actor_obs"],
                terms={
                    **observation.r1_student_direct_ir_observation.groups["actor_obs"].terms,
                    "previous_action": replace(
                        observation.r1_student_direct_ir_observation.groups["actor_obs"].terms[
                            "previous_action"
                        ],
                        func="holosoma.managers.observation.terms.wbt:actions",
                    ),
                },
            ),
            "critic_obs": replace(
                observation.r1_student_direct_ir_observation.groups["critic_obs"],
                terms={
                    **observation.r1_student_direct_ir_observation.groups["critic_obs"].terms,
                    "previous_action": replace(
                        observation.r1_student_direct_ir_observation.groups["critic_obs"].terms[
                            "previous_action"
                        ],
                        func="holosoma.managers.observation.terms.wbt:actions",
                    ),
                },
            ),
        },
    ),
    curriculum=curriculum.r1_26dof_final_ppo_curriculum,
)

__all__ = ["r1_fastsac", "r1_final_ppo", "r1_student", "r1_teacher"]
