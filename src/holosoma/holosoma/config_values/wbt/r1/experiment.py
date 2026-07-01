"""Whole Body Tracking experiment presets for the R1 robot."""

from dataclasses import replace

from holosoma.config_types.video import CartesianCameraConfig
from holosoma.config_values import action, logger as logger_values, robot, simulator
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
        interval=100,
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
    action=action.r1_26dof_joint_pos,
    command=command.r1_26dof_wbt_command_w_object_multi_teacher,
    termination=termination.r1_26dof_wbt_termination,
    randomization=randomization.r1_26dof_wbt_randomization_w_object,
    observation=observation.r1_26dof_wbt_observation_w_object_multi_teacher,
    reward=reward.r1_26dof_wbt_reward_w_object_multi_teacher,
    curriculum=curriculum.r1_26dof_wbt_curriculum,
)

__all__ = ["r1_teacher"]
