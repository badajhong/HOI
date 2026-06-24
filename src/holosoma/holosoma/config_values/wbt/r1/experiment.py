"""Whole Body Tracking experiment presets for the R1 robot."""

from dataclasses import replace

from holosoma.config_values import action, robot
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object_multi_teacher
from holosoma.config_values.wbt.r1 import command, curriculum, observation, randomization, reward, termination

r1_teacher = replace(
    g1_29dof_wbt_w_object_multi_teacher,
    training=replace(
        g1_29dof_wbt_w_object_multi_teacher.training,
        name="r1_teacher",
        num_envs=16384,
    ),
    robot=replace(
        robot.r1_26dof_w_object_multi_teacher,
        asset=replace(
            robot.r1_26dof_w_object_multi_teacher.asset,
            enable_self_collisions=True,
        ),
        object=replace(
            robot.r1_26dof_w_object_multi_teacher.object,
            object_urdf_path="train_r1/objects/largebox/largebox.urdf",
            object_parm="train_r1/objects/objects_parm.yaml",
        ),
        init_state=replace(robot.r1_26dof_w_object_multi_teacher.init_state, pos=[0.0, 0.0, 0.72]),
    ),
    action=action.r1_26dof_joint_pos,
    command=command.r1_26dof_wbt_command_w_object_multi_teacher,
    termination=termination.r1_26dof_wbt_termination,
    randomization=randomization.r1_26dof_wbt_randomization_w_object,
    observation=observation.r1_26dof_wbt_observation_w_object_multi_teacher,
    reward=reward.r1_26dof_wbt_reward_w_object_multi_teacher,
    curriculum=curriculum.r1_26dof_wbt_curriculum,
)

__all__ = ["r1_teacher"]
