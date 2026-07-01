"""Whole Body Tracking command presets for the R1 robot."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

R1_WBT_BODY_NAMES_TO_TRACK = [
    "pelvis_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
]

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.1, 0.1, 0.05],
    root_ang_vel=[0.1, 0.1, 0.1],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file="",
    body_names_to_track=R1_WBT_BODY_NAMES_TO_TRACK,
    body_name_ref=["waist_yaw_link"],
    use_adaptive_timesteps_sampler=False,
    noise_to_initial_pose=init_pose_config,
)

motion_config_w_object_multi_teacher = replace(
    motion_config,
    motion_file="",
    motion_folder="train_r1/rl",
    start_at_timestep_zero_prob=0.8,
    stable_state_reset_ratio=0.3,
    stable_state_reset_adaptive=True,
    stable_state_reset_warmup_steps=24000,
    stable_state_reset_min_alive_steps=48,
    stable_state_reset_update_interval=48,
    stable_state_reset_pool_size=65536,
    stable_state_reset_max_updates_per_step=128,
    stable_state_reset_min_pool_fill_ratio=0.02,
    stable_state_reset_full_pool_fill_ratio=0.10,
    stable_state_reset_min_average_episode_length=96.0,
    stable_state_reset_full_average_episode_length=240.0,
    stable_state_reset_bad_tracking_rate_threshold=0.35,
    stable_state_reset_bad_tracking_ema_alpha=0.05,
)

r1_26dof_wbt_command_w_object_multi_teacher = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object_multi_teacher,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)

__all__ = [
    "R1_WBT_BODY_NAMES_TO_TRACK",
    "r1_26dof_wbt_command_w_object_multi_teacher",
]
