"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""

    func: str
    """Import path for the command hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across command hooks."""

    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)


########################################################################################################################
# Motion command configuration
########################################################################################################################
@dataclass(frozen=True)
class NoiseToInitialPoseConfig:
    """Initial pose of the robot and object to those in the motion file."""

    overall_noise_scale: float = 0.0
    """Overall noise scale for the initial pose."""

    dof_pos: float = 0.0
    """Noise scale for the initial dof position."""

    root_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root position x, y, z."""

    root_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root rotation roll, pitch, yaw."""

    root_lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root linear velocity vx, vy, vz."""

    root_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root angular velocity wx, wy, wz."""

    object_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for object position x, y, z."""

    object_sector_radius: list[float] = field(default_factory=lambda: [0.0, 0.0])
    """Min/max XY displacement from the reference object pose in a robot-forward sector."""

    object_sector_half_angle_deg: float = 0.0
    """Half-angle of object spawn sector around the reference robot forward axis."""

    object_sector_min_front_clearance: float = 0.05
    """Minimum forward-plane distance between robot root and randomized object."""


@dataclass(frozen=True)
class MotionConfig:
    """Motion related configuration for Whole Body Tracking.

    NOTE:
    - Motion file is assumed to be in the format of:
      - joint_pos: (T, J)
      - joint_vel: (T, J)

      - body_pos_w: (T, B, 3)
      - body_quat_w: (T, B, 4) # wxyz -> xyzw
      - body_lin_vel_w: (T, B, 3)
      - body_ang_vel_w: (T, B, 3)

      If object is present in the motion file, it is assumed to be in the format of:
      - object_pos_w: (T, 3)
      - object_quat_w: (T, 4)
      - object_lin_vel_w: (T, 3)
      - object_ang_vel_w: (T, 3)

      If the motion clip assumes a terrain, the terrain has to be specified in holosoma/config/terrain/terrain_wbt.yaml
    """

    body_name_ref: list[str]
    """Body name of the reference frame (in general, torso_link). """
    body_names_to_track: list[str]
    """Key body names to track, used for reward/termination computation."""

    motion_file: str = ""
    """Motion file (.npz) that contains motion_clips to track. Either motion_file or motion_folder must be provided."""

    motion_folder: str = ""
    """Motion folder containing multiple .npz files to concatenate.

    Either motion_file or motion_folder must be provided.
    """

    contact_file: str = ""
    """Optional .npz file containing contact labels for motion_file.

    If empty, contact labels are read from motion_file when present.
    """

    object_contact_threshold: float | None = None
    """Optional shared object-contact threshold for contact reward and observations.

    If None, reward and observation term defaults are used.
    """

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    adaptive_timestep_lookback_steps: int = 0
    """When adaptive timestep sampling is enabled, start this many frames before the sampled hard timestep."""

    adaptive_timestep_sampling_ratio: float = 0.0
    """Fraction of resets sampled from adaptive hard timesteps; remaining non-start-zero resets stay uniform random."""

    hard_motion_sampling_ratio: float = 0.0
    """Fraction of reset environments sampled from low-completion motions.

    The remaining reset environments are sampled uniformly. A value of 0.5 means that roughly half of the reset batch
    uses hard-motion weighted sampling, while the other half keeps uniform motion exploration.
    """

    hard_motion_sampling_ema_alpha: float = 0.1
    """EMA update rate for per-motion start0 completion percent used by hard-motion sampling."""

    start_at_timestep_zero_prob: float = 0.2
    """Probability of starting at timestep zero."""

    adaptive_phase_zero: bool = False
    """Hold the interaction-start reference frame until the live robot reaches its shifted start pose."""

    phase_zero_position_threshold_m: float = 0.15
    phase_zero_yaw_threshold_rad: float = 0.35
    phase_zero_ready_hold_steps: int = 8
    phase_zero_linear_velocity_gain: float = 1.5
    phase_zero_angular_velocity_gain: float = 1.5
    phase_zero_max_linear_velocity: float = 0.6
    phase_zero_max_angular_velocity: float = 0.8

    freeze_at_timestep_zero_prob: float = 0.0
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    enable_default_pose_prepend: bool = True
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True."""

    enable_default_pose_append: bool = True
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)
