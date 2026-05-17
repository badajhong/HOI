"""Task configuration types for holosoma_inference."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class DebugConfig:
    """Debug overrides for quick testing."""

    force_upright_imu: bool = False
    """Override projected_gravity with [0, 0, -1] (perfectly upright)."""

    force_zero_angular_velocity: bool = False
    """Override base_ang_vel with [0, 0, 0]."""

    force_zero_action: bool = False
    """Zero out the scaled policy action (robot holds default pose)."""


@dataclass(frozen=True)
class TaskConfig:
    """Task execution configuration for policy inference."""

    model_path: str | list[str]
    """Path to ONNX model(s). Supports local paths and wandb:// URIs. Required field."""

    rl_rate: float = 50
    """Policy inference rate in Hz."""

    policy_action_scale: float = 0.25
    """Scaling factor applied to policy actions."""

    use_phase: bool = True
    """Whether to use gait phase observations."""

    gait_period: float = 1.0
    """Gait cycle period in seconds."""

    domain_id: int = 0
    """DDS domain ID for communication."""

    interface: str = "auto"
    """Network interface name. Use ``"auto"`` to auto-detect, or specify explicitly (e.g. ``"eth0"``)."""

    use_joystick: bool = False
    """Enable joystick control input."""

    joystick_type: str = "xbox"
    """Joystick type."""

    joystick_device: int = 0
    """Joystick device index."""

    use_sim_time: bool = False
    """Use synchronized simulation time for WBT policies."""

    wandb_download_dir: str = "/tmp"
    """Directory for downloading W&B checkpoints."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    use_ros: bool = False
    """Use ROS2 for rate limiting."""

    print_observations: bool = False
    """Print observation vectors for debugging."""

    motion_start_timestep: int = 0
    """Starting timestep for motion clip playback."""

    motion_end_timestep: int | None = None
    """Ending timestep for motion clip playback. If None, plays until the end."""

    di_model_path: str = ""
    """Legacy override for older two-stage exports that still use a separate DI encoder ONNX."""

    di_latent_path: str = ""
    """Legacy override for older two-stage exports that inject a precomputed DI latent vector."""

    depth_window_npy_path: str = ""
    """Optional path to a `.npy` depth window for fused WBT ONNX offline testing."""

    depth_window_zmq_port: int = 5556
    """ZMQ port used to receive simulator depth frames for fused WBT ONNX inference."""

    depth_window_zmq_host: str = "localhost"
    """Host used to receive simulator depth frames."""

    depth_window_zmq_timeout_ms: int = 1000
    """Initial receive timeout for the first simulator depth frame."""

    show_depth_window: bool = False
    """Show the latest simulator depth frame in an OpenCV debug window."""

    depth_window_display_scale: int = 4
    """Integer scale factor for the live depth debug window."""

    log_depth_window_input_stats: bool = True
    """Log the exact depth/proprioception slices that are fed into fused ONNX once per motion start."""

    depth_window_debug_npy_path: str = ""
    """Optional path to save the first live depth window that is fed into the fused ONNX model."""

    allow_missing_depth_window: bool = False
    """Allow fused WBT ONNX to run with zero depth when no live/static depth window is available."""

    object_type_id: int = 0
    """Object type index used to build `obj_type_one_hot` for residual WBT exports."""

    debug: DebugConfig = DebugConfig()
    """Debug overrides for quick testing."""
