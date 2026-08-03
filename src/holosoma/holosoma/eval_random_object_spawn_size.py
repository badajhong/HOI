"""Visualize R1 objects with random forward-sector positions and random sizes."""

from __future__ import annotations

import dataclasses
import math
import sys
from typing import Any, Literal

import tyro
from loguru import logger
from pydantic.dataclasses import dataclass

from holosoma.config_types.env import resolve_observation_term_overrides
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.config_values.wbt.r1.observation import r1_26dof_wbt_observation_w_object_multi_teacher
from holosoma.eval_randomize_object import (
    _apply_motion_overrides,
    _apply_object_overrides,
    _ensure_eval_runtime_randomization_defaults,
    _resolve_device,
    _with_default_experiment,
    _write_sim_data,
    _zero_active_object_velocity,
    _zero_pose_noise,
)
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.object_urdf import resolve_multi_object_urdf_config
from holosoma.utils.rotations import quat_apply_yaw
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


DEFAULT_EXPERIMENT = "exp:r1-fastsac"
DEFAULT_MOTION_FOLDER = "train_r1/rl"
DEFAULT_OBJECT_ASSET = "train_r1/objects"


@dataclass(frozen=True)
class SectorSpawnViewerConfig:
    """Configuration for random position-and-size spawn visualization."""

    motion_file: str | None = None
    motion_folder: str | None = DEFAULT_MOTION_FOLDER
    object_urdf_path: str | None = None
    object_urdf_asset: str | None = DEFAULT_OBJECT_ASSET
    start_at_timestep_zero_prob: float = 1.0
    """Always reset each selected motion at its first frame for a stable sector center."""

    radius_min_m: float = 0.0
    """Minimum displacement from the original reference object position."""

    radius_max_m: float = 0.50
    """Maximum displacement from the original reference object position."""

    sector_half_angle_deg: float = 30.0
    """Half-angle around the robot forward direction; total sector angle is twice this value."""

    num_radial_samples: int = 4
    num_angular_samples: int = 7
    num_envs: int = 16
    randomize_spawn_location: bool = True
    """Sample a new area-uniform sector position at every reset."""

    object_volume_ratio_min: float = 0.6
    object_volume_ratio_max: float = 1.4
    """Linear env-ordered volume-ratio range applied at simulator startup."""
    min_front_clearance_m: float = 0.05
    """Minimum signed forward distance of the spawned object from the robot root."""

    headless: bool = False
    max_steps: int | None = None
    freeze_after_control_steps: int | None = 2
    """Stop physics after this many control steps; None keeps physics running."""
    reset_interval_s: float | None = 5.0
    env_spacing: float = 3.0
    zero_object_velocity: bool = True

    draw_debug_markers: bool = True
    default_experiment: str = DEFAULT_EXPERIMENT
    device: Literal["cpu", "gpu"] = "gpu"


def sector_grid_offsets(
    num_envs: int,
    radius_min_m: float,
    radius_max_m: float,
    sector_half_angle_deg: float,
    num_radial_samples: int,
    num_angular_samples: int,
    *,
    device: str,
    sample_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic robot-frame XY offsets and their radius/angle labels."""
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")
    if num_radial_samples <= 0 or num_angular_samples <= 0:
        raise ValueError("num_radial_samples and num_angular_samples must be positive.")
    if radius_min_m < 0.0 or radius_max_m < radius_min_m:
        raise ValueError(
            f"Expected 0 <= radius_min_m <= radius_max_m, got {radius_min_m}, {radius_max_m}."
        )
    if not 0.0 <= sector_half_angle_deg < 90.0:
        raise ValueError("sector_half_angle_deg must be in [0, 90) to keep offsets forward-facing.")

    radii = torch.linspace(radius_min_m, radius_max_m, num_radial_samples, device=device)
    angles = torch.linspace(
        -math.radians(sector_half_angle_deg),
        math.radians(sector_half_angle_deg),
        num_angular_samples,
        device=device,
    )
    grid_radius, grid_angle = torch.meshgrid(radii, angles, indexing="ij")
    grid_radius = grid_radius.reshape(-1)
    grid_angle = grid_angle.reshape(-1)
    num_grid_points = int(grid_radius.numel())
    if num_envs <= num_grid_points:
        # Cover the whole sector even when only a few environments are used.
        indices = torch.linspace(0, num_grid_points - 1, num_envs, device=device).round().long()
    else:
        indices = torch.arange(num_envs, device=device) % num_grid_points
    indices = (indices + sample_offset) % num_grid_points
    selected_radius = grid_radius[indices]
    selected_angle = grid_angle[indices]
    offsets = torch.stack(
        (
            selected_radius * torch.cos(selected_angle),
            selected_radius * torch.sin(selected_angle),
        ),
        dim=-1,
    )
    return offsets, selected_radius, selected_angle


def _viewer_overrides(config: ExperimentConfig, cli_cfg: SectorSpawnViewerConfig) -> ExperimentConfig:
    config = _apply_motion_overrides(config, cli_cfg)  # type: ignore[arg-type]
    config = _apply_object_overrides(config, cli_cfg)  # type: ignore[arg-type]
    config = _zero_pose_noise(config, robot=True, obj=True)
    if cli_cfg.object_volume_ratio_min <= 0.0 or cli_cfg.object_volume_ratio_max < cli_cfg.object_volume_ratio_min:
        raise ValueError(
            "Expected 0 < object_volume_ratio_min <= object_volume_ratio_max, got "
            f"{cli_cfg.object_volume_ratio_min}, {cli_cfg.object_volume_ratio_max}."
        )
    scale_randomization = RandomizationManagerCfg(
        setup_terms={
            "set_object_scale_grid_startup": RandomizationTermCfg(
                func="holosoma.eval_randomize_object:set_object_scale_grid_startup",
                params={
                    "scale_min": cli_cfg.object_volume_ratio_min,
                    "scale_max": cli_cfg.object_volume_ratio_max,
                    "num_scales": cli_cfg.num_envs,
                    "envs_per_scale": 1,
                    "object_height": 0.0,
                    "enabled": True,
                    "reset_physics_after_usd_edit": True,
                },
            )
        }
    )
    return dataclasses.replace(
        config,
        teacher=None,
        student=None,
        ir_ae=None,
        ir_ae_body_source=None,
        di_ae=None,
        di_pro_ae=None,
        training=dataclasses.replace(
            config.training,
            num_envs=cli_cfg.num_envs,
            headless=cli_cfg.headless,
            max_eval_steps=cli_cfg.max_steps,
            export_onnx=False,
        ),
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                config.simulator.config,
                scene=dataclasses.replace(
                    config.simulator.config.scene,
                    env_spacing=cli_cfg.env_spacing,
                    replicate_physics=False,
                ),
            ),
        ),
        observation=r1_26dof_wbt_observation_w_object_multi_teacher,
        randomization=scale_randomization,
    )


def _spawn_sector_grid(env: Any, cli_cfg: SectorSpawnViewerConfig) -> None:
    command = env.command_manager.get_state("motion_command")
    if command is None or not getattr(command.motion, "has_object", False):
        raise RuntimeError("Forward-sector spawn viewer requires a MotionCommand with object motion.")

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    reference_object_position = command.object_pos_w.clone()
    current_states = command._active_object_states_w().clone()
    # Preserve grounding/scale compensation already applied by MotionCommand at reset.
    sector_centers = current_states[:, :3].clone()
    robot_positions = command.robot_root_pos_w.clone()
    forward_body = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    forward_body[:, 0] = 1.0
    forward_world = quat_apply_yaw(command.robot_root_quat_w, forward_body, w_last=True)
    forward_world[:, 2] = 0.0
    forward_world = forward_world / torch.linalg.norm(forward_world, dim=-1, keepdim=True).clamp_min(1e-6)
    left_world = torch.stack((-forward_world[:, 1], forward_world[:, 0], torch.zeros(env.num_envs, device=env.device)), dim=-1)

    sample_offset = int(getattr(env, "_sector_spawn_sample_offset", 0))
    if cli_cfg.randomize_spawn_location:
        if cli_cfg.radius_min_m < 0.0 or cli_cfg.radius_max_m < cli_cfg.radius_min_m:
            raise ValueError("Expected 0 <= radius_min_m <= radius_max_m.")
        if not 0.0 <= cli_cfg.sector_half_angle_deg < 90.0:
            raise ValueError("sector_half_angle_deg must be in [0, 90).")
        # Area-uniform sector sampling avoids concentrating samples near the
        # original reference object position.
        radii = torch.sqrt(
            torch.rand(env.num_envs, device=env.device)
            * (cli_cfg.radius_max_m**2 - cli_cfg.radius_min_m**2)
            + cli_cfg.radius_min_m**2
        )
        half_angle = math.radians(cli_cfg.sector_half_angle_deg)
        angles = (torch.rand(env.num_envs, device=env.device) * 2.0 - 1.0) * half_angle
        offsets_robot = torch.stack((radii * torch.cos(angles), radii * torch.sin(angles)), dim=-1)
    else:
        offsets_robot, radii, angles = sector_grid_offsets(
            env.num_envs,
            cli_cfg.radius_min_m,
            cli_cfg.radius_max_m,
            cli_cfg.sector_half_angle_deg,
            cli_cfg.num_radial_samples,
            cli_cfg.num_angular_samples,
            device=env.device,
            sample_offset=sample_offset,
        )
    env._sector_spawn_sample_offset = sample_offset + env.num_envs
    offsets_world = forward_world * offsets_robot[:, :1] + left_world * offsets_robot[:, 1:2]
    target_positions = sector_centers + offsets_world
    signed_front_distance = torch.sum(
        (target_positions[:, :2] - robot_positions[:, :2]) * forward_world[:, :2], dim=-1
    )
    correction = torch.clamp(cli_cfg.min_front_clearance_m - signed_front_distance, min=0.0)
    target_positions[:, :2] += correction[:, None] * forward_world[:, :2]
    signed_front_distance += correction

    current_states[:, :3] = target_positions
    current_states[:, 7:13] = 0.0
    command.set_simulator_object_states(env_ids, current_states)
    command.object_pos_reward_offset[env_ids] = target_positions - reference_object_position
    env._sector_spawn_centers_w = sector_centers
    env._sector_spawn_targets_w = target_positions
    # set_simulator_object_states() already performs the object-only write.
    # Do not write the entire scene here because that can overwrite the new pose.
    env.simulator.refresh_sim_tensors()
    actual_positions = command._active_object_states_w()[:, :3]
    spawn_error = torch.linalg.norm(actual_positions - target_positions, dim=-1)
    if torch.any(spawn_error > 1e-3):
        bad_ids = (spawn_error > 1e-3).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(
            "Object sector spawn was not applied to the simulator: "
            f"env_ids={bad_ids}, max_error={float(spawn_error.max().item()):.6f} m."
        )

    for env_id in range(env.num_envs):
        scale = getattr(env, "object_scale_factors", None)
        volume_ratio = float(scale[env_id].prod().item()) if scale is not None else 1.0
        size_deltas = getattr(env, "object_scale_xy_center_delta", None)
        size_delta = float(size_deltas[env_id].item()) if size_deltas is not None else 0.0
        logger.info(
            f"[sector_spawn] env={env_id:03d} volume_ratio={volume_ratio:.3f} "
            f"size_center_delta={size_delta:+.4f}m "
            f"radius={float(radii[env_id]):.3f}m "
            f"angle={math.degrees(float(angles[env_id])):+.1f}deg "
            f"front_distance={float(signed_front_distance[env_id]):.3f}m "
            f"center={sector_centers[env_id, :2].tolist()} target={target_positions[env_id, :2].tolist()}"
        )


def _reset_and_spawn(env: Any, cli_cfg: SectorSpawnViewerConfig) -> None:
    env.set_is_evaluating()
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_envs_idx(env_ids)
    refresh = getattr(env, "_refresh_envs_after_reset", None)
    if callable(refresh):
        refresh(env_ids)
    else:
        _write_sim_data(env)
    _spawn_sector_grid(env, cli_cfg)
    if cli_cfg.zero_object_velocity:
        _zero_active_object_velocity(env, env_ids)
    # Commit one physics/Fabric update before the first GUI frame.  Merely
    # changing an IsaacLab state tensor can otherwise leave the viewport on the
    # previous transform until a simulation step occurs.
    env.simulator.simulate_at_each_physics_step()
    env.simulator.refresh_sim_tensors()

    command = env.command_manager.get_state("motion_command")
    actual_positions = command._active_object_states_w()[:, :3]
    targets = env._sector_spawn_targets_w
    xy_error = torch.linalg.norm(actual_positions[:, :2] - targets[:, :2], dim=-1)
    logger.info(
        "[sector_spawn_visible] first physics frame committed; "
        f"max_xy_error={float(xy_error.max().item()):.4f}m, "
        f"env0_actual={actual_positions[0, :3].tolist()}, env0_target={targets[0, :3].tolist()}"
    )
    env.reset_buf[env_ids] = 0


def _run(env: Any, cli_cfg: SectorSpawnViewerConfig) -> None:
    actions = torch.zeros(env.num_envs, env.dim_actions, device=env.device)
    if env.action_manager is not None:
        env.action_manager.process_actions(actions)
    reset_steps = None
    if cli_cfg.reset_interval_s is not None and cli_cfg.reset_interval_s > 0.0:
        reset_steps = max(1, round(cli_cfg.reset_interval_s / env.dt))
    freeze_steps = cli_cfg.freeze_after_control_steps
    if freeze_steps is not None and freeze_steps < 0:
        raise ValueError(
            "freeze_after_control_steps must be non-negative or None, "
            f"got {cli_cfg.freeze_after_control_steps}."
        )

    logger.info(
        "Running random object position-and-size viewer. "
        f"Physics freeze_after_control_steps={freeze_steps}. Press Ctrl+C to stop."
    )
    step = 0
    render_step = 0
    physics_frozen = freeze_steps == 0
    try:
        while cli_cfg.max_steps is None or render_step < cli_cfg.max_steps:
            env.render(sync_frame_time=True)
            render_step += 1
            if not physics_frozen:
                for _ in range(env.simulator.simulator_config.sim.control_decimation):
                    if env.action_manager is not None:
                        env.action_manager.apply_actions()
                    env.simulator.simulate_at_each_physics_step()
                env.simulator.refresh_sim_tensors()
            if cli_cfg.draw_debug_markers and hasattr(env, "_draw_debug_vis"):
                env._draw_debug_vis()
            if physics_frozen:
                continue
            step += 1
            if freeze_steps is not None and step >= freeze_steps:
                physics_frozen = True
                logger.info(
                    f"Physics frozen at control_step={step}; "
                    "gravity and contacts will no longer advance."
                )
                continue
            if reset_steps is not None and step % reset_steps == 0:
                _reset_and_spawn(env, cli_cfg)
                if env.action_manager is not None:
                    env.action_manager.process_actions(actions)
    except KeyboardInterrupt:
        logger.info("Forward-sector spawn viewer interrupted.")


def main() -> None:
    init_eval_logging()
    cli_cfg, remaining = tyro.cli(SectorSpawnViewerConfig, return_unknown_args=True, add_help=False)
    experiment_args = _with_default_experiment(remaining, cli_cfg.default_experiment)
    config = tyro.cli(
        AnnotatedExperimentConfig,
        args=experiment_args,
        description="R1 random forward-sector position and object-size visualization.",
        config=TYRO_CONIFG,
    )
    config = resolve_multi_object_urdf_config(_viewer_overrides(config, cli_cfg))
    config = resolve_observation_term_overrides(config)
    sys.argv = [sys.argv[0]]
    env, _, simulation_app = setup_simulation_environment(config, device=_resolve_device(cli_cfg))
    try:
        _ensure_eval_runtime_randomization_defaults(env)
        _reset_and_spawn(env, cli_cfg)
        _run(env, cli_cfg)
    except Exception:
        # Log before closing IsaacSim: shutdown can be slow and otherwise hides
        # the exception that prevented the first rendered frame.
        logger.exception("Random object position-and-size viewer failed.")
        raise
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--motion-folder", "train_r1/rl",
        "--object-urdf-asset", "train_r1/objects",
        "--num-envs", "16",
        "--radius-min-m", "0.0",
        "--radius-max-m", "0.5",
        "--sector-half-angle-deg", "30",
        "--object-volume-ratio-min", "0.6",
        "--object-volume-ratio-max", "1.4",
        "--freeze-after-control-steps", "2",
        "--device", "gpu",
        "exp:r1-fastsac",
    ]
    main()
