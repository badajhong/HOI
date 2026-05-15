from __future__ import annotations

import dataclasses
import sys
from typing import Any, Literal

import tyro
from loguru import logger
from pydantic.dataclasses import dataclass

from holosoma.config_types.command import MotionConfig
from holosoma.config_types.env import resolve_observation_term_overrides
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.config_values.wbt.g1 import observation as wbt_observation
from holosoma.managers.randomization.terms.locomotion import randomize_object_scale_startup
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.object_urdf import resolve_multi_object_urdf_config
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG

DEFAULT_EXPERIMENT = "exp:g1-29dof-wbt-w-object-multi-res"
DEFAULT_MOTION_FILE = (
    "/home/rllab/haechan/holosoma/train/rl/suitcase/sub1_suitcase_001.npz"
)
DEFAULT_OBJECT_URDF_PATH = (
    "/home/rllab/haechan/holosoma/src/holosoma_retargeting/holosoma_retargeting/"
    "models/objects/suitcase/suitcase.urdf"
)
DEFAULT_VOLUME_RATIOS = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)


@dataclass(frozen=True)
class RandomizeObjectSpawnConfig:
    """Visualize initial object spawning with deterministic object volume ratios."""

    motion_file: str | None = DEFAULT_MOTION_FILE
    """Motion .npz used only to place the robot/object at the initial frame."""

    motion_folder: str | None = None
    """Motion folder for multi-object runs. Clears motion_file when provided."""

    object_urdf_path: str | None = DEFAULT_OBJECT_URDF_PATH
    """URDF path for a single object."""

    object_urdf_asset: str | None = None
    """Folder containing object URDFs for multi-object motion folders."""

    scale_min: float = DEFAULT_VOLUME_RATIOS[0]
    """Smallest object volume ratio to visualize."""

    scale_max: float = DEFAULT_VOLUME_RATIOS[-1]
    """Largest object volume ratio to visualize."""

    num_scales: int = len(DEFAULT_VOLUME_RATIOS)
    """Number of evenly spaced volume ratios between scale_min and scale_max."""

    envs_per_scale: int = 1
    """Number of consecutive environments assigned to each scale before repeating."""

    num_envs: int = len(DEFAULT_VOLUME_RATIOS)
    """Number of environments. Default shows one env for each volume ratio."""

    headless: bool = False
    """Run with an interactive IsaacSim viewer by default."""

    max_steps: int | None = None
    """Number of simulation control steps to run. None runs until interrupted."""

    env_spacing: float = 2.0
    """Distance between environments in the IsaacSim grid."""

    start_at_timestep_zero_prob: float = 1.0
    """Force reset to the initial frame of the chosen motion clip."""

    zero_object_velocity: bool = True
    """Zero object linear/angular velocity after initial placement so only gravity starts it moving."""

    reset_interval_s: float | None = 5.0
    """Reset all environments to the initial spawn every this many simulated seconds. None disables it."""

    draw_debug_markers: bool = True
    """Draw the WBT reference/current debug markers when the viewer is available."""

    disable_object_randomization: bool = True
    """Disable object material/mass/inertia noise; the deterministic volume-ratio grid remains enabled."""

    disable_robot_randomization: bool = True
    """Disable robot/push randomization for a cleaner spawn visualization."""

    default_experiment: str = DEFAULT_EXPERIMENT
    """Experiment subcommand to use when no exp:* subcommand is supplied."""

    device: Literal["cpu", "gpu"] = "cpu"
    """Simulation device choice. Use 'gpu' for cuda:0."""


def _volume_ratios(scale_min: float, scale_max: float, num_scales: int) -> list[float]:
    if num_scales <= 0:
        raise ValueError(f"num_scales must be positive, got {num_scales}.")
    if num_scales == 1:
        return [float(scale_min)]
    return [
        float(scale_min + (scale_max - scale_min) * i / (num_scales - 1))
        for i in range(num_scales)
    ]


def _volume_ratio_to_xyz_scale(volume_ratio: float) -> float:
    if volume_ratio <= 0.0:
        raise ValueError(f"Object volume ratio must be positive, got {volume_ratio}.")
    return volume_ratio ** (1.0 / 3.0)


def _replace_motion_config(motion_config: Any, updates: dict[str, Any]) -> Any:
    if isinstance(motion_config, MotionConfig):
        return dataclasses.replace(motion_config, **updates)
    if isinstance(motion_config, dict):
        updated_motion_config = dict(motion_config)
        updated_motion_config.update(updates)
        return updated_motion_config
    raise TypeError(f"Unsupported motion_config type: {type(motion_config).__name__}")


def _get_motion_config(config: ExperimentConfig) -> Any:
    if config.command is None:
        raise ValueError("This viewer requires a command config with setup_terms.motion_command.")

    setup_term = config.command.setup_terms.get("motion_command")
    if setup_term is None or "motion_config" not in setup_term.params:
        raise ValueError("This viewer requires setup_terms.motion_command.params.motion_config.")
    return setup_term.params["motion_config"]


def _apply_motion_overrides(config: ExperimentConfig, cli_cfg: RandomizeObjectSpawnConfig) -> ExperimentConfig:
    if cli_cfg.motion_file is not None and cli_cfg.motion_folder is not None:
        raise ValueError("Choose either --motion-file or --motion-folder, not both.")

    setup_term = config.command.setup_terms["motion_command"]
    params = dict(setup_term.params)
    motion_updates: dict[str, Any] = {
        "start_at_timestep_zero_prob": cli_cfg.start_at_timestep_zero_prob,
    }
    if cli_cfg.motion_file is not None:
        motion_updates["motion_file"] = cli_cfg.motion_file
        motion_updates["motion_folder"] = ""
    if cli_cfg.motion_folder is not None:
        motion_updates["motion_folder"] = cli_cfg.motion_folder
        motion_updates["motion_file"] = ""

    params["motion_config"] = _replace_motion_config(params["motion_config"], motion_updates)
    setup_terms = dict(config.command.setup_terms)
    setup_terms["motion_command"] = dataclasses.replace(setup_term, params=params)
    updated_config = dataclasses.replace(
        config,
        command=dataclasses.replace(config.command, setup_terms=setup_terms),
    )

    motion_config = _get_motion_config(updated_config)
    motion_file = motion_config.get("motion_file", "") if isinstance(motion_config, dict) else motion_config.motion_file
    motion_folder = (
        motion_config.get("motion_folder", "") if isinstance(motion_config, dict) else motion_config.motion_folder
    )
    if not motion_file and not motion_folder:
        raise ValueError("Pass --motion-file or --motion-folder so the initial object pose can be loaded.")

    return updated_config


def _apply_object_overrides(config: ExperimentConfig, cli_cfg: RandomizeObjectSpawnConfig) -> ExperimentConfig:
    if cli_cfg.object_urdf_path is not None and cli_cfg.object_urdf_asset is not None:
        raise ValueError("Choose either --object-urdf-path or --object-urdf-asset, not both.")
    if cli_cfg.object_urdf_path is None and cli_cfg.object_urdf_asset is None:
        return config

    if cli_cfg.object_urdf_asset is not None:
        object_updates = {
            "object_urdf_path": None,
            "object_urdf_asset": cli_cfg.object_urdf_asset,
            "object_urdf_folder": cli_cfg.object_urdf_asset,
            "object_urdf_name_to_path": {},
        }
    else:
        object_updates = {
            "object_urdf_path": cli_cfg.object_urdf_path,
            "object_urdf_asset": None,
            "object_urdf_folder": None,
            "object_urdf_name_to_path": {},
        }

    return dataclasses.replace(
        config,
        robot=dataclasses.replace(
            config.robot,
            object=dataclasses.replace(config.robot.object, **object_updates),
        ),
    )


def _is_object_randomization_term(term_name: str, term_cfg: Any) -> bool:
    func = getattr(term_cfg, "func", "")
    func_name = func.rsplit(":", 1)[-1] if isinstance(func, str) else ""
    return "object" in term_name.lower() or func_name in {
        "randomize_object_rigid_body_material_startup",
        "randomize_object_rigid_body_mass_startup",
        "randomize_object_rigid_body_inertia_startup",
        "randomize_object_scale_startup",
        "set_object_scale_grid_startup",
        "set_object_init_pose_noise",
    }


def _is_object_scale_term(term_name: str, term_cfg: Any) -> bool:
    func = getattr(term_cfg, "func", "")
    func_name = func.rsplit(":", 1)[-1] if isinstance(func, str) else ""
    return ("scale" in term_name.lower() and "object" in term_name.lower()) or func_name in {
        "randomize_object_scale_startup",
        "set_object_scale_grid_startup",
    }


def _is_robot_randomization_term(term_name: str, term_cfg: Any) -> bool:
    func = getattr(term_cfg, "func", "")
    func_name = func.rsplit(":", 1)[-1] if isinstance(func, str) else ""
    return term_name in {
        "push_randomizer_state",
        "actuator_randomizer_state",
        "setup_action_delay_buffers",
        "randomize_push_schedule",
        "randomize_action_delay",
        "randomize_dof_state",
        "apply_pushes",
    } or func_name in {
        "PushRandomizerState",
        "ActuatorRandomizerState",
        "setup_action_delay_buffers",
        "setup_dof_pos_bias",
        "randomize_robot_rigid_body_material_startup",
        "randomize_base_com_startup",
        "randomize_push_schedule",
        "randomize_action_delay",
        "randomize_dof_state",
        "apply_pushes",
    }


def _remove_randomization_terms(config: ExperimentConfig, predicate: Any) -> ExperimentConfig:
    if config.randomization is None:
        return config

    randomization = config.randomization
    updated_groups: dict[str, dict[str, Any]] = {}
    for group_name in ("setup_terms", "reset_terms", "step_terms"):
        terms = getattr(randomization, group_name, {}) or {}
        updated_groups[group_name] = {
            term_name: term_cfg
            for term_name, term_cfg in terms.items()
            if not predicate(term_name, term_cfg)
        }

    return dataclasses.replace(
        config,
        randomization=dataclasses.replace(randomization, **updated_groups),
    )


def _zero_pose_noise(config: ExperimentConfig, *, robot: bool, obj: bool) -> ExperimentConfig:
    setup_term = config.command.setup_terms["motion_command"]
    params = dict(setup_term.params)
    motion_config = params["motion_config"]

    noise_updates: dict[str, Any] = {}
    if robot:
        noise_updates.update(
            {
                "dof_pos": 0.0,
                "root_pos": [0.0, 0.0, 0.0],
                "root_rot": [0.0, 0.0, 0.0],
                "root_lin_vel": [0.0, 0.0, 0.0],
                "root_ang_vel": [0.0, 0.0, 0.0],
            }
        )
    if obj:
        noise_updates["object_pos"] = [0.0, 0.0, 0.0]

    if isinstance(motion_config, MotionConfig):
        noise_cfg = dataclasses.replace(motion_config.noise_to_initial_pose, **noise_updates)
        params["motion_config"] = dataclasses.replace(motion_config, noise_to_initial_pose=noise_cfg)
    elif isinstance(motion_config, dict):
        updated_motion_config = dict(motion_config)
        noise_cfg = updated_motion_config.get("noise_to_initial_pose")
        if noise_cfg is not None:
            if isinstance(noise_cfg, dict):
                updated_noise_cfg = dict(noise_cfg)
                updated_noise_cfg.update(noise_updates)
            else:
                updated_noise_cfg = dataclasses.replace(noise_cfg, **noise_updates)
            updated_motion_config["noise_to_initial_pose"] = updated_noise_cfg
        params["motion_config"] = updated_motion_config
    else:
        raise TypeError(f"Unsupported motion_config type: {type(motion_config).__name__}")

    setup_terms = dict(config.command.setup_terms)
    setup_terms["motion_command"] = dataclasses.replace(setup_term, params=params)
    return dataclasses.replace(config, command=dataclasses.replace(config.command, setup_terms=setup_terms))


def _reset_isaacsim_physics_after_usd_scale_edit(env: Any) -> None:
    """Rebuild IsaacSim tensor views after per-env USD scale edits."""
    simulator = getattr(env, "simulator", None)
    if simulator is None or simulator.__class__.__name__ != "IsaacSim":
        return

    sim = getattr(simulator, "sim", None)
    if sim is None or not hasattr(sim, "reset"):
        return

    logger.info("Resetting IsaacSim physics after startup object volume-ratio edits.")
    sim.reset()
    simulator.refresh_sim_tensors()


def set_object_scale_grid_startup(
    env: Any,
    env_ids: list[int] | torch.Tensor | None = None,
    *,
    scale_min: float,
    scale_max: float,
    num_scales: int,
    envs_per_scale: int,
    object_height: float | None = 0.0,
    enabled: bool = True,
    reset_physics_after_usd_edit: bool = True,
    **_: Any,
) -> None:
    """Assign deterministic per-env object volume ratios using randomize_object_scale_startup."""
    if not enabled:
        return
    if envs_per_scale <= 0:
        raise ValueError(f"envs_per_scale must be positive, got {envs_per_scale}.")

    selected_env_ids = (
        torch.arange(env.num_envs, dtype=torch.long)
        if env_ids is None
        else torch.as_tensor(env_ids, dtype=torch.long).to(device="cpu")
    )
    if selected_env_ids.numel() == 0:
        return

    volume_ratios = _volume_ratios(scale_min, scale_max, num_scales)
    for scale_index, volume_ratio in enumerate(volume_ratios):
        scale_env_ids = selected_env_ids[((selected_env_ids // envs_per_scale) % num_scales) == scale_index]
        if scale_env_ids.numel() == 0:
            continue

        randomize_object_scale_startup(
            env,
            env_ids=scale_env_ids,
            scale_value=volume_ratio,
            object_height=object_height,
            enabled=True,
        )
        xyz_scale = _volume_ratio_to_xyz_scale(volume_ratio)
        logger.info(
            f"Object volume ratio {volume_ratio:.3f} "
            f"(internal xyz scale {xyz_scale:.6f}) assigned to env ids: {scale_env_ids.tolist()}"
        )

    if reset_physics_after_usd_edit:
        _reset_isaacsim_physics_after_usd_scale_edit(env)


def _apply_scale_grid_randomization(
    config: ExperimentConfig,
    cli_cfg: RandomizeObjectSpawnConfig,
) -> ExperimentConfig:
    randomization = config.randomization or RandomizationManagerCfg()
    setup_terms = dict(randomization.setup_terms)
    setup_terms["set_object_scale_grid_startup"] = RandomizationTermCfg(
        func="holosoma.eval_randomize_object:set_object_scale_grid_startup",
        params={
            # scale_min/scale_max are kept as CLI names, but their values are object volume ratios.
            "scale_min": cli_cfg.scale_min,
            "scale_max": cli_cfg.scale_max,
            "num_scales": cli_cfg.num_scales,
            "envs_per_scale": cli_cfg.envs_per_scale,
            "object_height": 0.0,
            "enabled": True,
            "reset_physics_after_usd_edit": True,
        },
    )
    return dataclasses.replace(
        config,
        randomization=dataclasses.replace(randomization, setup_terms=setup_terms),
    )


def _apply_runtime_overrides(config: ExperimentConfig, cli_cfg: RandomizeObjectSpawnConfig) -> ExperimentConfig:
    training = dataclasses.replace(
        config.training,
        num_envs=cli_cfg.num_envs,
        headless=cli_cfg.headless,
        max_eval_steps=cli_cfg.max_steps,
        export_onnx=False,
    )
    scene = dataclasses.replace(
        config.simulator.config.scene,
        env_spacing=cli_cfg.env_spacing,
        replicate_physics=False,
    )
    simulator = dataclasses.replace(
        config.simulator,
        config=dataclasses.replace(config.simulator.config, scene=scene),
    )
    return dataclasses.replace(
        config,
        teacher=None,
        student=None,
        ir_ae=None,
        ir_ae_body_source=None,
        di_ae=None,
        di_pro_ae=None,
        training=training,
        simulator=simulator,
        observation=wbt_observation.g1_29dof_wbt_observation_w_object_multi,
    )


def apply_spawn_viewer_overrides(
    config: ExperimentConfig,
    cli_cfg: RandomizeObjectSpawnConfig,
) -> ExperimentConfig:
    if cli_cfg.num_envs < cli_cfg.num_scales * cli_cfg.envs_per_scale:
        raise ValueError(
            "num_envs must be at least num_scales * envs_per_scale "
            f"({cli_cfg.num_scales} * {cli_cfg.envs_per_scale}), got {cli_cfg.num_envs}."
        )

    _get_motion_config(config)
    config = _apply_motion_overrides(config, cli_cfg)
    config = _apply_object_overrides(config, cli_cfg)
    config = _apply_runtime_overrides(config, cli_cfg)

    if cli_cfg.disable_object_randomization:
        config = _remove_randomization_terms(config, _is_object_randomization_term)
        config = _zero_pose_noise(config, robot=False, obj=True)
    else:
        config = _remove_randomization_terms(config, _is_object_scale_term)

    if cli_cfg.disable_robot_randomization:
        config = _remove_randomization_terms(config, _is_robot_randomization_term)
        config = _zero_pose_noise(config, robot=True, obj=False)

    return _apply_scale_grid_randomization(config, cli_cfg)


def _resolve_device(cli_cfg: RandomizeObjectSpawnConfig) -> str | None:
    return "cuda:0" if cli_cfg.device == "gpu" else "cpu"


def _ensure_eval_runtime_randomization_defaults(env: Any) -> None:
    if not hasattr(env, "_randomize_ctrl_delay"):
        env._randomize_ctrl_delay = False
    if not hasattr(env, "_ctrl_delay_step_range"):
        env._ctrl_delay_step_range = [0, 0]
    if not hasattr(env, "_randomize_dof_pos_bias"):
        env._randomize_dof_pos_bias = False
    if not hasattr(env, "_dof_pos_bias_range"):
        env._dof_pos_bias_range = [0.0, 0.0]
    if not hasattr(env, "_randomize_base_com"):
        env._randomize_base_com = False
    if not hasattr(env, "_base_com_range"):
        env._base_com_range = {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}
    if not hasattr(env, "_randomize_push_robots"):
        env._randomize_push_robots = False
    if not hasattr(env, "_push_robots_enabled"):
        env._push_robots_enabled = False
    if not hasattr(env, "_pending_torque_rfi"):
        env._pending_torque_rfi = (False, 0.0)


def _write_sim_data(env: Any) -> None:
    scene = getattr(env.simulator, "scene", None)
    if scene is not None and hasattr(scene, "write_data_to_sim"):
        scene.write_data_to_sim()
    env.simulator.refresh_sim_tensors()


def _zero_active_object_velocity(env: Any, env_ids: torch.Tensor) -> None:
    motion_command = env.command_manager.get_state("motion_command")
    if motion_command is None or not getattr(getattr(motion_command, "motion", None), "has_object", False):
        return

    object_states = motion_command._active_object_states_w()[env_ids].clone()
    object_states[:, 7:13] = 0.0
    motion_command.set_simulator_object_states(env_ids, object_states)
    _write_sim_data(env)


def _reset_to_initial_spawn(env: Any, *, zero_object_velocity: bool) -> None:
    env.set_is_evaluating()
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_envs_idx(env_ids)
    refresh_after_reset = getattr(env, "_refresh_envs_after_reset", None)
    if callable(refresh_after_reset):
        refresh_after_reset(env_ids)
    else:
        _write_sim_data(env)

    if zero_object_velocity:
        _zero_active_object_velocity(env, env_ids)

    env.reset_buf[env_ids] = 0
    logger.info("Spawned objects at the initial motion frame.")


def _run_physics_viewer(env: Any, cli_cfg: RandomizeObjectSpawnConfig) -> None:
    actions = torch.zeros(env.num_envs, env.dim_actions, device=env.device, requires_grad=False)
    if env.action_manager is not None:
        env.action_manager.process_actions(actions)

    reset_interval_steps = None
    if cli_cfg.reset_interval_s is not None and cli_cfg.reset_interval_s > 0.0:
        reset_interval_steps = max(1, round(cli_cfg.reset_interval_s / env.dt))
        logger.info(
            f"Resetting objects to the initial spawn every {cli_cfg.reset_interval_s:.2f}s "
            f"({reset_interval_steps} control steps)."
        )

    step = 0
    logger.info("Running object spawn viewer. Press Ctrl+C to stop.")
    try:
        while cli_cfg.max_steps is None or step < cli_cfg.max_steps:
            env.render(sync_frame_time=True)
            for _ in range(env.simulator.simulator_config.sim.control_decimation):
                if env.action_manager is not None:
                    env.action_manager.apply_actions()
                env.simulator.simulate_at_each_physics_step()
            env.simulator.refresh_sim_tensors()

            if cli_cfg.draw_debug_markers and hasattr(env, "_draw_debug_vis"):
                env._draw_debug_vis()
            step += 1
            if reset_interval_steps is not None and step % reset_interval_steps == 0:
                _reset_to_initial_spawn(env, zero_object_velocity=cli_cfg.zero_object_velocity)
                if env.action_manager is not None:
                    env.action_manager.process_actions(actions)
    except KeyboardInterrupt:
        logger.info("Object spawn viewer interrupted.")


def run_spawn_viewer(tyro_config: ExperimentConfig, cli_cfg: RandomizeObjectSpawnConfig) -> None:
    tyro_config = resolve_observation_term_overrides(tyro_config)
    sys.argv = [sys.argv[0]]
    env, _, simulation_app = setup_simulation_environment(tyro_config, device=_resolve_device(cli_cfg))

    try:
        _ensure_eval_runtime_randomization_defaults(env)
        _reset_to_initial_spawn(env, zero_object_velocity=cli_cfg.zero_object_velocity)
        _run_physics_viewer(env, cli_cfg)
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


def _with_default_experiment(args: list[str], default_experiment: str) -> list[str]:
    if any(arg.startswith("exp:") for arg in args):
        return args
    return [default_experiment, *args]


def main() -> None:
    init_eval_logging()
    spawn_cfg, remaining_args = tyro.cli(
        RandomizeObjectSpawnConfig,
        return_unknown_args=True,
        add_help=False,
    )
    experiment_args = _with_default_experiment(remaining_args, spawn_cfg.default_experiment)
    config = tyro.cli(
        AnnotatedExperimentConfig,
        args=experiment_args,
        description="Object spawn visualization config.",
        config=TYRO_CONIFG,
    )
    config = apply_spawn_viewer_overrides(config, spawn_cfg)
    config = resolve_multi_object_urdf_config(config)
    logger.info("Starting deterministic object-volume spawn visualization.")
    run_spawn_viewer(config, spawn_cfg)


if __name__ == "__main__":
    main()
