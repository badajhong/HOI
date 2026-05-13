from __future__ import annotations

import ast
import dataclasses
import itertools
import os
import sys
from typing import Any, Literal

import tyro
from loguru import logger
from pydantic.dataclasses import dataclass

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.command import MotionConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.video import CartesianCameraConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.object_urdf import resolve_multi_object_urdf_config
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG

DEFAULT_CHECKPOINT = "/home/rllab/haechan/holosoma/logs/WholeBodyTracking/teacher_suitcase/model_17000.pt"
DEFAULT_MOTION_FILE = "/home/rllab/haechan/holosoma/train/rl/tripod/sub1_tripod_117.npz"
DEFAULT_OBJECT_URDF_PATH = "/home/rllab/haechan/holosoma/train/objects/tripod/tripod.urdf"
DEFAULT_NUM_ENVS = 1
DEFAULT_HEADLESS = False
DEFAULT_CAMERA_OFFSET = [3.0, -3.0, 1.5]
DEFAULT_START_AT_TIMESTEP_ZERO_PROB = 1.0
DEFAULT_DEVICE = "cpu"


@dataclass(frozen=True)
class TeacherEvalConfig(CheckpointConfig):
    """Evaluate a teacher checkpoint while choosing the motion/object files."""

    checkpoint: str | None = DEFAULT_CHECKPOINT
    """Teacher checkpoint to evaluate."""

    motion_file: str | None = DEFAULT_MOTION_FILE
    """Evaluate one motion .npz file. Clears motion_folder when provided."""

    motion_folder: str | None = None
    """Evaluate all .npz files in a folder. Clears motion_file when provided."""

    object_urdf_path: str | None = DEFAULT_OBJECT_URDF_PATH
    """URDF path for a single-object evaluation."""

    object_urdf_asset: str | None = None
    """Folder containing object URDFs for multi-object motion folders."""

    num_envs: int | None = DEFAULT_NUM_ENVS
    """Override evaluation environment count."""

    headless: bool | None = DEFAULT_HEADLESS
    """Override rendering mode. Use false for an interactive viewer."""

    max_eval_steps: int | None = None
    """Stop evaluation after this many policy steps."""

    export_onnx: bool | None = None
    """Override whether evaluation exports ONNX next to the checkpoint."""

    env_spacing: float | None = None
    """Override simulator environment spacing."""

    camera_offset: list[float] | None = dataclasses.field(default_factory=lambda: list(DEFAULT_CAMERA_OFFSET))
    """Cartesian video camera offset [x, y, z]."""

    start_at_timestep_zero_prob: float | None = DEFAULT_START_AT_TIMESTEP_ZERO_PROB
    """Override the motion command reset probability for starting at frame zero."""

    disable_object_randomization: bool = True
    """Disable object material/mass/inertia/scale randomization and object reset position noise during eval."""

    disable_robot_randomization: bool = True
    """Disable robot material/base-COM/DOF/actuator/push randomization and robot reset pose noise during eval."""

    device: Literal["cpu", "gpu"] = DEFAULT_DEVICE
    """Simulation device choice. Use 'gpu' for cuda:0."""


def _replace_motion_config(motion_config: Any, updates: dict[str, Any]) -> Any:
    if isinstance(motion_config, MotionConfig):
        return dataclasses.replace(motion_config, **updates)
    if isinstance(motion_config, dict):
        updated_motion_config = dict(motion_config)
        updated_motion_config.update(updates)
        return updated_motion_config
    raise TypeError(f"Unsupported motion_config type: {type(motion_config).__name__}")


def _apply_motion_overrides(config: ExperimentConfig, cli_cfg: TeacherEvalConfig) -> ExperimentConfig:
    if cli_cfg.motion_file is not None and cli_cfg.motion_folder is not None:
        raise ValueError("Choose either --motion-file or --motion-folder, not both.")

    motion_updates: dict[str, Any] = {}
    if cli_cfg.motion_file is not None:
        motion_updates["motion_file"] = cli_cfg.motion_file
        motion_updates["motion_folder"] = ""
    if cli_cfg.motion_folder is not None:
        motion_updates["motion_folder"] = cli_cfg.motion_folder
        motion_updates["motion_file"] = ""
    if cli_cfg.start_at_timestep_zero_prob is not None:
        motion_updates["start_at_timestep_zero_prob"] = cli_cfg.start_at_timestep_zero_prob

    if not motion_updates:
        return config

    if config.command is None:
        raise ValueError("Cannot override motion files because this experiment has no command config.")

    setup_term = config.command.setup_terms.get("motion_command")
    if setup_term is None:
        raise ValueError("Cannot override motion files because setup_terms.motion_command is missing.")

    params = dict(setup_term.params)
    if "motion_config" not in params:
        raise ValueError("Cannot override motion files because motion_command.params.motion_config is missing.")

    params["motion_config"] = _replace_motion_config(params["motion_config"], motion_updates)
    setup_terms = dict(config.command.setup_terms)
    setup_terms["motion_command"] = dataclasses.replace(setup_term, params=params)
    return dataclasses.replace(config, command=dataclasses.replace(config.command, setup_terms=setup_terms))


def _apply_object_overrides(config: ExperimentConfig, cli_cfg: TeacherEvalConfig) -> ExperimentConfig:
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

    robot_config = dataclasses.replace(
        config.robot,
        object=dataclasses.replace(config.robot.object, **object_updates),
    )
    return dataclasses.replace(config, robot=robot_config)


def _apply_training_overrides(config: ExperimentConfig, cli_cfg: TeacherEvalConfig) -> ExperimentConfig:
    training_updates: dict[str, Any] = {}
    if cli_cfg.num_envs is not None:
        training_updates["num_envs"] = cli_cfg.num_envs
    if cli_cfg.headless is not None:
        training_updates["headless"] = cli_cfg.headless
    if cli_cfg.max_eval_steps is not None:
        training_updates["max_eval_steps"] = cli_cfg.max_eval_steps
    if cli_cfg.export_onnx is not None:
        training_updates["export_onnx"] = cli_cfg.export_onnx

    if not training_updates:
        return config

    return dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, **training_updates),
    )


def _apply_simulator_overrides(config: ExperimentConfig, cli_cfg: TeacherEvalConfig) -> ExperimentConfig:
    if cli_cfg.env_spacing is None:
        return config

    scene_config = dataclasses.replace(config.simulator.config.scene, env_spacing=cli_cfg.env_spacing)
    simulator_init_config = dataclasses.replace(config.simulator.config, scene=scene_config)
    simulator_config = dataclasses.replace(config.simulator, config=simulator_init_config)
    return dataclasses.replace(config, simulator=simulator_config)


def _apply_camera_overrides(config: ExperimentConfig, cli_cfg: TeacherEvalConfig) -> ExperimentConfig:
    if cli_cfg.camera_offset is None:
        return config
    if len(cli_cfg.camera_offset) != 3:
        raise ValueError("--camera-offset must contain exactly three values: [x, y, z].")

    camera = config.logger.video.camera
    if isinstance(camera, CartesianCameraConfig):
        camera = dataclasses.replace(camera, offset=list(cli_cfg.camera_offset))
    else:
        camera = CartesianCameraConfig(offset=list(cli_cfg.camera_offset))

    video_config = dataclasses.replace(config.logger.video, camera=camera)
    logger_config = dataclasses.replace(config.logger, video=video_config)
    return dataclasses.replace(config, logger=logger_config)


def _is_object_randomization_term(term_name: str, term_cfg: Any) -> bool:
    func = getattr(term_cfg, "func", "")
    return "object" in term_name.lower() or (isinstance(func, str) and ":randomize_object_" in func)


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


def _remove_randomization_terms(config: ExperimentConfig, label: str, predicate: Any) -> ExperimentConfig:
    if config.randomization is None:
        return config

    randomization = config.randomization
    removed_terms: list[str] = []
    updated_groups: dict[str, dict[str, Any]] = {}
    for group_name in ("setup_terms", "reset_terms", "step_terms"):
        terms = getattr(randomization, group_name, {}) or {}
        kept_terms = {}
        for term_name, term_cfg in terms.items():
            if predicate(term_name, term_cfg):
                removed_terms.append(f"{group_name}.{term_name}")
                continue
            kept_terms[term_name] = term_cfg
        updated_groups[group_name] = kept_terms

    if removed_terms:
        logger.info(f"Disabled {label} randomization terms for eval: {removed_terms}")

    return dataclasses.replace(
        config,
        randomization=dataclasses.replace(randomization, **updated_groups),
    )


def _remove_object_randomization_terms(config: ExperimentConfig) -> ExperimentConfig:
    return _remove_randomization_terms(config, "object", _is_object_randomization_term)


def _remove_robot_randomization_terms(config: ExperimentConfig) -> ExperimentConfig:
    return _remove_randomization_terms(config, "robot", _is_robot_randomization_term)


def _zero_object_pose_noise(motion_config: Any) -> Any:
    if isinstance(motion_config, MotionConfig):
        noise_cfg = dataclasses.replace(motion_config.noise_to_initial_pose, object_pos=[0.0, 0.0, 0.0])
        return dataclasses.replace(motion_config, noise_to_initial_pose=noise_cfg)

    if isinstance(motion_config, dict):
        updated_motion_config = dict(motion_config)
        noise_cfg = updated_motion_config.get("noise_to_initial_pose")
        if noise_cfg is None:
            return updated_motion_config
        if isinstance(noise_cfg, dict):
            updated_noise_cfg = dict(noise_cfg)
            updated_noise_cfg["object_pos"] = [0.0, 0.0, 0.0]
        else:
            updated_noise_cfg = dataclasses.replace(noise_cfg, object_pos=[0.0, 0.0, 0.0])
        updated_motion_config["noise_to_initial_pose"] = updated_noise_cfg
        return updated_motion_config

    raise TypeError(f"Unsupported motion_config type: {type(motion_config).__name__}")


def _zero_robot_pose_noise(motion_config: Any) -> Any:
    noise_updates = {
        "dof_pos": 0.0,
        "root_pos": [0.0, 0.0, 0.0],
        "root_rot": [0.0, 0.0, 0.0],
        "root_lin_vel": [0.0, 0.0, 0.0],
        "root_ang_vel": [0.0, 0.0, 0.0],
    }

    if isinstance(motion_config, MotionConfig):
        noise_cfg = dataclasses.replace(motion_config.noise_to_initial_pose, **noise_updates)
        return dataclasses.replace(motion_config, noise_to_initial_pose=noise_cfg)

    if isinstance(motion_config, dict):
        updated_motion_config = dict(motion_config)
        noise_cfg = updated_motion_config.get("noise_to_initial_pose")
        if noise_cfg is None:
            return updated_motion_config
        if isinstance(noise_cfg, dict):
            updated_noise_cfg = dict(noise_cfg)
            updated_noise_cfg.update(noise_updates)
        else:
            updated_noise_cfg = dataclasses.replace(noise_cfg, **noise_updates)
        updated_motion_config["noise_to_initial_pose"] = updated_noise_cfg
        return updated_motion_config

    raise TypeError(f"Unsupported motion_config type: {type(motion_config).__name__}")


def _remove_object_pose_noise(config: ExperimentConfig) -> ExperimentConfig:
    if config.command is None:
        return config

    setup_term = config.command.setup_terms.get("motion_command")
    if setup_term is None or "motion_config" not in setup_term.params:
        return config

    params = dict(setup_term.params)
    params["motion_config"] = _zero_object_pose_noise(params["motion_config"])
    setup_terms = dict(config.command.setup_terms)
    setup_terms["motion_command"] = dataclasses.replace(setup_term, params=params)
    logger.info("Disabled object reset position noise for eval.")
    return dataclasses.replace(config, command=dataclasses.replace(config.command, setup_terms=setup_terms))


def _remove_robot_pose_noise(config: ExperimentConfig) -> ExperimentConfig:
    if config.command is None:
        return config

    setup_term = config.command.setup_terms.get("motion_command")
    if setup_term is None or "motion_config" not in setup_term.params:
        return config

    params = dict(setup_term.params)
    params["motion_config"] = _zero_robot_pose_noise(params["motion_config"])
    setup_terms = dict(config.command.setup_terms)
    setup_terms["motion_command"] = dataclasses.replace(setup_term, params=params)
    logger.info("Disabled robot reset pose noise for eval.")
    return dataclasses.replace(config, command=dataclasses.replace(config.command, setup_terms=setup_terms))


def _apply_object_randomization_overrides(config: ExperimentConfig, cli_cfg: TeacherEvalConfig) -> ExperimentConfig:
    if not cli_cfg.disable_object_randomization:
        return config

    config = _remove_object_randomization_terms(config)
    return _remove_object_pose_noise(config)


def _apply_robot_randomization_overrides(config: ExperimentConfig, cli_cfg: TeacherEvalConfig) -> ExperimentConfig:
    if not cli_cfg.disable_robot_randomization:
        return config

    config = _remove_robot_randomization_terms(config)
    return _remove_robot_pose_noise(config)


def apply_teacher_eval_overrides(config: ExperimentConfig, cli_cfg: TeacherEvalConfig) -> ExperimentConfig:
    config = _apply_motion_overrides(config, cli_cfg)
    config = _apply_object_overrides(config, cli_cfg)
    config = _apply_training_overrides(config, cli_cfg)
    config = _apply_simulator_overrides(config, cli_cfg)
    config = _apply_camera_overrides(config, cli_cfg)
    config = _apply_object_randomization_overrides(config, cli_cfg)
    return _apply_robot_randomization_overrides(config, cli_cfg)


def _resolve_device(cli_cfg: TeacherEvalConfig) -> str:
    return "cuda:0" if cli_cfg.device == "gpu" else "cpu"


def _strip_wrapping_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _parse_camera_offset_arg(value: str) -> list[str]:
    value = _strip_wrapping_quotes(value)
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
        raise ValueError("--camera-offset must be a list of exactly three values, e.g. '[3.0, -3.0, 1.5]'.")
    return [str(float(item)) for item in parsed]


def _normalize_camera_offset_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--camera-offset" and i + 1 < len(argv) and _strip_wrapping_quotes(argv[i + 1]).startswith("["):
            normalized.append(arg)
            normalized.extend(_parse_camera_offset_arg(argv[i + 1]))
            i += 2
            continue
        if arg.startswith("--camera-offset="):
            value = arg.split("=", 1)[1]
            if _strip_wrapping_quotes(value).startswith("["):
                normalized.append("--camera-offset")
                normalized.extend(_parse_camera_offset_arg(value))
            else:
                normalized.append(arg)
            i += 1
            continue
        normalized.append(arg)
        i += 1
    return normalized


def _checkpoint_object_keys(saved_config: ExperimentConfig) -> list[str]:
    object_name_to_path = getattr(saved_config.robot.object, "object_urdf_name_to_path", None) or {}
    return sorted(str(object_key) for object_key in object_name_to_path)


def _preserve_checkpoint_object_type_space(env: Any, saved_config: ExperimentConfig) -> None:
    """Keep obj_type_one_hot width compatible with the loaded teacher checkpoint."""
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return

    motion_command = command_manager.get_state("motion_command")
    if motion_command is None or not getattr(getattr(motion_command, "motion", None), "has_object", False):
        return

    checkpoint_object_keys = _checkpoint_object_keys(saved_config)
    if not checkpoint_object_keys:
        return

    current_num_types = max(int(getattr(motion_command, "num_object_types", 1)), 1)
    if len(checkpoint_object_keys) <= current_num_types:
        return

    clip_object_keys = list(getattr(motion_command.motion, "clip_object_keys", []) or [])
    unknown_clip_keys = sorted(
        {str(object_key) for object_key in clip_object_keys if object_key is not None}
        - set(checkpoint_object_keys)
    )
    if unknown_clip_keys:
        logger.warning(
            "Cannot preserve checkpoint object type space because eval motions contain "
            f"object keys not present in the checkpoint config: {unknown_clip_keys}"
        )
        return

    object_key_to_id = {object_key: object_id for object_id, object_key in enumerate(checkpoint_object_keys)}
    motion_command.object_key_to_id = object_key_to_id
    motion_command.num_object_types = len(checkpoint_object_keys)

    motion_command.object_type_id_per_clip = torch.zeros(
        len(clip_object_keys), dtype=torch.long, device=motion_command.device
    )
    for clip_idx, object_key in enumerate(clip_object_keys):
        if object_key is not None:
            motion_command.object_type_id_per_clip[clip_idx] = object_key_to_id[str(object_key)]

    if hasattr(motion_command, "object_type_ids") and hasattr(motion_command, "clip_ids"):
        motion_command.object_type_ids[:] = motion_command.object_type_id_per_clip[motion_command.clip_ids]

    logger.info(
        "Preserved checkpoint object type space for eval: "
        f"{checkpoint_object_keys} (current clips: {sorted({str(k) for k in clip_object_keys if k is not None})})"
    )


def _ensure_eval_runtime_randomization_defaults(env: Any) -> None:
    """Initialize runtime flags normally created by disabled randomization hooks."""
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


@torch.no_grad()
def evaluate_teacher_policy_with_summary(algo: BaseAlgo, max_eval_steps: int | None = None) -> dict[str, float | int]:
    if max_eval_steps is None:
        raise ValueError("A finite max_eval_steps is required to print a summary.")
    if not all(hasattr(algo, attr) for attr in ("actor_obs_keys", "critic_obs_keys", "num_act")):
        raise TypeError(f"{algo.__class__.__name__} does not expose the PPO eval interface.")

    algo._create_eval_callbacks()  # type: ignore[attr-defined]
    algo._pre_evaluate_policy()  # type: ignore[attr-defined]
    actor_state = algo._create_actor_state()  # type: ignore[attr-defined]
    algo.eval_policy = algo.get_inference_policy()  # type: ignore[attr-defined]

    obs_dict = algo.env.reset_all()
    init_actions = torch.zeros(algo.env.num_envs, algo.num_act, device=algo.device)  # type: ignore[attr-defined]
    actor_state.update({"obs": obs_dict, "actions": init_actions})

    critic_obs = torch.cat([actor_state["obs"][k] for k in algo.critic_obs_keys], dim=1)  # type: ignore[attr-defined]
    actor_state["obs"]["critic_obs"] = critic_obs

    rollout_rewards = torch.zeros(algo.env.num_envs, device=algo.device)
    episode_rewards = torch.zeros(algo.env.num_envs, device=algo.device)
    episode_lengths = torch.zeros(algo.env.num_envs, device=algo.device)
    completed_episode_rewards: list[float] = []
    completed_episode_lengths: list[float] = []
    steps_run = 0

    for step in itertools.islice(itertools.count(), max_eval_steps):
        actor_state["step"] = step
        actor_state = algo._pre_eval_env_step(actor_state)  # type: ignore[attr-defined]
        actor_state = algo.env_step(actor_state)

        rewards = actor_state["rewards"].detach().to(device=algo.device).view(-1)
        dones = actor_state["dones"].detach().to(device=algo.device).view(-1).bool()
        rollout_rewards += rewards
        episode_rewards += rewards
        episode_lengths += 1.0

        if dones.any():
            completed_episode_rewards.extend(episode_rewards[dones].detach().cpu().tolist())
            completed_episode_lengths.extend(episode_lengths[dones].detach().cpu().tolist())
            episode_rewards[dones] = 0.0
            episode_lengths[dones] = 0.0

        actor_state = algo._post_eval_env_step(actor_state)  # type: ignore[attr-defined]
        steps_run += 1

    algo._post_evaluate_policy()  # type: ignore[attr-defined]

    rollout_mean_reward = float((rollout_rewards / max(steps_run, 1)).mean().detach().cpu())
    summary: dict[str, float | int] = {
        "steps": steps_run,
        "num_envs": int(algo.env.num_envs),
        "completed_episodes": len(completed_episode_rewards),
        "mean_reward_per_step": rollout_mean_reward,
        "mean_rollout_return_per_env": float(rollout_rewards.mean().detach().cpu()),
    }
    if completed_episode_rewards:
        rewards_tensor = torch.tensor(completed_episode_rewards)
        lengths_tensor = torch.tensor(completed_episode_lengths)
        summary.update(
            {
                "mean_episode_return": float(rewards_tensor.mean()),
                "min_episode_return": float(rewards_tensor.min()),
                "max_episode_return": float(rewards_tensor.max()),
                "mean_episode_length": float(lengths_tensor.mean()),
            }
        )
    return summary


@torch.no_grad()
def watch_teacher_policy(algo: BaseAlgo, log_interval: int = 100) -> None:
    if not all(hasattr(algo, attr) for attr in ("actor_obs_keys", "critic_obs_keys", "num_act")):
        raise TypeError(f"{algo.__class__.__name__} does not expose the PPO eval interface.")

    algo._create_eval_callbacks()  # type: ignore[attr-defined]
    algo._pre_evaluate_policy()  # type: ignore[attr-defined]
    actor_state = algo._create_actor_state()  # type: ignore[attr-defined]
    algo.eval_policy = algo.get_inference_policy()  # type: ignore[attr-defined]

    obs_dict = algo.env.reset_all()
    init_actions = torch.zeros(algo.env.num_envs, algo.num_act, device=algo.device)  # type: ignore[attr-defined]
    actor_state.update({"obs": obs_dict, "actions": init_actions})

    critic_obs = torch.cat([actor_state["obs"][k] for k in algo.critic_obs_keys], dim=1)  # type: ignore[attr-defined]
    actor_state["obs"]["critic_obs"] = critic_obs

    logger.info("Watch loop started. Close the IsaacSim window or press Ctrl+C to stop.")
    try:
        for step in itertools.count():
            actor_state["step"] = step
            actor_state = algo._pre_eval_env_step(actor_state)  # type: ignore[attr-defined]
            actor_state = algo.env_step(actor_state)
            actor_state = algo._post_eval_env_step(actor_state)  # type: ignore[attr-defined]
            if step == 0 or (step + 1) % log_interval == 0:
                reward_mean = float(actor_state["rewards"].detach().mean().cpu())
                done_count = int(actor_state["dones"].detach().sum().cpu())
                logger.info(f"Watch step {step + 1}: reward_mean={reward_mean:.4f}, done_count={done_count}")
    except KeyboardInterrupt:
        logger.info("Watch mode interrupted by user.")
    finally:
        algo._post_evaluate_policy()  # type: ignore[attr-defined]


def run_teacher_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    device: str | None = None,
) -> None:
    # AppLauncher parses sys.argv itself. Keep only script name so teacher-specific
    # args such as "--headless False" do not leak into IsaacLab's parser.
    sys.argv = [sys.argv[0]]

    env, resolved_device, simulation_app = setup_simulation_environment(tyro_config, device=device)

    try:
        _preserve_checkpoint_object_type_space(env, saved_config)
        _ensure_eval_runtime_randomization_defaults(env)

        eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
        eval_log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving eval logs to {eval_log_dir}")
        tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

        assert checkpoint_cfg.checkpoint is not None
        checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
        checkpoint_path = str(checkpoint)

        algo_class = get_class(tyro_config.algo._target_)
        algo: BaseAlgo = algo_class(
            device=resolved_device,
            env=env,
            config=tyro_config.algo.config,
            log_dir=str(eval_log_dir),
            multi_gpu_cfg=None,
        )
        algo.setup()
        algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
        algo.load(checkpoint_path)

        checkpoint_dir = os.path.dirname(checkpoint_path)
        exported_policy_dir_path = os.path.join(checkpoint_dir, "exported")
        os.makedirs(exported_policy_dir_path, exist_ok=True)
        exported_policy_name = checkpoint_path.split("/")[-1]
        exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")

        if tyro_config.training.export_onnx:
            exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
            if not hasattr(algo, "export"):
                raise AttributeError(
                    f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
                )
            algo.export(onnx_file_path=exported_onnx_path)  # type: ignore[attr-defined]
            logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

        if tyro_config.training.max_eval_steps is None:
            logger.info("Starting watch mode. The simulation will keep running until you press Ctrl+C.")
            watch_teacher_policy(algo)
        else:
            summary = evaluate_teacher_policy_with_summary(algo, max_eval_steps=tyro_config.training.max_eval_steps)
            logger.info("Teacher eval summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")
    except Exception:
        logger.exception("Teacher evaluation failed before watch mode could continue.")
        raise
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    sys.argv = [sys.argv[0], *_normalize_camera_offset_args(sys.argv[1:])]
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        tyro.cli(TeacherEvalConfig)
        return

    teacher_eval_cfg, remaining_args = tyro.cli(TeacherEvalConfig, return_unknown_args=True, add_help=False)
    checkpoint_cfg = CheckpointConfig(checkpoint=teacher_eval_cfg.checkpoint)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = apply_teacher_eval_overrides(saved_cfg.get_eval_config(), teacher_eval_cfg)
    eval_cfg = resolve_multi_object_urdf_config(eval_cfg)
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Optional full ExperimentConfig overrides on top of teacher eval defaults.",
        config=TYRO_CONIFG,
    )
    logger.info(f"Evaluating teacher checkpoint: {checkpoint_cfg.checkpoint}")
    run_teacher_eval_with_tyro(
        overwritten_tyro_config,
        checkpoint_cfg,
        saved_cfg,
        saved_wandb_path,
        device=_resolve_device(teacher_eval_cfg),
    )


if __name__ == "__main__":
    main()
