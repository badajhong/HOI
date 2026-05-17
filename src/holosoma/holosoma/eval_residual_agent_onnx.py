from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import tyro
from loguru import logger
from pydantic.dataclasses import dataclass

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.command import MotionConfig
from holosoma.config_types.env import resolve_observation_term_overrides
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.eval_student_agent import StudentEvalConfig, _resolve_device
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
from holosoma.utils.rotations import quat_rotate_inverse
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG

DEFAULT_CHECKPOINT = "/home/rllab/haechan/holosoma/logs/20260515_results/model_03000.onnx"
DEFAULT_OBJECT_SCALE = 0.6
DEFAULT_OBJECT_SCALE_EVAL_VALUES: tuple[float, ...] = ()
DEFAULT_OBJECT_DYNAMIC_FRICTION = 0.2
DEFAULT_OBJECT_STATIC_FRICTION = 0.2
FUSED_RESIDUAL_ONNX_KIND = "wbt_student_residual_fused"

@dataclass(frozen=True)
class ResidualEvalConfig(StudentEvalConfig):
    checkpoint: str | None = DEFAULT_CHECKPOINT
    """Residual checkpoint to evaluate. Prefer a fused .onnx for this script."""

    motion_file: str | None = None
    """Optional motion override. None uses the motion config embedded in ONNX metadata/saved config."""

    motion_folder: str | None = None
    """Optional motion-folder override. None uses the saved config."""

    object_urdf_path: str | None = None
    """Optional object URDF override. None uses the saved config."""

    di_pro_ae: str | None = None
    """Disabled by default for ONNX eval because the DI-pro encoder is embedded in the fused ONNX."""

    num_envs: int | None = 1
    """Override evaluation environment count."""

    headless: bool | None = False
    """Override rendering mode. Use false for an interactive viewer."""

    max_eval_steps: int | None = 2000
    """Stop evaluation after this many policy steps so a summary can be printed."""

    export_onnx: bool | None = False
    """Override whether evaluation exports ONNX next to the checkpoint."""

    disable_object_randomization: bool = True
    """Disable object material/mass/inertia/scale randomization during eval."""

    disable_robot_randomization: bool = True
    """Disable robot material/base-COM/DOF/actuator/push randomization during eval."""

    object_scale: float | None = DEFAULT_OBJECT_SCALE
    """Optional single object volume ratio for eval. None preserves the original saved-config object spawn."""

    object_scale_eval_values: tuple[float, ...] = DEFAULT_OBJECT_SCALE_EVAL_VALUES
    """Optional multi-volume eval sweep. Ignored when object_scale is set."""

    object_scale_height: float | None = 0.0
    """Object height hint forwarded to deterministic scale startup. Non-positive uses URDF-derived bounds."""

    object_scale_reset_physics: bool = False
    """Deprecated eval option. Physics resets during reset-time USD edits invalidate IsaacSim tensor views."""

    object_static_friction: float = DEFAULT_OBJECT_STATIC_FRICTION
    """Optional fixed object static friction for eval. None keeps the object asset/default material."""

    object_dynamic_friction: float = DEFAULT_OBJECT_DYNAMIC_FRICTION
    """Optional fixed object dynamic friction for eval. None defaults to static friction when static is set."""

    object_restitution: float = 0.0
    """Fixed object restitution used when object friction is overridden."""


def _replace_motion_config(motion_config: Any, updates: dict[str, Any]) -> Any:
    if isinstance(motion_config, MotionConfig):
        return dataclasses.replace(motion_config, **updates)
    if isinstance(motion_config, dict):
        updated_motion_config = dict(motion_config)
        updated_motion_config.update(updates)
        return updated_motion_config
    raise TypeError(f"Unsupported motion_config type: {type(motion_config).__name__}")


def _apply_motion_overrides(config: ExperimentConfig, cli_cfg: ResidualEvalConfig) -> ExperimentConfig:
    if config.command is None:
        raise ValueError("Cannot override motion files because this experiment has no command config.")

    setup_term = config.command.setup_terms.get("motion_command")
    if setup_term is None:
        raise ValueError("Cannot override motion files because setup_terms.motion_command is missing.")

    params = dict(setup_term.params)
    if "motion_config" not in params:
        raise ValueError("Cannot override motion files because motion_command.params.motion_config is missing.")

    motion_updates: dict[str, Any] = {}
    if cli_cfg.motion_folder is not None:
        motion_updates["motion_folder"] = cli_cfg.motion_folder
        motion_updates["motion_file"] = ""
    elif cli_cfg.motion_file is not None:
        motion_updates["motion_file"] = cli_cfg.motion_file
        motion_updates["motion_folder"] = ""
    if cli_cfg.start_at_timestep_zero_prob is not None:
        motion_updates["start_at_timestep_zero_prob"] = cli_cfg.start_at_timestep_zero_prob

    params["motion_config"] = _replace_motion_config(params["motion_config"], motion_updates)
    setup_terms = dict(config.command.setup_terms)
    setup_terms["motion_command"] = dataclasses.replace(setup_term, params=params)
    return dataclasses.replace(config, command=dataclasses.replace(config.command, setup_terms=setup_terms))


def _apply_object_overrides(config: ExperimentConfig, cli_cfg: ResidualEvalConfig) -> ExperimentConfig:
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
        robot=dataclasses.replace(config.robot, object=dataclasses.replace(config.robot.object, **object_updates)),
    )


def _apply_training_overrides(config: ExperimentConfig, cli_cfg: ResidualEvalConfig) -> ExperimentConfig:
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
    return dataclasses.replace(config, training=dataclasses.replace(config.training, **training_updates))


def _apply_simulator_overrides(config: ExperimentConfig, cli_cfg: ResidualEvalConfig) -> ExperimentConfig:
    if cli_cfg.env_spacing is None:
        return config

    scene_config = dataclasses.replace(config.simulator.config.scene, env_spacing=cli_cfg.env_spacing)
    simulator_init_config = dataclasses.replace(config.simulator.config, scene=scene_config)
    return dataclasses.replace(config, simulator=dataclasses.replace(config.simulator, config=simulator_init_config))


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

    return dataclasses.replace(config, randomization=dataclasses.replace(randomization, **updated_groups))


def _remove_object_randomization_terms(config: ExperimentConfig) -> ExperimentConfig:
    return _remove_randomization_terms(config, "object", _is_object_randomization_term)


def _remove_robot_randomization_terms(config: ExperimentConfig) -> ExperimentConfig:
    return _remove_randomization_terms(config, "robot", _is_robot_randomization_term)


def _is_auto(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.lower() == "auto")


def _volume_ratio_to_xyz_scale_value(volume_ratio: float) -> float:
    if volume_ratio <= 0.0:
        raise ValueError(f"Object volume ratio must be positive, got {volume_ratio}.")
    return volume_ratio ** (1.0 / 3.0)


def _get_object_scale_randomization_params(config: ExperimentConfig) -> dict[str, Any] | None:
    if config.randomization is None:
        return None

    setup_terms = getattr(config.randomization, "setup_terms", {}) or {}
    term_cfg = setup_terms.get("randomize_object_scale_startup")
    if term_cfg is None:
        term_cfg = next(
            (
                candidate
                for candidate in setup_terms.values()
                if str(getattr(candidate, "func", "")).endswith(":randomize_object_scale_startup")
            ),
            None,
        )
    if term_cfg is None:
        return None

    params = dict(getattr(term_cfg, "params", {}) or {})
    if not bool(params.get("enabled", True)):
        return None
    return params


def _target_scale_value(scale_value: Any) -> float:
    if isinstance(scale_value, (int, float)):
        return float(scale_value)
    raise ValueError(
        "randomize_object_scale_startup.params.scale_value must be a scalar object volume ratio, "
        f"got {scale_value!r}."
    )


def _randomized_scale_range(params: dict[str, Any]) -> tuple[float, float] | None:
    scale_range = params.get("scale_range")
    if isinstance(scale_range, dict):
        raise ValueError(
            "randomize_object_scale_startup.params.scale_range must be a 2-value object volume ratio range."
        )
    if scale_range is not None:
        if len(scale_range) != 2:
            raise ValueError(f"scale_range must have two values, got {scale_range!r}.")
        return float(scale_range[0]), float(scale_range[1])
    return None


def _num_bins_from_size(bin_min: float, bin_max: float, bin_size: float) -> int:
    raw_bins = (bin_max - bin_min) / bin_size
    rounded_bins = round(raw_bins)
    if abs(raw_bins - rounded_bins) < 1e-6:
        return max(int(rounded_bins), 1)
    return max(math.ceil(raw_bins), 1)


def _resolve_object_scale_bin_params(params: dict[str, Any], randomization_params: dict[str, Any]) -> dict[str, Any]:
    resolved_params = dict(params)

    if not _is_auto(resolved_params.get("scale_values", "auto")):
        return resolved_params

    if randomization_params.get("scale_values") is not None:
        resolved_params["scale_values"] = randomization_params["scale_values"]
        return resolved_params

    if randomization_params.get("scale_value") is not None:
        resolved_params["scale_values"] = _target_scale_value(
            randomization_params["scale_value"],
        )
        return resolved_params

    scale_range = _randomized_scale_range(randomization_params)
    if scale_range is None:
        return resolved_params

    bin_min, bin_max = scale_range
    if _is_auto(resolved_params.get("bin_min", "auto")):
        resolved_params["bin_min"] = bin_min
    if _is_auto(resolved_params.get("bin_max", "auto")):
        resolved_params["bin_max"] = bin_max
    if _is_auto(resolved_params.get("num_bins", "auto")):
        bin_size = float(resolved_params.get("bin_size", 0.1))
        resolved_params["num_bins"] = _num_bins_from_size(bin_min, bin_max, bin_size)
    return resolved_params


def _resolve_object_scale_observation_autos(config: ExperimentConfig) -> ExperimentConfig:
    """Bake object-scale bin metadata into observations before eval disables scale randomization."""
    if config.observation is None:
        return config

    randomization_params = _get_object_scale_randomization_params(config)
    if randomization_params is None:
        return config

    changed_terms: list[str] = []
    updated_groups = {}
    for group_name, group_cfg in config.observation.groups.items():
        updated_terms = {}
        group_changed = False
        for term_name, term_cfg in group_cfg.terms.items():
            if not str(getattr(term_cfg, "func", "")).endswith(":ObjectScaleBinInput"):
                updated_terms[term_name] = term_cfg
                continue

            resolved_params = _resolve_object_scale_bin_params(dict(term_cfg.params), randomization_params)
            if resolved_params == term_cfg.params:
                updated_terms[term_name] = term_cfg
                continue

            updated_terms[term_name] = dataclasses.replace(term_cfg, params=resolved_params)
            changed_terms.append(f"{group_name}.{term_name}")
            group_changed = True

        updated_groups[group_name] = (
            dataclasses.replace(group_cfg, terms=updated_terms) if group_changed else group_cfg
        )

    if not changed_terms:
        return config

    logger.info(f"Resolved object scale observation auto params for eval: {changed_terms}")
    return dataclasses.replace(
        config,
        observation=dataclasses.replace(config.observation, groups=updated_groups),
    )


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


def _apply_object_randomization_overrides(config: ExperimentConfig, cli_cfg: ResidualEvalConfig) -> ExperimentConfig:
    if not cli_cfg.disable_object_randomization:
        return config

    config = _resolve_object_scale_observation_autos(config)
    config = _remove_object_randomization_terms(config)
    return _remove_object_pose_noise(config)


def _apply_robot_randomization_overrides(config: ExperimentConfig, cli_cfg: ResidualEvalConfig) -> ExperimentConfig:
    if not cli_cfg.disable_robot_randomization:
        return config

    config = _remove_robot_randomization_terms(config)
    return _remove_robot_pose_noise(config)


def _apply_fixed_object_friction(config: ExperimentConfig, cli_cfg: ResidualEvalConfig) -> ExperimentConfig:
    static_friction = cli_cfg.object_static_friction
    dynamic_friction = cli_cfg.object_dynamic_friction
    if static_friction is None and dynamic_friction is None:
        return config

    if static_friction is None:
        static_friction = dynamic_friction
    if dynamic_friction is None:
        dynamic_friction = static_friction
    assert static_friction is not None
    assert dynamic_friction is not None

    randomization = config.randomization or RandomizationManagerCfg()
    setup_terms = dict(randomization.setup_terms)
    setup_terms["residual_eval_object_material_startup"] = RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_material_startup",
        params={
            "static_friction_range": [float(static_friction), float(static_friction)],
            "dynamic_friction_range": [float(dynamic_friction), float(dynamic_friction)],
            "restitution_range": [float(cli_cfg.object_restitution), float(cli_cfg.object_restitution)],
            "enabled": True,
        },
    )
    logger.info(
        "Configured deterministic object material for eval: "
        f"static_friction={static_friction:g}, dynamic_friction={dynamic_friction:g}, "
        f"restitution={cli_cfg.object_restitution:g}"
    )
    return dataclasses.replace(config, randomization=dataclasses.replace(randomization, setup_terms=setup_terms))


def randomize_object_scale_startup(
    env: Any,
    env_ids: list[int] | torch.Tensor | None = None,
    *,
    scale_value: float,
    object_height: float | None = 0.0,
    enabled: bool = True,
    **_: Any,
) -> None:
    """Initialize scale metadata for objects already spawned from a volume-ratio override."""
    if not enabled:
        return

    selected_env_ids = (
        torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        if env_ids is None
        else torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    )
    if selected_env_ids.numel() == 0:
        return

    fixed_scale = torch.as_tensor(scale_value, device=env.device, dtype=torch.float32)
    if fixed_scale.ndim == 0:
        fixed_scale = torch.tensor(
            _volume_ratio_to_xyz_scale_value(float(fixed_scale.item())),
            device=env.device,
            dtype=torch.float32,
        )
        scale_tensor = fixed_scale.repeat(selected_env_ids.numel(), 3).view(selected_env_ids.numel(), 3)
    else:
        raise ValueError(f"object_scale must be a scalar object volume ratio, got shape {tuple(fixed_scale.shape)}.")

    if not hasattr(env, "object_scale_factors"):
        env.object_scale_factors = torch.ones(env.num_envs, 3, device=env.device, dtype=torch.float32)
    if not hasattr(env, "object_scale_factors_z"):
        env.object_scale_factors_z = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)

    simulator = getattr(env, "simulator", None)
    if simulator is not None:
        from holosoma.managers.randomization.terms.locomotion import (  # noqa: PLC0415
            _get_object_scene_entity_names,
            _setup_object_scale_reference_bounds,
        )

        object_names = _get_object_scene_entity_names(simulator)
        if object_names and not hasattr(env, "object_local_bbox_center_by_actor"):
            _setup_object_scale_reference_bounds(env, object_names, object_height)

    env.object_scale_factors[selected_env_ids] = scale_tensor
    env.object_scale_factors_z[selected_env_ids] = scale_tensor[:, 2]
    logger.info(
        "Initialized eval object scale metadata with internal XYZ scale "
        f"{scale_tensor[0].detach().cpu().tolist()}"
    )


def _scale_label(scale: float) -> str:
    return f"{scale:g}".replace("-", "m").replace(".", "p")


def _apply_fixed_object_spawn_scale(
    config: ExperimentConfig,
    scale: float,
    object_height: float | None,
) -> ExperimentConfig:
    randomization = config.randomization or RandomizationManagerCfg()
    setup_terms = dict(randomization.setup_terms)
    setup_terms["residual_eval_object_scale_startup"] = RandomizationTermCfg(
        func="holosoma.eval_residual_agent:randomize_object_scale_startup",
        params={
            "scale_value": float(scale),
            "object_height": object_height,
            "enabled": True,
        },
    )

    scene_config = dataclasses.replace(config.simulator.config.scene, replicate_physics=False)
    simulator_init_config = dataclasses.replace(config.simulator.config, scene=scene_config)
    logger.info(f"Configured spawn-time object volume ratio for eval: {scale:g}")
    return dataclasses.replace(
        config,
        randomization=dataclasses.replace(randomization, setup_terms=setup_terms),
        simulator=dataclasses.replace(config.simulator, config=simulator_init_config),
    )


def _resolve_object_scale_eval_values(cli_cfg: ResidualEvalConfig) -> tuple[float, ...]:
    if cli_cfg.object_scale is not None:
        return (float(cli_cfg.object_scale),)
    return tuple(float(scale) for scale in cli_cfg.object_scale_eval_values)


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


def _resolve_eval_config(config: ExperimentConfig, cli_cfg: ResidualEvalConfig) -> ExperimentConfig:
    config = _apply_motion_overrides(config, cli_cfg)
    config = _apply_object_overrides(config, cli_cfg)
    config = _apply_training_overrides(config, cli_cfg)
    config = _apply_simulator_overrides(config, cli_cfg)
    config = _apply_object_randomization_overrides(config, cli_cfg)
    config = _apply_robot_randomization_overrides(config, cli_cfg)
    return _apply_fixed_object_friction(config, cli_cfg)


def _is_onnx_checkpoint(checkpoint: str | None) -> bool:
    return bool(checkpoint) and str(checkpoint).lower().endswith(".onnx")


def _json_metadata_value(raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _load_onnx_metadata(onnx_path: str | Path) -> dict[str, Any]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return {entry.key: _json_metadata_value(entry.value) for entry in model.metadata_props}


def _load_onnx_experiment_config(onnx_path: str | Path) -> tuple[ExperimentConfig, str | None, dict[str, Any]]:
    metadata = _load_onnx_metadata(onnx_path)
    config_data = metadata.get("experiment_config")
    if not isinstance(config_data, dict):
        raise ValueError(
            f"ONNX checkpoint '{onnx_path}' does not contain an experiment_config metadata entry. "
            "Re-export the policy from a checkpoint that saves ONNX metadata."
        )

    export_spec = metadata.get("policy_export_spec")
    if not isinstance(export_spec, dict):
        raise ValueError(
            f"ONNX checkpoint '{onnx_path}' does not contain policy_export_spec metadata. "
            "This evaluator needs the fused-policy input layout."
        )
    if export_spec.get("kind") != FUSED_RESIDUAL_ONNX_KIND:
        raise ValueError(
            f"Unsupported ONNX policy kind '{export_spec.get('kind')}'. "
            f"Expected '{FUSED_RESIDUAL_ONNX_KIND}'."
        )

    return ExperimentConfig(**config_data), metadata.get("wandb_run_path"), metadata


def _minimal_onnx_observation_config(config: ExperimentConfig) -> ObservationManagerCfg:
    clip_observations = 100.0
    if config.observation is not None:
        clip_observations = config.observation.clip_observations

    return ObservationManagerCfg(
        groups={
            "actor_obs": ObsGroupCfg(
                concatenate=True,
                enable_noise=False,
                history_length=1,
                terms={
                    "base_ang_vel": ObsTermCfg(
                        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
                        scale=1.0,
                        noise=0.0,
                    )
                },
            )
        },
        clip_observations=clip_observations,
    )


def _enable_robot_depth_camera(config: ExperimentConfig) -> ExperimentConfig:
    simulator_init_config = dataclasses.replace(config.simulator.config, enable_robot_depth_camera=True)
    return dataclasses.replace(config, simulator=dataclasses.replace(config.simulator, config=simulator_init_config))


def _resolve_onnx_eval_config(config: ExperimentConfig, cli_cfg: ResidualEvalConfig) -> ExperimentConfig:
    """Prepare the simulator config for direct ONNXRuntime inference.

    The fused ONNX embeds the frozen student actor and DI/DI-pro encoder, so the
    environment observation manager must not instantiate those checkpoint-backed
    terms during eval. WBT reset/reward still use the motion_command config from
    the saved run because the current ONNX export does not contain the full body
    and object trajectories required by MotionCommand.
    """
    has_motion_override = (
        cli_cfg.motion_file is not None
        or cli_cfg.motion_folder is not None
        or cli_cfg.start_at_timestep_zero_prob is not None
    )
    if has_motion_override:
        config = _apply_motion_overrides(config, cli_cfg)
    config = _apply_object_overrides(config, cli_cfg)
    config = _apply_training_overrides(config, cli_cfg)
    config = _apply_simulator_overrides(config, cli_cfg)
    config = _apply_object_randomization_overrides(config, cli_cfg)
    config = _apply_robot_randomization_overrides(config, cli_cfg)
    config = _apply_fixed_object_friction(config, cli_cfg)
    config = _enable_robot_depth_camera(config)
    return dataclasses.replace(
        config,
        teacher=None,
        student=None,
        ir_ae=None,
        ir_ae_body_source=None,
        di_ae=None,
        di_pro_ae=None,
        observation=_minimal_onnx_observation_config(config),
    )


def _append_enable_cameras_arg_for_onnx() -> None:
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
        logger.info("Enabled IsaacSim cameras for fused ONNX depth input via --enable_cameras.")


def _make_actor_state(algo: BaseAlgo) -> dict[str, Any]:
    if not all(hasattr(algo, attr) for attr in ("actor_obs_keys", "critic_obs_keys", "num_act")):
        raise TypeError(f"{algo.__class__.__name__} does not expose the PPO eval interface.")

    actor_state = algo._create_actor_state()  # type: ignore[attr-defined]
    obs_dict = algo.env.reset_all()
    init_actions = torch.zeros(algo.env.num_envs, algo.num_act, device=algo.device)  # type: ignore[attr-defined]
    actor_state.update({"obs": obs_dict, "actions": init_actions})

    critic_obs = torch.cat([actor_state["obs"][key] for key in algo.critic_obs_keys], dim=1)  # type: ignore[attr-defined]
    actor_state["obs"]["critic_obs"] = critic_obs
    return actor_state


@torch.no_grad()
def evaluate_residual_policy_with_summary(algo: BaseAlgo, max_eval_steps: int) -> dict[str, float | int]:
    algo._create_eval_callbacks()  # type: ignore[attr-defined]
    algo._pre_evaluate_policy()  # type: ignore[attr-defined]
    actor_state = _make_actor_state(algo)
    algo.eval_policy = algo.get_inference_policy()  # type: ignore[attr-defined]

    rollout_rewards = torch.zeros(algo.env.num_envs, device=algo.device)  # type: ignore[attr-defined]
    episode_rewards = torch.zeros(algo.env.num_envs, device=algo.device)  # type: ignore[attr-defined]
    episode_lengths = torch.zeros(algo.env.num_envs, device=algo.device)  # type: ignore[attr-defined]
    completed_episode_rewards: list[float] = []
    completed_episode_lengths: list[float] = []

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

    algo._post_evaluate_policy()  # type: ignore[attr-defined]

    summary: dict[str, float | int] = {
        "steps": max_eval_steps,
        "num_envs": int(algo.env.num_envs),
        "completed_episodes": len(completed_episode_rewards),
        "mean_reward_per_step": float((rollout_rewards / max(max_eval_steps, 1)).mean().detach().cpu()),
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
def watch_residual_policy(algo: BaseAlgo, log_interval: int = 100) -> None:
    algo._create_eval_callbacks()  # type: ignore[attr-defined]
    algo._pre_evaluate_policy()  # type: ignore[attr-defined]
    actor_state = _make_actor_state(algo)
    algo.eval_policy = algo.get_inference_policy()  # type: ignore[attr-defined]

    logger.info("Watch loop started. Close the viewer or press Ctrl+C to stop.")
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


class FusedResidualOnnxRuntimePolicy:
    """Direct ONNXRuntime runner for the fused residual/student/DI policy export."""

    def __init__(self, onnx_path: str | Path, export_spec: dict[str, Any], env: Any, device: str):
        if env.num_envs != 1:
            raise ValueError(
                "The current fused ONNX export has a fixed batch size of 1. "
                f"Run ONNX eval with --num-envs=1, got {env.num_envs}."
            )

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.output_names = [output.name for output in self.session.get_outputs()]

        input_names = [input_info.name for input_info in self.session.get_inputs()]
        if "obs" not in input_names or "time_step" not in input_names:
            raise ValueError(f"Expected ONNX inputs ['obs', 'time_step'], got {input_names}.")
        self.obs_input_name = "obs"
        self.time_input_name = "time_step"

        self.device = device
        self.export_spec = export_spec
        self.num_envs = int(env.num_envs)
        self.num_actions = int(export_spec["num_actions"])
        self.input_dim = int(self.session.get_inputs()[input_names.index("obs")].shape[1])
        self.shared_obs_dims = {str(k): int(v) for k, v in export_spec["shared_obs_dims"].items()}
        self.shared_obs_scales = dict(export_spec.get("shared_obs_scales") or {})
        metadata_terms = [str(term) for term in export_spec["shared_obs_terms"]]
        self.shared_obs_terms = sorted(metadata_terms)
        if metadata_terms != self.shared_obs_terms:
            logger.info(
                "Using alphabetic observation term order for ONNX input to match ObservationManager: "
                f"{self.shared_obs_terms}"
            )

        self.depth_shape = tuple(int(value) for value in export_spec["depth_window_shape"])
        self.depth_window = torch.zeros(self.num_envs, *self.depth_shape, device=device, dtype=torch.float32)
        self.depth_initialized = torch.zeros(self.num_envs, device=device, dtype=torch.bool)

        proprioception_shape = export_spec.get("proprioception_window_shape")
        self.proprioception_shape = (
            tuple(int(value) for value in proprioception_shape) if proprioception_shape is not None else None
        )
        if self.proprioception_shape is not None:
            self.proprioception_window = torch.zeros(
                self.num_envs,
                *self.proprioception_shape,
                device=device,
                dtype=torch.float32,
            )
            self.proprioception_initialized = torch.zeros(self.num_envs, device=device, dtype=torch.bool)
        else:
            self.proprioception_window = None
            self.proprioception_initialized = None

        self.real_object_scale_dim = int(export_spec.get("real_object_scale_dim") or 0)
        if self.real_object_scale_dim:
            raise NotImplementedError(
                "This ONNX eval path does not yet support fused exports that require real_object_scale input."
            )

        self.prev_actions = torch.zeros(self.num_envs, self.num_actions, device=device, dtype=torch.float32)
        expected_input_dim = (
            sum(self.shared_obs_dims.values())
            + int(export_spec["depth_window_flat_dim"])
            + int(export_spec.get("proprioception_window_flat_dim") or 0)
            + self.real_object_scale_dim
        )
        if expected_input_dim != self.input_dim:
            raise ValueError(
                f"ONNX metadata input dim {expected_input_dim} does not match graph input {self.input_dim}."
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.prev_actions.zero_()
            self.depth_window.zero_()
            self.depth_initialized.zero_()
            if self.proprioception_window is not None and self.proprioception_initialized is not None:
                self.proprioception_window.zero_()
                self.proprioception_initialized.zero_()
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        self.prev_actions[env_ids] = 0.0
        self.depth_window[env_ids] = 0.0
        self.depth_initialized[env_ids] = False
        if self.proprioception_window is not None and self.proprioception_initialized is not None:
            self.proprioception_window[env_ids] = 0.0
            self.proprioception_initialized[env_ids] = False

    def _run_session(self, obs: torch.Tensor, time_steps: torch.Tensor) -> dict[str, torch.Tensor]:
        obs_np = obs.detach().cpu().numpy().astype(np.float32, copy=False)
        time_np = time_steps.detach().view(self.num_envs, 1).cpu().numpy().astype(np.float32, copy=False)
        outputs = self.session.run(
            self.output_names,
            {
                self.obs_input_name: obs_np,
                self.time_input_name: time_np,
            },
        )
        return {
            name: torch.as_tensor(value, device=self.device, dtype=torch.float32)
            for name, value in zip(self.output_names, outputs, strict=True)
        }

    def _lookup_embedded_motion(self, time_steps: torch.Tensor) -> dict[str, torch.Tensor]:
        dummy_obs = torch.zeros(self.num_envs, self.input_dim, device=self.device, dtype=torch.float32)
        return self._run_session(dummy_obs, time_steps)

    def _current_time_steps(self, env: Any, fallback_step: int) -> torch.Tensor:
        motion_command = None
        if getattr(env, "command_manager", None) is not None:
            motion_command = env.command_manager.get_state("motion_command")
        if motion_command is not None and hasattr(motion_command, "time_steps"):
            return motion_command.time_steps.detach().to(device=self.device, dtype=torch.float32)
        return torch.full((self.num_envs,), float(fallback_step), device=self.device, dtype=torch.float32)

    def _read_depth_frames(self, env: Any) -> torch.Tensor:
        depth_camera = getattr(env.simulator, "robot_depth_camera", None)
        if depth_camera is None:
            raise RuntimeError(
                "Fused ONNX policy needs live robot depth frames, but simulator.robot_depth_camera is missing."
            )

        depth_output = depth_camera.data.output.get("distance_to_image_plane")
        if depth_output is None:
            raise RuntimeError("Robot depth camera has no 'distance_to_image_plane' output.")

        depth_frames = depth_output.to(device=self.device, dtype=torch.float32)
        if depth_frames.ndim == 4 and depth_frames.shape[-1] == 1:
            depth_frames = depth_frames[..., 0]
        if depth_frames.ndim != 3:
            raise RuntimeError(f"Unexpected robot depth batch shape {tuple(depth_frames.shape)}.")

        _, expected_height, expected_width = self.depth_shape
        if depth_frames.shape[1:] == (expected_width, expected_height):
            depth_frames = depth_frames.transpose(1, 2)
        if depth_frames.shape[1:] != (expected_height, expected_width):
            raise RuntimeError(
                f"Unexpected robot depth spatial shape {tuple(depth_frames.shape[1:])}; "
                f"expected {(expected_height, expected_width)}."
            )
        return torch.nan_to_num(depth_frames, nan=0.0, posinf=0.0, neginf=0.0)

    def _update_depth_window(self, env: Any) -> torch.Tensor:
        depth_frames = self._read_depth_frames(env)
        new_mask = ~self.depth_initialized
        if torch.any(new_mask):
            self.depth_window[new_mask] = depth_frames[new_mask].unsqueeze(1).repeat(1, self.depth_shape[0], 1, 1)

        existing_mask = self.depth_initialized
        if torch.any(existing_mask):
            existing_history = self.depth_window[existing_mask].clone()
            existing_history[:, :-1] = existing_history[:, 1:].clone()
            existing_history[:, -1] = depth_frames[existing_mask]
            self.depth_window[existing_mask] = existing_history

        self.depth_initialized[:] = True
        return self.depth_window

    def _base_ang_vel(self, env: Any) -> torch.Tensor:
        if hasattr(env, "base_quat"):
            env.base_quat[:] = env.simulator.base_quat[:]
            base_quat = env.base_quat
        else:
            base_quat = env.simulator.base_quat
        return quat_rotate_inverse(base_quat, env.simulator.robot_root_states[:, 10:13], w_last=True)

    def _current_proprioception(self, env: Any, base_ang_vel: torch.Tensor) -> torch.Tensor:
        dof_pos = env.simulator.dof_pos - env.default_dof_pos
        dof_vel = env.simulator.dof_vel
        return torch.cat((base_ang_vel, dof_pos, dof_vel), dim=-1)

    def _update_proprioception_window(self, proprioception: torch.Tensor) -> torch.Tensor | None:
        if self.proprioception_window is None or self.proprioception_initialized is None:
            return None

        new_mask = ~self.proprioception_initialized
        if torch.any(new_mask):
            self.proprioception_window[new_mask] = proprioception[new_mask].unsqueeze(1).repeat(
                1,
                self.proprioception_shape[0],  # type: ignore[index]
                1,
            )

        existing_mask = self.proprioception_initialized
        if torch.any(existing_mask):
            existing_history = self.proprioception_window[existing_mask].clone()
            existing_history[:, :-1] = existing_history[:, 1:].clone()
            existing_history[:, -1] = proprioception[existing_mask]
            self.proprioception_window[existing_mask] = existing_history

        self.proprioception_initialized[:] = True
        return self.proprioception_window

    def _object_type_one_hot(self, env: Any) -> torch.Tensor:
        expected_dim = self.shared_obs_dims.get("obj_type_one_hot", 1)
        motion_command = None
        if getattr(env, "command_manager", None) is not None:
            motion_command = env.command_manager.get_state("motion_command")
        if motion_command is None or not hasattr(motion_command, "object_type_ids"):
            return torch.ones(self.num_envs, expected_dim, device=self.device, dtype=torch.float32)

        object_type_ids = motion_command.object_type_ids.to(device=self.device, dtype=torch.long)
        object_type_ids = torch.clamp(object_type_ids, min=0, max=max(expected_dim - 1, 0))
        return torch.nn.functional.one_hot(object_type_ids, num_classes=expected_dim).float()

    def _scale_term(self, term_name: str, value: torch.Tensor) -> torch.Tensor:
        scale = self.shared_obs_scales.get(term_name, 1.0)
        if isinstance(scale, list | tuple):
            scale_tensor = torch.tensor(scale, device=self.device, dtype=value.dtype).view(1, -1)
            return value * scale_tensor
        return value * float(scale)

    def _build_shared_obs(self, env: Any, motion_outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        base_ang_vel = self._base_ang_vel(env)
        term_values = {
            "actions": self.prev_actions,
            "base_ang_vel": base_ang_vel,
            "dof_pos": env.simulator.dof_pos - env.default_dof_pos,
            "dof_vel": env.simulator.dof_vel,
            "motion_command_joint_pos": motion_outputs["joint_pos"],
            "obj_type_one_hot": self._object_type_one_hot(env),
        }

        pieces = []
        for term_name in self.shared_obs_terms:
            if term_name not in term_values:
                raise RuntimeError(f"Unsupported fused ONNX shared observation term '{term_name}'.")
            value = self._scale_term(term_name, term_values[term_name])
            expected_dim = self.shared_obs_dims[term_name]
            if value.shape[1] != expected_dim:
                raise RuntimeError(
                    f"Observation term '{term_name}' has dim {value.shape[1]}, expected {expected_dim}."
                )
            pieces.append(value)
        return torch.cat(pieces, dim=-1), base_ang_vel

    def build_input(self, env: Any, fallback_step: int) -> tuple[torch.Tensor, torch.Tensor]:
        time_steps = self._current_time_steps(env, fallback_step)
        motion_outputs = self._lookup_embedded_motion(time_steps)
        shared_obs, base_ang_vel = self._build_shared_obs(env, motion_outputs)
        depth_window = self._update_depth_window(env).flatten(1)

        pieces = [shared_obs, depth_window]
        if self.proprioception_shape is not None:
            proprioception = self._current_proprioception(env, base_ang_vel)
            proprioception_window = self._update_proprioception_window(proprioception)
            assert proprioception_window is not None
            pieces.append(proprioception_window.flatten(1))

        actor_obs = torch.cat(pieces, dim=-1)
        if actor_obs.shape[1] != self.input_dim:
            raise RuntimeError(f"Built ONNX input dim {actor_obs.shape[1]}, expected {self.input_dim}.")
        return actor_obs, time_steps

    def act(self, env: Any, fallback_step: int) -> torch.Tensor:
        actor_obs, time_steps = self.build_input(env, fallback_step)
        outputs = self._run_session(actor_obs, time_steps)
        actions = outputs["actions"]
        self.prev_actions[:] = actions
        if hasattr(env, "student_prev_actions"):
            env.student_prev_actions[:] = actions
        if hasattr(env, "student_base_actions"):
            env.student_base_actions[:] = actions
        return actions

    def handle_dones(self, dones: torch.Tensor) -> None:
        done_env_ids = dones.nonzero(as_tuple=False).flatten()
        self.reset(done_env_ids)


@torch.no_grad()
def evaluate_onnx_policy_with_summary(
    env: Any,
    policy: FusedResidualOnnxRuntimePolicy,
    max_eval_steps: int,
) -> dict[str, float | int]:
    env.set_is_evaluating()
    env.reset_all()
    policy.reset()

    rollout_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    completed_episode_rewards: list[float] = []
    completed_episode_lengths: list[float] = []

    actor_state: dict[str, torch.Tensor] = {"actions": torch.zeros(env.num_envs, env.dim_actions, device=env.device)}
    for step in itertools.islice(itertools.count(), max_eval_steps):
        actor_state["actions"] = policy.act(env, step)
        _, rewards, dones, _ = env.step(actor_state)

        rewards = rewards.detach().to(device=env.device).view(-1)
        dones = dones.detach().to(device=env.device).view(-1).bool()
        rollout_rewards += rewards
        episode_rewards += rewards
        episode_lengths += 1.0

        if dones.any():
            completed_episode_rewards.extend(episode_rewards[dones].detach().cpu().tolist())
            completed_episode_lengths.extend(episode_lengths[dones].detach().cpu().tolist())
            episode_rewards[dones] = 0.0
            episode_lengths[dones] = 0.0
            policy.handle_dones(dones)

    summary: dict[str, float | int] = {
        "steps": max_eval_steps,
        "num_envs": int(env.num_envs),
        "completed_episodes": len(completed_episode_rewards),
        "mean_reward_per_step": float((rollout_rewards / max(max_eval_steps, 1)).mean().detach().cpu()),
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
def watch_onnx_policy(env: Any, policy: FusedResidualOnnxRuntimePolicy, log_interval: int = 100) -> None:
    env.set_is_evaluating()
    env.reset_all()
    policy.reset()

    actor_state: dict[str, torch.Tensor] = {"actions": torch.zeros(env.num_envs, env.dim_actions, device=env.device)}
    logger.info("ONNX watch loop started. Close the viewer or press Ctrl+C to stop.")
    try:
        for step in itertools.count():
            actor_state["actions"] = policy.act(env, step)
            _, rewards, dones, _ = env.step(actor_state)
            dones = dones.detach().view(-1).bool()
            if dones.any():
                policy.handle_dones(dones)
            if step == 0 or (step + 1) % log_interval == 0:
                reward_mean = float(rewards.detach().mean().cpu())
                done_count = int(dones.detach().sum().cpu())
                logger.info(f"ONNX watch step {step + 1}: reward_mean={reward_mean:.4f}, done_count={done_count}")
    except KeyboardInterrupt:
        logger.info("ONNX watch mode interrupted by user.")


def run_onnx_residual_eval_with_tyro(
    tyro_config: ExperimentConfig,
    onnx_path: str | Path,
    metadata: dict[str, Any],
    *,
    device: str,
    object_scale: float | None = None,
) -> None:
    _append_enable_cameras_arg_for_onnx()
    tyro_config = resolve_observation_term_overrides(tyro_config)
    env, device, simulation_app = setup_simulation_environment(tyro_config, device=device)
    _ensure_eval_runtime_randomization_defaults(env)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    if object_scale is not None:
        eval_log_dir = eval_log_dir.with_name(f"{eval_log_dir.name}-scale_{_scale_label(object_scale)}")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving ONNX eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    export_spec = metadata["policy_export_spec"]
    policy = FusedResidualOnnxRuntimePolicy(onnx_path, export_spec, env, device)
    logger.info(
        "Loaded fused ONNX policy. Student actor and DI/DI-pro encoder are embedded in the graph; "
        "no student/DI checkpoint files are loaded by this eval path."
    )
    logger.warning(
        "The current fused ONNX export contains joint_pos/joint_vel/ref pose arrays only. "
        "WBT simulator reset, rewards, and termination still use motion_command from the saved config."
    )

    try:
        if tyro_config.training.max_eval_steps is None:
            watch_onnx_policy(env, policy)
        else:
            summary = evaluate_onnx_policy_with_summary(env, policy, tyro_config.training.max_eval_steps)
            scale_suffix = "" if object_scale is None else f" at object_scale={object_scale:g}"
            logger.info(f"Residual ONNX eval summary{scale_suffix}:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


def run_residual_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    *,
    device: str,
    object_scale: float | None = None,
) -> None:
    tyro_config = resolve_observation_term_overrides(tyro_config)
    env, device, simulation_app = setup_simulation_environment(tyro_config, device=device)
    _ensure_eval_runtime_randomization_defaults(env)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    if object_scale is not None:
        eval_log_dir = eval_log_dir.with_name(f"{eval_log_dir.name}-scale_{_scale_label(object_scale)}")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
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
        watch_residual_policy(algo)
    else:
        summary = evaluate_residual_policy_with_summary(algo, max_eval_steps=tyro_config.training.max_eval_steps)
        scale_suffix = "" if object_scale is None else f" at object_scale={object_scale:g}"
        logger.info(f"Residual eval summary{scale_suffix}:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        tyro.cli(ResidualEvalConfig)
        return

    residual_eval_cfg, remaining_args = tyro.cli(ResidualEvalConfig, return_unknown_args=True, add_help=False)
    if _is_onnx_checkpoint(residual_eval_cfg.checkpoint):
        assert residual_eval_cfg.checkpoint is not None
        onnx_path = Path(residual_eval_cfg.checkpoint).expanduser()
        saved_cfg, _, metadata = _load_onnx_experiment_config(onnx_path)
        eval_cfg = saved_cfg.get_eval_config()
        eval_cfg = _resolve_onnx_eval_config(eval_cfg, residual_eval_cfg)
        eval_cfg = resolve_multi_object_urdf_config(eval_cfg)

        overwritten_tyro_config = tyro.cli(
            ExperimentConfig,
            default=eval_cfg,
            args=remaining_args,
            description="Overriding config on top of what's loaded from ONNX metadata.",
            config=TYRO_CONIFG,
        )
        print("overwritten_tyro_config: ", overwritten_tyro_config)
        device = _resolve_device(residual_eval_cfg)
        object_scale_eval_values = _resolve_object_scale_eval_values(residual_eval_cfg)
        if object_scale_eval_values:
            num_scales = len(object_scale_eval_values)
            for scale_index, object_scale in enumerate(object_scale_eval_values, start=1):
                logger.info(
                    f"Starting residual ONNX eval scale pass {scale_index}/{num_scales}: "
                    f"object_scale={object_scale:g}"
                )
                scaled_tyro_config = _apply_fixed_object_spawn_scale(
                    overwritten_tyro_config,
                    object_scale,
                    residual_eval_cfg.object_scale_height,
                )
                run_onnx_residual_eval_with_tyro(
                    scaled_tyro_config,
                    onnx_path,
                    metadata,
                    device=device,
                    object_scale=object_scale,
                )
            return

        run_onnx_residual_eval_with_tyro(
            overwritten_tyro_config,
            onnx_path,
            metadata,
            device=device,
        )
        return

    checkpoint_cfg = CheckpointConfig(checkpoint=residual_eval_cfg.checkpoint)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)

    eval_cfg = saved_cfg.get_eval_config()
    eval_cfg = dataclasses.replace(
        eval_cfg,
        teacher=residual_eval_cfg.teacher,
        ir_ae=residual_eval_cfg.ir_ae or eval_cfg.ir_ae,
        di_ae=residual_eval_cfg.di_ae or eval_cfg.di_ae,
        di_pro_ae=residual_eval_cfg.di_pro_ae or eval_cfg.di_pro_ae,
    )
    eval_cfg = _resolve_eval_config(eval_cfg, residual_eval_cfg)
    eval_cfg = resolve_multi_object_urdf_config(eval_cfg)

    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    print("overwritten_tyro_config: ", overwritten_tyro_config)
    device = _resolve_device(residual_eval_cfg)
    object_scale_eval_values = _resolve_object_scale_eval_values(residual_eval_cfg)
    if object_scale_eval_values:
        num_scales = len(object_scale_eval_values)
        for scale_index, object_scale in enumerate(object_scale_eval_values, start=1):
            logger.info(
                f"Starting residual eval scale pass {scale_index}/{num_scales}: object_scale={object_scale:g}"
            )
            scaled_tyro_config = _apply_fixed_object_spawn_scale(
                overwritten_tyro_config,
                object_scale,
                residual_eval_cfg.object_scale_height,
            )
            run_residual_eval_with_tyro(
                scaled_tyro_config,
                checkpoint_cfg,
                saved_cfg,
                saved_wandb_path,
                device=device,
                object_scale=object_scale,
            )
        return

    run_residual_eval_with_tyro(
        overwritten_tyro_config,
        checkpoint_cfg,
        saved_cfg,
        saved_wandb_path,
        device=device,
    )


if __name__ == "__main__":
    main()
