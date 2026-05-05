from __future__ import annotations

import dataclasses

from loguru import logger
from pydantic.dataclasses import dataclass

from holosoma.config_types.action import ActionManagerCfg
from holosoma.config_types.command import CommandManagerCfg
from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_types.logger import LoggerConfig
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_types.reward import RewardManagerCfg
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.termination import TerminationManagerCfg
from holosoma.config_types.terrain import TerrainManagerCfg

_DEPTH_CAMERA_ASSET_VARIANTS: dict[str, dict[str, tuple[str, str, bool]]] = {
    "g1/g1_29dof.urdf": {
        "original": ("g1/g1_29dof.urdf", "g1/g1_29dof.xml", True),
        "realsense": ("g1/g1_29dof_realsense.urdf", "g1/g1_29dof_realsense.xml", False),
    },
    "g1/g1_29dof_realsense.urdf": {
        "original": ("g1/g1_29dof.urdf", "g1/g1_29dof.xml", True),
        "realsense": ("g1/g1_29dof_realsense.urdf", "g1/g1_29dof_realsense.xml", False),
    },
}

_ROBOT_DEPTH_ASSET_MODE_CHOICES = {"auto", "original", "realsense"}


@dataclass(frozen=True)
class EnvConfig:
    """Collection of configs needed for constructing env classes."""

    env_class: str

    simulator: SimulatorConfig
    terrain: TerrainManagerCfg
    observation: ObservationManagerCfg | None
    action: ActionManagerCfg | None
    reward: RewardManagerCfg | None
    termination: TerminationManagerCfg | None
    randomization: RandomizationManagerCfg | None
    command: CommandManagerCfg | None
    curriculum: CurriculumManagerCfg | None
    robot: RobotConfig
    training: TrainingConfig
    logger: LoggerConfig
    teacher: str | None = None
    student: str | None = None
    ir_ae: str | None = None
    ir_ae_body_source: str | None = None
    di_ae: str | None = None


def _replace_observation_term_params(
    observation_cfg: ObservationManagerCfg | None,
    *,
    group_name: str,
    term_name: str,
    updates: dict[str, object | None],
) -> ObservationManagerCfg | None:
    """Return observation config with selected term params updated."""
    if observation_cfg is None:
        return observation_cfg

    group_cfg = observation_cfg.groups.get(group_name)
    if group_cfg is None:
        return observation_cfg

    term_cfg = group_cfg.terms.get(term_name)
    if term_cfg is None:
        return observation_cfg

    new_params = dict(term_cfg.params)
    changed = False
    for key, value in updates.items():
        if value is None or value == "":
            continue
        if new_params.get(key) == value:
            continue
        new_params[key] = value
        changed = True

    if not changed:
        return observation_cfg

    new_terms = dict(group_cfg.terms)
    new_terms[term_name] = dataclasses.replace(term_cfg, params=new_params)
    new_groups = dict(observation_cfg.groups)
    new_groups[group_name] = dataclasses.replace(group_cfg, terms=new_terms)
    return dataclasses.replace(observation_cfg, groups=new_groups)


def _ensure_di_ae_latent_group(observation_cfg: ObservationManagerCfg | None) -> ObservationManagerCfg | None:
    """Ensure residual observation configs expose a shared di_ae_latent group."""
    if observation_cfg is None:
        return observation_cfg

    existing_group = observation_cfg.groups.get("di_ae_latent")
    if existing_group is not None and "di_ae_latent" in existing_group.terms:
        return observation_cfg

    student_base_group = observation_cfg.groups.get("student_base_action")
    student_base_term = None
    if student_base_group is not None:
        student_base_term = student_base_group.terms.get("student_base_action")

    if student_base_group is None and existing_group is None:
        return observation_cfg

    legacy_params = dict(student_base_term.params) if student_base_term is not None else {}
    di_ae_term = ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:DIAELatent",
        params={
            "checkpoint_path": legacy_params.get("checkpoint_path", ""),
            "condition_text": legacy_params.get("condition_text", ""),
            "robot_depth_asset_mode": legacy_params.get("robot_depth_asset_mode", "auto"),
            "debug_save_depth_images": legacy_params.get("debug_save_depth_images", False),
            "debug_depth_save_interval": legacy_params.get("debug_depth_save_interval", 200),
            "debug_depth_env_ids": legacy_params.get("debug_depth_env_ids", "0"),
        },
        scale=1.0,
        noise=0.0,
    )
    di_ae_group = ObsGroupCfg(
        concatenate=True,
        enable_noise=False,
        history_length=1,
        terms={"di_ae_latent": di_ae_term},
    )

    new_groups: dict[str, ObsGroupCfg] = {}
    inserted = False
    for group_name, group_cfg in observation_cfg.groups.items():
        if group_name == "di_ae_latent":
            continue
        if group_name == "student_base_action" and not inserted:
            new_groups["di_ae_latent"] = di_ae_group
            inserted = True
        new_groups[group_name] = group_cfg
    if not inserted:
        new_groups["di_ae_latent"] = di_ae_group

    return dataclasses.replace(observation_cfg, groups=new_groups)


def _observation_term_requires_depth_camera(term_cfg: ObsTermCfg) -> bool:
    if term_cfg.func in {
        "holosoma.managers.observation.terms.wbt:DIAELatent",
        "holosoma.managers.observation.terms.wbt:FrozenStudentBaseAction",
    }:
        return True

    if term_cfg.func in {
        "holosoma.managers.observation.terms.wbt:AELatent",
        "holosoma.managers.observation.terms.wbt:StudentLatent",
    }:
        source = str(term_cfg.params.get("source", "")).strip().lower()
        return source == "di" or bool(term_cfg.params.get("di_checkpoint_path"))

    return False


def _config_requires_robot_depth_camera(
    tyro_config: ExperimentConfig,
    observation_cfg: ObservationManagerCfg | None,
) -> bool:
    if bool(getattr(tyro_config, "di_ae", None)):
        return True
    if observation_cfg is None:
        return False

    for group_cfg in observation_cfg.groups.values():
        for term_cfg in group_cfg.terms.values():
            if _observation_term_requires_depth_camera(term_cfg):
                return True
    return False


def _resolve_depth_camera_robot_asset(
    tyro_config: ExperimentConfig,
    observation_cfg: ObservationManagerCfg | None,
) -> ExperimentConfig:
    requested_mode = "auto"
    observation_requested_modes: set[str] = set()
    if observation_cfg is not None:
        for group_cfg in observation_cfg.groups.values():
            for term_cfg in group_cfg.terms.values():
                if not _observation_term_requires_depth_camera(term_cfg):
                    continue
                raw_mode = str(term_cfg.params.get("robot_depth_asset_mode", "") or "").strip().lower()
                if not raw_mode:
                    continue
                if raw_mode not in _ROBOT_DEPTH_ASSET_MODE_CHOICES:
                    raise ValueError(
                        "Unsupported observation-term robot_depth_asset_mode "
                        f"'{raw_mode}'. Expected one of: {sorted(_ROBOT_DEPTH_ASSET_MODE_CHOICES)}."
                    )
                observation_requested_modes.add(raw_mode)

    if len(observation_requested_modes) > 1:
        raise ValueError(
            "Conflicting robot_depth_asset_mode values were found across depth-related observation terms: "
            f"{sorted(observation_requested_modes)}. Use a single consistent mode."
        )
    if observation_requested_modes:
        requested_mode = next(iter(observation_requested_modes))

    depth_camera_required = _config_requires_robot_depth_camera(tyro_config, observation_cfg)
    if not depth_camera_required and requested_mode == "auto":
        return tyro_config

    asset_cfg = tyro_config.robot.asset
    asset_variants = _DEPTH_CAMERA_ASSET_VARIANTS.get(asset_cfg.urdf_file)
    if asset_variants is None:
        logger.warning(
            "No known robot asset variants for depth-camera override on "
            f"urdf_file='{asset_cfg.urdf_file}'. Keeping the current asset config unchanged."
        )
        return tyro_config

    target_mode = "realsense" if depth_camera_required and requested_mode == "auto" else requested_mode
    target_variant = asset_variants.get(target_mode)
    if target_variant is None:
        logger.warning(
            f"Robot depth asset mode '{target_mode}' is not available for urdf_file='{asset_cfg.urdf_file}'. "
            "Keeping the current asset config unchanged."
        )
        return tyro_config

    target_urdf_file, target_xml_file, target_collapse_fixed_joints = target_variant
    if (
        asset_cfg.urdf_file == target_urdf_file
        and asset_cfg.xml_file == target_xml_file
        and asset_cfg.collapse_fixed_joints is target_collapse_fixed_joints
    ):
        return tyro_config

    new_asset_cfg = dataclasses.replace(
        asset_cfg,
        urdf_file=target_urdf_file,
        xml_file=target_xml_file,
        collapse_fixed_joints=target_collapse_fixed_joints,
    )
    new_robot_cfg = dataclasses.replace(tyro_config.robot, asset=new_asset_cfg)
    if depth_camera_required and target_mode == "realsense":
        logger.info(
            "Depth latent observation requires robot-mounted camera support. "
            f"Switching robot assets to urdf='{target_urdf_file}', xml='{target_xml_file}', "
            "collapse_fixed_joints=False."
        )
    elif depth_camera_required and target_mode == "original":
        logger.info(
            "Depth latent observation requested with robot_depth_asset_mode='original'. "
            f"Using original robot assets urdf='{target_urdf_file}', xml='{target_xml_file}', "
            "and relying on simulator-side depth-camera mounting when available."
        )
    else:
        logger.info(
            f"Applying robot_depth_asset_mode='{target_mode}': "
            f"urdf='{target_urdf_file}', xml='{target_xml_file}', "
            f"collapse_fixed_joints={target_collapse_fixed_joints}."
        )
    return dataclasses.replace(tyro_config, robot=new_robot_cfg)


def resolve_observation_term_overrides(tyro_config: ExperimentConfig) -> ExperimentConfig:
    """Project legacy top-level latent overrides into observation-term params.

    Observation terms are the canonical place to configure latent encoders used
    by student and residual policies. We still accept legacy top-level
    ``ir_ae`` / ``ir_ae_body_source`` / ``di_ae`` / ``student`` overrides and
    mirror them into the relevant observation terms for backward compatibility.
    """
    observation_cfg = tyro_config.observation
    observation_cfg = _ensure_di_ae_latent_group(observation_cfg)
    observation_cfg = _replace_observation_term_params(
        observation_cfg,
        group_name="ae_latent",
        term_name="ae_latent",
        updates={
            "checkpoint_path": tyro_config.ir_ae,
            "di_checkpoint_path": tyro_config.di_ae,
            "body_source": tyro_config.ir_ae_body_source,
        },
    )
    observation_cfg = _replace_observation_term_params(
        observation_cfg,
        group_name="student_latent",
        term_name="student_latent",
        updates={
            "checkpoint_path": tyro_config.ir_ae,
            "di_checkpoint_path": tyro_config.di_ae,
            "body_source": tyro_config.ir_ae_body_source,
        },
    )
    observation_cfg = _replace_observation_term_params(
        observation_cfg,
        group_name="ir_ae_latent",
        term_name="ir_ae_latent",
        updates={
            "checkpoint_path": tyro_config.ir_ae,
            "di_checkpoint_path": tyro_config.di_ae,
            "body_source": tyro_config.ir_ae_body_source,
        },
    )
    observation_cfg = _replace_observation_term_params(
        observation_cfg,
        group_name="student_base_action",
        term_name="student_base_action",
        updates={
            "student_checkpoint": tyro_config.student,
            "latent_obs_group": "di_ae_latent",
        },
    )
    observation_cfg = _replace_observation_term_params(
        observation_cfg,
        group_name="di_ae_latent",
        term_name="di_ae_latent",
        updates={
            "checkpoint_path": tyro_config.di_ae,
        },
    )

    resolved_config = tyro_config
    if observation_cfg is not tyro_config.observation:
        resolved_config = dataclasses.replace(tyro_config, observation=observation_cfg)
    return _resolve_depth_camera_robot_asset(resolved_config, observation_cfg)


def get_tyro_env_config(tyro_config: ExperimentConfig) -> EnvConfig:
    """Convert ExperimentConfig to EnvConfig for environment construction.

    Parameters
    ----------
    tyro_config : ExperimentConfig
        The experiment configuration containing all settings.

    Returns
    -------
    EnvConfig
        Environment configuration with extracted fields.
    """
    tyro_config = resolve_observation_term_overrides(tyro_config)
    return EnvConfig(
        env_class=tyro_config.env_class,
        teacher=tyro_config.teacher,
        student=tyro_config.student,
        ir_ae=tyro_config.ir_ae,
        ir_ae_body_source=tyro_config.ir_ae_body_source,
        di_ae=tyro_config.di_ae,
        training=tyro_config.training,
        simulator=tyro_config.simulator,
        terrain=tyro_config.terrain,
        observation=tyro_config.observation,
        action=tyro_config.action,
        reward=tyro_config.reward,
        termination=tyro_config.termination,
        randomization=tyro_config.randomization,
        command=tyro_config.command,
        curriculum=tyro_config.curriculum,
        robot=tyro_config.robot,
        logger=tyro_config.logger,
    )
