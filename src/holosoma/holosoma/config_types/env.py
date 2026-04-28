from __future__ import annotations

import dataclasses

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
            "debug_save_depth_images": legacy_params.get("debug_save_depth_images", False),
            "debug_depth_save_interval": legacy_params.get("debug_depth_save_interval", 200),
            "debug_depth_env_ids": legacy_params.get("debug_depth_env_ids", (0,)),
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
        group_name="ir_ae_latent",
        term_name="ir_ae_latent",
        updates={
            "checkpoint_path": tyro_config.ir_ae,
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

    if observation_cfg is tyro_config.observation:
        return tyro_config
    return dataclasses.replace(tyro_config, observation=observation_cfg)


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
