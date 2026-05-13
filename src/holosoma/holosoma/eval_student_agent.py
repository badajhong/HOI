from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any, Literal

import tyro
from loguru import logger
from pydantic.dataclasses import dataclass

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.command import MotionConfig
from holosoma.config_types.env import resolve_observation_term_overrides
from holosoma.config_types.experiment import ExperimentConfig
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
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG

DEFAULT_CHECKPOINT = "/home/rllab/haechan/holosoma/logs/student/20260510_student/model_07000.pt"
DEFAULT_DI_PRO_AE = "/home/rllab/haechan/holosoma/logs/AE/20260509_ae_suitcase/best.pt"
DEFAULT_MOTION_FILE = (
    "/home/rllab/haechan/holosoma/src/holosoma_retargeting/holosoma_retargeting/"
    "converted_res/object_interaction/sub1_suitcase_001.npz"
)
DEFAULT_OBJECT_URDF_PATH = (
    "/home/rllab/haechan/holosoma/src/holosoma_retargeting/holosoma_retargeting/"
    "models/objects/suitcase/suitcase.urdf"
)


@dataclass(frozen=True)
class StudentEvalConfig(CheckpointConfig):
    checkpoint: str | None = DEFAULT_CHECKPOINT
    """Student checkpoint to evaluate."""

    motion_file: str | None = DEFAULT_MOTION_FILE
    """Evaluate one motion .npz file. Clears motion_folder when motion_folder is not provided."""

    motion_folder: str | None = None
    """Evaluate all .npz files in a folder. Takes precedence over motion_file when provided."""

    object_urdf_path: str | None = DEFAULT_OBJECT_URDF_PATH
    """URDF path for a single-object evaluation."""

    object_urdf_asset: str | None = None
    """Folder containing object URDFs for multi-object motion folders."""

    ir_ae: str | None = None
    """Optional latent AE checkpoint override for student evaluation."""

    di_ae: str | None = None
    """Optional depth latent AE checkpoint override for student evaluation."""

    di_pro_ae: str | None = DEFAULT_DI_PRO_AE
    """Optional depth+proprioception latent AE checkpoint override for student evaluation."""

    teacher: str | None = None
    """Optional teacher checkpoint override. Defaults to disabled during eval."""

    num_envs: int | None = None
    """Override evaluation environment count."""

    headless: bool | None = None
    """Override rendering mode. Use false for an interactive viewer."""

    max_eval_steps: int | None = None
    """Stop evaluation after this many policy steps."""

    export_onnx: bool | None = None
    """Override whether evaluation exports ONNX next to the checkpoint."""

    env_spacing: float | None = None
    """Override simulator environment spacing."""

    start_at_timestep_zero_prob: float | None = 1.0
    """Override the motion command reset probability for starting at frame zero."""

    device: Literal["cpu", "gpu"] = "cpu"
    """Simulation device choice. Use 'gpu' for cuda:0."""


def _replace_motion_config(motion_config: Any, updates: dict[str, Any]) -> Any:
    if isinstance(motion_config, MotionConfig):
        return dataclasses.replace(motion_config, **updates)
    if isinstance(motion_config, dict):
        updated_motion_config = dict(motion_config)
        updated_motion_config.update(updates)
        return updated_motion_config
    raise TypeError(f"Unsupported motion_config type: {type(motion_config).__name__}")


def _apply_motion_overrides(config: ExperimentConfig, cli_cfg: StudentEvalConfig) -> ExperimentConfig:
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


def _apply_object_overrides(config: ExperimentConfig, cli_cfg: StudentEvalConfig) -> ExperimentConfig:
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


def _apply_training_overrides(config: ExperimentConfig, cli_cfg: StudentEvalConfig) -> ExperimentConfig:
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


def _apply_simulator_overrides(config: ExperimentConfig, cli_cfg: StudentEvalConfig) -> ExperimentConfig:
    if cli_cfg.env_spacing is None:
        return config

    scene_config = dataclasses.replace(config.simulator.config.scene, env_spacing=cli_cfg.env_spacing)
    simulator_init_config = dataclasses.replace(config.simulator.config, scene=scene_config)
    return dataclasses.replace(config, simulator=dataclasses.replace(config.simulator, config=simulator_init_config))


def apply_student_eval_overrides(config: ExperimentConfig, cli_cfg: StudentEvalConfig) -> ExperimentConfig:
    config = _apply_motion_overrides(config, cli_cfg)
    config = _apply_object_overrides(config, cli_cfg)
    config = _apply_training_overrides(config, cli_cfg)
    return _apply_simulator_overrides(config, cli_cfg)


def _resolve_device(cli_cfg: StudentEvalConfig) -> str:
    return "cuda:0" if cli_cfg.device == "gpu" else "cpu"


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    *,
    device: str,
):
    tyro_config = resolve_observation_term_overrides(tyro_config)

    # Use shared simulation environment setup
    env, device, simulation_app = setup_simulation_environment(tyro_config, device=device)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
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
    exported_policy_name = checkpoint_path.split("/")[-1]  # example: model_5000.pt
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")  # example: model_5000.onnx

    if tyro_config.training.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        algo.export(onnx_file_path=exported_onnx_path)  # type: ignore[attr-defined]
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    algo.evaluate_policy(
        max_eval_steps=tyro_config.training.max_eval_steps,
    )

    # Cleanup simulation app
    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        tyro.cli(StudentEvalConfig)
        return

    student_eval_cfg, remaining_args = tyro.cli(StudentEvalConfig, return_unknown_args=True, add_help=False)
    checkpoint_cfg = CheckpointConfig(checkpoint=student_eval_cfg.checkpoint)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    eval_cfg = dataclasses.replace(
        eval_cfg,
        teacher=student_eval_cfg.teacher,
        ir_ae=student_eval_cfg.ir_ae or eval_cfg.ir_ae,
        di_ae=student_eval_cfg.di_ae or eval_cfg.di_ae,
        di_pro_ae=student_eval_cfg.di_pro_ae or eval_cfg.di_pro_ae,
    )
    eval_cfg = apply_student_eval_overrides(eval_cfg, student_eval_cfg)
    eval_cfg = resolve_multi_object_urdf_config(eval_cfg)
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    print("overwritten_tyro_config: ", overwritten_tyro_config)
    run_eval_with_tyro(
        overwritten_tyro_config,
        checkpoint_cfg,
        saved_cfg,
        saved_wandb_path,
        device=_resolve_device(student_eval_cfg),
    )


if __name__ == "__main__":
    main()
