from __future__ import annotations

import dataclasses
import glob
import os
from pathlib import Path

from loguru import logger

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.path import resolve_data_file_path


def extract_object_key_from_motion_name(npz_path: str) -> str:
    """Extract object key from motion filename: sub{number}_{object}_{something}.npz."""
    stem = Path(npz_path).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Could not parse object key from motion file '{npz_path}'. Expected: sub{{number}}_{{object}}_{{something}}.npz"
        )
    return parts[1]


def resolve_object_urdf_for_key(object_urdf_folder: str, object_key: str) -> str:
    """Resolve a URDF for an object key from a folder with common layouts."""
    candidates = [
        Path(object_urdf_folder) / object_key / f"{object_key}.urdf",
        Path(object_urdf_folder) / f"{object_key}.urdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    nested = sorted((Path(object_urdf_folder) / object_key).glob("*.urdf"))
    if nested:
        return str(nested[0])

    raise FileNotFoundError(
        f"No URDF found for object key '{object_key}' under object-urdf-folder '{object_urdf_folder}'. "
        "Supported layouts: <folder>/<key>/<key>.urdf or <folder>/<key>.urdf"
    )


def resolve_multi_object_urdf_config(tyro_config: ExperimentConfig) -> ExperimentConfig:
    """Populate object key -> URDF path mapping when evaluation/training uses a motion folder."""
    object_cfg = tyro_config.robot.object
    object_urdf_folder = object_cfg.object_urdf_asset or object_cfg.object_urdf_folder
    if not object_urdf_folder:
        return tyro_config

    setup_term = tyro_config.command.setup_terms.get("motion_command")
    if setup_term is None or "motion_config" not in setup_term.params:
        return tyro_config

    motion_cfg = setup_term.params["motion_config"]
    if isinstance(motion_cfg, dict):
        motion_folder = motion_cfg.get("motion_folder", "")
        motion_file = motion_cfg.get("motion_file", "")
    else:
        motion_folder = getattr(motion_cfg, "motion_folder", "")
        motion_file = getattr(motion_cfg, "motion_file", "")

    if not motion_folder and not motion_file:
        return tyro_config

    resolved_folder = resolve_data_file_path(object_urdf_folder)
    if not Path(resolved_folder).exists():
        raise FileNotFoundError(f"object_urdf_folder does not exist: {resolved_folder}")

    motion_files: list[str]
    if motion_folder:
        resolved_motion_folder = resolve_data_file_path(motion_folder)
        motion_files = sorted(glob.glob(os.path.join(resolved_motion_folder, "*.npz")))
        if not motion_files:
            raise ValueError(f"No .npz motion files found in motion_folder: {resolved_motion_folder}")
    else:
        motion_files = [resolve_data_file_path(motion_file)]

    object_keys = sorted({extract_object_key_from_motion_name(mf) for mf in motion_files})
    object_name_to_path = {key: resolve_object_urdf_for_key(resolved_folder, key) for key in object_keys}

    default_object_path = object_cfg.object_urdf_path or next(iter(object_name_to_path.values()))
    new_object_cfg = dataclasses.replace(
        object_cfg,
        object_urdf_asset=resolved_folder,
        object_urdf_folder=resolved_folder,
        object_urdf_name_to_path=object_name_to_path,
        object_urdf_path=default_object_path,
    )
    new_robot_cfg = dataclasses.replace(tyro_config.robot, object=new_object_cfg)
    logger.info(f"Resolved object URDFs from folder: {object_name_to_path}")
    return dataclasses.replace(tyro_config, robot=new_robot_cfg)
