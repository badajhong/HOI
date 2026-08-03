from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from pydantic.dataclasses import dataclass

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.wbt.r1 import observation as r1_observation_values
from holosoma.config_values.wbt.r1 import reward as r1_reward_values
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.depth import (
    ROBOT_DEPTH_MAX_M,
    ROBOT_DEPTH_MIN_M,
    ROBOT_DEPTH_HORIZONTAL_FOV_DEG,
    ROBOT_DEPTH_OUTPUT_RESOLUTION_WH,
    ROBOT_DEPTH_RAW_RESOLUTION_WH,
    ROBOT_DEPTH_VERTICAL_FOV_DEG,
    preprocess_robot_depth_array,
)
from holosoma.utils.eval_utils import (
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.object_urdf import resolve_multi_object_urdf_config
from holosoma.utils.rotations import quat_rotate_inverse
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.surface_features import SurfaceFeatureComputer
from holosoma.utils.task_phase import TwoPhaseSchedule
from holosoma.utils.tyro_utils import TYRO_CONIFG


ORIGINAL_G1_URDF_FILE = "g1/g1_29dof.urdf"
ORIGINAL_G1_XML_FILE = "g1/g1_29dof.xml"
DEFAULT_IR_CHECKPOINT = (
    "./logs/teacher/"
    "20260727_070811-r1_teacher-locomotion/model_30000.pt"
)
DEFAULT_IR_NUM_ENVS = 24
DEFAULT_IR_LOG_BASE_DIR = "./logs"
DEFAULT_IR_PROJECT = "ir_di_pro_largebox"
REALSENSE_CAMERA_BODY_LINK = "realsense_d435_link"
DEPTH_CAMERA_FRAME_LINK = "realsense_d435_depth_optical_frame"
DEPTH_CAMERA_PRIM_NAME = "realsense_d435_depth"
DEPTH_CAMERA_FALLBACK_PARENT_LINK = "torso_link"
DEPTH_CAMERA_FALLBACK_PARENT_CANDIDATES = ("torso_link", "waist_yaw_link", "pelvis_link", "pelvis")
DEPTH_CAMERA_FALLBACK_POS = (0.085, 0.0, 0.42)
R1_DEPTH_CAMERA_PARENT_LINK = "waist_yaw_link"
R1_DEPTH_CAMERA_MOUNT_PRESETS = {
    "cam1": (
        (0.085, 0.0, 0.42),
        (0.2126281001816864, -0.6743797093191588, 0.6743809086432211, -0.212630404056609),
    ),
    "cam2": (
        (0.075, 0.0, 0.21),
        (0.7071067811865476, 0.0, 0.7071067811865475, 0.0),
    ),
}
DEPTH_CAMERA_FALLBACK_ROT_ROS_WXYZ = R1_DEPTH_CAMERA_MOUNT_PRESETS["cam1"][1]
# IsaacSim camera config uses (width, height). Saved depth_window tensors use
# [window, height, width], so telemetry stores both conventions explicitly.
DEPTH_RESOLUTION = ROBOT_DEPTH_OUTPUT_RESOLUTION_WH
IR_SURFACE_FEATURE_COMPONENT_NAMES = (
    "phi",
    "grad_phi_x",
    "grad_phi_y",
    "grad_phi_z",
    "v_t_x",
    "v_t_y",
    "v_t_z",
    "v_norm_x",
    "v_norm_y",
    "v_norm_z",
    "v_tan_x",
    "v_tan_y",
    "v_tan_z",
)
IR_SURFACE_FEATURE_BODY_SOURCE_CHOICES = ("pelvis", "hands", "feet", "all")
IR_SURFACE_FEATURE_BODY_SOURCE_BASE_CHOICES = ("pelvis", "hands", "feet")
IR_SURFACE_FEATURE_BODY_SOURCE_ALL_RESOLVED = ("hands", "pelvis", "feet")
IR_HAND_BODY_LABELS = ("left_hand", "right_hand")
IR_FOOT_BODY_LABELS = ("left_foot", "right_foot")
IR_LEFT_HAND_BODY_NAME_CANDIDATES = (
    "left_hand_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
)
IR_RIGHT_HAND_BODY_NAME_CANDIDATES = (
    "right_hand_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
)
IR_FOOT_BODY_NAME_CANDIDATES_BY_ROBOT_TYPE = {
    "g1": {
        "left": ("left_foot_contact_point", "left_ankle_roll_link", "left_foot_link"),
        "right": ("right_foot_contact_point", "right_ankle_roll_link", "right_foot_link"),
    },
    "r1": {
        "left": ("left_ankle_roll_link", "left_foot_front_inner_link", "left_foot_front_outer_link"),
        "right": ("right_ankle_roll_link", "right_foot_front_inner_link", "right_foot_front_outer_link"),
    },
}
IR_PELVIS_BODY_NAME_CANDIDATES_BY_ROBOT_TYPE = {
    "g1": ("pelvis", "pelvis_link"),
    "r1": ("pelvis_link", "pelvis"),
}
PROPRIOCEPTION_COMPONENT_NAMES = (
    "base_ang_vel",
    "dof_pos",
    "dof_vel",
)


@dataclass(frozen=True)
class IRCheckpointConfig:
    checkpoint: str | None = DEFAULT_IR_CHECKPOINT
    """Path to a local checkpoint file, or W&B URI in the format `wandb://<entity>/<project>/<run_id>[/<checkpoint_name>]`."""

    max_eval_steps: int | None = None
    """Maximum number of evaluation steps inside a single episode."""

    num_eval_episodes: int | None = None
    """Number of episodes to collect per environment before ending the IR run. None means keep running until manually stopped."""

    evaluate_all_motions: bool = True
    """Evaluate every motion clip once per environment, sequentially from its first frame."""

    all_motions_iterations: int = 10
    """Number of complete all-motion sweeps to collect per environment."""

    headless: bool = True
    """Run IR telemetry collection without showing the simulator window."""

    surface_feature_log_env_ids: tuple[int, ...] = (0,)
    """Environment ids to print for live IR ir_t features during playback."""

    surface_feature_body_source: str = "all"
    """Surface-feature body source selection. Supported values: 'pelvis', 'hands', 'feet', or 'all'."""

    surface_feature_body_name: str | None = None
    """Optional pelvis rigid-body override; otherwise it is resolved from robot_type."""

    left_hand_body_name: str | None = None
    """Optional override for the left-hand rigid body when `surface_feature_body_source` includes 'hands'."""

    right_hand_body_name: str | None = None
    """Optional override for the right-hand rigid body when `surface_feature_body_source` includes 'hands'."""

    left_foot_body_name: str | None = None
    """Optional override for the left-foot rigid body when `surface_feature_body_source` includes 'feet'."""

    right_foot_body_name: str | None = None
    """Optional override for the right-foot rigid body when `surface_feature_body_source` includes 'feet'."""

    save_camera_images: bool = False
    """Save per-step depth camera preview images under the IR telemetry folder."""

    show_camera_marker: bool = False
    """Show RGB local axes and a red forward marker at each live IsaacSim depth-camera prim."""

    depth_camera_location: str = "cam1"
    """Depth-camera mount preset: 'cam1' for head or 'cam2' for the forward-facing upper body."""

    camera_position_noise_m: float = 0.02
    """Per-environment camera-mount XYZ position noise half-range in meters."""

    depth_pixel_noise_max_std_m: float = 0.005
    """Gaussian depth-noise standard deviation at the 5 m far limit; scales quadratically with distance."""

    depth_dropout_probability: float = 0.001
    """Independent probability that a valid depth pixel is replaced by zero."""


def _normalize_bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value '{value}'. Expected true/false.")


def _normalize_ir_cli_bool_equals_args(args: list[str]) -> list[str]:
    """Allow `--flag=True/False` spellings for Tyro bool flags used by IR CLI."""
    bool_flags = (
        "--evaluate-all-motions",
        "--headless",
        "--save-camera-images",
        "--show-camera-marker",
    )
    normalized_args: list[str] = []
    for arg in args:
        rewritten = False
        for flag in bool_flags:
            prefix = f"{flag}="
            if not arg.startswith(prefix):
                continue
            flag_value = _normalize_bool_value(arg[len(prefix) :])
            normalized_args.append(flag if flag_value else f"--no-{flag[2:]}")
            rewritten = True
            break
        if not rewritten:
            normalized_args.append(arg)
    return normalized_args


@dataclasses.dataclass(frozen=True)
class DepthCameraMountSpec:
    source_urdf_path: str
    mount_mode: str
    scene_parent_link: str
    parent_link: str
    camera_body_link: str
    optical_frame_link: str
    translation: tuple[float, float, float]
    quaternion_ros_wxyz: tuple[float, float, float, float]
    camera_body_xyz: tuple[float, float, float]
    camera_body_rpy: tuple[float, float, float]
    optical_frame_xyz: tuple[float, float, float]
    optical_frame_rpy: tuple[float, float, float]

    def to_json_dict(self) -> dict:
        return {
            "source_urdf_path": self.source_urdf_path,
            "mount_mode": self.mount_mode,
            "scene_parent_link": self.scene_parent_link,
            "parent_link": self.parent_link,
            "camera_body_link": self.camera_body_link,
            "optical_frame_link": self.optical_frame_link,
            "translation": list(self.translation),
            "quaternion_ros_wxyz": list(self.quaternion_ros_wxyz),
            "camera_body_xyz": list(self.camera_body_xyz),
            "camera_body_rpy": list(self.camera_body_rpy),
            "optical_frame_xyz": list(self.optical_frame_xyz),
            "optical_frame_rpy": list(self.optical_frame_rpy),
        }


def _parse_xyz_or_rpy(origin_value: str | None) -> tuple[float, float, float]:
    if not origin_value:
        return (0.0, 0.0, 0.0)
    values = tuple(float(v) for v in origin_value.split())
    if len(values) != 3:
        raise ValueError(f"Expected 3 values but got {len(values)} from '{origin_value}'")
    return values


def _rpy_to_rotation_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float64,
    )
    rot_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def _rotation_matrix_to_quaternion_wxyz(rotation_matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation_matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        scale = math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        scale = math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
        z = 0.25 * scale

    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat.tolist())


def _quaternion_wxyz_to_rotation_matrix(quaternion_wxyz: np.ndarray | tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in quaternion_wxyz]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _to_numpy_float32(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _ir_t_mode_for_body_source(body_source: str) -> str:
    return f"surface_phi_grad_v_vnorm_vtan_{body_source}_v1"


def _canonical_surface_feature_body_source(body_sources: tuple[str, ...]) -> str:
    if body_sources == ("pelvis",):
        return "pelvis"
    if body_sources == ("hands",):
        return "hands"
    if body_sources == IR_SURFACE_FEATURE_BODY_SOURCE_ALL_RESOLVED:
        return "all"
    return ",".join(body_sources)


def _parse_surface_feature_body_sources(body_source: str) -> tuple[str, ...]:
    normalized = body_source.strip().lower()
    normalized = {"foot": "feet", "foots": "feet"}.get(normalized, normalized)
    if not normalized:
        raise ValueError("surface_feature_body_source must not be empty.")

    if normalized in IR_SURFACE_FEATURE_BODY_SOURCE_BASE_CHOICES:
        return (normalized,)
    if normalized == "all":
        return IR_SURFACE_FEATURE_BODY_SOURCE_ALL_RESOLVED

    # Backward-compatible alias for older comma-separated combinations such as "hands,pelvis".
    parts = [
        {"foot": "feet", "foots": "feet"}.get(part.strip(), part.strip())
        for part in normalized.split(",")
        if part.strip()
    ]
    invalid_parts = [part for part in parts if part not in IR_SURFACE_FEATURE_BODY_SOURCE_BASE_CHOICES]
    if invalid_parts or not parts:
        raise ValueError(
            "surface_feature_body_source contains unsupported values "
            f"{invalid_parts}. Expected entries from {IR_SURFACE_FEATURE_BODY_SOURCE_CHOICES}."
        )

    ordered_unique_parts: list[str] = []
    for part in parts:
        if part not in ordered_unique_parts:
            ordered_unique_parts.append(part)
    return tuple(ordered_unique_parts)


def _ir_t_mode_for_body_sources(body_sources: tuple[str, ...]) -> str:
    return _ir_t_mode_for_body_source(_canonical_surface_feature_body_source(body_sources))


def _ir_t_component_names_for_body_source(body_source: str) -> tuple[str, ...]:
    if body_source == "pelvis":
        return IR_SURFACE_FEATURE_COMPONENT_NAMES
    if body_source == "hands":
        return tuple(
            f"{body_label}_{component_name}"
            for body_label in IR_HAND_BODY_LABELS
            for component_name in IR_SURFACE_FEATURE_COMPONENT_NAMES
        )
    if body_source == "feet":
        return tuple(
            f"{body_label}_{component_name}"
            for body_label in IR_FOOT_BODY_LABELS
            for component_name in IR_SURFACE_FEATURE_COMPONENT_NAMES
        )
    raise ValueError(
        f"Unsupported surface_feature_body_source '{body_source}'. Expected one of {IR_SURFACE_FEATURE_BODY_SOURCE_CHOICES}."
    )


def _ir_t_component_names_for_body_sources(body_sources: tuple[str, ...]) -> tuple[str, ...]:
    if len(body_sources) == 1:
        return _ir_t_component_names_for_body_source(body_sources[0])

    component_names: list[str] = []
    for body_source in body_sources:
        if body_source == "pelvis":
            component_names.extend(f"pelvis_{component_name}" for component_name in IR_SURFACE_FEATURE_COMPONENT_NAMES)
            continue
        component_names.extend(_ir_t_component_names_for_body_source(body_source))
    return tuple(component_names)


def _resolve_surface_feature_body_name(
    available_body_names: list[str],
    explicit_name: str | None,
    candidate_names: tuple[str, ...],
    body_label: str,
) -> str:
    if explicit_name is not None:
        if explicit_name not in available_body_names:
            raise RuntimeError(
                f"Configured {body_label} body '{explicit_name}' was not found in simulator body names: {available_body_names}."
            )
        return explicit_name

    for candidate_name in candidate_names:
        if candidate_name in available_body_names:
            return candidate_name

    raise RuntimeError(
        f"Could not auto-resolve the {body_label} body name. Tried {list(candidate_names)} against available bodies: "
        f"{available_body_names}"
    )


def _robot_family(robot_type: str) -> str:
    normalized = robot_type.lower()
    if normalized.startswith("r1"):
        return "r1"
    if normalized.startswith("g1"):
        return "g1"
    raise ValueError(
        f"Unsupported robot_type '{robot_type}' for automatic foot-body resolution. "
        "Provide --left-foot-body-name and --right-foot-body-name explicitly."
    )


def _compose_local_transform(
    xyz_a: tuple[float, float, float],
    rpy_a: tuple[float, float, float],
    xyz_b: tuple[float, float, float],
    rpy_b: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    rot_a = _rpy_to_rotation_matrix(rpy_a)
    rot_b = _rpy_to_rotation_matrix(rpy_b)
    composed_rot = rot_a @ rot_b
    composed_xyz = np.asarray(xyz_a, dtype=np.float64) + rot_a @ np.asarray(xyz_b, dtype=np.float64)
    composed_quat = _rotation_matrix_to_quaternion_wxyz(composed_rot)
    return tuple(float(v) for v in composed_xyz.tolist()), composed_quat


def _resolve_robot_urdf_path(tyro_config: ExperimentConfig) -> Path:
    asset_cfg = tyro_config.robot.asset
    asset_root = asset_cfg.asset_root or ""
    if asset_root.startswith("@holosoma/"):
        asset_root = asset_root.replace("@holosoma", get_holosoma_root(), 1)

    urdf_file = asset_cfg.urdf_file
    if urdf_file is None:
        raise ValueError("Robot asset config has no urdf_file; IR depth camera mount requires a URDF.")

    if asset_root:
        return (Path(asset_root) / urdf_file).resolve()
    return Path(urdf_file).resolve()


def _find_joint_by_child(robot_root: ET.Element, child_link_name: str) -> ET.Element:
    for joint in robot_root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link") == child_link_name:
            return joint
    raise ValueError(f"Could not find a URDF joint whose child link is '{child_link_name}'.")


def _joint_origin_xyz_rpy(joint: ET.Element) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    origin = joint.find("origin")
    if origin is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz = _parse_xyz_or_rpy(origin.get("xyz"))
    rpy = _parse_xyz_or_rpy(origin.get("rpy"))
    return xyz, rpy


def _fallback_depth_mount_from_simulator_defaults(
    urdf_path: Path,
    camera_location: str,
) -> DepthCameraMountSpec:
    robot_root = ET.parse(urdf_path).getroot()
    available_links = {link.get("name") for link in robot_root.findall("link")}
    is_r1 = R1_DEPTH_CAMERA_PARENT_LINK in available_links and "pelvis_link" in available_links
    if is_r1:
        fallback_parent = R1_DEPTH_CAMERA_PARENT_LINK
        fallback_position, fallback_rotation = R1_DEPTH_CAMERA_MOUNT_PRESETS[camera_location]
    else:
        fallback_parent = next(
            (candidate for candidate in DEPTH_CAMERA_FALLBACK_PARENT_CANDIDATES if candidate in available_links),
            DEPTH_CAMERA_FALLBACK_PARENT_LINK,
        )
        fallback_position = DEPTH_CAMERA_FALLBACK_POS
        fallback_rotation = DEPTH_CAMERA_FALLBACK_ROT_ROS_WXYZ
    return DepthCameraMountSpec(
        source_urdf_path=str(urdf_path),
        mount_mode="simulator_fallback_parent",
        scene_parent_link=fallback_parent,
        parent_link=fallback_parent,
        camera_body_link=fallback_parent,
        optical_frame_link=DEPTH_CAMERA_FRAME_LINK,
        translation=fallback_position,
        quaternion_ros_wxyz=fallback_rotation,
        camera_body_xyz=fallback_position,
        camera_body_rpy=(0.0, 0.0, 0.0),
        optical_frame_xyz=(0.0, 0.0, 0.0),
        optical_frame_rpy=(0.0, 0.0, 0.0),
    )


def _resolve_depth_mount_spec(tyro_config: ExperimentConfig) -> DepthCameraMountSpec:
    urdf_path = _resolve_robot_urdf_path(tyro_config)
    robot_root = ET.parse(urdf_path).getroot()

    try:
        camera_body_joint = _find_joint_by_child(robot_root, REALSENSE_CAMERA_BODY_LINK)
        optical_frame_joint = _find_joint_by_child(robot_root, DEPTH_CAMERA_FRAME_LINK)
    except ValueError:
        logger.info(
            "Current robot URDF has no dedicated RealSense links. "
            "IR depth camera will use IsaacSim's fallback torso mount."
        )
        camera_location = str(tyro_config.simulator.config.robot_depth_camera_location).lower()
        return _fallback_depth_mount_from_simulator_defaults(urdf_path, camera_location)

    camera_body_parent = camera_body_joint.find("parent")
    optical_parent = optical_frame_joint.find("parent")
    if camera_body_parent is None or optical_parent is None:
        raise ValueError("RealSense URDF joints are missing parent links.")

    camera_body_parent_link = camera_body_parent.get("link")
    optical_parent_link = optical_parent.get("link")
    if camera_body_parent_link is None or optical_parent_link is None:
        raise ValueError("RealSense URDF joints are missing parent link names.")
    if optical_parent_link != REALSENSE_CAMERA_BODY_LINK:
        raise ValueError(
            f"Expected optical frame '{DEPTH_CAMERA_FRAME_LINK}' to be parented under '{REALSENSE_CAMERA_BODY_LINK}', "
            f"but URDF says parent='{optical_parent_link}'."
        )

    camera_body_xyz, camera_body_rpy = _joint_origin_xyz_rpy(camera_body_joint)
    optical_frame_xyz, optical_frame_rpy = _joint_origin_xyz_rpy(optical_frame_joint)
    translation, quaternion_ros_wxyz = _compose_local_transform(
        xyz_a=camera_body_xyz,
        rpy_a=camera_body_rpy,
        xyz_b=optical_frame_xyz,
        rpy_b=optical_frame_rpy,
    )

    return DepthCameraMountSpec(
        source_urdf_path=str(urdf_path),
        mount_mode="urdf_optical_frame",
        scene_parent_link=DEPTH_CAMERA_FRAME_LINK,
        parent_link=camera_body_parent_link,
        camera_body_link=REALSENSE_CAMERA_BODY_LINK,
        optical_frame_link=DEPTH_CAMERA_FRAME_LINK,
        translation=translation,
        quaternion_ros_wxyz=quaternion_ros_wxyz,
        camera_body_xyz=camera_body_xyz,
        camera_body_rpy=camera_body_rpy,
        optical_frame_xyz=optical_frame_xyz,
        optical_frame_rpy=optical_frame_rpy,
    )


class IRTelemetryRecorder:
    """Collect per-step IR telemetry and export per-episode JSON files."""

    def __init__(
        self,
        algo: BaseAlgo,
        ir_cfg: IRCheckpointConfig,
        log_dir: str,
        depth_camera_mount: DepthCameraMountSpec,
    ):
        self.algo = algo
        self.env = algo.env
        self.ir_cfg = ir_cfg
        self.log_dir = Path(log_dir)
        self.telemetry_dir = self.log_dir / "telemetry"
        self.depth_image_dir = self.telemetry_dir / "depth_images"
        self.hdf5_path = self.telemetry_dir / "telemetry.h5"
        self.window_size = 5

        self.log_env_ids = set(ir_cfg.surface_feature_log_env_ids)
        self.surface_feature_body_sources = _parse_surface_feature_body_sources(ir_cfg.surface_feature_body_source)
        self.surface_feature_body_source = _canonical_surface_feature_body_source(self.surface_feature_body_sources)
        self.surface_feature_body_name = ir_cfg.surface_feature_body_name
        self.left_hand_body_name = ir_cfg.left_hand_body_name
        self.right_hand_body_name = ir_cfg.right_hand_body_name
        self.left_foot_body_name = ir_cfg.left_foot_body_name
        self.right_foot_body_name = ir_cfg.right_foot_body_name
        robot_asset_cfg = getattr(self.env.robot_config, "asset", None)
        self.robot_type = str(getattr(robot_asset_cfg, "robot_type", "")).lower()
        self.max_eval_steps = ir_cfg.max_eval_steps
        self.num_eval_episodes = ir_cfg.num_eval_episodes
        self.evaluate_all_motions = ir_cfg.evaluate_all_motions
        self.all_motions_iterations = ir_cfg.all_motions_iterations
        self.num_motion_clips: int | None = None
        self.depth_resolution = DEPTH_RESOLUTION
        self.depth_camera_mount = depth_camera_mount
        self.save_camera_images = ir_cfg.save_camera_images
        self.show_camera_marker = ir_cfg.show_camera_marker
        self.depth_camera_location = ir_cfg.depth_camera_location.strip().lower()
        self.camera_position_noise_m = ir_cfg.camera_position_noise_m
        self._camera_position_offsets_m: list[list[float]] = []
        self.depth_pixel_noise_max_std_m = ir_cfg.depth_pixel_noise_max_std_m
        self.depth_dropout_probability = ir_cfg.depth_dropout_probability
        self.ir_t_mode = _ir_t_mode_for_body_sources(self.surface_feature_body_sources)
        self.ir_t_component_names = list(_ir_t_component_names_for_body_sources(self.surface_feature_body_sources))

        body_labels: list[str] = []
        for body_source in self.surface_feature_body_sources:
            if body_source == "pelvis":
                body_labels.append("pelvis")
            elif body_source == "hands":
                body_labels.extend(IR_HAND_BODY_LABELS)
            else:
                body_labels.extend(IR_FOOT_BODY_LABELS)
        self._surface_feature_body_labels = tuple(body_labels)
        self._surface_feature_body_names: tuple[str, ...] = ()
        self._surface_feature_body_indices: tuple[int, ...] = ()
        self._surface_feature_computer: SurfaceFeatureComputer | None = None
        self._task_phase_schedule: TwoPhaseSchedule | None = None
        self._run_complete = False

        self._episode_indices: list[int] = []
        self._episode_steps: list[int] = []
        self._ir_windows: list[list[list[float]]] = []
        self._depth_windows: list[list[list[list[float]]]] = []
        self._proprioception_windows: list[list[list[float]]] = []
        self._episode_entries: list[list[dict]] = []
        self._exported_episode_count = 0
        self._completed_episode_counts: list[int] = []

    @property
    def run_complete(self) -> bool:
        return self._run_complete

    def _object_keys_for_envs(self, motion_command, num_envs: int) -> list[str | None]:
        object_key_to_id = getattr(motion_command, "object_key_to_id", None) or {}
        if not object_key_to_id:
            return [None] * num_envs
        id_to_key = {int(idx): key for key, idx in object_key_to_id.items()}
        object_type_ids = motion_command.object_type_ids.detach().cpu().tolist()
        return [id_to_key.get(int(type_id)) for type_id in object_type_ids]

    def _surface_feature_body_name_map(self) -> dict[str, str]:
        return {
            body_label: body_name
            for body_label, body_name in zip(self._surface_feature_body_labels, self._surface_feature_body_names)
        }

    def _resolve_surface_feature_bodies(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        available_body_names = self.env.body_names
        resolved_body_names: list[str] = []
        resolved_body_indices: list[int] = []

        for body_source in self.surface_feature_body_sources:
            if body_source == "pelvis":
                family = _robot_family(self.robot_type)
                pelvis_body_name = _resolve_surface_feature_body_name(
                    available_body_names=available_body_names,
                    explicit_name=self.surface_feature_body_name,
                    candidate_names=IR_PELVIS_BODY_NAME_CANDIDATES_BY_ROBOT_TYPE[family],
                    body_label=f"pelvis ({self.robot_type})",
                )
                resolved_body_names.append(pelvis_body_name)
                resolved_body_indices.append(available_body_names.index(pelvis_body_name))
                continue

            if body_source == "hands":
                left_body_name = _resolve_surface_feature_body_name(
                    available_body_names=available_body_names,
                    explicit_name=self.left_hand_body_name,
                    candidate_names=IR_LEFT_HAND_BODY_NAME_CANDIDATES,
                    body_label="left hand",
                )
                right_body_name = _resolve_surface_feature_body_name(
                    available_body_names=available_body_names,
                    explicit_name=self.right_hand_body_name,
                    candidate_names=IR_RIGHT_HAND_BODY_NAME_CANDIDATES,
                    body_label="right hand",
                )
                body_kind = "hand"
            else:
                family = _robot_family(self.robot_type)
                candidates = IR_FOOT_BODY_NAME_CANDIDATES_BY_ROBOT_TYPE[family]
                left_body_name = _resolve_surface_feature_body_name(
                    available_body_names=available_body_names,
                    explicit_name=self.left_foot_body_name,
                    candidate_names=candidates["left"],
                    body_label=f"left foot ({self.robot_type})",
                )
                right_body_name = _resolve_surface_feature_body_name(
                    available_body_names=available_body_names,
                    explicit_name=self.right_foot_body_name,
                    candidate_names=candidates["right"],
                    body_label=f"right foot ({self.robot_type})",
                )
                body_kind = "foot"

            if left_body_name == right_body_name:
                raise RuntimeError(
                    f"Left and right {body_kind} body names resolved to the same body '{left_body_name}'."
                )

            resolved_body_names.extend((left_body_name, right_body_name))
            resolved_body_indices.extend(
                (available_body_names.index(left_body_name), available_body_names.index(right_body_name))
            )

        return tuple(resolved_body_names), tuple(resolved_body_indices)

    def _compute_surface_feature_batches(
        self,
        motion_command,
        object_keys: list[str | None],
    ) -> dict[str, dict[str, torch.Tensor]]:
        if self._surface_feature_computer is None:
            raise RuntimeError("IR surface feature computer was not initialized before evaluation stepping.")

        body_feature_batches: dict[str, dict[str, torch.Tensor]] = {}
        for body_label, body_index in zip(self._surface_feature_body_labels, self._surface_feature_body_indices):
            body_pos_w = self.env.simulator._rigid_body_pos[:, body_index, :]
            body_lin_vel_w = self.env.simulator._rigid_body_vel[:, body_index, :]
            body_feature_batches[body_label] = self._surface_feature_computer.compute_batch(
                body_pos_w=body_pos_w,
                body_lin_vel_w=body_lin_vel_w,
                object_pos_w=motion_command.simulator_object_pos_w,
                object_quat_w=motion_command.simulator_object_quat_w,
                object_keys=object_keys,
            )
        return body_feature_batches

    def _combine_surface_feature_batches(
        self,
        body_feature_batches: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if self.surface_feature_body_sources == ("pelvis",):
            return body_feature_batches["pelvis"]

        ir_t_parts: list[torch.Tensor] = []
        for body_source in self.surface_feature_body_sources:
            if body_source == "pelvis":
                ir_t_parts.append(body_feature_batches["pelvis"]["ir_t"])
            elif body_source == "hands":
                ir_t_parts.append(body_feature_batches["left_hand"]["ir_t"])
                ir_t_parts.append(body_feature_batches["right_hand"]["ir_t"])
            else:
                ir_t_parts.append(body_feature_batches["left_foot"]["ir_t"])
                ir_t_parts.append(body_feature_batches["right_foot"]["ir_t"])

        combined_ir_t = torch.cat(ir_t_parts, dim=-1)
        combined_features: dict[str, torch.Tensor] = {"ir_t": combined_ir_t}
        for body_label, body_features in body_feature_batches.items():
            for feature_name, feature_tensor in body_features.items():
                if feature_name == "ir_t":
                    continue
                combined_features[f"{body_label}_{feature_name}"] = feature_tensor
        return combined_features

    def _surface_feature_entry_for_env(
        self,
        body_name: str,
        body_features: dict[str, torch.Tensor],
        env_id: int,
    ) -> dict[str, float | list[float] | str]:
        return {
            "body_name": body_name,
            "phi": float(body_features["phi"][env_id, 0].item()),
            "grad_phi": [float(v) for v in body_features["grad_phi"][env_id].tolist()],
            "v_t": [float(v) for v in body_features["v_t"][env_id].tolist()],
            "v_norm": [float(v) for v in body_features["v_norm"][env_id].tolist()],
            "v_tan": [float(v) for v in body_features["v_tan"][env_id].tolist()],
        }

    def _env_episode_target_reached(self, env_id: int) -> bool:
        return self.num_eval_episodes is not None and self._completed_episode_counts[env_id] >= self.num_eval_episodes

    def _all_episode_targets_reached(self) -> bool:
        return (
            self.num_eval_episodes is not None
            and bool(self._completed_episode_counts)
            and all(count >= self.num_eval_episodes for count in self._completed_episode_counts)
        )

    def _proprioception_term_dims(self) -> dict[str, int]:
        num_dof = len(self.env.dof_names)
        return {
            "base_ang_vel": 3,
            "dof_pos": num_dof,
            "dof_vel": num_dof,
        }

    def _proprioception_dim(self) -> int:
        return sum(self._proprioception_term_dims().values())

    def _proprioception_window_shape_t_f(self) -> tuple[int, int]:
        return (self.window_size, self._proprioception_dim())

    def _compute_proprioception_batch(self) -> dict[str, torch.Tensor]:
        # Match the semantics used in the WBT observation preset:
        # base_ang_vel in base frame, dof_pos relative to default_dof_pos, and raw dof_vel.
        base_ang_vel = quat_rotate_inverse(
            self.env.base_quat,
            self.env.simulator.robot_root_states[:, 10:13],
            w_last=True,
        )
        dof_pos = self.env.simulator.dof_pos - self.env.default_dof_pos
        dof_vel = self.env.simulator.dof_vel
        proprioception = torch.cat((base_ang_vel, dof_pos, dof_vel), dim=-1)
        return {
            "base_ang_vel": base_ang_vel,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "proprioception": proprioception,
        }

    def _reset_env_buffers(self, env_id: int) -> None:
        self._episode_steps[env_id] = 0
        self._ir_windows[env_id] = []
        self._depth_windows[env_id] = []
        self._proprioception_windows[env_id] = []
        self._episode_entries[env_id] = []
        self._episode_indices[env_id] += 1

    def _build_window(self, history: list, current_value):
        if not history:
            history[:] = [copy.deepcopy(current_value) for _ in range(self.window_size)]
        else:
            history.append(copy.deepcopy(current_value))
            if len(history) > self.window_size:
                del history[0 : len(history) - self.window_size]
        return copy.deepcopy(history)

    def _build_window_1d(self, history: list, current_value: list) -> list:
        """Optimized window builder for 1-D float lists. Avoids copy.deepcopy."""
        value_copy = list(current_value)
        if not history:
            history[:] = [list(current_value) for _ in range(self.window_size)]
        else:
            history.append(value_copy)
            if len(history) > self.window_size:
                del history[0 : len(history) - self.window_size]
        return [row[:] for row in history]

    def _build_window_2d(self, history: list, current_value: list) -> list:
        """Optimized window builder for 2-D float lists. Avoids copy.deepcopy."""
        frame_copy = [row[:] for row in current_value]
        if not history:
            history[:] = [[row[:] for row in frame_copy] for _ in range(self.window_size)]
        else:
            history.append(frame_copy)
            if len(history) > self.window_size:
                del history[0 : len(history) - self.window_size]
        return [[row[:] for row in frame] for frame in history]

    def _episode_hdf5_group_name(self, env_id: int, episode_index: int) -> str:
        return f"episode_env{env_id:03d}_idx{episode_index:03d}"

    def _depth_preview_file_name(self, env_id: int, episode_index: int, episode_step: int) -> Path:
        return Path(f"env_{env_id:03d}") / f"episode_{episode_index:03d}" / f"step_{episode_step:06d}_depth.png"

    def _save_depth_preview(self, env_id: int, episode_index: int, episode_step: int, depth_frame: list[list[float]]) -> str:
        from PIL import Image  # noqa: PLC0415

        depth_path_rel = self._depth_preview_file_name(env_id, episode_index, episode_step)
        depth_path_abs = self.depth_image_dir / depth_path_rel
        depth_path_abs.parent.mkdir(parents=True, exist_ok=True)

        depth_array = np.asarray(depth_frame, dtype=np.float32)
        finite_mask = np.isfinite(depth_array) & (depth_array > 0.0)
        preview = np.zeros_like(depth_array, dtype=np.uint8)
        if finite_mask.any():
            valid = depth_array[finite_mask]
            lo = float(valid.min())
            hi = float(np.percentile(valid, 99.0))
            if hi <= lo:
                hi = lo + 1e-6
            normalized = np.clip((depth_array - lo) / (hi - lo), 0.0, 1.0)
            preview = (normalized * 255.0).astype(np.uint8)

        Image.fromarray(preview, mode="L").save(depth_path_abs)
        return str(Path("depth_images") / depth_path_rel)

    def _get_simulator_depth_camera(self):
        depth_camera = getattr(self.env.simulator, "robot_depth_camera", None)
        if depth_camera is None:
            raise RuntimeError(
                "Simulator did not create a robot-mounted depth camera. "
                "Expected IsaacSim to register 'robot_depth_camera' from either the URDF optical frame "
                "or the fallback torso mount."
            )
        return depth_camera

    def _camera_prim_path_template(self) -> str:
        depth_camera = self._get_simulator_depth_camera()
        prim_path = getattr(getattr(depth_camera, "cfg", None), "prim_path", None)
        if not prim_path:
            raise RuntimeError("Robot depth camera is missing cfg.prim_path.")
        return str(prim_path)

    def _camera_parent_prim_path(self, env_id: int) -> str:
        return self._camera_prim_path(env_id).rsplit("/", 1)[0]

    def _camera_prim_path(self, env_id: int) -> str:
        prim_template = self._camera_prim_path_template()
        return re.sub(r"env_\.\*/", f"env_{env_id}/", prim_template)

    def _create_camera_location_markers(self, stage) -> None:
        """Attach RGB local axes and a red forward dart to each camera prim."""
        from pxr import Gf, UsdGeom  # noqa: PLC0415

        marker_paths: list[str] = []
        for env_id in range(self.env.num_envs):
            marker_path = f"{self._camera_prim_path(env_id)}/LocationMarker"
            UsdGeom.Xform.Define(stage, marker_path)

            origin = UsdGeom.Sphere.Define(stage, f"{marker_path}/Origin")
            origin.GetRadiusAttr().Set(0.012)
            origin.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

            axis_specs = (
                ("RollX", Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 90.0, 0.0)),
                ("PitchY", Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(-90.0, 0.0, 0.0)),
                ("YawZ", Gf.Vec3f(0.0, 0.25, 1.0), Gf.Vec3f(0.0, 0.0, 0.0)),
            )
            for axis_name, color, rotation_xyz in axis_specs:
                axis = UsdGeom.Cylinder.Define(stage, f"{marker_path}/{axis_name}")
                axis.GetRadiusAttr().Set(0.003)
                axis.GetHeightAttr().Set(0.10)
                axis.GetDisplayColorAttr().Set([color])
                axis.AddRotateXYZOp().Set(rotation_xyz)

            # USD cameras look along local -Z. Keep this dart and all axes
            # inside the 0.07 m near clip to avoid contaminating depth.
            dart = UsdGeom.Cone.Define(stage, f"{marker_path}/ViewDirection")
            dart.GetRadiusAttr().Set(0.012)
            dart.GetHeightAttr().Set(0.05)
            dart.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            dart.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
            dart.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
            marker_paths.append(marker_path)
        logger.info(
            "Created IsaacSim depth-camera frame markers: "
            f"X=red Y=green Z=blue forward_dart=red paths={marker_paths}"
        )

    def _randomize_camera_mount_positions(self, stage) -> None:
        """Apply one fixed XYZ camera-mount offset per environment."""
        from pxr import Gf, UsdGeom  # noqa: PLC0415

        noise_m = float(self.camera_position_noise_m)
        if noise_m < 0.0:
            raise ValueError(f"camera_position_noise_m must be non-negative, got {noise_m}.")

        offsets = (torch.rand((self.env.num_envs, 3), device="cpu") * 2.0 - 1.0) * noise_m
        self._camera_position_offsets_m = offsets.tolist()

        for env_id, offset in enumerate(self._camera_position_offsets_m):
            camera_prim_path = self._camera_prim_path(env_id)
            camera_prim = stage.GetPrimAtPath(camera_prim_path)
            if not camera_prim.IsValid():
                raise RuntimeError(f"Cannot randomize missing depth-camera prim: {camera_prim_path}")

            xformable = UsdGeom.Xformable(camera_prim)
            translate_op = next(
                (
                    op
                    for op in xformable.GetOrderedXformOps()
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                ),
                None,
            )
            if translate_op is None:
                raise RuntimeError(
                    f"Depth-camera prim has no translate xform op required for mount randomization: {camera_prim_path}"
                )

            nominal_position = translate_op.Get()
            if nominal_position is None:
                nominal_position = Gf.Vec3d(0.0, 0.0, 0.0)
            randomized_position = type(nominal_position)(
                float(nominal_position[0]) + offset[0],
                float(nominal_position[1]) + offset[1],
                float(nominal_position[2]) + offset[2],
            )
            translate_op.Set(randomized_position)

        logger.info(
            "Applied per-environment depth-camera mount position randomization: "
            f"xyz_uniform_range_m=[{-noise_m}, {noise_m}], envs={self.env.num_envs}."
        )

    def _read_depth_frame(self, env_id: int) -> list[list[float]]:
        expected_hw = (self.depth_resolution[1], self.depth_resolution[0])
        depth_tensor = self.env.simulator.get_robot_depth_frame(env_id)
        if depth_tensor is None:
            if self._depth_windows[env_id]:
                return [row[:] for row in self._depth_windows[env_id][-1]]
            return [[0.0] * expected_hw[1] for _ in range(expected_hw[0])]

        depth_array = _to_numpy_float32(depth_tensor)
        depth_array = np.squeeze(depth_array)
        valid_shapes = {
            expected_hw,
            (ROBOT_DEPTH_RAW_RESOLUTION_WH[1], ROBOT_DEPTH_RAW_RESOLUTION_WH[0]),
        }
        if depth_array.shape not in valid_shapes:
            raise RuntimeError(
                f"Unexpected depth frame shape {depth_array.shape} for env {env_id}; expected one of {valid_shapes}."
            )

        depth_array = preprocess_robot_depth_array(depth_array)
        depth_array = self._apply_depth_sensor_noise(depth_array)
        return depth_array.tolist()

    def _read_all_depth_frames(self) -> list[list[list[float]]]:
        """Read depth frames for all envs at once, reusing per-env fallbacks."""
        expected_hw = (self.depth_resolution[1], self.depth_resolution[0])
        results: list[list[list[float]]] = []
        get_frame = self.env.simulator.get_robot_depth_frame
        for env_id in range(self.env.num_envs):
            depth_tensor = get_frame(env_id)
            if depth_tensor is None:
                if self._depth_windows[env_id]:
                    results.append([row[:] for row in self._depth_windows[env_id][-1]])
                else:
                    results.append([[0.0] * expected_hw[1] for _ in range(expected_hw[0])])
                continue
            depth_array = _to_numpy_float32(depth_tensor)
            depth_array = np.squeeze(depth_array)
            valid_shapes = {
                expected_hw,
                (ROBOT_DEPTH_RAW_RESOLUTION_WH[1], ROBOT_DEPTH_RAW_RESOLUTION_WH[0]),
            }
            if depth_array.shape not in valid_shapes:
                raise RuntimeError(
                    f"Unexpected depth frame shape {depth_array.shape} for env {env_id}; expected one of {valid_shapes}."
                )
            depth_array = preprocess_robot_depth_array(depth_array)
            depth_array = self._apply_depth_sensor_noise(depth_array)
            results.append(depth_array.tolist())
        return results

    def _apply_depth_sensor_noise(self, depth_array: np.ndarray) -> np.ndarray:
        """Apply small range-dependent pixel noise and sparse dropout."""
        max_std_m = float(self.depth_pixel_noise_max_std_m)
        dropout_probability = float(self.depth_dropout_probability)
        if max_std_m < 0.0:
            raise ValueError(f"depth_pixel_noise_max_std_m must be non-negative, got {max_std_m}.")
        if not 0.0 <= dropout_probability <= 1.0:
            raise ValueError(
                "depth_dropout_probability must be in [0, 1], "
                f"got {dropout_probability}."
            )

        noisy_depth = np.asarray(depth_array, dtype=np.float32).copy()
        valid = np.isfinite(noisy_depth) & (noisy_depth > 0.0)
        if max_std_m > 0.0 and valid.any():
            normalized_range = np.clip(noisy_depth / ROBOT_DEPTH_MAX_M, 0.0, 1.0)
            pixel_std_m = max_std_m * np.square(normalized_range)
            gaussian_noise = np.random.normal(size=noisy_depth.shape).astype(np.float32)
            noisy_depth[valid] += gaussian_noise[valid] * pixel_std_m[valid]
            noisy_depth[valid] = np.clip(
                noisy_depth[valid],
                ROBOT_DEPTH_MIN_M,
                ROBOT_DEPTH_MAX_M,
            )

        if dropout_probability > 0.0 and valid.any():
            dropout_mask = np.random.random(size=noisy_depth.shape) < dropout_probability
            noisy_depth[valid & dropout_mask] = 0.0

        return noisy_depth

    def _depth_frame_shape_hw(self) -> tuple[int, int]:
        return (int(self.depth_resolution[1]), int(self.depth_resolution[0]))

    def _depth_window_shape_t_h_w(self) -> tuple[int, int, int]:
        height, width = self._depth_frame_shape_hw()
        return (self.window_size, height, width)

    def _export_episode_hdf5(self, episode_data: dict, group_name: str) -> str:
        try:
            import h5py  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "h5py is required for IR telemetry collection. Install h5py to write telemetry.h5."
            ) from exc

        entries = episode_data.get("entries", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError("Cannot export HDF5 telemetry for an episode with no entries.")

        metadata = {key: value for key, value in episode_data.items() if key != "entries"}
        ir_windows = np.asarray([entry["ir_window"] for entry in entries], dtype=np.float32)
        depth_windows = np.asarray([entry["depth_window"] for entry in entries], dtype=np.float32)
        next_depth_windows = np.concatenate((depth_windows[1:], depth_windows[-1:]), axis=0)
        proprioception_windows = np.asarray(
            [entry["proprioception_window"] for entry in entries],
            dtype=np.float32,
        )
        reference_velocity_commands = np.asarray(
            [entry["reference_velocity_command"] for entry in entries],
            dtype=np.float32,
        )
        task_phases = np.asarray([entry["task_phase"] for entry in entries], dtype=np.int64)
        teacher_actions = np.asarray([entry["teacher_action"] for entry in entries], dtype=np.float32)
        actor_observations = np.asarray([entry["actor_observation"] for entry in entries], dtype=np.float32)
        critic_observations = np.asarray([entry["critic_observation"] for entry in entries], dtype=np.float32)
        next_actor_observations = np.asarray(
            [entry["next_actor_observation"] for entry in entries], dtype=np.float32
        )
        next_critic_observations = np.asarray(
            [entry["next_critic_observation"] for entry in entries], dtype=np.float32
        )
        depth_frames = np.asarray([entry["depth_window"][-1] for entry in entries], dtype=np.float32)
        next_depth_frames = np.concatenate((depth_frames[1:], depth_frames[-1:]), axis=0)
        next_depth_valid = np.ones(len(entries), dtype=np.bool_)
        next_depth_valid[-1] = False
        sac_rewards = np.asarray([entry["sac_reward"] for entry in entries], dtype=np.float32)
        dones = np.asarray([entry["done"] for entry in entries], dtype=np.bool_)
        terminations = np.asarray([entry["terminated"] for entry in entries], dtype=np.bool_)
        truncations = np.asarray([entry["truncated"] for entry in entries], dtype=np.bool_)
        episode_ids = np.asarray([entry["episode_index"] for entry in entries], dtype=np.int64)
        timesteps = np.asarray([entry["episode_step"] for entry in entries], dtype=np.int64)
        reward_term_names = sorted(entries[0].get("sac_reward_terms_raw", {}))

        group_path = f"episodes/{group_name}"
        with h5py.File(self.hdf5_path, "a") as hdf5_file:
            hdf5_file.attrs["format"] = "holosoma_ir_telemetry"
            hdf5_file.attrs["format_version"] = 2
            episodes_group = hdf5_file.require_group("episodes")
            if group_name in episodes_group:
                del episodes_group[group_name]
            episode_group = episodes_group.create_group(group_name)
            episode_group.create_dataset("ir_windows", data=ir_windows)
            episode_group.create_dataset("depth_windows", data=depth_windows)
            episode_group.create_dataset("next_depth_windows", data=next_depth_windows)
            episode_group.create_dataset("proprioception_windows", data=proprioception_windows)
            episode_group.create_dataset("reference_velocity_commands", data=reference_velocity_commands)
            episode_group.create_dataset("task_phases", data=task_phases)
            episode_group.create_dataset("teacher_actions", data=teacher_actions)
            episode_group.create_dataset("actor_observations", data=actor_observations)
            episode_group.create_dataset("critic_observations", data=critic_observations)
            episode_group.create_dataset("next_actor_observations", data=next_actor_observations)
            episode_group.create_dataset("next_critic_observations", data=next_critic_observations)
            episode_group.create_dataset("depth_frames", data=depth_frames)
            episode_group.create_dataset("next_depth_frames", data=next_depth_frames)
            episode_group.create_dataset("next_depth_valid", data=next_depth_valid)
            episode_group.create_dataset("sac_rewards", data=sac_rewards)
            episode_group.create_dataset("dones", data=dones)
            episode_group.create_dataset("terminations", data=terminations)
            episode_group.create_dataset("truncations", data=truncations)
            episode_group.create_dataset("episode_ids", data=episode_ids)
            episode_group.create_dataset("timesteps", data=timesteps)
            reward_terms_group = episode_group.create_group("sac_reward_terms_raw")
            for term_name in reward_term_names:
                values = np.asarray(
                    [entry["sac_reward_terms_raw"][term_name] for entry in entries],
                    dtype=np.float32,
                )
                reward_terms_group.create_dataset(term_name, data=values)
            episode_group.attrs["metadata_json"] = json.dumps(metadata, separators=(",", ":"))
        return group_path

    def _export_episode(self, episode_data: dict) -> None:
        env_id = int(episode_data["env_id"])
        episode_index = int(episode_data["episode_index"])
        group_name = self._episode_hdf5_group_name(env_id, episode_index)
        self._export_episode_hdf5(episode_data, group_name)
        self._exported_episode_count += 1

    def _finalize_episode(self, env_id: int, reason: str, global_step: int | None) -> None:
        first_entry = self._episode_entries[env_id][0] if self._episode_entries[env_id] else {}
        episode_index = self._episode_indices[env_id]
        episode_data = {
            "env_id": env_id,
            "episode_index": episode_index,
            "num_steps": len(self._episode_entries[env_id]),
            "termination_reason": reason,
            "motion_clip_id": first_entry.get("motion_clip_id"),
            "motion_name": first_entry.get("motion_name"),
            "motion_iteration": (
                episode_index // self.num_motion_clips + 1
                if self.evaluate_all_motions and self.num_motion_clips
                else None
            ),
            "all_motions_iterations": self.all_motions_iterations,
            "max_eval_steps": self.max_eval_steps,
            "num_eval_episodes": self.num_eval_episodes,
            "num_eval_episodes_scope": "per_env",
            "surface_feature_body_source": self.surface_feature_body_source,
            "surface_feature_body_sources": list(self.surface_feature_body_sources),
            "surface_feature_body_names": self._surface_feature_body_name_map(),
            "robot_type": self.robot_type,
            "surface_feature_body_name": self.surface_feature_body_name,
            "left_hand_body_name": self.left_hand_body_name,
            "right_hand_body_name": self.right_hand_body_name,
            "ir_t_mode": self.ir_t_mode,
            "ir_t_components": self.ir_t_component_names,
            "ir_t_dim": len(self.ir_t_component_names),
            "save_camera_images": self.save_camera_images,
            "depth_camera_location": self.depth_camera_location,
            "camera_position_noise_m": self.camera_position_noise_m,
            "camera_position_offset_m": self._camera_position_offsets_m[env_id],
            "depth_pixel_noise_max_std_m": self.depth_pixel_noise_max_std_m,
            "depth_pixel_noise_model": "gaussian_quadratic_in_range",
            "depth_dropout_probability": self.depth_dropout_probability,
            "telemetry_format": "hdf5",
            "telemetry_hdf5_file": self.hdf5_path.name,
            # Legacy field: camera config order is [width, height].
            "depth_resolution": list(self.depth_resolution),
            "depth_resolution_order": "width_height",
            "depth_raw_resolution": list(ROBOT_DEPTH_RAW_RESOLUTION_WH),
            "depth_min_m": ROBOT_DEPTH_MIN_M,
            "depth_max_m": ROBOT_DEPTH_MAX_M,
            "depth_horizontal_fov_deg": ROBOT_DEPTH_HORIZONTAL_FOV_DEG,
            "depth_vertical_fov_deg": ROBOT_DEPTH_VERTICAL_FOV_DEG,
            "depth_frame_shape": list(self._depth_frame_shape_hw()),
            "depth_frame_shape_order": "height_width",
            "depth_window_shape": list(self._depth_window_shape_t_h_w()),
            "depth_window_shape_order": "time_height_width",
            "proprioception_components": list(PROPRIOCEPTION_COMPONENT_NAMES),
            "proprioception_term_dims": self._proprioception_term_dims(),
            "proprioception_dim": self._proprioception_dim(),
            "proprioception_window_shape": list(self._proprioception_window_shape_t_f()),
            "proprioception_window_shape_order": "time_feature",
            "proprioception_dof_names": list(self.env.dof_names),
            "reference_velocity_command_components": [
                "linear_velocity_x",
                "linear_velocity_y",
                "angular_velocity_z",
            ],
            "reference_velocity_command_frame": "reference_root_body",
            "task_phase_definitions": {"0": "approach", "1": "interaction_and_finish"},
            "replay_actor_observation_group": "replay_actor_obs",
            "replay_critic_observation_group": "replay_critic_obs",
            "fastsac_task_object_identity_included": True,
            "depth_latent_included": False,
            "next_depth_terminal_policy": "repeat_last_depth_frame",
            "sac_reward_terms_raw": sorted(first_entry.get("sac_reward_terms_raw", {})),
            "depth_camera_prim_name": DEPTH_CAMERA_PRIM_NAME,
            "depth_camera_mount": self.depth_camera_mount.to_json_dict(),
            "completed_at_global_step": global_step,
            "entries": self._episode_entries[env_id],
        }
        self._export_episode(episode_data)
        self._completed_episode_counts[env_id] += 1
        self._reset_env_buffers(env_id)

        if self._all_episode_targets_reached():
            self._run_complete = True

        total_completed = sum(self._completed_episode_counts)
        logger.info(
            f"[ir_episode_complete] env={env_id} episode={episode_data['episode_index']} "
            f"steps={episode_data['num_steps']} reason={reason} "
            f"env_completed={self._completed_episode_counts[env_id]} total_completed={total_completed}"
        )

    def on_pre_evaluate_policy(self) -> None:
        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is None:
            raise RuntimeError("motion_command not found; IR telemetry requires a motion command.")
        if not getattr(motion_command.motion, "has_object", False):
            raise RuntimeError("IR telemetry requires a motion with an object.")
        self._task_phase_schedule = TwoPhaseSchedule(motion_command)
        if self.all_motions_iterations <= 0:
            raise ValueError(
                f"all_motions_iterations must be positive, got {self.all_motions_iterations}"
            )
        if not self.evaluate_all_motions and self.all_motions_iterations != 1:
            raise ValueError("all_motions_iterations requires evaluate_all_motions=True.")
        if self.evaluate_all_motions:
            num_clips = len(motion_command.motion.clip_ranges)
            if num_clips == 0:
                raise RuntimeError("evaluate_all_motions requires at least one loaded motion clip.")
            motion_command.enable_eval_clip_sweep()
            self.num_motion_clips = num_clips
            self.num_eval_episodes = num_clips * self.all_motions_iterations
            logger.info(
                f"Enabled deterministic all-motion evaluation: clips={num_clips}, "
                f"iterations={self.all_motions_iterations}, "
                f"episodes_per_env={self.num_eval_episodes}, each starting at the first frame."
            )
        if self.max_eval_steps is not None and self.max_eval_steps <= 0:
            raise ValueError(f"max_eval_steps must be positive when provided, got {self.max_eval_steps}")
        if self.num_eval_episodes is not None and self.num_eval_episodes <= 0:
            raise ValueError(f"num_eval_episodes must be positive when provided, got {self.num_eval_episodes}")

        self._surface_feature_body_names, self._surface_feature_body_indices = self._resolve_surface_feature_bodies()
        if "pelvis" in self._surface_feature_body_labels:
            pelvis_index = self._surface_feature_body_labels.index("pelvis")
            self.surface_feature_body_name = self._surface_feature_body_names[pelvis_index]
        try:
            self._surface_feature_computer = SurfaceFeatureComputer.from_object_config(
                self.env.robot_config.object,
                mesh_mode="full",
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize GPU IR surface feature computer: {exc}") from exc

        depth_camera = self._get_simulator_depth_camera()
        camera_resolution = (int(depth_camera.cfg.width), int(depth_camera.cfg.height))
        if camera_resolution != ROBOT_DEPTH_RAW_RESOLUTION_WH:
            raise RuntimeError(
                f"Robot depth camera resolution is {camera_resolution}; expected raw "
                f"{ROBOT_DEPTH_RAW_RESOLUTION_WH} before uniform sampling."
            )
        self.depth_resolution = ROBOT_DEPTH_OUTPUT_RESOLUTION_WH

        import omni.usd  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        missing_prim_paths = [
            self._camera_prim_path(env_id)
            for env_id in range(self.env.num_envs)
            if not stage.GetPrimAtPath(self._camera_prim_path(env_id)).IsValid()
        ]
        if missing_prim_paths:
            raise RuntimeError(
                f"Simulator depth camera prims were not found on the live stage: {missing_prim_paths}"
            )
        self._randomize_camera_mount_positions(stage)
        if self.show_camera_marker:
            self._create_camera_location_markers(stage)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        if self.save_camera_images:
            self.depth_image_dir.mkdir(parents=True, exist_ok=True)

        self._episode_indices = [0 for _ in range(self.env.num_envs)]
        self._episode_steps = [0 for _ in range(self.env.num_envs)]
        self._ir_windows = [[] for _ in range(self.env.num_envs)]
        self._depth_windows = [[] for _ in range(self.env.num_envs)]
        self._proprioception_windows = [[] for _ in range(self.env.num_envs)]
        self._episode_entries = [[] for _ in range(self.env.num_envs)]
        self._exported_episode_count = 0
        self._completed_episode_counts = [0 for _ in range(self.env.num_envs)]
        self._run_complete = False

        logger.info(
            f"IR telemetry enabled for body_source='{self.surface_feature_body_source}' "
            f"body_names={list(self._surface_feature_body_names)} with window_size={self.window_size}, "
            f"ir_t_mode={self.ir_t_mode}, ir_t_dim={len(self.ir_t_component_names)}, "
            f"proprioception_dim={self._proprioception_dim()}, "
            f"max_eval_steps={self.max_eval_steps}, "
            f"num_eval_episodes_per_env={self.num_eval_episodes}, save_camera_images={self.save_camera_images}, "
            f"depth_pixel_noise_max_std_m={self.depth_pixel_noise_max_std_m}, "
            f"depth_dropout_probability={self.depth_dropout_probability}, "
            "telemetry_format=hdf5, "
            f"depth_resolution_wh={self.depth_resolution}, depth_window_shape_t_h_w={self._depth_window_shape_t_h_w()}."
        )
        logger.info(
            "Resolved depth camera mount spec: "
            f"mount_mode={self.depth_camera_mount.mount_mode}, "
            f"scene_parent_link={self.depth_camera_mount.scene_parent_link}, "
            f"parent_link={self.depth_camera_mount.parent_link}, "
            f"optical_frame_link={self.depth_camera_mount.optical_frame_link}, "
            f"translation={list(self.depth_camera_mount.translation)}, "
            f"quaternion_ros_wxyz={list(self.depth_camera_mount.quaternion_ros_wxyz)}, "
            f"source_urdf='{self.depth_camera_mount.source_urdf_path}'"
        )
        logger.info(
            f"IR depth camera is managed by IsaacSim scene sensor at prim: {self._camera_parent_prim_path(0)}"
        )

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        if not self._surface_feature_body_indices or self._run_complete:
            return actor_state

        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is None:
            return actor_state
        object_keys = self._object_keys_for_envs(motion_command, self.env.num_envs)
        body_feature_batches = self._compute_surface_feature_batches(motion_command=motion_command, object_keys=object_keys)
        features = self._combine_surface_feature_batches(body_feature_batches)
        proprioception_batches = self._compute_proprioception_batch()

        # ir_t: [num_envs, 13] for pelvis, [num_envs, 26] for hands, or concatenated across requested sources.
        # ir_window: [5, ir_t_dim] after the unchanged windowing logic below.
        actor_state["ir_features"] = features
        actor_state["ir_surface_features_by_body"] = body_feature_batches
        actor_state["proprioception_features"] = proprioception_batches
        actor_state["ir_object_keys"] = object_keys
        global_step = int(actor_state.get("step", -1))

        # Batch-convert all GPU tensors to Python lists once (outside per-env loop).
        ir_t_all: list = features["ir_t"].tolist()
        base_ang_vel_all: list = proprioception_batches["base_ang_vel"].tolist()
        dof_pos_all: list = proprioception_batches["dof_pos"].tolist()
        dof_vel_all: list = proprioception_batches["dof_vel"].tolist()
        proprioception_all: list = proprioception_batches["proprioception"].tolist()
        reference_root_lin_vel_b = quat_rotate_inverse(
            motion_command.root_quat_w,
            motion_command.root_lin_vel_w,
            w_last=True,
        )
        reference_root_ang_vel_b = quat_rotate_inverse(
            motion_command.root_quat_w,
            motion_command.root_ang_vel_w,
            w_last=True,
        )
        reference_velocity_commands = torch.stack(
            (
                reference_root_lin_vel_b[:, 0],
                reference_root_lin_vel_b[:, 1],
                reference_root_ang_vel_b[:, 2],
            ),
            dim=-1,
        ).tolist()
        if self._task_phase_schedule is None:
            raise RuntimeError("Task phase schedule was not initialized before evaluation stepping.")
        task_phases = self._task_phase_schedule.phase(motion_command).tolist()
        teacher_actions = actor_state["actions"].detach().cpu().tolist()
        replay_actor_obs = actor_state["obs"]["replay_actor_obs"].detach().cpu().tolist()
        replay_critic_obs = actor_state["obs"]["replay_critic_obs"].detach().cpu().tolist()
        all_depth_frames: list = self._read_all_depth_frames()
        clip_ids = motion_command.clip_ids.detach().cpu().tolist()
        clip_files = getattr(motion_command.motion, "clip_files", [])

        for env_id in range(self.env.num_envs):
            if self._env_episode_target_reached(env_id):
                continue

            current_ir_t = ir_t_all[env_id]
            current_ir_window = self._build_window_1d(self._ir_windows[env_id], current_ir_t)
            current_depth_frame = all_depth_frames[env_id]
            current_depth_window = self._build_window_2d(self._depth_windows[env_id], current_depth_frame)
            current_base_ang_vel = base_ang_vel_all[env_id]
            current_dof_pos = dof_pos_all[env_id]
            current_dof_vel = dof_vel_all[env_id]
            current_proprioception = proprioception_all[env_id]
            current_proprioception_window = self._build_window_1d(
                self._proprioception_windows[env_id],
                current_proprioception,
            )
            depth_image_file: str | None = None
            if self.save_camera_images:
                depth_image_file = self._save_depth_preview(
                    env_id=env_id,
                    episode_index=self._episode_indices[env_id],
                    episode_step=self._episode_steps[env_id],
                    depth_frame=current_depth_frame,
                )
            entry = {
                "global_step": global_step,
                "episode_index": self._episode_indices[env_id],
                "episode_step": self._episode_steps[env_id],
                "env_id": env_id,
                "object_key": object_keys[env_id],
                "motion_clip_id": int(clip_ids[env_id]),
                "motion_name": (
                    Path(str(clip_files[int(clip_ids[env_id])])).stem
                    if int(clip_ids[env_id]) < len(clip_files)
                    else f"clip_{int(clip_ids[env_id])}"
                ),
                "surface_feature_body_source": self.surface_feature_body_source,
                "surface_feature_body_sources": list(self.surface_feature_body_sources),
                "ir_t": current_ir_t,
                "ir_window": current_ir_window,
                "base_ang_vel": current_base_ang_vel,
                "dof_pos": current_dof_pos,
                "dof_vel": current_dof_vel,
                "proprioception": current_proprioception,
                "proprioception_window": current_proprioception_window,
                "reference_velocity_command": reference_velocity_commands[env_id],
                "task_phase": int(task_phases[env_id]),
                "teacher_action": teacher_actions[env_id],
                "actor_observation": replay_actor_obs[env_id],
                "critic_observation": replay_critic_obs[env_id],
                "depth_window": current_depth_window,
                "depth_image_file": depth_image_file,
            }
            if self.surface_feature_body_sources == ("pelvis",):
                entry.update(
                    self._surface_feature_entry_for_env(
                        body_name=self._surface_feature_body_names[0],
                        body_features=body_feature_batches["pelvis"],
                        env_id=env_id,
                    )
                )
            else:
                if "pelvis" in self.surface_feature_body_sources:
                    pelvis_body_index = self._surface_feature_body_labels.index("pelvis")
                    entry["pelvis_surface_features"] = self._surface_feature_entry_for_env(
                        body_name=self._surface_feature_body_names[pelvis_body_index],
                        body_features=body_feature_batches["pelvis"],
                        env_id=env_id,
                    )
                if "hands" in self.surface_feature_body_sources:
                    left_hand_index = self._surface_feature_body_labels.index("left_hand")
                    right_hand_index = self._surface_feature_body_labels.index("right_hand")
                    entry["left_hand_surface_features"] = self._surface_feature_entry_for_env(
                        body_name=self._surface_feature_body_names[left_hand_index],
                        body_features=body_feature_batches["left_hand"],
                        env_id=env_id,
                    )
                    entry["right_hand_surface_features"] = self._surface_feature_entry_for_env(
                        body_name=self._surface_feature_body_names[right_hand_index],
                        body_features=body_feature_batches["right_hand"],
                        env_id=env_id,
                    )
                if "feet" in self.surface_feature_body_sources:
                    for foot_label in IR_FOOT_BODY_LABELS:
                        foot_index = self._surface_feature_body_labels.index(foot_label)
                        entry[f"{foot_label}_surface_features"] = self._surface_feature_entry_for_env(
                            body_name=self._surface_feature_body_names[foot_index],
                            body_features=body_feature_batches[foot_label],
                            env_id=env_id,
                        )
            self._episode_entries[env_id].append(entry)

            if env_id in self.log_env_ids:
                if self.surface_feature_body_sources == ("pelvis",):
                    body_features = body_feature_batches["pelvis"]
                    logger.info(
                        f"[ir_window] step={global_step} env={env_id} episode={self._episode_indices[env_id]} "
                        f"episode_step={self._episode_steps[env_id]} object={object_keys[env_id]} "
                        f"body={self._surface_feature_body_names[0]} "
                        f"phi={float(body_features['phi'][env_id, 0].item()):.4f} "
                        f"grad_phi={[round(float(v), 4) for v in body_features['grad_phi'][env_id].tolist()]} "
                        f"v_norm={[round(float(v), 4) for v in body_features['v_norm'][env_id].tolist()]} "
                        f"v_tan={[round(float(v), 4) for v in body_features['v_tan'][env_id].tolist()]} "
                        f"depth_shape_t_h_w={self._depth_window_shape_t_h_w()}"
                    )
                else:
                    body_phi_summary = " ".join(
                        f"{body_label}={self._surface_feature_body_names[index]} "
                        f"{body_label}_phi={float(body_feature_batches[body_label]['phi'][env_id, 0].item()):.4f}"
                        for index, body_label in enumerate(self._surface_feature_body_labels)
                    )
                    logger.info(
                        f"[ir_window] step={global_step} env={env_id} episode={self._episode_indices[env_id]} "
                        f"episode_step={self._episode_steps[env_id]} object={object_keys[env_id]} "
                        f"{body_phi_summary} "
                        f"depth_shape_t_h_w={self._depth_window_shape_t_h_w()}"
                    )

        return actor_state

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        if self._run_complete:
            return actor_state

        dones = actor_state.get("dones")
        if dones is None:
            return actor_state

        global_step = int(actor_state.get("step", -1))
        rewards = actor_state.get("rewards")
        extras = actor_state.get("extras") or {}
        time_outs = extras.get("time_outs")
        reward_manager = getattr(self.env, "reward_manager", None)
        raw_terms = getattr(reward_manager, "last_raw_term_values", {}) if reward_manager is not None else {}
        raw_terms_cpu = {name: values.detach().cpu().tolist() for name, values in raw_terms.items()}
        next_obs = actor_state.get("obs") or {}
        final_obs = extras.get("final_observations") or {}
        maxed_env_ids: list[int] = []

        for env_id in range(self.env.num_envs):
            if self._run_complete:
                break

            if self._env_episode_target_reached(env_id):
                continue

            self._episode_steps[env_id] += 1
            reached_limit = self.max_eval_steps is not None and self._episode_steps[env_id] >= self.max_eval_steps
            is_done = bool(dones[env_id].item())
            if self._episode_entries[env_id]:
                entry = self._episode_entries[env_id][-1]
                next_actor = final_obs.get("replay_actor_obs") if is_done else next_obs.get("replay_actor_obs")
                next_critic = final_obs.get("replay_critic_obs") if is_done else next_obs.get("replay_critic_obs")
                if next_actor is None or next_critic is None:
                    raise RuntimeError("Replay next observations are missing from environment observations.")
                entry["sac_reward"] = float(rewards[env_id].item()) if rewards is not None else 0.0
                entry["done"] = is_done
                is_truncated = bool(time_outs[env_id].item()) if time_outs is not None else False
                entry["terminated"] = is_done and not is_truncated
                entry["truncated"] = is_truncated
                entry["next_actor_observation"] = next_actor[env_id].detach().cpu().tolist()
                entry["next_critic_observation"] = next_critic[env_id].detach().cpu().tolist()
                entry["sac_reward_terms_raw"] = {
                    name: float(values[env_id]) for name, values in raw_terms_cpu.items()
                }

            if is_done:
                self._finalize_episode(env_id, reason="done", global_step=global_step)
            elif reached_limit:
                entry = self._episode_entries[env_id][-1]
                entry["done"] = True
                entry["terminated"] = False
                entry["truncated"] = True
                self._finalize_episode(env_id, reason="max_eval_steps", global_step=global_step)
                if not self._run_complete and not self._env_episode_target_reached(env_id):
                    maxed_env_ids.append(env_id)

        if maxed_env_ids and not self._run_complete:
            env_ids_tensor = torch.tensor(maxed_env_ids, device=self.env.device, dtype=torch.long)
            self.env.reset_envs_idx(env_ids_tensor)
            refresh_env_ids = self.env._ensure_long_tensor(self.env._get_envs_to_refresh())
            if refresh_env_ids.numel() > 0:
                self.env._refresh_envs_after_reset(refresh_env_ids)
            self.env._compute_observations()
            self.env._post_compute_observations_callback()
            self.env._clip_observations()
            actor_state["obs"] = self.env.obs_buf_dict
            critic_obs = torch.cat([actor_state["obs"][k] for k in self.algo.critic_obs_keys], dim=1)
            actor_state["obs"]["critic_obs"] = critic_obs

        return actor_state

    def on_post_evaluate_policy(self) -> None:
        target_reached = self._all_episode_targets_reached()
        if not target_reached:
            for env_id in range(self.env.num_envs):
                if self._episode_entries[env_id]:
                    self._finalize_episode(env_id, reason="run_end", global_step=None)

        logger.info(
            f"Exported IR telemetry for {self._exported_episode_count} episode(s) to {self.telemetry_dir} "
            "with telemetry_format=hdf5"
        )


def _with_original_robot_assets(tyro_config: ExperimentConfig) -> ExperimentConfig:
    asset_cfg = tyro_config.robot.asset
    if asset_cfg.urdf_file not in {ORIGINAL_G1_URDF_FILE, "g1/g1_29dof_realsense.urdf"}:
        logger.info(
            "IR robot asset normalization skipped because the current URDF is not a known G1 variant: "
            f"urdf={asset_cfg.urdf_file}, xml={asset_cfg.xml_file}"
        )
        return tyro_config

    new_asset_cfg = dataclasses.replace(
        asset_cfg,
        urdf_file=ORIGINAL_G1_URDF_FILE,
        xml_file=ORIGINAL_G1_XML_FILE,
        collapse_fixed_joints=True,
    )
    new_robot_cfg = dataclasses.replace(tyro_config.robot, asset=new_asset_cfg)
    logger.info(
        "Normalized IR robot assets to original G1 variants: "
        f"urdf={ORIGINAL_G1_URDF_FILE}, xml={ORIGINAL_G1_XML_FILE}, collapse_fixed_joints=True"
    )
    return dataclasses.replace(tyro_config, robot=new_robot_cfg)


def _force_original_depth_camera_asset_mode(tyro_config: ExperimentConfig) -> ExperimentConfig:
    observation_cfg = tyro_config.observation
    if observation_cfg is None:
        return tyro_config

    changed = False
    new_groups = dict(observation_cfg.groups)
    for group_name, group_cfg in observation_cfg.groups.items():
        new_terms = dict(group_cfg.terms)
        group_changed = False
        for term_name, term_cfg in group_cfg.terms.items():
            params = dict(term_cfg.params)
            if params.get("robot_depth_asset_mode") == "original":
                continue
            if "robot_depth_asset_mode" not in params:
                continue
            params["robot_depth_asset_mode"] = "original"
            new_terms[term_name] = dataclasses.replace(term_cfg, params=params)
            group_changed = True
            changed = True
        if group_changed:
            new_groups[group_name] = dataclasses.replace(group_cfg, terms=new_terms)

    if not changed:
        return tyro_config

    logger.info("Forced IR depth-related observation terms to robot_depth_asset_mode='original'.")
    return dataclasses.replace(tyro_config, observation=dataclasses.replace(observation_cfg, groups=new_groups))


def _ensure_isaacsim_cameras_enabled() -> None:
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
        logger.info("Enabled IsaacSim cameras for IR depth capture via --enable_cameras.")


def run_ir_evaluation(
    algo: BaseAlgo,
    ir_cfg: IRCheckpointConfig,
    ir_log_dir: Path,
    depth_camera_mount: DepthCameraMountSpec,
) -> None:
    env = algo.env
    telemetry = IRTelemetryRecorder(
        algo=algo,
        ir_cfg=ir_cfg,
        log_dir=str(ir_log_dir),
        depth_camera_mount=depth_camera_mount,
    )

    if hasattr(algo, "_eval_mode"):
        algo._eval_mode()  # type: ignore[attr-defined]
    env.set_is_evaluating()

    telemetry.on_pre_evaluate_policy()
    algo.eval_policy = algo.get_inference_policy()  # type: ignore[attr-defined]

    actor_state = algo._create_actor_state()  # type: ignore[attr-defined]
    obs_dict = env.reset_all()
    init_actions = torch.zeros(env.num_envs, algo.num_act, device=algo.device)  # type: ignore[attr-defined]
    actor_state.update({"obs": obs_dict, "actions": init_actions})

    critic_obs = torch.cat([actor_state["obs"][k] for k in algo.critic_obs_keys], dim=1)  # type: ignore[attr-defined]
    actor_state["obs"]["critic_obs"] = critic_obs

    total_eval_steps = getattr(getattr(algo, "config", None), "max_eval_steps", None)

    step = 0
    try:
        while not telemetry.run_complete:
            if total_eval_steps is not None and step >= total_eval_steps:
                logger.info(f"Reached total evaluation step limit: {total_eval_steps}")
                break

            actor_state["step"] = step
            actor_state = algo._pre_eval_env_step(actor_state)  # type: ignore[attr-defined]
            actor_state = telemetry.on_pre_eval_env_step(actor_state)
            actor_state = algo.env_step(actor_state)  # type: ignore[attr-defined]
            actor_state = telemetry.on_post_eval_env_step(actor_state)
            step += 1
    finally:
        telemetry.on_post_evaluate_policy()


def run_ir_with_tyro(
    tyro_config: ExperimentConfig,
    ir_cfg: IRCheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
):
    _ensure_isaacsim_cameras_enabled()
    tyro_config = _with_original_robot_assets(tyro_config)
    tyro_config = _force_original_depth_camera_asset_mode(tyro_config)
    tyro_config = resolve_multi_object_urdf_config(tyro_config)
    depth_camera_mount = _resolve_depth_mount_spec(tyro_config)

    ir_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="ir")
    ir_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving IR logs to {ir_log_dir}")
    tyro_config.save_config(str(ir_log_dir / CONFIG_NAME))

    env, device, simulation_app = setup_simulation_environment(tyro_config)

    try:
        assert ir_cfg.checkpoint is not None
        checkpoint = load_checkpoint(ir_cfg.checkpoint, str(ir_log_dir))
        checkpoint_path = str(checkpoint)

        algo_class = get_class(tyro_config.algo._target_)
        algo: BaseAlgo = algo_class(
            device=device,
            env=env,
            config=tyro_config.algo.config,
            log_dir=str(ir_log_dir),
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

        try:
            run_ir_evaluation(
                algo=algo,
                ir_cfg=ir_cfg,
                ir_log_dir=ir_log_dir,
                depth_camera_mount=depth_camera_mount,
            )
        except Exception:
            logger.exception(
                "IR evaluation failed after setup. This is often a camera prim / asset configuration issue, not a GPU OOM."
            )
            raise
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    normalized_args = _normalize_ir_cli_bool_equals_args(sys.argv[1:])
    # IsaacLab's AppLauncher later parses the process-level argv with argparse,
    # where boolean flags such as --headless do not accept "=True" spellings.
    sys.argv = [sys.argv[0]] + normalized_args
    ir_cfg, remaining_args = tyro.cli(
        IRCheckpointConfig,
        args=normalized_args,
        return_unknown_args=True,
        add_help=False,
    )
    saved_cfg, saved_wandb_path = load_saved_experiment_config(ir_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    object_cfg = dataclasses.replace(
        eval_cfg.robot.object,
        object_urdf_asset="train_r1/objects",
        object_urdf_folder="train_r1/objects",
        object_urdf_name_to_path={},
    )
    eval_cfg = dataclasses.replace(
        eval_cfg,
        robot=dataclasses.replace(eval_cfg.robot, object=object_cfg),
        reward=r1_reward_values.r1_26dof_fastsac_reward,
        observation=dataclasses.replace(
            eval_cfg.observation,
            groups={
                **eval_cfg.observation.groups,
                "replay_actor_obs": r1_observation_values.r1_26dof_fastsac_observation.groups["actor_obs"],
                # Match the runtime r1-fastsac privileged critic schema exactly.
                # In particular this includes direct IR plus task/object identity,
                # and records the action actually executed by the teacher rollout.
                "replay_critic_obs": dataclasses.replace(
                    r1_observation_values.r1_student_privileged_critic_obs,
                    terms={
                        **r1_observation_values.r1_student_privileged_critic_obs.terms,
                        "previous_action": dataclasses.replace(
                            r1_observation_values.r1_student_privileged_critic_obs.terms["previous_action"],
                            func="holosoma.managers.observation.terms.wbt:actions",
                        ),
                    },
                ),
            },
        ),
    )
    eval_cfg = dataclasses.replace(
        eval_cfg,
        training=dataclasses.replace(
            eval_cfg.training,
            num_envs=DEFAULT_IR_NUM_ENVS,
            project=DEFAULT_IR_PROJECT,
            headless=False,
        ),
        logger=dataclasses.replace(
            eval_cfg.logger,
            base_dir=DEFAULT_IR_LOG_BASE_DIR,
        ),
    )
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    camera_location = ir_cfg.depth_camera_location.strip().lower()
    if camera_location not in R1_DEPTH_CAMERA_MOUNT_PRESETS:
        raise ValueError(
            f"Unsupported --depth-camera-location='{ir_cfg.depth_camera_location}'. "
            "Expected 'cam1' or 'cam2'."
        )
    overwritten_tyro_config = dataclasses.replace(
        overwritten_tyro_config,
        simulator=dataclasses.replace(
            overwritten_tyro_config.simulator,
            config=dataclasses.replace(
                overwritten_tyro_config.simulator.config,
                robot_depth_camera_location=camera_location,
            ),
        ),
    )
    if ir_cfg.headless and not overwritten_tyro_config.training.headless:
        overwritten_tyro_config = dataclasses.replace(
            overwritten_tyro_config,
            training=dataclasses.replace(overwritten_tyro_config.training, headless=True),
        )
        logger.info("IR telemetry collection will run headless; simulator viewer is disabled.")
    logger.info(
        f"Running IR evaluation with num_envs={overwritten_tyro_config.training.num_envs}, "
        f"headless={overwritten_tyro_config.training.headless}, "
        f"episode_max_eval_steps={ir_cfg.max_eval_steps}, num_eval_episodes_per_env={ir_cfg.num_eval_episodes}, "
        f"all_motions_iterations={ir_cfg.all_motions_iterations}, "
        f"depth_camera_location={camera_location}, "
        f"camera_position_noise_m={ir_cfg.camera_position_noise_m}, "
        f"depth_pixel_noise_max_std_m={ir_cfg.depth_pixel_noise_max_std_m}, "
        f"depth_dropout_probability={ir_cfg.depth_dropout_probability}, "
        f"depth_resolution_wh={DEPTH_RESOLUTION}, "
        f"depth_frame_shape_hw={(DEPTH_RESOLUTION[1], DEPTH_RESOLUTION[0])}"
    )
    run_ir_with_tyro(overwritten_tyro_config, ir_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()
