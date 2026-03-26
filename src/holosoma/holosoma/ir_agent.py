# python src/holosoma/holosoma/ir_agent.py   --checkpoint=/home/rllab/haechan/holosoma/logs/WholeBodyTracking/20260324_040235-g1_29dof_wbt_manager-locomotion/model_15000.pt   --num-eval-episodes=1000   --max-eval-steps=200

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from pydantic.dataclasses import dataclass

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.object_urdf import resolve_multi_object_urdf_config
from holosoma.utils.pelvis_surface_features import PelvisSurfaceFeatureComputer
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


REALSENSE_URDF_FILE = "g1/g1_29dof_realsense.urdf"
REALSENSE_XML_FILE = "g1/g1_29dof_realsense.xml"
REALSENSE_CAMERA_BODY_LINK = "realsense_d435_link"
DEPTH_CAMERA_FRAME_LINK = "realsense_d435_depth_optical_frame"
DEPTH_CAMERA_PRIM_NAME = "realsense_d435_depth"
DEPTH_RESOLUTION = (80, 60)


@dataclass(frozen=True)
class IRCheckpointConfig:
    checkpoint: str | None = None
    """Path to a local checkpoint file, or W&B URI in the format `wandb://<entity>/<project>/<run_id>[/<checkpoint_name>]`."""

    max_eval_steps: int | None = None
    """Maximum number of evaluation steps inside a single episode."""

    num_eval_episodes: int | None = None
    """Total number of episodes to collect before ending the IR run. None means keep running until manually stopped."""

    pelvis_surface_feature_log_env_ids: tuple[int, ...] = (0,)
    """Environment ids to print for live pelvis-based surface features during playback."""

    pelvis_body_name: str = "pelvis"
    """Rigid body name from the simulated robot to use as x_t."""

    pelvis_surface_feature_mesh_mode: str = "box"
    """How to treat mesh collision geometry for live pelvis surface features: 'box' or 'full'."""

    show_camera_marker: bool = True
    """Deprecated. Camera marker visualization was removed from ir_agent.py and this flag has no effect."""


@dataclasses.dataclass(frozen=True)
class DepthCameraMountSpec:
    source_urdf_path: str
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


def _load_realsense_depth_mount_from_urdf(tyro_config: ExperimentConfig) -> DepthCameraMountSpec:
    urdf_path = _resolve_robot_urdf_path(tyro_config)
    robot_root = ET.parse(urdf_path).getroot()

    camera_body_joint = _find_joint_by_child(robot_root, REALSENSE_CAMERA_BODY_LINK)
    optical_frame_joint = _find_joint_by_child(robot_root, DEPTH_CAMERA_FRAME_LINK)

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
        self.window_size = 5
        self.log_env_ids = set(ir_cfg.pelvis_surface_feature_log_env_ids)
        self.pelvis_body_name = ir_cfg.pelvis_body_name
        self.mesh_mode = ir_cfg.pelvis_surface_feature_mesh_mode
        self.max_eval_steps = ir_cfg.max_eval_steps
        self.num_eval_episodes = ir_cfg.num_eval_episodes
        self.depth_resolution = DEPTH_RESOLUTION
        self.depth_camera_mount = depth_camera_mount
        self.show_camera_marker = ir_cfg.show_camera_marker

        self._feature_computer: PelvisSurfaceFeatureComputer | None = None
        self._pelvis_body_index: int | None = None
        self._run_complete = False

        self._episode_indices: list[int] = []
        self._episode_steps: list[int] = []
        self._u_windows: list[list[list[float]]] = []
        self._depth_windows: list[list[list[list[float]]]] = []
        self._episode_entries: list[list[dict]] = []
        self._index_entries: list[dict] = []
        self._completed_episode_count = 0

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

    def _reset_env_buffers(self, env_id: int) -> None:
        self._episode_steps[env_id] = 0
        self._u_windows[env_id] = []
        self._depth_windows[env_id] = []
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

    def _episode_file_name(self, env_id: int, episode_index: int) -> str:
        return f"episode_env{env_id:03d}_idx{episode_index:03d}.json"

    def _write_index_file(self) -> None:
        index_path = self.telemetry_dir / "episodes_index.json"
        with index_path.open("w", encoding="utf-8") as file:
            json.dump(self._index_entries, file, indent=2)

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
                "Expected IsaacSim to register 'robot_depth_camera' from the URDF optical frame."
            )
        return depth_camera

    def _camera_parent_prim_path(self, env_id: int) -> str:
        # Attach the camera directly under the imported URDF optical-frame link so the
        # sensor inherits the robot motion through the articulation hierarchy.
        return f"/World/envs/env_{env_id}/Robot/{self.depth_camera_mount.optical_frame_link}"

    def _camera_prim_path(self, env_id: int) -> str:
        return f"{self._camera_parent_prim_path(env_id)}/{DEPTH_CAMERA_PRIM_NAME}"

    def _read_depth_frame(self, env_id: int) -> list[list[float]]:
        expected_hw = (self.depth_resolution[1], self.depth_resolution[0])
        depth_tensor = self.env.simulator.get_robot_depth_frame(env_id)
        if depth_tensor is None:
            if self._depth_windows[env_id]:
                return copy.deepcopy(self._depth_windows[env_id][-1])
            return [[0.0 for _ in range(expected_hw[1])] for _ in range(expected_hw[0])]

        depth_array = _to_numpy_float32(depth_tensor)
        depth_array = np.squeeze(depth_array)
        if depth_array.shape == (self.depth_resolution[0], self.depth_resolution[1]):
            depth_array = depth_array.T
        if depth_array.shape != expected_hw:
            raise RuntimeError(
                f"Unexpected depth frame shape {depth_array.shape} for env {env_id}; expected {expected_hw}."
            )

        depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)
        return depth_array.tolist()

    def _export_episode(self, episode_data: dict) -> None:
        file_name = self._episode_file_name(episode_data["env_id"], episode_data["episode_index"])
        file_path = self.telemetry_dir / file_name
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(episode_data, file, indent=2)

        self._index_entries.append(
            {
                "env_id": episode_data["env_id"],
                "episode_index": episode_data["episode_index"],
                "file": file_name,
                "num_steps": episode_data["num_steps"],
                "termination_reason": episode_data["termination_reason"],
            }
        )
        self._write_index_file()

    def _finalize_episode(self, env_id: int, reason: str, global_step: int | None) -> None:
        episode_data = {
            "env_id": env_id,
            "episode_index": self._episode_indices[env_id],
            "num_steps": len(self._episode_entries[env_id]),
            "termination_reason": reason,
            "max_eval_steps": self.max_eval_steps,
            "num_eval_episodes": self.num_eval_episodes,
            "pelvis_body_name": self.pelvis_body_name,
            "mesh_mode": self.mesh_mode,
            "depth_resolution": list(self.depth_resolution),
            "depth_camera_prim_name": DEPTH_CAMERA_PRIM_NAME,
            "depth_camera_mount": self.depth_camera_mount.to_json_dict(),
            "completed_at_global_step": global_step,
            "entries": self._episode_entries[env_id],
        }
        self._export_episode(episode_data)
        self._completed_episode_count += 1
        self._reset_env_buffers(env_id)

        if self.num_eval_episodes is not None and self._completed_episode_count >= self.num_eval_episodes:
            self._run_complete = True

        logger.info(
            f"[ir_episode_complete] env={env_id} episode={episode_data['episode_index']} "
            f"steps={episode_data['num_steps']} reason={reason} total_completed={self._completed_episode_count}"
        )

    def on_pre_evaluate_policy(self) -> None:
        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is None:
            raise RuntimeError("motion_command not found; IR telemetry requires a motion command.")
        if not getattr(motion_command.motion, "has_object", False):
            raise RuntimeError("IR telemetry requires a motion with an object.")
        if self.pelvis_body_name not in self.env.body_names:
            raise RuntimeError(
                f"Pelvis body '{self.pelvis_body_name}' was not found in simulator body names: {self.env.body_names}."
            )
        if self.max_eval_steps is not None and self.max_eval_steps <= 0:
            raise ValueError(f"max_eval_steps must be positive when provided, got {self.max_eval_steps}")
        if self.num_eval_episodes is not None and self.num_eval_episodes <= 0:
            raise ValueError(f"num_eval_episodes must be positive when provided, got {self.num_eval_episodes}")

        self._pelvis_body_index = self.env.body_names.index(self.pelvis_body_name)
        depth_camera = self._get_simulator_depth_camera()
        self.depth_resolution = (int(depth_camera.cfg.width), int(depth_camera.cfg.height))

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
        self._feature_computer = PelvisSurfaceFeatureComputer.from_object_config(
            self.env.robot_config.object,
            mesh_mode=self.mesh_mode,
        )
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.depth_image_dir.mkdir(parents=True, exist_ok=True)

        self._episode_indices = [0 for _ in range(self.env.num_envs)]
        self._episode_steps = [0 for _ in range(self.env.num_envs)]
        self._u_windows = [[] for _ in range(self.env.num_envs)]
        self._depth_windows = [[] for _ in range(self.env.num_envs)]
        self._episode_entries = [[] for _ in range(self.env.num_envs)]
        self._index_entries = []
        self._completed_episode_count = 0
        self._run_complete = False
        self._write_index_file()

        logger.info(
            f"IR telemetry enabled for body '{self.pelvis_body_name}' with window_size={self.window_size}, "
            f"mesh_mode={self.mesh_mode}, max_eval_steps={self.max_eval_steps}, "
            f"num_eval_episodes={self.num_eval_episodes}, depth_resolution={self.depth_resolution}."
        )
        logger.info(
            "Loaded depth camera mount from URDF: "
            f"parent_link={self.depth_camera_mount.parent_link}, "
            f"optical_frame_link={self.depth_camera_mount.optical_frame_link}, "
            f"translation={list(self.depth_camera_mount.translation)}, "
            f"quaternion_ros_wxyz={list(self.depth_camera_mount.quaternion_ros_wxyz)}, "
            f"source_urdf='{self.depth_camera_mount.source_urdf_path}'"
        )
        logger.info(
            f"IR depth camera is managed by IsaacSim scene sensor at URDF optical-frame prim: {self._camera_parent_prim_path(0)}"
        )
        if self.show_camera_marker:
            logger.info("show_camera_marker is deprecated and ignored; marker visualization was removed from ir_agent.py.")

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        if self._feature_computer is None or self._pelvis_body_index is None or self._run_complete:
            return actor_state

        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is None:
            return actor_state

        pelvis_pos_w = self.env.simulator._rigid_body_pos[:, self._pelvis_body_index, :]
        pelvis_lin_vel_w = self.env.simulator._rigid_body_vel[:, self._pelvis_body_index, :]
        object_keys = self._object_keys_for_envs(motion_command, self.env.num_envs)
        features = self._feature_computer.compute_batch(
            pelvis_pos_w=pelvis_pos_w,
            pelvis_lin_vel_w=pelvis_lin_vel_w,
            object_pos_w=motion_command.simulator_object_pos_w,
            object_quat_w=motion_command.simulator_object_quat_w,
            object_keys=object_keys,
        )

        actor_state["pelvis_surface_features"] = features
        actor_state["ir_object_keys"] = object_keys
        global_step = int(actor_state.get("step", -1))

        for env_id in range(self.env.num_envs):
            current_u_t = [float(v) for v in features["u_t"][env_id].tolist()]
            current_u_window = self._build_window(self._u_windows[env_id], current_u_t)
            current_depth_frame = self._read_depth_frame(env_id)
            current_depth_window = self._build_window(self._depth_windows[env_id], current_depth_frame)
            # depth_image_file = self._save_depth_preview(
            #     env_id=env_id,
            #     episode_index=self._episode_indices[env_id],
            #     episode_step=self._episode_steps[env_id],
            #     depth_frame=current_depth_frame,
            # )
            entry = {
                "global_step": global_step,
                "episode_index": self._episode_indices[env_id],
                "episode_step": self._episode_steps[env_id],
                "env_id": env_id,
                "object_key": object_keys[env_id],
                "phi": float(features["phi"][env_id, 0].item()),
                "grad_phi": [float(v) for v in features["grad_phi"][env_id].tolist()],
                "v_norm": [float(v) for v in features["v_norm"][env_id].tolist()],
                "v_tan": [float(v) for v in features["v_tan"][env_id].tolist()],
                "u_t": current_u_t,
                "u_window": current_u_window,
                "depth_window": current_depth_window,
                # "depth_image_file": depth_image_file,
            }
            self._episode_entries[env_id].append(entry)

            if env_id in self.log_env_ids:
                logger.info(
                    f"[ir_u_window] step={global_step} env={env_id} episode={self._episode_indices[env_id]} "
                    f"episode_step={self._episode_steps[env_id]} depth_shape={self.depth_resolution}"
                )

        return actor_state

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        if self._run_complete:
            return actor_state

        dones = actor_state.get("dones")
        if dones is None:
            return actor_state

        global_step = int(actor_state.get("step", -1))
        maxed_env_ids: list[int] = []

        for env_id in range(self.env.num_envs):
            if self._run_complete:
                break

            self._episode_steps[env_id] += 1
            reached_limit = self.max_eval_steps is not None and self._episode_steps[env_id] >= self.max_eval_steps
            is_done = bool(dones[env_id].item())

            if is_done:
                self._finalize_episode(env_id, reason="done", global_step=global_step)
            elif reached_limit:
                self._finalize_episode(env_id, reason="max_eval_steps", global_step=global_step)
                if not self._run_complete:
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
        target_reached = self.num_eval_episodes is not None and self._completed_episode_count >= self.num_eval_episodes
        if not target_reached:
            for env_id in range(self.env.num_envs):
                if self._episode_entries[env_id]:
                    self._finalize_episode(env_id, reason="run_end", global_step=None)

        self._write_index_file()
        logger.info(
            f"Exported IR telemetry for {len(self._index_entries)} episode(s) to {self.telemetry_dir}"
        )


def _with_realsense_robot_assets(tyro_config: ExperimentConfig) -> ExperimentConfig:
    asset_cfg = tyro_config.robot.asset
    new_asset_cfg = dataclasses.replace(
        asset_cfg,
        urdf_file=REALSENSE_URDF_FILE,
        xml_file=REALSENSE_XML_FILE,
        collapse_fixed_joints=False,
    )
    new_robot_cfg = dataclasses.replace(tyro_config.robot, asset=new_asset_cfg)
    logger.info(
        f"Switched IR robot assets to RealSense variants: urdf={REALSENSE_URDF_FILE}, xml={REALSENSE_XML_FILE}, collapse_fixed_joints=False"
    )
    return dataclasses.replace(tyro_config, robot=new_robot_cfg)


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
    tyro_config = _with_realsense_robot_assets(tyro_config)
    tyro_config = resolve_multi_object_urdf_config(tyro_config)
    depth_camera_mount = _load_realsense_depth_mount_from_urdf(tyro_config)

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
    ir_cfg, remaining_args = tyro.cli(IRCheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(ir_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    logger.info(
        f"Running IR evaluation with num_envs={overwritten_tyro_config.training.num_envs}, "
        f"episode_max_eval_steps={ir_cfg.max_eval_steps}, num_eval_episodes={ir_cfg.num_eval_episodes}, "
        f"depth_resolution={DEPTH_RESOLUTION}"
    )
    run_ir_with_tyro(overwritten_tyro_config, ir_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()
