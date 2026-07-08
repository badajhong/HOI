from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Literal

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from scipy import sparse as sp  # type: ignore[import-untyped]
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from tqdm import tqdm
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting.config_types.retargeter import DEFAULT_CONTACT_HUMAN_JOINT_REGEX, FootLockConfig

# Add src to path for direct execution
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import with type ignore for mypy compatibility
from mujoco_utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    _world_mesh_from_geom,
)
from utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
    transform_points_local_to_world,
    transform_points_world_to_local,
)
from viser_utils import create_motion_control_sliders  # type: ignore[import-not-found,no-redef]  # noqa: E402


class InteractionMeshRetargeter:
    """
    A class to perform kinematic retargeting from human motion to a robot,
    preserving spatial relationships using an interaction mesh.
    """

    VIRTUAL_CONTACT_BODY_SPECS = {
        "left_hand_contact_link": ("left_wrist_roll_link", np.array([0.07, 0.0, 0.0], dtype=np.float64)),
        "right_hand_contact_link": ("right_wrist_roll_link", np.array([0.07, 0.0, 0.0], dtype=np.float64)),
    }

    FOOT_CONTACT_VISUAL_ALIASES = {
        "left_ankle_constraint_A_link": ("left_ankle_roll_link",),
        "left_ankle_constraint_B_link": ("left_ankle_roll_link",),
        "left_foot_front_outer_link": ("left_ankle_roll_link",),
        "left_foot_front_inner_link": ("left_ankle_roll_link",),
        "left_foot_rear_outer_link": ("left_ankle_roll_link",),
        "left_foot_rear_inner_link": ("left_ankle_roll_link",),
        "right_ankle_constraint_A_link": ("right_ankle_roll_link",),
        "right_ankle_constraint_B_link": ("right_ankle_roll_link",),
        "right_foot_front_outer_link": ("right_ankle_roll_link",),
        "right_foot_front_inner_link": ("right_ankle_roll_link",),
        "right_foot_rear_outer_link": ("right_ankle_roll_link",),
        "right_foot_rear_inner_link": ("right_ankle_roll_link",),
    }

    def __init__(
        self,
        task_constants: ModuleType,
        object_urdf_path: str,
        q_a_init_idx: int = -7,
        activate_foot_sticking: bool = True,
        activate_obj_non_penetration: bool = True,
        activate_joint_limits: bool = True,
        step_size: float = 0.2,
        collision_detection_threshold: float = 0.1,
        penetration_tolerance: float = 1e-3,
        surface_penetration_tolerance: float | None = None,
        object_penetration_tolerance: float | None = None,
        foot_sticking_tolerance: float = 1e-3,
        foot_lock: FootLockConfig | None = None,
        visualize: bool = False,
        contact_visualization: bool = False,
        contact_source: Literal["robot", "human"] = "human",
        contact_threshold: float = 0.05,
        contact_target_topk: int = 5,
        contact_human_joint_regex: str = DEFAULT_CONTACT_HUMAN_JOINT_REGEX,
        debug: bool = False,
        w_nominal_tracking_init: float = 5.0,
        nominal_tracking_tau: float = 10.0,
    ):
        """This kinematic retargeter solves the diffIK problem with hard constraints in SQP style.
        During each SQP iteration, the problem is solved with the following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation in the object frame.
            2. [Constraint] Enforce the non-penetration constraints w/ the ground and (if activated) the object.
            3. [Constraint] Enforce the foot sticking constraints if activated.
            4. [Constraint] Enforce the joint limits if activated.
            5. [Constraint] Enforce trust region of dq.
        The constraints are linearized and the costs are quadratic with a trust region.

        Args:
            q_a_init_idx: the index in robot's configuration where the optimization variables start. -7: starts from the
            floating base, -3: starts from the translation of the floating base, 0: starts from the actuated DOF,
            12: starts from waist, 15: starts from left shoulder
            step_size: trust region for each SQP iteration.
            collision_detection_threshold: only start to detect collision
            when the distance is smaller than this threshold.
            penetration_tolerance: tolerance for penetration when enforcing non-penetration constraints.
            surface_penetration_tolerance: optional tolerance for ground/surface contacts. Defaults to
                penetration_tolerance.
            object_penetration_tolerance: optional tolerance for robot-object contacts. Defaults to
                penetration_tolerance.
            foot_sticking_tolerance: tolerance for foot sticking constraints in x, y.
            foot_lock: configuration for explicit frame-range based foot locking constraints.
            nominal_tracking_tau: the time constant for the nominal tracking cost.
        """

        self.robot_model_path = task_constants.ROBOT_URDF_FILE
        self.object_model_path = object_urdf_path
        self.object_name = task_constants.OBJECT_NAME
        self.collision_detection_threshold = collision_detection_threshold
        self.activate_foot_sticking = activate_foot_sticking
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_joint_limits = activate_joint_limits
        self.foot_links = dict(zip(task_constants.FOOT_STICKING_LINKS, task_constants.FOOT_STICKING_LINKS))
        self.penetration_tolerance = penetration_tolerance
        self.surface_penetration_tolerance = (
            penetration_tolerance if surface_penetration_tolerance is None else surface_penetration_tolerance
        )
        self.object_penetration_tolerance = (
            penetration_tolerance if object_penetration_tolerance is None else object_penetration_tolerance
        )
        self.step_size = step_size
        self.visualize = visualize
        self.contact_visualization = contact_visualization
        self.contact_source = contact_source
        self.contact_threshold = contact_threshold
        self.contact_target_topk = max(int(contact_target_topk), 1)
        self.contact_human_joint_regex = contact_human_joint_regex
        self.debug = debug
        self.demo_joints = task_constants.DEMO_JOINTS
        self.laplacian_match_links = task_constants.JOINTS_MAPPING
        self.task_constants = task_constants

        self.smplh_mapped_joint_indices = [self.demo_joints.index(name) for name in self.laplacian_match_links]

        # Setup weights and parameters
        self.laplacian_weights = 10
        self.smooth_weight = 0.2
        # Tolerance for foot sticking constraints in x, y.
        self.foot_sticking_tolerance = foot_sticking_tolerance
        self._init_foot_lock(foot_lock)

        # Setup visualization if requested
        if self.visualize:
            self._setup_visualization()

        # Load Mujoco model
        scene_xml_file = getattr(self.task_constants, "SCENE_XML_FILE", "")
        if self.object_name == "ground":
            robot_xml_path = self.robot_model_path.replace(".urdf", ".xml")
        elif scene_xml_file:
            robot_xml_path = scene_xml_file
        else:
            robot_xml_path = self.robot_model_path.replace(".urdf", "_w_" + self.object_name + ".xml")

        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print("Loading robot model from: ", robot_xml_path)

        self.robot_data = mujoco.MjData(self.robot_model)

        if self.robot_data.qpos.shape[0] > 7 + self.task_constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False

        self.nq = self.robot_model.nq

        self.q_a_init_idx = q_a_init_idx
        self.q_a_indices = np.arange(7 + self.q_a_init_idx, 7 + self.task_constants.ROBOT_DOF)

        self.nq_a = len(self.q_a_indices)

        # Create complete limits with floating base (-inf, inf) and actuated joint limits
        n_floating_base = 7
        joint_names = [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)]
        actuated_joints = [(i, name) for i, name in enumerate(joint_names) if name]  # Filter out None names

        large_number = 1e6
        complete_lower_limits = np.concatenate(
            [-large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 0]]
        )
        complete_upper_limits = np.concatenate(
            [large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 1]]
        )

        self.q_a_lb = complete_lower_limits[self.q_a_indices]
        self.q_a_ub = complete_upper_limits[self.q_a_indices]

        self.q_a_lb[np.array(list(self.task_constants.MANUAL_LB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_LB.values()
        )
        self.q_a_ub[np.array(list(self.task_constants.MANUAL_UB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_UB.values()
        )

        # Prevent too much waist twist
        self.Q_diag = np.zeros(self.nq_a) * 1e-3
        self.Q_diag[np.array(list(self.task_constants.MANUAL_COST.keys())).astype(int)] = list(
            self.task_constants.MANUAL_COST.values()
        )

        self.w_nominal_tracking_init = w_nominal_tracking_init
        self.nominal_tracking_tau = nominal_tracking_tau
        self.track_nominal_indices = task_constants.NOMINAL_TRACKING_INDICES

    def _init_foot_lock(self, foot_lock: FootLockConfig | None) -> None:
        """Initialize foot lock configuration and normalize window mappings."""
        self.foot_lock = foot_lock or FootLockConfig()
        self._foot_lock_windows: dict[str, tuple[tuple[int, int], ...]] = {"left": (), "right": ()}
        if self.foot_lock.windows is None:
            return
        for key, windows in self.foot_lock.windows.items():
            key_lower = key.lower()
            side = None
            if key_lower.startswith("l") or ("left" in key_lower):
                side = "left"
            elif key_lower.startswith("r") or ("right" in key_lower):
                side = "right"
            if side is None:
                continue

            normalized_windows: list[tuple[int, int]] = []
            for window in windows:
                if len(window) != 2:
                    raise ValueError(f"Invalid foot lock window for {key}: {window}")
                start, end = int(window[0]), int(window[1])
                if end < start:
                    raise ValueError(f"Invalid foot lock window with end < start for {key}: {window}")
                normalized_windows.append((start, end))
            self._foot_lock_windows[side] = tuple(normalized_windows)

    def _link_name_from_viser_mesh_path(self, mesh_path: str) -> str | None:
        """Resolve a Viser mesh path back to a URDF link name."""
        link_names = set(self.robot_urdf.link_map)
        for part in reversed(mesh_path.split("/")):
            if part in link_names:
                return part
            stem = Path(part).stem
            if stem in link_names:
                return stem
        return None

    def _setup_visualization(self):
        """Setup Viser visualization components."""
        self.server = viser.ViserServer()

        # 1) Ensure a world frame exists (absolute path!)
        try:
            self.server.scene.add_frame("/world", show_axes=False)
        except Exception:
            print("Starting viser")

        # Create parent frames for robot and object
        self.robot_base = self.server.scene.add_frame("/world/robot", show_axes=False)

        print("robot_model_path: ", self.robot_model_path)

        # Load robot URDF
        self.robot_urdf = yourdfpy.URDF.load(
            self.robot_model_path,
            load_meshes=True,
            build_scene_graph=True,
        )

        print("Viser using robot URDF: ", self.robot_model_path)

        # Create ViserUrdf instance for robot, attaching it to the robot_base frame
        self.viser_robot = ViserUrdf(
            self.server,
            urdf_or_path=self.robot_urdf,
            root_node_name="/world/robot",  # This links to the robot_base frame we created
            mesh_color_override=(245, 245, 245),
        )
        self.robot_mesh_handles_by_link: dict[str, list[viser.SceneNodeHandle]] = {}
        for mesh_handle in self.viser_robot._meshes:  # noqa: SLF001
            link_name = self._link_name_from_viser_mesh_path(mesh_handle.name)
            if link_name is not None:
                self.robot_mesh_handles_by_link.setdefault(link_name, []).append(mesh_handle)

        self.contact_viser_robot = None
        self.contact_mesh_handles_by_link: dict[str, list[viser.SceneNodeHandle]] = {}
        if self.contact_visualization:
            self.server.scene.add_frame("/world/robot/contact_overlay", show_axes=False)
            self.contact_viser_robot = ViserUrdf(
                self.server,
                urdf_or_path=self.robot_urdf,
                root_node_name="/world/robot/contact_overlay",
                mesh_color_override=(255, 45, 35),
            )
            for mesh_handle in self.contact_viser_robot._meshes:  # noqa: SLF001
                mesh_handle.visible = False
                link_name = self._link_name_from_viser_mesh_path(mesh_handle.name)
                if link_name is not None:
                    self.contact_mesh_handles_by_link.setdefault(link_name, []).append(mesh_handle)

        # Similarly for object
        if self.object_model_path:
            self.object_base = self.server.scene.add_frame("/world/object", show_axes=False)

            self.object_urdf = yourdfpy.URDF.load(
                self.object_model_path,
                load_meshes=True,
                build_scene_graph=True,
            )

            # Create ViserUrdf instance for object, attaching it to the object_base frame
            self.viser_object = ViserUrdf(
                self.server,
                urdf_or_path=self.object_urdf,
                root_node_name="/world/object",  # This links to the object_base frame we created
            )
            print("Viser using object URDF: ", self.object_model_path)

        else:
            self.viser_object = None

        # Check the number of actuated joints and their names
        robot_joint_limits = self.viser_robot.get_actuated_joint_limits()
        print("\nRobot joints:")
        print("Number of actuated joints:", len(robot_joint_limits))
        print("Joint names:", list(robot_joint_limits.keys()))

        # Initialize robot with this configuration
        robot_initial_config = np.zeros(len(robot_joint_limits))
        self.viser_robot.update_cfg(robot_initial_config)

        # Add grid
        self.server.scene.add_grid(
            "/world/grid",
            width=8,
            height=8,
            position=(0.0, 0.0, 0.0),
        )

    def draw_mesh_from_geom(self, model, data, geom_id, geom_name, name="/mesh", color=(50, 150, 255), opacity=0.5):
        """
        Draw a single MuJoCo mesh geom (already baked to world coords) in viser.
        color is [0, 255] RGB ints; opacity is [0,1].
        """
        if not hasattr(self, "server"):
            return
        V, F = _world_mesh_from_geom(model, data, geom_id, geom_name)
        self.server.scene.add_mesh_simple(
            name,
            vertices=V.astype(np.float32),
            faces=F.astype(np.int32),
            position=(0.0, 0.0, 0.0),  # already world-frame
            color=tuple(int(c) for c in color),
            opacity=float(opacity),
        )

    def draw_mesh_pair_with_contact(
        self,
        model,
        data,
        geom_id1,
        geom_id2,
        geom1_name,
        geom2_name,
        fromto=None,
        group_name="pair",
        color1=(50, 150, 255),
        color2=(255, 120, 60),
        opacity=0.45,
        show_segment=True,
    ):
        """
        Draw two meshes and (optionally) a contact/query segment.
        Uses the existing self.draw_keypoints(...) to visualize points.
        """
        # Note: sometime geom does not have mesh, mesh_id will be -1
        if int(model.geom_dataid[geom_id1]) == -1 or int(model.geom_dataid[geom_id2]) == -1:
            return

        base = f"/{group_name}"
        # meshes
        self.draw_mesh_from_geom(model, data, geom_id1, geom1_name, name=f"{base}/mesh1", color=color1, opacity=opacity)
        self.draw_mesh_from_geom(model, data, geom_id2, geom2_name, name=f"{base}/mesh2", color=color2, opacity=opacity)

        # contact points (q: green, c: red) via your draw_keypoints
        if fromto is not None:
            q = np.asarray(fromto[:3], dtype=float)
            c = np.asarray(fromto[3:], dtype=float)

            # your existing helper (rgba expects floats 0..1)
            self.draw_keypoints(q, name=f"{group_name}_q", rgba=(0.0, 1.0, 0.0, 1.0))
            self.draw_keypoints(c, name=f"{group_name}_c", rgba=(1.0, 0.0, 0.0, 1.0))

    def _human_joint_to_robot_body(self, joint_name: str) -> str | None:
        mapping = self.laplacian_match_links
        if joint_name in mapping:
            return mapping[joint_name]

        left_hand_tokens = ("L_Index", "L_Middle", "L_Pinky", "L_Ring", "L_Thumb")
        right_hand_tokens = ("R_Index", "R_Middle", "R_Pinky", "R_Ring", "R_Thumb")
        if joint_name.startswith(left_hand_tokens):
            return mapping.get("L_HandCenter", mapping.get("L_Wrist"))
        if joint_name.startswith(right_hand_tokens):
            return mapping.get("R_HandCenter", mapping.get("R_Wrist"))
        return None

    def _resolve_contact_body_base_and_offset(self, body_name: str) -> tuple[int, np.ndarray] | None:
        virtual_spec = self.VIRTUAL_CONTACT_BODY_SPECS.get(body_name)
        if virtual_spec is None:
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                return body_id, np.zeros(3, dtype=np.float64)
            return None

        parent_body_name, local_offset = virtual_spec
        parent_body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, parent_body_name)
        if parent_body_id == -1:
            return None
        return parent_body_id, np.asarray(local_offset, dtype=np.float64)

    def _contact_body_parent_name(self, body_name: str) -> str:
        virtual_spec = self.VIRTUAL_CONTACT_BODY_SPECS.get(body_name)
        if virtual_spec is None:
            return body_name
        return virtual_spec[0]

    def _contact_body_parent_index(self, body_name: str) -> int:
        parent_name = self._contact_body_parent_name(body_name)
        return mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, parent_name)

    def _contact_body_local_offsets(self, body_names: list[str]) -> np.ndarray:
        offsets: list[np.ndarray] = []
        for body_name in body_names:
            resolved = self._resolve_contact_body_base_and_offset(body_name)
            if resolved is None:
                offsets.append(np.zeros(3, dtype=np.float64))
            else:
                offsets.append(resolved[1])
        return np.asarray(offsets, dtype=np.float32)

    def _contact_body_marker_position(self, body_id: int, local_offset: np.ndarray) -> np.ndarray:
        pos = self.robot_data.xpos[body_id].copy()
        if np.any(local_offset):
            rot = self.robot_data.xmat[body_id].reshape(3, 3)
            pos = pos + rot @ local_offset
        return pos

    def _selected_contact_targets(self) -> tuple[list[str], list[int], list[str], list[tuple[int, int]]]:
        include = re.compile(self.contact_human_joint_regex)
        human_names: list[str] = []
        human_indices: list[int] = []
        robot_names: list[str] = []
        robot_name_to_idx: dict[str, int] = {}
        pairs: list[tuple[int, int]] = []

        for human_idx, human_name in enumerate(self.demo_joints):
            if not include.search(human_name):
                continue
            robot_name = self._human_joint_to_robot_body(human_name)
            if robot_name is None:
                continue
            if self._resolve_contact_body_base_and_offset(robot_name) is None:
                continue
            if robot_name not in robot_name_to_idx:
                robot_name_to_idx[robot_name] = len(robot_names)
                robot_names.append(robot_name)
            human_names.append(human_name)
            human_indices.append(human_idx)
            pairs.append((len(human_names) - 1, robot_name_to_idx[robot_name]))

        return human_names, human_indices, robot_names, pairs

    def _contact_sample_points(self, fallback_points: np.ndarray) -> np.ndarray:
        if self.object_model_path:
            object_model_path = Path(self.object_model_path)
            candidate_paths = [object_model_path.with_name("sample_points.npy")]
            if not object_model_path.is_absolute():
                package_root = Path(__file__).resolve().parents[1]
                candidate_paths.append(package_root / object_model_path.with_name("sample_points.npy"))
                candidate_paths.append(Path.cwd() / object_model_path.with_name("sample_points.npy"))
            for candidate_path in candidate_paths:
                if candidate_path.exists():
                    return np.asarray(np.load(candidate_path), dtype=np.float64)
        return np.asarray(fallback_points, dtype=np.float64)

    def _distances_to_object_surface(
        self,
        query_points_w: np.ndarray,
        qpos_sequence: np.ndarray,
        object_points_local: np.ndarray,
    ) -> np.ndarray:
        query_points_w = np.asarray(query_points_w, dtype=np.float64)
        qpos_sequence = np.asarray(qpos_sequence, dtype=np.float64)
        object_points_local = np.asarray(object_points_local, dtype=np.float64)
        if query_points_w.ndim != 3 or query_points_w.shape[-1] != 3:
            raise ValueError(f"query_points_w must have shape (T, N, 3), got {query_points_w.shape}")
        if query_points_w.shape[0] != qpos_sequence.shape[0]:
            raise ValueError("query_points_w and qpos_sequence must have the same frame count.")

        tree = cKDTree(object_points_local)
        distances = np.empty(query_points_w.shape[:2], dtype=np.float32)
        for frame_idx, q in enumerate(qpos_sequence):
            if self.has_dynamic_object and q.shape[0] >= 7:
                object_pos = q[-7:-4]
                object_quat = q[-4:]
            else:
                object_pos = np.zeros(3)
                object_quat = np.array([1.0, 0.0, 0.0, 0.0])
            query_local = transform_points_world_to_local(object_quat, object_pos, query_points_w[frame_idx])
            distances[frame_idx] = tree.query(query_local, k=1)[0].astype(np.float32)
        return distances

    def _object_surface_topk_from_local_queries(
        self,
        query_points_local: np.ndarray,
        object_points_local: np.ndarray,
        topk: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_points_local = np.asarray(query_points_local, dtype=np.float64)
        object_points_local = np.asarray(object_points_local, dtype=np.float64)
        requested_topk = self.contact_target_topk if topk is None else max(int(topk), 1)
        if object_points_local.ndim != 2 or object_points_local.shape[1] != 3:
            raise ValueError(f"object_points_local must have shape (N, 3), got {object_points_local.shape}")
        if object_points_local.shape[0] == 0:
            raise ValueError("object_points_local is empty; cannot compute contact target points.")

        query_shape = query_points_local.shape[:-1]
        query_topk = min(requested_topk, object_points_local.shape[0])
        tree = cKDTree(object_points_local)
        distances, indices = tree.query(query_points_local.reshape(-1, 3), k=query_topk)
        if query_topk == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        distances = distances.reshape(*query_shape, query_topk).astype(np.float32)
        indices = indices.reshape(*query_shape, query_topk).astype(np.int32)
        target_points = object_points_local[indices].astype(np.float32)

        if query_topk < requested_topk:
            pad_count = requested_topk - query_topk
            distances = np.concatenate(
                [distances, np.repeat(distances[..., -1:], pad_count, axis=-1)],
                axis=-1,
            )
            indices = np.concatenate(
                [indices, np.repeat(indices[..., -1:], pad_count, axis=-1)],
                axis=-1,
            )
            target_points = np.concatenate(
                [target_points, np.repeat(target_points[..., -1:, :], pad_count, axis=-2)],
                axis=-2,
            )

        return distances, indices, target_points

    def _object_surface_topk_from_world_points(
        self,
        query_points_w: np.ndarray,
        qpos_sequence: np.ndarray,
        object_points_local: np.ndarray,
        topk: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_points_w = np.asarray(query_points_w, dtype=np.float64)
        qpos_sequence = np.asarray(qpos_sequence, dtype=np.float64)
        if query_points_w.ndim != 3 or query_points_w.shape[-1] != 3:
            raise ValueError(f"query_points_w must have shape (T, N, 3), got {query_points_w.shape}")
        if query_points_w.shape[0] != qpos_sequence.shape[0]:
            raise ValueError("query_points_w and qpos_sequence must have the same frame count.")

        query_points_local = np.empty_like(query_points_w, dtype=np.float64)
        for frame_idx, q in enumerate(qpos_sequence):
            if self.has_dynamic_object and q.shape[0] >= 7:
                object_pos = q[-7:-4]
                object_quat = q[-4:]
            else:
                object_pos = np.zeros(3)
                object_quat = np.array([1.0, 0.0, 0.0, 0.0])
            query_points_local[frame_idx] = transform_points_world_to_local(
                object_quat,
                object_pos,
                query_points_w[frame_idx],
            )
        return self._object_surface_topk_from_local_queries(query_points_local, object_points_local, topk=topk)

    def _mujoco_geom_names(self) -> list[str]:
        if not hasattr(self, "_geom_names") or len(self._geom_names) != self.robot_model.ngeom:
            self._geom_names = [
                mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                for g in range(self.robot_model.ngeom)
            ]
        return self._geom_names

    def _mujoco_body_names(self) -> list[str]:
        if not hasattr(self, "_body_names") or len(self._body_names) != self.robot_model.nbody:
            self._body_names = [
                mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                for b in range(self.robot_model.nbody)
            ]
        return self._body_names

    def _contact_object_geom_ids(self) -> list[int]:
        token = self.object_name.lower()
        geom_names = self._mujoco_geom_names()
        body_names = self._mujoco_body_names()
        geom_ids = []
        for geom_id, geom_name in enumerate(geom_names):
            body_name = body_names[int(self.robot_model.geom_bodyid[geom_id])]
            if token in geom_name.lower() or token in body_name.lower():
                geom_ids.append(geom_id)
        return geom_ids

    def _contact_robot_geom_ids_by_body(self, robot_names: list[str]) -> list[list[int]]:
        object_geom_ids = set(self._contact_object_geom_ids())
        geom_ids_by_body: list[list[int]] = []
        for robot_name in robot_names:
            if robot_name in self.VIRTUAL_CONTACT_BODY_SPECS:
                geom_ids_by_body.append([])
                continue
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, robot_name)
            if body_id == -1:
                geom_ids_by_body.append([])
                continue
            geom_ids_by_body.append(
                [
                    geom_id
                    for geom_id in range(self.robot_model.ngeom)
                    if int(self.robot_model.geom_bodyid[geom_id]) == body_id and geom_id not in object_geom_ids
                ]
            )
        return geom_ids_by_body

    def _get_robot_contact_positions(self, q: np.ndarray, body_names: list[str]) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.shape[0] == self.robot_data.qpos.shape[0]:
            self.robot_data.qpos[:] = q
        elif q.shape[0] == self.robot_data.qpos.shape[0] + 7:
            self.robot_data.qpos[:] = q[:-7]
        else:
            raise ValueError(
                f"q frame has length {q.shape[0]}, expected {self.robot_data.qpos.shape[0]} "
                f"for contact marker positions."
            )
        mujoco.mj_forward(self.robot_model, self.robot_data)

        positions = []
        for body_name in body_names:
            resolved = self._resolve_contact_body_base_and_offset(body_name)
            if resolved is None:
                raise ValueError(f"Contact body {body_name} not found in Mujoco model or virtual specs.")
            body_id, local_offset = resolved
            positions.append(self._contact_body_marker_position(body_id, local_offset))
        return np.asarray(positions, dtype=np.float64)

    def _robot_object_geom_distances(
        self,
        qpos_sequence: np.ndarray,
        robot_names: list[str],
        object_points_local: np.ndarray,
    ) -> np.ndarray:
        qpos_sequence = np.asarray(qpos_sequence, dtype=np.float64)
        object_geom_ids = self._contact_object_geom_ids()
        if not object_geom_ids:
            body_positions = np.stack([self._get_robot_contact_positions(q, robot_names) for q in qpos_sequence], axis=0)
            sample_points = self._contact_sample_points(object_points_local)
            return self._distances_to_object_surface(body_positions, qpos_sequence, sample_points)

        geom_ids_by_body = self._contact_robot_geom_ids_by_body(robot_names)
        distances = np.full((qpos_sequence.shape[0], len(robot_names)), np.inf, dtype=np.float32)
        distance_cutoff = max(float(self.contact_threshold) + 1e-6, float(self.collision_detection_threshold), 1.0)
        fromto = np.zeros(6, dtype=np.float64)

        for frame_idx, q in enumerate(qpos_sequence):
            if q.shape[0] != self.robot_data.qpos.shape[0]:
                raise ValueError(
                    f"qpos frame has length {q.shape[0]}, expected {self.robot_data.qpos.shape[0]} for contact labels."
                )
            self.robot_data.qpos[:] = q
            mujoco.mj_forward(self.robot_model, self.robot_data)

            for body_col, robot_geom_ids in enumerate(geom_ids_by_body):
                min_dist = np.inf
                for robot_geom_id in robot_geom_ids:
                    for object_geom_id in object_geom_ids:
                        fromto[:] = 0.0
                        dist = mujoco.mj_geomDistance(
                            self.robot_model,
                            self.robot_data,
                            robot_geom_id,
                            object_geom_id,
                            distance_cutoff,
                            fromto,
                        )
                        min_dist = min(min_dist, float(dist))
                        if min_dist <= self.contact_threshold:
                            break
                    if min_dist <= self.contact_threshold:
                        break
                distances[frame_idx, body_col] = min_dist

        fallback_cols = [idx for idx, geom_ids in enumerate(geom_ids_by_body) if not geom_ids]
        if fallback_cols:
            fallback_names = [robot_names[idx] for idx in fallback_cols]
            body_positions = np.stack(
                [self._get_robot_contact_positions(q, fallback_names) for q in qpos_sequence],
                axis=0,
            )
            sample_points = self._contact_sample_points(object_points_local)
            distances[:, fallback_cols] = self._distances_to_object_surface(
                body_positions,
                qpos_sequence,
                sample_points,
            )
        return distances

    def _robot_object_geom_contact_targets(
        self,
        qpos_sequence: np.ndarray,
        robot_names: list[str],
        object_points_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        qpos_sequence = np.asarray(qpos_sequence, dtype=np.float64)
        sample_points = self._contact_sample_points(object_points_local)
        topk = self.contact_target_topk

        fallback_positions = np.stack(
            [self._get_robot_contact_positions(q, robot_names) for q in qpos_sequence],
            axis=0,
        )
        fallback_topk_distances, fallback_topk_indices, fallback_target_points = (
            self._object_surface_topk_from_world_points(
                fallback_positions,
                qpos_sequence,
                sample_points,
                topk=topk,
            )
        )

        object_geom_ids = self._contact_object_geom_ids()
        if not object_geom_ids:
            return (
                fallback_topk_distances[..., 0],
                fallback_target_points,
                fallback_topk_distances,
                fallback_topk_indices,
            )

        geom_ids_by_body = self._contact_robot_geom_ids_by_body(robot_names)
        distances = np.full((qpos_sequence.shape[0], len(robot_names)), np.inf, dtype=np.float32)
        query_points_obj = np.empty((qpos_sequence.shape[0], len(robot_names), 3), dtype=np.float64)

        fallback_query_distances = fallback_topk_distances[..., 0]
        for frame_idx, q in enumerate(qpos_sequence):
            if q.shape[0] != self.robot_data.qpos.shape[0]:
                raise ValueError(
                    f"qpos frame has length {q.shape[0]}, expected {self.robot_data.qpos.shape[0]} "
                    "for contact labels."
                )

            self.robot_data.qpos[:] = q
            mujoco.mj_forward(self.robot_model, self.robot_data)

            if self.has_dynamic_object and q.shape[0] >= 7:
                object_pos = q[-7:-4]
                object_quat = q[-4:]
            else:
                object_pos = np.zeros(3)
                object_quat = np.array([1.0, 0.0, 0.0, 0.0])

            fallback_local = transform_points_world_to_local(
                object_quat,
                object_pos,
                fallback_positions[frame_idx],
            )
            query_points_obj[frame_idx] = fallback_local
            distances[frame_idx] = fallback_query_distances[frame_idx]

            distance_cutoff = max(float(self.contact_threshold) + 1e-6, float(self.collision_detection_threshold), 1.0)
            fromto = np.zeros(6, dtype=np.float64)
            for body_col, robot_geom_ids in enumerate(geom_ids_by_body):
                if not robot_geom_ids:
                    continue

                min_dist = np.inf
                best_object_point_w: np.ndarray | None = None
                for robot_geom_id in robot_geom_ids:
                    for object_geom_id in object_geom_ids:
                        fromto[:] = 0.0
                        dist = mujoco.mj_geomDistance(
                            self.robot_model,
                            self.robot_data,
                            robot_geom_id,
                            object_geom_id,
                            distance_cutoff,
                            fromto,
                        )
                        if float(dist) < min_dist:
                            min_dist = float(dist)
                            best_object_point_w = fromto[3:].copy()
                        if min_dist <= self.contact_threshold:
                            break
                    if min_dist <= self.contact_threshold:
                        break

                distances[frame_idx, body_col] = min_dist
                if best_object_point_w is not None and min_dist < distance_cutoff:
                    query_points_obj[frame_idx, body_col] = transform_points_world_to_local(
                        object_quat,
                        object_pos,
                        best_object_point_w[None, :],
                    )[0]

        target_distances, target_indices, target_points = self._object_surface_topk_from_local_queries(
            query_points_obj,
            sample_points,
            topk=topk,
        )
        return distances, target_points, target_distances, target_indices

    def _compute_contact_labels(
        self,
        qpos_sequence: np.ndarray,
        human_joint_motions: np.ndarray,
        object_points_local: np.ndarray,
    ) -> dict[str, np.ndarray]:
        human_names, human_indices, robot_names, pairs = self._selected_contact_targets()
        empty = {
            "contact_object_label": np.zeros((qpos_sequence.shape[0], 0), dtype=bool),
            "contact_object_distance": np.zeros((qpos_sequence.shape[0], 0), dtype=np.float32),
            "contact_object_names": np.asarray([], dtype=str),
            "contact_object_indices": np.asarray([], dtype=np.int32),
            "contact_object_parent_names": np.asarray([], dtype=str),
            "contact_object_parent_indices": np.asarray([], dtype=np.int32),
            "contact_object_local_offsets": np.zeros((0, 3), dtype=np.float32),
            "contact_object_target_points_obj": np.zeros(
                (qpos_sequence.shape[0], 0, self.contact_target_topk, 3),
                dtype=np.float32,
            ),
            "contact_object_target_distances": np.zeros(
                (qpos_sequence.shape[0], 0, self.contact_target_topk),
                dtype=np.float32,
            ),
            "contact_object_target_indices": np.zeros(
                (qpos_sequence.shape[0], 0, self.contact_target_topk),
                dtype=np.int32,
            ),
            "contact_object_target_valid": np.zeros((qpos_sequence.shape[0], 0), dtype=bool),
            "contact_object_target_topk": np.asarray(self.contact_target_topk, dtype=np.int32),
            "contact_object_source": np.asarray(self.contact_source),
            "contact_object_threshold_m": np.asarray(self.contact_threshold, dtype=np.float32),
        }
        if not robot_names:
            return empty

        if self.contact_source == "human":
            sample_points = self._contact_sample_points(object_points_local)
            query_points = np.asarray(human_joint_motions[:, human_indices], dtype=np.float64)
            human_target_distances, human_target_indices, human_target_points = (
                self._object_surface_topk_from_world_points(
                    query_points,
                    qpos_sequence,
                    sample_points,
                    topk=self.contact_target_topk,
                )
            )
            human_distances = human_target_distances[..., 0]
            human_labels = human_distances <= self.contact_threshold
            robot_labels = np.zeros((qpos_sequence.shape[0], len(robot_names)), dtype=bool)
            robot_distances = np.full((qpos_sequence.shape[0], len(robot_names)), np.inf, dtype=np.float32)
            robot_target_distances = np.full(
                (qpos_sequence.shape[0], len(robot_names), self.contact_target_topk),
                np.inf,
                dtype=np.float32,
            )
            robot_target_indices = np.full(
                (qpos_sequence.shape[0], len(robot_names), self.contact_target_topk),
                -1,
                dtype=np.int32,
            )
            robot_target_points = np.zeros(
                (qpos_sequence.shape[0], len(robot_names), self.contact_target_topk, 3),
                dtype=np.float32,
            )
            for human_col, robot_col in pairs:
                replace = human_distances[:, human_col] < robot_distances[:, robot_col]
                robot_labels[:, robot_col] |= human_labels[:, human_col]
                robot_distances[replace, robot_col] = human_distances[replace, human_col]
                robot_target_distances[replace, robot_col] = human_target_distances[replace, human_col]
                robot_target_indices[replace, robot_col] = human_target_indices[replace, human_col]
                robot_target_points[replace, robot_col] = human_target_points[replace, human_col]
            return {
                "contact_object_label": robot_labels,
                "contact_object_distance": robot_distances,
                "contact_object_names": np.asarray(robot_names),
                "contact_object_indices": np.asarray(
                    [mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in robot_names],
                    dtype=np.int32,
                ),
                "contact_object_parent_names": np.asarray(
                    [self._contact_body_parent_name(name) for name in robot_names],
                    dtype=str,
                ),
                "contact_object_parent_indices": np.asarray(
                    [self._contact_body_parent_index(name) for name in robot_names],
                    dtype=np.int32,
                ),
                "contact_object_local_offsets": self._contact_body_local_offsets(robot_names),
                "contact_object_target_points_obj": robot_target_points,
                "contact_object_target_distances": robot_target_distances,
                "contact_object_target_indices": robot_target_indices,
                "contact_object_target_valid": robot_labels,
                "contact_object_target_topk": np.asarray(self.contact_target_topk, dtype=np.int32),
                "contact_object_source": np.asarray("human_smplh"),
                "contact_object_threshold_m": np.asarray(self.contact_threshold, dtype=np.float32),
                "contact_human_joint_names": np.asarray(human_names),
                "contact_human_joint_indices": np.asarray(human_indices, dtype=np.int32),
                "contact_human_joint_regex": np.asarray(self.contact_human_joint_regex),
            }

        robot_distances, robot_target_points, robot_target_distances, robot_target_indices = (
            self._robot_object_geom_contact_targets(qpos_sequence, robot_names, object_points_local)
        )
        robot_labels = robot_distances <= self.contact_threshold
        return {
            "contact_object_label": robot_labels,
            "contact_object_distance": robot_distances,
            "contact_object_names": np.asarray(robot_names),
            "contact_object_indices": np.asarray(
                [mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in robot_names],
                dtype=np.int32,
            ),
            "contact_object_parent_names": np.asarray(
                [self._contact_body_parent_name(name) for name in robot_names],
                dtype=str,
            ),
            "contact_object_parent_indices": np.asarray(
                [self._contact_body_parent_index(name) for name in robot_names],
                dtype=np.int32,
            ),
            "contact_object_local_offsets": self._contact_body_local_offsets(robot_names),
            "contact_object_target_points_obj": robot_target_points,
            "contact_object_target_distances": robot_target_distances,
            "contact_object_target_indices": robot_target_indices,
            "contact_object_target_valid": robot_labels,
            "contact_object_target_topk": np.asarray(self.contact_target_topk, dtype=np.int32),
            "contact_object_source": np.asarray("retarget_robot_geom"),
            "contact_object_threshold_m": np.asarray(self.contact_threshold, dtype=np.float32),
            "contact_human_joint_names": np.asarray(human_names),
            "contact_human_joint_indices": np.asarray(human_indices, dtype=np.int32),
            "contact_human_joint_regex": np.asarray(self.contact_human_joint_regex),
        }

    def retarget_motion(
        self,
        human_joint_motions,
        object_poses,
        object_poses_augmented,
        object_points_local_demo,
        object_points_local,
        foot_sticking_sequences,
        q_a_init=None,
        q_nominal_list=None,
        original=True,
        dest_res_path=None,
    ):
        """
        The main function to retarget an entire motion sequence frame by frame.

        Args:
            human_joint_motions (np.ndarray): (num_frames, num_joints, 3) array.
            object_poses (np.ndarray): (num_frames, 7) array of demo object poses (quat, trans).
            object_poses_augmented (np.ndarray): (num_frames, 7) array of augmented object poses (quat, trans).
            object_points_local_demo (np.ndarray): Demo object points in local frame (rest pose).
            object_points_local (np.ndarray): Current object points in local frame (rest pose).
            foot_sticking_sequences (list): List of foot sticking sequences for each frame.
            q_a_init (np.ndarray, optional): Initial robot configuration.
            q_a_nominal (np.ndarray, optional): Nominal robot configuration.

        Returns:
            tuple: (retargeted_motions, obj_pts_demo_list, obj_pts_list, tetrahedra)
        """
        num_frames = human_joint_motions.shape[0]
        if q_nominal_list is not None:
            q_locked_list = q_nominal_list
        else:
            q_locked_list = np.zeros((num_frames, self.nq))
            q_locked_list[0, self.q_a_indices] = q_a_init

        q_locked_list[:, -7:] = object_poses_augmented
        q = np.copy(q_locked_list[0])
        retargeted_motions = [q]

        tetrahedra = []
        obj_pts_demo_list = []  # scaled object pts
        obj_pts_list = []  # original size object pts

        print(f"\nStarting motion retargeting for {num_frames} frames...")

        with tqdm(range(num_frames)) as pbar:
            for i in pbar:
                # Get object poses and transform points
                object_quat_demo = object_poses[i, 3:]
                object_trans_demo = object_poses[i, :3]

                # Get human joint positions and create interaction mesh in object frame
                human_mapped_joints = human_joint_motions[i, self.smplh_mapped_joint_indices]

                if self.object_name == "ground":
                    human_mapped_joints_in_object = human_mapped_joints
                else:
                    human_mapped_joints_in_object = transform_points_world_to_local(
                        object_quat_demo, object_trans_demo, human_mapped_joints
                    )

                source_vertices, source_tetrahedra = create_interaction_mesh(
                    np.vstack([human_mapped_joints_in_object, object_points_local_demo])
                )
                tetrahedra.append(source_tetrahedra)

                if self.debug:
                    # Only for visualization
                    object_quat = object_poses_augmented[i, 3:]
                    object_trans = object_poses_augmented[i, :3]
                    obj_pts_demo = transform_points_local_to_world(
                        object_quat_demo, object_trans_demo, object_points_local_demo
                    )
                    obj_pts = transform_points_local_to_world(object_quat, object_trans, object_points_local)

                    obj_pts_demo_list.append(obj_pts_demo)
                    obj_pts_list.append(obj_pts)
                    human_kpts_handle_list = self.draw_keypoints(human_mapped_joints, name="human_kpts")  # 15 X 3
                    obj_kpts_demo_handle_list = self.draw_keypoints(
                        obj_pts_demo, name="object_demo_kpts", rgba=(1, 0, 0, 1)
                    )  # 100 X 3
                    obj_kpts_handle_list = self.draw_keypoints(
                        obj_pts, name="object_kpts", rgba=(0, 1, 1, 1)
                    )  # 100 X 3

                # Create adjacency list and calculate target Laplacian coordinates
                adj_list = get_adjacency_list(source_tetrahedra, len(source_vertices))
                target_laplacian = calculate_laplacian_coordinates(source_vertices, adj_list)

                # Run optimization
                if original:
                    w_nominal_tracking = self.w_nominal_tracking_init
                else:
                    w_nominal_tracking = self.w_nominal_tracking_init * np.exp(-i / self.nominal_tracking_tau)

                q, cost = self.iterate(
                    q_locked=q_locked_list[i],
                    q_n=q,
                    q_t_last=retargeted_motions[-1],
                    target_laplacian=target_laplacian,
                    adj_list=adj_list,
                    obj_pts_local=object_points_local,
                    foot_sticking=foot_sticking_sequences[i],
                    w_nominal_tracking=w_nominal_tracking,
                    q_a_nominal=(q_nominal_list[i, self.q_a_indices] if q_nominal_list is not None else None),
                    init_t=i == 0,
                    n_iter=50 if i == 0 else 10,
                    frame_idx=i,
                )
                if self.debug:
                    robot_link_positions = self._get_robot_link_positions(
                        q, self.laplacian_match_links.values()
                    )  # 15 X 3
                    robot_kpts_handle_list = self.draw_keypoints(
                        robot_link_positions, name="robot_kpts", rgba=(0, 1, 0, 1)
                    )

                retargeted_motions.append(q)
                if self.visualize and self.debug:
                    self.draw_q(q)

                pbar.set_postfix(cost=cost)

        # Remove previous debug visualization
        if self.debug:
            for handle in human_kpts_handle_list:
                handle.remove()
            human_kpts_handle_list.clear()

            for handle in obj_kpts_demo_handle_list:
                handle.remove()
            obj_kpts_demo_handle_list.clear()

            for handle in obj_kpts_handle_list:
                handle.remove()
            obj_kpts_handle_list.clear()

            for handle in robot_kpts_handle_list:
                handle.remove()
            robot_kpts_handle_list.clear()

        retargeted_motions_np = np.array(retargeted_motions)[1:]
        contact_arrays: dict[str, np.ndarray] = {}
        if self.contact_visualization:
            contact_arrays = self._compute_contact_labels(
                retargeted_motions_np,
                human_joint_motions,
                object_points_local,
            )

        # Save results
        np.savez(
            dest_res_path,
            qpos=retargeted_motions_np,
            human_joints=human_joint_motions,
            fps=30,
            cost=cost,
            **contact_arrays,
        )
        print("Saving results to path:", dest_res_path)

        if self.visualize:
            robot_dof = len(self.viser_robot.get_actuated_joint_limits())

            create_motion_control_sliders(
                server=self.server,
                viser_robot=self.viser_robot,
                robot_base_frame=self.robot_base,
                motion_sequence=np.asarray(retargeted_motions)[1:],
                robot_dof=robot_dof,
                viser_object=self.viser_object,
                object_base_frame=getattr(self, "object_base", None) if self.viser_object else None,
                contains_object_in_qpos=bool(self.viser_object) and bool(self.has_dynamic_object),
                initial_fps=30,
                initial_interp_mult=2,
                loop=False,
                contact_urdf=getattr(self, "contact_viser_robot", None),
                robot_mesh_handles_by_link=(
                    self.robot_mesh_handles_by_link if self.contact_visualization else None
                ),
                contact_mesh_handles_by_link=(
                    self.contact_mesh_handles_by_link if self.contact_visualization else None
                ),
                contact_labels=contact_arrays.get("contact_object_label") if contact_arrays else None,
                contact_body_names=(
                    [str(name) for name in contact_arrays["contact_object_names"].tolist()]
                    if contact_arrays and "contact_object_names" in contact_arrays
                    else None
                ),
                contact_visual_link_aliases=self.FOOT_CONTACT_VISUAL_ALIASES,
            )

            # 4) optional: visibility toggle
            with self.server.gui.add_folder("Visibility"):
                show_meshes_cb = self.server.gui.add_checkbox("Show meshes", self.viser_robot.show_visual)

                @show_meshes_cb.on_update
                def _(_):
                    self.viser_robot.show_visual = show_meshes_cb.value
                    if getattr(self, "contact_viser_robot", None) is not None:
                        self.contact_viser_robot.show_visual = show_meshes_cb.value
                    if self.viser_object is not None:
                        self.viser_object.show_visual = show_meshes_cb.value

        return (
            retargeted_motions_np,
            obj_pts_demo_list,
            obj_pts_list,
            tetrahedra,
        )

    def solve_single_iteration(
        self,
        q_locked: np.ndarray,
        q_a_n_last: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        verbose=False,
        init_t=False,
        frame_idx: int = 0,
    ):
        """The main function to solve a single iteration of the DiffIK problem.
        Args:
            q_locked: the locked robot and object configuration.
            q_a_n_last: the last optimized robot configuration at current time step.
            q_t_last: the robot and object configuration at the last time step.
            foot_sticking: a sequence of booleans indicating whether the foot [left, right] is sticking to the ground.
            smpl_joints: the (possibly scaled) SMPL joint positions to match for IK.
            q_ref: the reference robot configuration.
            smpl_joints_original: the original SMPL joint positions (used for contact matching).
            obj_original: the original object pose (used for contact matching).
            init_t: the current time step is the first time step.
            frame_idx: frame index used by explicit foot lock window constraints.
        """
        assert len(q_a_n_last) == self.nq_a

        # Lock the object pose and set the current robot slice to last accepted solution
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last

        # Compute Laplacian pieces
        J_OC_dict, p_OC_dict, _ = self._calc_manipulator_jacobians(
            q, links=self.laplacian_match_links, obj_frame=(self.object_name != "ground")
        )
        robot_link_keys = list(self.laplacian_match_links.keys())
        V_r = len(robot_link_keys)
        V_o = len(obj_pts_local)
        V = V_r + V_o

        # Stack Jacobians for robot points
        J_V = np.zeros((3 * V, self.nq_a))
        for i, key in enumerate(robot_link_keys):
            J_V[3 * i : 3 * (i + 1), :] = J_OC_dict[key]

        robot_pts_local = np.array([p_OC_dict[k] for k in robot_link_keys])
        vertices = np.vstack([robot_pts_local, obj_pts_local])  # (V x 3)

        L = calculate_laplacian_matrix(vertices, adj_list)  # (V x V), EXPECT SPARSE OR SMALL
        if not sp.issparse(L):
            L = sp.csr_matrix(L)

        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
        J_L = Kron @ J_V

        lap0 = L @ vertices
        lap0_vec = lap0.reshape(-1)  # (3V,)
        target_lap_vec = target_laplacian.reshape(-1)  # (3V,)

        w_v = (self.laplacian_weights * np.ones(V)).astype(float)  # (V,)
        sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

        # Decision variables
        dqa = cp.Variable(len(self.q_a_indices), name="dqa")
        lap_var = cp.Variable(3 * V, name="laplacian")

        # Constraints list
        constraints = []

        # Linear equality
        constraints += [cp.Constant(J_L[:, self.q_a_indices]) @ dqa - lap_var == -lap0_vec]

        # Foot constraints (sticking + foot lock window Z pinning)
        apply_foot_sticking = (self.q_a_init_idx < 12) and self.activate_foot_sticking
        apply_foot_lock = (self.q_a_init_idx < 12) and self.foot_lock.enable
        if apply_foot_sticking or apply_foot_lock:
            J_WF_dict, p_WF_dict, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)

            # Foot sticking: constrain XY to stay near previous frame position
            if apply_foot_sticking:
                _, p_WF_t_last_dict, _ = self._calc_manipulator_jacobians(
                    q_t_last, links=self.foot_links, obj_frame=False
                )
                left_key = right_key = None
                for key in foot_sticking:
                    if key.lower().startswith("l"):
                        left_key = key
                    elif key.lower().startswith("r"):
                        right_key = key
                if left_key is None or right_key is None:
                    raise ValueError("foot_sticking must include one left* and one right* key")

                for key, J_WF in J_WF_dict.items():
                    apply_left = ("left" in key) and foot_sticking[left_key]
                    apply_right = ("right" in key) and foot_sticking[right_key]
                    if apply_left or apply_right:
                        p_lb = p_WF_t_last_dict[key] - p_WF_dict[key] - self.foot_sticking_tolerance
                        p_ub = p_lb + 2 * self.foot_sticking_tolerance  # symmetric window

                        Jxy = J_WF[:2, self.q_a_indices]  # (2 x nq_act)
                        constraints += [
                            Jxy @ dqa >= p_lb[:2],
                            Jxy @ dqa <= p_ub[:2],
                        ]

            # Foot lock windows: pin Z to floor within configured frame ranges
            if apply_foot_lock:
                for key, J_WF in J_WF_dict.items():
                    if not self._is_foot_locked_in_window(key, frame_idx):
                        continue

                    z_anchor = self.foot_lock.z_floor
                    z_delta = z_anchor - p_WF_dict[key][2]
                    Jz = J_WF[2, self.q_a_indices]
                    constraints += [
                        Jz @ dqa >= z_delta - self.foot_lock.tolerance,
                        Jz @ dqa <= z_delta + self.foot_lock.tolerance,
                    ]

        # Non-penetration constraints
        Js, phis = self._update_jacobians_and_phis_from_q(q)
        for key, phi in phis.items():
            Ja_n_full = Js[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            rhs = -phi - self._penetration_tolerance_for_geom_pair(key)
            constraints += [Ja_n @ dqa >= rhs]

        # Joint limits constraints (actuated)
        if self.activate_joint_limits:
            constraints += [
                dqa >= (self.q_a_lb - q_a_n_last),
                dqa <= (self.q_a_ub - q_a_n_last),
            ]

        # Step size constraints (Lorentz cone)
        constraints += [cp.SOC(self.step_size, dqa)]

        # Objective
        obj_terms = []

        obj_terms.append(cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)))

        # nominal tracking for selected indices
        if (w_nominal_tracking > 0) and (q_a_nominal is not None):
            idx = np.array(self.track_nominal_indices, dtype=int)
            if idx.size > 0:
                z = dqa[idx] - (q_a_nominal[idx] - q_a_n_last[idx])
                obj_terms.append(w_nominal_tracking * cp.sum_squares(z))

        # Q_diag cost
        Qd = np.asarray(self.Q_diag, dtype=float).reshape(-1)
        obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_n_last)))

        # Smoothness cost
        dqa_smooth = q_t_last[self.q_a_indices] - q_a_n_last
        if np.isscalar(self.smooth_weight):
            obj_terms.append(self.smooth_weight * cp.sum_squares(dqa - dqa_smooth))
        else:
            Wsmooth = np.asarray(self.smooth_weight, dtype=float)
            if Wsmooth.ndim == 1:
                obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Wsmooth), dqa - dqa_smooth)))
            else:
                # if a full matrix was supplied, fall back to quad_form
                obj_terms.append(cp.quad_form(dqa - dqa_smooth, Wsmooth))

        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

        # -------- Solve with Clarabel --------
        solver_kwargs = {"verbose": verbose}
        cvxpy_canon_backend = os.environ.get("CVXPY_CANON_BACKEND")
        if cvxpy_canon_backend:
            solver_kwargs["canon_backend"] = cvxpy_canon_backend
        problem.solve(solver=cp.CLARABEL, **solver_kwargs)
        if (problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) and init_t:
            constraints = [c for c in constraints if not isinstance(c, cp.constraints.second_order.SOC)]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
            problem.solve(solver=cp.CLARABEL, **solver_kwargs)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"CVXPY solve failed: {problem.status}")

        dqa_star = dqa.value
        cost = problem.value

        q_star = np.copy(q)
        q_star[self.q_a_indices] = dqa_star + q_a_n_last
        q_star[3:7] /= np.linalg.norm(q_star[3:7]) + 1e-12

        return q_star, cost

    def _is_foot_locked_in_window(self, foot_link_key: str, frame_idx: int) -> bool:
        """Check whether a foot link is locked by configured frame windows."""
        key_lower = foot_link_key.lower()
        side = None
        if "left" in key_lower:
            side = "left"
        elif "right" in key_lower:
            side = "right"
        if side is None:
            return False

        return any(start <= frame_idx <= end for start, end in self._foot_lock_windows.get(side, ()))

    def _penetration_tolerance_for_geom_pair(self, geom_pair: tuple[int, int]) -> float:
        """Select a non-penetration tolerance for a MuJoCo geom pair."""
        g1, g2 = geom_pair
        name1 = self._geom_names[g1].lower()
        name2 = self._geom_names[g2].lower()
        if "ground" in name1 or "ground" in name2:
            return float(self.surface_penetration_tolerance)

        object_name = str(self.object_name or "").lower()
        if object_name and object_name != "ground" and (object_name in name1 or object_name in name2):
            return float(self.object_penetration_tolerance)

        return float(self.penetration_tolerance)

    def iterate(
        self,
        q_locked: np.ndarray,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        init_t: bool = False,
        n_iter: int = 10,
        frame_idx: int = 0,
    ):
        """Iterate the solver for multiple iterations."""
        last_cost = np.inf
        for _ in range(n_iter):
            q_a_n_last = q_n[self.q_a_indices]
            q_n, cost = self.solve_single_iteration(
                q_locked=q_locked,
                q_a_n_last=q_a_n_last,
                q_t_last=q_t_last,
                target_laplacian=target_laplacian,
                adj_list=adj_list,
                obj_pts_local=obj_pts_local,
                foot_sticking=foot_sticking,
                q_a_nominal=q_a_nominal,
                w_nominal_tracking=w_nominal_tracking,
                init_t=init_t,
                frame_idx=frame_idx,
            )
            if np.isclose(cost, last_cost):
                break
            last_cost = cost
        return q_n, cost

    def draw_q(self, q: np.ndarray):
        """Draw a single robot configuration."""
        # Update robot joint configurations
        robot_joint_positions = q[7 : 7 + self.task_constants.ROBOT_DOF]
        self.viser_robot.update_cfg(robot_joint_positions)
        if getattr(self, "contact_viser_robot", None) is not None:
            self.contact_viser_robot.update_cfg(robot_joint_positions)

        # Update robot base pose using set_transform
        robot_quat = q[3:7]  # Base orientation
        robot_pos = q[:3]  # Base position

        # Update robot base frame
        self.robot_base.position = robot_pos
        self.robot_base.wxyz = robot_quat  # Assuming quaternion is in wxyz order

        # Update object pose if it exists
        if hasattr(self, "viser_object") and self.viser_object is not None:
            if self.has_dynamic_object:
                object_quat = q[-4:]
                object_pos = q[-7:-4]
            else:
                object_quat = np.asarray([1, 0, 0, 0])
                object_pos = np.zeros(3)

            # Update object base frame
            self.object_base.position = object_pos
            self.object_base.wxyz = object_quat  # Assuming quaternion is in wxyz order

    def draw_keypoints(self, p, name="keypoint", rgba=(0, 0, 1, 1)):
        """Draw keypoints in visualization."""
        if not hasattr(self, "server"):
            return None

        # Create a sphere mesh using trimesh
        sphere = trimesh.primitives.Sphere(radius=0.02)
        vertices = sphere.vertices
        faces = sphere.faces

        color = tuple(int(c * 255) for c in rgba[:3])
        opacity = float(rgba[3])

        kpts_handle_list = []

        # Draw keypoints
        if len(p.shape) == 1:
            # Single point
            kpts_handle = self.server.scene.add_mesh_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                position=p,
                color=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)
        elif len(p.shape) == 2:
            # Multiple points
            kpts_handle = self.server.scene.add_batched_meshes_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                batched_positions=p,
                batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (p.shape[0], 1)),
                batched_colors=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)

        return kpts_handle_list

    def visualize_motion(
        self,
        human_joint_motions,
        obj_pts_demo,
        obj_pts,
        retargeted_motions,
        tetrahedra,
        dt=1 / 30,
        visualize_tetrahedra=False,
    ):
        for i in range(len(human_joint_motions)):
            object_pts_demo = obj_pts_demo[i]
            object_pts = obj_pts[i]
            self.draw_keypoints(human_joint_motions[i, self.smplh_mapped_joint_indices], name="human")
            self.draw_keypoints(object_pts_demo, name="object_demo", rgba=(1, 0, 0, 1))
            self.draw_keypoints(object_pts, name="object", rgba=(0, 1, 0, 1))
            self.draw_q(retargeted_motions[i])
            robot_link_positions = self._get_robot_link_positions(
                retargeted_motions[i], self.laplacian_match_links.values()
            )
            self.draw_keypoints(robot_link_positions, name="robot", rgba=(0, 1, 0, 1))
            input()
            if visualize_tetrahedra:
                self.visualize_tetrahedra(
                    np.vstack(
                        [
                            human_joint_motions[i, self.smplh_mapped_joint_indices],
                            object_pts_demo,
                        ]
                    ),
                    tetrahedra[i],
                    name="human_tetrahedra",
                )
                self.visualize_tetrahedra(
                    np.vstack([robot_link_positions, object_pts]),
                    tetrahedra[i],
                    name="robot_tetrahedra",
                    rgba=(0, 1, 1, 1),
                )
            else:
                time.sleep(dt)

    def visualize_tetrahedra(self, vertices, tetrahedra, name="tetrahedra", color=(0, 0, 0, 1)):
        # Convert color to 0-255 range
        color_255 = np.array(color[:3]) * 255

        # Prepare points and colors for all edges
        points = []
        colors = []

        for tet in tetrahedra:
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = tet[i], tet[j]
                    points.extend([vertices[u], vertices[v]])
                    colors.extend([color_255, color_255])

        # Convert to numpy arrays
        points = np.array(points)
        colors = np.array(colors)

        # Add line segments for all edges at once
        self.server.scene.add_line_segments(
            f"/{name}",
            points=points,
            colors=colors,
            line_width=0.01,
        )

    def _compute_jacobian_for_contact_relative(self, geom1, geom2, geom1_name, geom2_name, fromto, dist):
        # Get closest points from fromto buffer
        pos1 = fromto[:3]  # closest point on geom1
        pos2 = fromto[3:]  # closest point on geom2

        v = pos1 - pos2
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-12:
            nhat_BA_W = np.sign(dist) * (v / norm_v)
        # Degenerate: points coincide. Heuristics fallback.
        # If one side is a plane/ground, use its known normal.
        elif "ground" in geom2_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, 1.0]) * (1.0 if dist >= 0 else -1.0)
        elif "ground" in geom1_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, -1.0]) * (1.0 if dist >= 0 else -1.0)
        else:
            nhat_BA_W = np.array([0.0, 0.0, 0.0])

        J_bodyA = self._calc_contact_jacobian_from_point(geom1.bodyid, pos1, input_world=True)
        J_bodyB = self._calc_contact_jacobian_from_point(geom2.bodyid, pos2, input_world=True)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        return nhat_BA_W @ Jc

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        m.geom_margin[:] = threshold

        # Run collision. This runs broad→narrow and fills d.contact.
        mujoco.mj_collision(m, d)

        # Collect unique candidate pairs that involve at least one masked geom
        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore margins to keep physics untouched
        m.geom_margin[:] = self._saved_margins

        return candidates

    def _update_jacobians_and_phis_from_q(self, q: np.ndarray):
        self.robot_data.qpos[:] = q

        mujoco.mj_forward(self.robot_model, self.robot_data)  # kinematics & AABBs valid

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # 1) Fast prefilter via mj_collision with temporary margins
        candidates = self._prefilter_pairs_with_mj_collision(threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        # 2) Precise distance only on candidates (early-exit at threshold)
        contype, conaff = m.geom_contype, m.geom_conaffinity

        def masks_ok(g1, g2):
            if contype[g1] == 0 and conaff[g1] == 0:
                return False
            if contype[g2] == 0 and conaff[g2] == 0:
                return False
            if self.object_name in self._geom_names[g1] and "ground" in self._geom_names[g2]:
                return False
            if "ground" in self._geom_names[g1] and self.object_name in self._geom_names[g2]:
                return False
            return (
                self.object_name in self._geom_names[g1]
                or self.object_name in self._geom_names[g2]
                or "ground" in self._geom_names[g1]
                or "ground" in self._geom_names[g2]
            )

        for g1, g2 in candidates:
            # Optional: keep your own filters here (e.g., skip object-ground, only keep interaction with object/ground)
            if not masks_ok(g1, g2):
                continue

            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, g1, g2, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(g1), m.geom(g2), self._geom_names[g1], self._geom_names[g2], fromto, dist
                )
                Js[(g1, g2)] = J_rel
                phis[(g1, g2)] = float(dist)

                # For debug
                # self.draw_mesh_pair_with_contact(self.robot_model, self.robot_data, g1, g2,   \
                #     self._geom_names[g1], self._geom_names[g2], fromto=fromto)

        return Js, phis

    def _world_to_body_frame(self, p_w: np.ndarray, body_idx: int) -> np.ndarray:
        """Transform point from world frame to body frame."""
        p_w = np.asarray(p_w).reshape(3)
        body_pos = self.robot_data.xpos[body_idx].reshape(3)
        body_mat = self.robot_data.xmat[body_idx].reshape(3, 3)
        return body_mat.T @ (p_w - body_pos)

    def _get_geometry_name(self, geom_id: int) -> str:
        """Get geometry name from ID."""
        return mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

    def _build_transform_qdot_to_qvel_fast(self, use_world_omega=True):
        """
        Return T(q) (nv x nq) such that v = T(q) @ qdot.
        - Free root: qpos=[x,y,z, qw,qx,qy,qz], qvel=[vx,vy,vz, ωx,ωy,ωz]
        where ω and v are WORLD-expressed in MuJoCo.
        - 23 hinge joints: v = qdot.

        If use_world_omega=False, uses BODY-omega mapping (for debugging).
        """
        nq, nv = self.robot_model.nq, self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        # ---- root free joint (assumed joint 0) ----
        j0 = 0
        assert self.robot_model.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE
        qadr = self.robot_model.jnt_qposadr[j0]  # 0
        dadr = self.robot_model.jnt_dofadr[j0]  # 0

        # Linear block: v_lin = xyz_dot
        T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)

        # Angular block: ω_* = 2 * E_*(q) * quat_dot
        w, x, y, z = self.robot_data.qpos[qadr + 3 : qadr + 7]

        def get_e_world(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, qz, -qy],
                    [-qy, -qz, qw, qx],
                    [-qz, qy, -qx, qw],
                ]
            )

        def get_e_body(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, -qz, qy],
                    [-qy, qz, qw, -qx],
                    [-qz, -qy, qx, qw],
                ]
            )

        E_fn = get_e_world if use_world_omega else get_e_body

        # ---- FREE joint #1 (human/root): use model addresses, but this should be the first joint ----
        j_free1 = 0
        assert self.robot_model.jnt_type[j_free1] == mujoco.mjtJoint.mjJNT_FREE
        qadr1 = int(self.robot_model.jnt_qposadr[j_free1])  # expect 0
        dadr1 = int(self.robot_model.jnt_dofadr[j_free1])  # start of its 6 qvel dofs

        qw, qx, qy, qz = self.robot_data.qpos[qadr1 + 3 : qadr1 + 7]
        E1 = 2.0 * E_fn(qw, qx, qy, qz)
        # linear-first: v_W = rdot, ω_W = 2E(q) * quat_dot
        T[dadr1 + 0 : dadr1 + 3, qadr1 + 0 : qadr1 + 3] = np.eye(3)  # v block
        T[dadr1 + 3 : dadr1 + 6, qadr1 + 3 : qadr1 + 7] = E1  # ω block

        if self.has_dynamic_object:
            # ---- FREE joint #2 (object): assume it's the last FREE joint; fill its 6x7 block ----
            # Find it by type (safer than hardcoding tail indices)
            free_joints = [
                j for j in range(self.robot_model.njnt) if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            ]
            assert len(free_joints) >= 2, "Expected two FREE joints (human + object)."
            j_free2 = free_joints[1]  # second FREE joint
            qadr2 = int(self.robot_model.jnt_qposadr[j_free2])  # expect nq-7
            dadr2 = int(self.robot_model.jnt_dofadr[j_free2])  # its 6 qvel dofs (often at nv-6)

            qw, qx, qy, qz = self.robot_data.qpos[qadr2 + 3 : qadr2 + 7]
            E2 = 2.0 * E_fn(qw, qx, qy, qz)
            T[dadr2 + 0 : dadr2 + 3, qadr2 + 0 : qadr2 + 3] = np.eye(3)  # v block
            T[dadr2 + 3 : dadr2 + 6, qadr2 + 3 : qadr2 + 7] = E2  # ω block

        # ---- remaining hinge/slide joints: v = qdot ----
        for j in range(1, self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint block not implemented.")

        return T

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray, input_world=False):
        """
        Translational Jacobian J(q) (3 x nq) such that
        v_point_world = J(q) @ qdot.

        Fast analytic version: J_qdot = J_v @ T(q)
        """

        p_body = np.asarray(p_body, dtype=float).reshape(3)

        # 1) Make sure kinematics are current once
        mujoco.mj_forward(self.robot_model, self.robot_data)

        # 2) World point (3,1) for mj_jac
        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]

        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)

        # 3) J_v: translational Jacobian wrt generalized velocities (3 x nv)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))  # Jp = J_v

        T = self._build_transform_qdot_to_qvel_fast()

        return Jp @ T

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str],
        obj_frame: bool = False,
        point_offsets: np.ndarray | None = None,
    ):
        """Compute position-based Jacobians using MuJoCo."""
        J_XC_dict = {}
        p_XC_dict = {}

        if obj_frame:
            if self.has_dynamic_object:
                obj_quat = q[-4:]
                obj_pos = q[-7:-4]
                obj_rot = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()
                obj_rot_inv = obj_rot.T
            else:
                obj_rot = Rotation.from_quat([0, 0, 0, 1]).as_matrix()
                obj_rot_inv = obj_rot.T
                obj_pos = np.zeros(3)

        q_mujoco = q.copy()
        self.robot_data.qpos[:] = q_mujoco

        mujoco.mj_forward(self.robot_model, self.robot_data)

        for name, link_name in links.items():
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

            if point_offsets is not None:
                pC_B = point_offsets
            else:
                pC_B = np.zeros(3)

            J = self._calc_contact_jacobian_from_point(body_id, pC_B)
            pos_world = self.robot_data.xpos[body_id]

            if obj_frame:
                p_XC = obj_rot_inv @ (pos_world - obj_pos)
                J_XC = obj_rot_inv @ J
            else:
                p_XC = pos_world
                J_XC = J

            # Store reduced Jacobian and position with hard copies to avoid aliasing
            J_XC_dict[name] = np.array(J_XC[:, self.q_a_indices], dtype=float, copy=True)  # FIX (copy)
            p_XC_dict[name] = np.array(p_XC, dtype=float, copy=True)

        P_WO = {"position": obj_pos, "rotation": obj_rot} if obj_frame else None

        return J_XC_dict, p_XC_dict, P_WO

    def _get_robot_link_positions(self, q, link_names):
        """Get robot link positions for given configuration using Mujoco."""
        mujoco_q = q.copy()

        # Set the configuration
        if mujoco_q.shape != self.robot_data.qpos.shape:
            self.robot_data.qpos = mujoco_q[:-7]  # Exclude object information from q
        else:
            self.robot_data.qpos = mujoco_q
        # Forward kinematics to update all positions
        mujoco.mj_forward(self.robot_model, self.robot_data)

        robot_link_positions = []

        for link_name in link_names:
            # Get body ID from name
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id == -1:
                raise ValueError(f"Body {link_name} not found in Mujoco model")

            # Get position in world frame
            # xpos gives us the position of the body's center of mass in world coordinates
            pos = self.robot_data.xpos[body_id].copy()
            robot_link_positions.append(pos)

        return np.array(robot_link_positions)
