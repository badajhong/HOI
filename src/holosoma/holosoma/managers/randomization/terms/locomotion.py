"""Randomization terms for locomotion environments."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.simulator import MujocoBackend
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
from holosoma.managers.randomization.base import RandomizationTermBase
from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError
from holosoma.simulator import mujoco_required_field
from holosoma.simulator.shared.field_decorators import MUJOCO_FIELD_ATTR
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.torch_utils import torch_rand_float

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg

    from holosoma.simulator.isaacsim.isaacsim import IsaacSim


def _ensure_env_ids_tensor(env: Any, env_ids: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
    """Convert environment indices to a tensor on the correct device."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)


def _get_joint_action_term(env: Any) -> JointPositionActionTerm | None:
    """Return the joint-position action term registered with the action manager."""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None

    get_term = getattr(action_manager, "get_term", None)
    if callable(get_term):
        term = get_term("joint_control")
        if isinstance(term, JointPositionActionTerm):
            return term

    iter_terms = getattr(action_manager, "iter_terms", None)
    if callable(iter_terms):
        for _, term in iter_terms():
            if isinstance(term, JointPositionActionTerm):
                return term

    return None


def _get_object_scene_entity_names(simulator: Any) -> list[str]:
    """Resolve object entity names for both single-object and multi-object scenes."""
    scene_keys = list(simulator.scene.keys())
    if "object" in scene_keys:
        return ["object"]

    object_names = sorted(name for name in scene_keys if name.startswith("object_"))
    if object_names:
        return object_names

    return []


def _get_scene_entity_prim_paths(simulator: Any, entity_name: str) -> list[str]:
    """Resolve concrete prim paths for a scene entity across all environments."""
    entity = simulator.scene[entity_name]

    entity_cfg = getattr(entity, "cfg", None)
    prim_path_expr = getattr(entity_cfg, "prim_path", None)
    if isinstance(prim_path_expr, str) and prim_path_expr:
        try:
            import isaaclab.sim as sim_utils
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("IsaacSim prim-path resolution requires isaaclab.") from exc

        prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
        if prim_paths:
            return prim_paths

    root_physx_view = getattr(entity, "root_physx_view", None)
    prim_paths = getattr(root_physx_view, "prim_paths", None)
    if prim_paths is not None:
        return list(prim_paths)

    return []


def _parse_xyz_attr(value: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Parse a URDF xyz/rpy-style attribute into a float vector."""
    if not value:
        return np.asarray(default, dtype=np.float32)
    return np.asarray([float(v) for v in value.split()], dtype=np.float32)


def _rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw angles to a 3x3 rotation matrix."""
    roll, pitch, yaw = rpy.tolist()
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _aabb_corners(bounds: np.ndarray) -> np.ndarray:
    """Return the 8 corners for an axis-aligned bounding box."""
    mins, maxs = bounds
    return np.asarray(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )


def _transform_bounds(bounds: np.ndarray, translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Transform an axis-aligned bounding box and return its new bounds."""
    corners = _aabb_corners(bounds)
    transformed = (rotation @ corners.T).T + translation
    return np.stack([transformed.min(axis=0), transformed.max(axis=0)], axis=0).astype(np.float32)


def _resolve_urdf_referenced_file(urdf_dir: Path, referenced_path: str) -> Path:
    """Resolve a URDF-referenced mesh path against common project layouts."""
    path = Path(referenced_path)
    if path.is_absolute():
        return path.resolve()

    candidates = [(urdf_dir / path).resolve()]
    parents = list(urdf_dir.parents)
    if len(parents) >= 2:
        candidates.append((parents[1] / path).resolve())
    candidates.append(Path(resolve_data_file_path(referenced_path)).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _load_urdf_geometry_bounds(geometry_elem: ET.Element, urdf_dir: Path) -> np.ndarray | None:
    """Load local bounds for a URDF geometry element."""
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("URDF mesh bound extraction requires trimesh.") from exc

    mesh_elem = geometry_elem.find("mesh")
    if mesh_elem is not None:
        filename = mesh_elem.get("filename")
        if not filename:
            return None
        mesh_path = _resolve_urdf_referenced_file(urdf_dir, filename)
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
        scale = _parse_xyz_attr(mesh_elem.get("scale"), default=(1.0, 1.0, 1.0))
        scaled_corners = _aabb_corners(bounds) * scale[None, :]
        return np.stack([scaled_corners.min(axis=0), scaled_corners.max(axis=0)], axis=0).astype(np.float32)

    box_elem = geometry_elem.find("box")
    if box_elem is not None:
        size = _parse_xyz_attr(box_elem.get("size"))
        half = 0.5 * size
        return np.stack([-half, half], axis=0).astype(np.float32)

    sphere_elem = geometry_elem.find("sphere")
    if sphere_elem is not None:
        radius = float(sphere_elem.get("radius", "0.0"))
        half = np.asarray([radius, radius, radius], dtype=np.float32)
        return np.stack([-half, half], axis=0).astype(np.float32)

    cylinder_elem = geometry_elem.find("cylinder")
    if cylinder_elem is not None:
        radius = float(cylinder_elem.get("radius", "0.0"))
        length = float(cylinder_elem.get("length", "0.0"))
        half = np.asarray([radius, radius, 0.5 * length], dtype=np.float32)
        return np.stack([-half, half], axis=0).astype(np.float32)

    return None


def _load_urdf_geometry_support_points(geometry_elem: ET.Element, urdf_dir: Path) -> np.ndarray | None:
    """Load representative support points for a URDF geometry element."""
    mesh_elem = geometry_elem.find("mesh")
    if mesh_elem is not None:
        try:
            import trimesh
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("URDF mesh support extraction requires trimesh.") from exc

        filename = mesh_elem.get("filename")
        if not filename:
            return None
        mesh_path = _resolve_urdf_referenced_file(urdf_dir, filename)
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        try:
            support_mesh = mesh.convex_hull
        except Exception:  # pragma: no cover - robust fallback
            support_mesh = mesh

        points = np.asarray(support_mesh.vertices, dtype=np.float32)
        scale = _parse_xyz_attr(mesh_elem.get("scale"), default=(1.0, 1.0, 1.0))
        return points * scale[None, :]

    box_elem = geometry_elem.find("box")
    if box_elem is not None:
        size = _parse_xyz_attr(box_elem.get("size"))
        half = 0.5 * size
        return _aabb_corners(np.stack([-half, half], axis=0).astype(np.float32))

    sphere_elem = geometry_elem.find("sphere")
    if sphere_elem is not None:
        radius = float(sphere_elem.get("radius", "0.0"))
        return np.asarray(
            [
                [radius, 0.0, 0.0],
                [-radius, 0.0, 0.0],
                [0.0, radius, 0.0],
                [0.0, -radius, 0.0],
                [0.0, 0.0, radius],
                [0.0, 0.0, -radius],
            ],
            dtype=np.float32,
        )

    cylinder_elem = geometry_elem.find("cylinder")
    if cylinder_elem is not None:
        radius = float(cylinder_elem.get("radius", "0.0"))
        length = float(cylinder_elem.get("length", "0.0"))
        angles = np.linspace(0.0, 2.0 * np.pi, num=16, endpoint=False, dtype=np.float32)
        circle = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        top = np.concatenate([circle, np.full((circle.shape[0], 1), 0.5 * length, dtype=np.float32)], axis=1)
        bottom = np.concatenate([circle, np.full((circle.shape[0], 1), -0.5 * length, dtype=np.float32)], axis=1)
        return np.concatenate([top, bottom], axis=0).astype(np.float32)

    return None


def _extract_urdf_collision_support_points(urdf_path: str) -> np.ndarray | None:
    """Collect representative collision/support points in the URDF root frame."""
    urdf_path = resolve_data_file_path(urdf_path)
    urdf_file = Path(urdf_path).resolve()
    root = ET.parse(urdf_file).getroot()

    support_points: list[np.ndarray] = []
    urdf_dir = urdf_file.parent
    for link_elem in root.findall("link"):
        collision_elems = list(link_elem.findall("collision"))
        if not collision_elems:
            collision_elems = list(link_elem.findall("visual"))

        for collision_elem in collision_elems:
            geometry_elem = collision_elem.find("geometry")
            if geometry_elem is None:
                continue

            local_points = _load_urdf_geometry_support_points(geometry_elem, urdf_dir)
            if local_points is None or local_points.size == 0:
                continue

            origin_elem = collision_elem.find("origin")
            translation = _parse_xyz_attr(None if origin_elem is None else origin_elem.get("xyz"))
            rotation = _rpy_to_rotation_matrix(
                _parse_xyz_attr(None if origin_elem is None else origin_elem.get("rpy"))
            )
            transformed = (rotation @ local_points.T).T + translation[None, :]
            support_points.append(transformed.astype(np.float32))

    if not support_points:
        return None

    return np.concatenate(support_points, axis=0).astype(np.float32)


def _extract_urdf_collision_bounds(urdf_path: str) -> np.ndarray | None:
    """Compute union bounds of collision geometries in the URDF root frame."""
    urdf_path = resolve_data_file_path(urdf_path)
    urdf_file = Path(urdf_path).resolve()
    root = ET.parse(urdf_file).getroot()

    bounds_list: list[np.ndarray] = []
    urdf_dir = urdf_file.parent
    for link_elem in root.findall("link"):
        collision_elems = list(link_elem.findall("collision"))
        if not collision_elems:
            collision_elems = list(link_elem.findall("visual"))

        for collision_elem in collision_elems:
            geometry_elem = collision_elem.find("geometry")
            if geometry_elem is None:
                continue

            local_bounds = _load_urdf_geometry_bounds(geometry_elem, urdf_dir)
            if local_bounds is None:
                continue

            origin_elem = collision_elem.find("origin")
            translation = _parse_xyz_attr(None if origin_elem is None else origin_elem.get("xyz"))
            rotation = _rpy_to_rotation_matrix(
                _parse_xyz_attr(None if origin_elem is None else origin_elem.get("rpy"))
            )
            bounds_list.append(_transform_bounds(local_bounds, translation, rotation))

    if not bounds_list:
        return None

    all_mins = np.stack([bounds[0] for bounds in bounds_list], axis=0)
    all_maxs = np.stack([bounds[1] for bounds in bounds_list], axis=0)
    return np.stack([all_mins.min(axis=0), all_maxs.max(axis=0)], axis=0).astype(np.float32)


def _resolve_object_actor_urdf_paths(env: Any) -> dict[str, str]:
    """Map simulator object actor names to their URDF file paths."""
    object_cfg = env.robot_config.object
    object_name_to_path = getattr(object_cfg, "object_urdf_name_to_path", {}) or {}
    if object_name_to_path:
        return {
            f"object_{object_key}": resolve_data_file_path(urdf_path)
            for object_key, urdf_path in object_name_to_path.items()
        }

    object_urdf_path = getattr(object_cfg, "object_urdf_path", None)
    if object_urdf_path:
        return {"object": resolve_data_file_path(object_urdf_path)}

    return {}


def _setup_object_scale_reference_bounds(env: Any, object_names: Sequence[str], object_height: float | None) -> None:
    """Store per-object local bounds used to keep scaled objects grounded on reset."""
    object_actor_paths = _resolve_object_actor_urdf_paths(env)
    if not object_actor_paths:
        return

    env.object_local_bbox_center_by_actor = {}
    env.object_local_bbox_half_extent_by_actor = {}
    env.object_local_support_points_by_actor = {}

    for actor_name in object_names:
        urdf_path = object_actor_paths.get(actor_name)
        if urdf_path is None:
            logger.warning(f"No URDF path found for actor '{actor_name}' while setting up object scale bounds.")
            continue

        bounds = _extract_urdf_collision_bounds(urdf_path)
        if bounds is None:
            logger.warning(f"Could not extract collision bounds from object URDF '{urdf_path}'.")
            continue

        mins, maxs = bounds
        center = 0.5 * (mins + maxs)
        half_extent = 0.5 * (maxs - mins)
        env.object_local_bbox_center_by_actor[actor_name] = torch.tensor(
            center, device=env.device, dtype=torch.float32
        )
        env.object_local_bbox_half_extent_by_actor[actor_name] = torch.tensor(
            half_extent, device=env.device, dtype=torch.float32
        )
        support_points = _extract_urdf_collision_support_points(urdf_path)
        if support_points is not None and support_points.size > 0:
            env.object_local_support_points_by_actor[actor_name] = torch.tensor(
                support_points, device=env.device, dtype=torch.float32
            )
        logger.info(
            f"[Randomization] Object bounds for '{actor_name}': "
            f"min={mins.tolist()}, max={maxs.tolist()}, height={float(maxs[2] - mins[2]):.4f}"
        )

    if object_height is not None and object_height > 0.0:
        env.object_scale_height = float(object_height)
    elif object_names:
        first_center = env.object_local_bbox_center_by_actor.get(object_names[0])
        first_half = env.object_local_bbox_half_extent_by_actor.get(object_names[0])
        if first_center is not None and first_half is not None:
            env.object_scale_height = float((2.0 * first_half[2]).item())


def _isaacsim_randomize_rigid_body_mass(
    simulator: IsaacSim,
    env_ids_cpu: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: str,
):
    try:
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim mass randomization requires isaaclab.") from exc
    func = mdp.randomize_rigid_body_mass(
        EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "env_ids": env_ids_cpu,
                "asset_cfg": asset_cfg,
                "mass_distribution_params": mass_distribution_params,
                "operation": operation,
            },
        ),
        env=simulator,
    )
    func(
        simulator,
        env_ids_cpu,
        asset_cfg=asset_cfg,
        mass_distribution_params=mass_distribution_params,
        operation=operation,
    )


def _isaacsim_randomize_rigid_body_material(
    simulator: IsaacSim,
    env_ids_cpu: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
):
    try:
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim material randomization requires isaaclab.") from exc
    func = mdp.randomize_rigid_body_material(
        EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "env_ids": env_ids_cpu,
                "asset_cfg": asset_cfg,
                "static_friction_range": static_friction_range,
                "dynamic_friction_range": dynamic_friction_range,
                "restitution_range": restitution_range,
                "num_buckets": num_buckets,
            },
        ),
        simulator,
    )
    func(
        simulator,
        env_ids_cpu,
        asset_cfg=asset_cfg,
        static_friction_range=static_friction_range,
        dynamic_friction_range=dynamic_friction_range,
        restitution_range=restitution_range,
        num_buckets=num_buckets,
    )


class PushRandomizerState(RandomizationTermBase):
    """Stateful randomizer that owns push scheduling buffers and counters."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        interval = params.get("push_interval_s", [5, 16])
        self.push_interval_range: Sequence[float] = [float(interval[0]), float(interval[1])]
        vector_max = params.get("max_push_vel")
        if vector_max is None:
            raise ValueError("PushRandomizerState requires `max_push_vel` to be specified.")
        self._max_push_vel_tensor = torch.empty(0, dtype=torch.float32, device=env.device)
        self._set_max_push_tensor(vector_max)
        self.enabled: bool = bool(params.get("enabled", True))
        logger.info(
            f"[Randomization] PushRandomizerState initialized (enabled={self.enabled}, \
                max_push_vel={self._max_push_vel_tensor.tolist()}, \
                interval_s={self.push_interval_range})",
        )

        self.push_interval_s: torch.Tensor | None = None
        self.push_robot_counter: torch.Tensor | None = None
        self.push_robot_plot_counter: torch.Tensor | None = None

    def setup(self) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs

        self.push_interval_s = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.push_robot_counter = torch.zeros(num_envs, dtype=torch.int, device=device)
        self.push_robot_plot_counter = torch.zeros(num_envs, dtype=torch.int, device=device)

        all_ids = torch.arange(num_envs, device=device, dtype=torch.long)
        self._resample_intervals(all_ids)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        idx = self._ensure_indices(env_ids)
        if idx.numel() == 0:
            return
        self.push_robot_counter[idx] = 0
        self.push_robot_plot_counter[idx] = 0

    def step(self) -> None:
        if not self.enabled:
            return
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        self.push_robot_counter += 1
        self.push_robot_plot_counter += 1

    # ------------------------------------------------------------------ #
    # Public helpers for other randomization hooks
    # ------------------------------------------------------------------ #

    def configure(
        self,
        *,
        enabled: bool | None = None,
        push_interval_s: Sequence[float] | None = None,
        max_push_vel: Sequence[float] | None = None,
    ) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if push_interval_s is not None:
            self.push_interval_range = [float(push_interval_s[0]), float(push_interval_s[1])]
        if max_push_vel is not None:
            self._set_max_push_tensor(max_push_vel)

    def resample(self, env_ids: torch.Tensor | None = None) -> None:
        idx = self._ensure_indices(env_ids)
        if idx.numel() == 0:
            return
        self._resample_intervals(idx)

    def due_envs(self, dt: float) -> torch.Tensor:
        if not self.enabled:
            return torch.empty(0, device=self.env.device, dtype=torch.long)
        if self.push_interval_s is None or self.push_robot_counter is None:
            return torch.empty(0, device=self.env.device, dtype=torch.long)
        interval_steps = (self.push_interval_s / dt).to(torch.int)
        return (self.push_robot_counter == interval_steps).nonzero(as_tuple=False).flatten()

    def zero_counters(self, env_ids: torch.Tensor) -> None:
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        self.push_robot_counter[env_ids] = 0
        self.push_robot_plot_counter[env_ids] = 0

    @property
    def max_push_vel(self) -> torch.Tensor:
        return self._max_push_vel_tensor

    def _ensure_indices(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)

    def _resample_intervals(self, env_ids: torch.Tensor) -> None:
        if self.push_interval_s is None:
            return
        low, high = self.push_interval_range
        low_i = max(1, int(low))
        high_i = max(low_i + 1, int(high))
        samples = torch_rand_float(low_i, high_i, (env_ids.shape[0], 1), device=self.env.device).squeeze(1)
        self.push_interval_s[env_ids] = samples

    def _set_max_push_tensor(self, values: Sequence[float]) -> None:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.env.device).flatten()
        if tensor.numel() == 0:
            raise ValueError("max_push_vel must contain at least one value.")
        self._max_push_vel_tensor = tensor.clone()


class ActuatorRandomizerState(RandomizationTermBase):
    """Stateful actuator randomizer managing PD gain and RFI scales."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}

        kp_range = params.get("kp_range", [1.0, 1.0])
        kd_range = params.get("kd_range", [1.0, 1.0])
        rfi_lim_range = params.get("rfi_lim_range", [1.0, 1.0])

        self.enable_pd_gain = bool(params.get("enable_pd_gain", True))
        self.enable_rfi_lim = bool(params.get("enable_rfi_lim", False))

        self.kp_range: Sequence[float] = [float(kp_range[0]), float(kp_range[1])]
        self.kd_range: Sequence[float] = [float(kd_range[0]), float(kd_range[1])]
        self.rfi_lim_range: Sequence[float] = [float(rfi_lim_range[0]), float(rfi_lim_range[1])]

        self.rfi_lim = float(params.get("rfi_lim", 0.1))

        self.kp_scale: torch.Tensor | None = None
        self.kd_scale: torch.Tensor | None = None
        self.rfi_lim_scale: torch.Tensor | None = None

    def setup(self) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs
        num_dof = env.num_dof

        self.kp_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)
        self.kd_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)
        self.rfi_lim_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)

        term = _get_joint_action_term(env)
        if term is not None:
            term.attach_actuator_scales(self.kp_scale, self.kd_scale, self.rfi_lim_scale)
        else:
            logger.debug(
                "JointPositionActionTerm not ready during ActuatorRandomizerState.setup(); "
                "the term will attach shared actuator scales once its setup() runs."
            )

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if self.kp_scale is None or self.kd_scale is None or self.rfi_lim_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() must be called before reset().")

        idx = _ensure_env_ids_tensor(self.env, env_ids)
        if idx.numel() == 0:
            return

        device = self.env.device

        if self.enable_pd_gain:
            self.kp_scale[idx] = torch_rand_float(
                self.kp_range[0], self.kp_range[1], (idx.shape[0], self.env.num_dof), device=device
            )
            self.kd_scale[idx] = torch_rand_float(
                self.kd_range[0], self.kd_range[1], (idx.shape[0], self.env.num_dof), device=device
            )
        else:
            self.kp_scale[idx] = 1.0
            self.kd_scale[idx] = 1.0

        if self.enable_rfi_lim:
            self.rfi_lim_scale[idx] = torch_rand_float(
                self.rfi_lim_range[0], self.rfi_lim_range[1], (idx.shape[0], self.env.num_dof), device=device
            )
        else:
            self.rfi_lim_scale[idx] = 1.0

    def step(self) -> None:
        """No per-step behaviour required."""

    @property
    def kp_scale_tensor(self) -> torch.Tensor:
        if self.kp_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.kp_scale

    @property
    def kd_scale_tensor(self) -> torch.Tensor:
        if self.kd_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.kd_scale

    @property
    def rfi_lim_scale_tensor(self) -> torch.Tensor:
        if self.rfi_lim_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.rfi_lim_scale


def setup_action_delay_buffers(env, *, ctrl_delay_step_range: Sequence[int], enabled: bool = True, **_) -> None:
    """Initialize action delay index buffer during setup.

    Note: The action_queue itself is managed by the action manager.
    This only sets up the delay index that determines which queued action to use.
    """
    env._randomize_ctrl_delay = bool(enabled)
    env._ctrl_delay_step_range = list(ctrl_delay_step_range)

    if not enabled:
        return

    # Initialize action delay indices (determines which action from the queue to use)
    env.action_delay_idx = torch.randint(
        ctrl_delay_step_range[0],
        ctrl_delay_step_range[1] + 1,
        (env.num_envs,),
        device=env.device,
        requires_grad=False,
    )


def setup_torque_rfi(env, *, enabled: bool = False, rfi_lim: float = 0.1, **_) -> None:
    """Configure torque RFI at startup."""
    term = _get_joint_action_term(env)
    env._pending_torque_rfi = (bool(enabled), float(rfi_lim))
    if term is None:
        return
    term.configure_torque_rfi(enabled=env._pending_torque_rfi[0], rfi_lim=env._pending_torque_rfi[1])


def setup_dof_pos_bias(env, *, dof_pos_bias_range: Sequence[float], enabled: bool = False, **_) -> None:
    """Apply startup DOF position bias randomization."""
    env._randomize_dof_pos_bias = bool(enabled)
    env._dof_pos_bias_range = list(dof_pos_bias_range)

    if not enabled:
        return

    default_dof_pos_bias = torch_rand_float(
        dof_pos_bias_range[0],
        dof_pos_bias_range[1],
        (env.num_envs, env.num_dof),
        device=env.device,
    )
    env.default_dof_pos = env.default_dof_pos_base + default_dof_pos_bias


def randomize_push_schedule(
    env,
    env_ids,
    *,
    push_interval_s: Sequence[float] | None = None,
    enabled: bool | None = None,
    max_push_vel: Sequence[float] | None = None,
    **_,
) -> None:
    """Resample push intervals for selected environments."""
    state = env.randomization_manager.get_state("push_randomizer_state")
    if state is None:
        raise AttributeError("PushRandomizerState is not registered with the randomization manager.")

    state.configure(enabled=enabled, push_interval_s=push_interval_s, max_push_vel=max_push_vel)
    env._randomize_push_robots = state.enabled
    env._max_push_vel = state.max_push_vel.clone()

    if not state.enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    state.zero_counters(idx)
    state.resample(idx)


def randomize_pd_gains(
    env, env_ids, *, kp_range: Sequence[float], kd_range: Sequence[float], enabled: bool = True, **_
):
    """Randomize proportional and derivative gain scales."""
    state = env.randomization_manager.get_state("actuator_randomizer_state")
    term = _get_joint_action_term(env)
    if state is None:
        if term is None:
            logger.warning("JointPositionActionTerm not found; PD gain randomization skipped.")
            return

        idx = _ensure_env_ids_tensor(env, env_ids)
        if idx.numel() == 0:
            return

        if not enabled:
            kp_scale, kd_scale = term.get_pd_scale_tensors()
            term.update_pd_scales(idx, torch.ones_like(kp_scale[idx]), torch.ones_like(kd_scale[idx]))
            return

        kp_samples = torch_rand_float(kp_range[0], kp_range[1], (idx.shape[0], env.num_dof), device=env.device)
        kd_samples = torch_rand_float(kd_range[0], kd_range[1], (idx.shape[0], env.num_dof), device=env.device)
        term.update_pd_scales(idx, kp_samples, kd_samples)
        return

    state.enable_pd_gain = bool(enabled)
    state.kp_range = [float(kp_range[0]), float(kp_range[1])]
    state.kd_range = [float(kd_range[0]), float(kd_range[1])]
    state.reset(env_ids)


def randomize_rfi_limits(
    env,
    env_ids,
    *,
    rfi_lim_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize residual force injection limits."""
    state = env.randomization_manager.get_state("actuator_randomizer_state")
    term = _get_joint_action_term(env)
    if state is None:
        if term is None:
            logger.warning("JointPositionActionTerm not found; RFI randomization skipped.")
            return

        idx = _ensure_env_ids_tensor(env, env_ids)
        if idx.numel() == 0:
            return

        if not enabled:
            term.update_rfi_scales(idx, torch.ones_like(term.get_rfi_scale_tensor()[idx]))
            return

        rfi_samples = torch_rand_float(
            rfi_lim_range[0], rfi_lim_range[1], (idx.shape[0], env.num_dof), device=env.device
        )
        term.update_rfi_scales(idx, rfi_samples)
        return

    state.enable_rfi_lim = bool(enabled)
    state.rfi_lim_range = [float(rfi_lim_range[0]), float(rfi_lim_range[1])]
    state.reset(env_ids)


def randomize_action_delay(
    env,
    env_ids,
    *,
    ctrl_delay_step_range: Sequence[int] | None = None,
    enabled: bool | None = None,
    **_,
) -> None:
    """Randomize control delay indices.

    If ``ctrl_delay_step_range``/``enabled`` are omitted the values captured during
    ``setup_action_delay_buffers`` are reused.
    """
    if enabled is not None:
        env._randomize_ctrl_delay = bool(enabled)
    elif not hasattr(env, "_randomize_ctrl_delay"):
        raise AttributeError(
            "randomize_action_delay() requires setup_action_delay_buffers to run before it can infer 'enabled'."
        )

    if ctrl_delay_step_range is not None:
        env._ctrl_delay_step_range = list(ctrl_delay_step_range)
    elif not hasattr(env, "_ctrl_delay_step_range"):
        raise AttributeError(
            "randomize_action_delay() requires setup_action_delay_buffers \
                to run before it can infer ctrl_delay_step_range."
        )

    if not env._randomize_ctrl_delay:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    # Reset action queue in the action manager
    if hasattr(env.action_manager, "action_queue"):
        env.action_manager.action_queue[idx] *= 0.0

    delay_low = int(env._ctrl_delay_step_range[0])
    delay_high = int(env._ctrl_delay_step_range[1])
    if delay_high < delay_low:
        raise ValueError("ctrl_delay_step_range upper bound must be >= lower bound.")

    # Randomize delay indices
    env.action_delay_idx[idx] = torch.randint(
        delay_low,
        delay_high + 1,
        (idx.shape[0],),
        device=env.device,
        requires_grad=False,
    )


def randomize_dof_state(
    env,
    env_ids,
    *,
    joint_pos_scale_range: Sequence[float],
    joint_pos_bias_range: Sequence[float],
    joint_vel_range: Sequence[float],
    randomize_dof_pos_bias: bool = False,
    **_,
) -> None:
    """Randomize DOF positions and velocities."""
    env._joint_pos_scale_range = list(joint_pos_scale_range)
    env._joint_pos_bias_range = list(joint_pos_bias_range)
    env._joint_vel_range = list(joint_vel_range)
    env._randomize_dof_pos_bias = bool(randomize_dof_pos_bias)

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    scale_factor = torch_rand_float(
        joint_pos_scale_range[0],
        joint_pos_scale_range[1],
        (idx.shape[0], env.num_dof),
        device=env.device,
    )
    if randomize_dof_pos_bias:
        bias_offset = torch_rand_float(
            joint_pos_bias_range[0],
            joint_pos_bias_range[1],
            (idx.shape[0], env.num_dof),
            device=env.device,
        )
    else:
        bias_offset = torch.zeros((idx.shape[0], env.num_dof), device=env.device)

    env.simulator.dof_pos[idx] = env.default_dof_pos[idx] * scale_factor + bias_offset
    env.simulator.dof_vel[idx] = torch_rand_float(
        joint_vel_range[0],
        joint_vel_range[1],
        (idx.shape[0], env.num_dof),
        device=env.device,
    )


@mujoco_required_field("body_ipos")
def randomize_base_com_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    base_com_range: dict[str, Sequence[float]],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize base (torso) center of mass.

    Note: Uses ADDITION operation to offset CoM position (e.g., x: [-0.01, 0.01] m).
    """
    env._randomize_base_com = bool(enabled)
    env._base_com_range = base_com_range
    if not enabled:
        return

    logger.info(
        f"[Randomization] Base CoM: "
        f"x={base_com_range.get('x', [0, 0])}, "
        f"y={base_com_range.get('y', [0, 0])}, "
        f"z={base_com_range.get('z', [0, 0])} (operation=add)"
    )

    simulator = env.simulator

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    if hasattr(simulator, "gym"):
        gym = simulator.gym
        torso_name = env.robot_config.torso_name
        if not hasattr(simulator, "_base_com_bias"):
            simulator._base_com_bias = torch.zeros(
                env.num_envs, 3, dtype=torch.float, device=env.device, requires_grad=False
            )

        for env_id in idx.tolist():
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
            body_index = gym.find_actor_rigid_body_handle(env_ptr, actor, torso_name)
            if body_index < 0:
                raise RuntimeError(f"Body '{torso_name}' not found when randomizing base COM.")

            xrange = base_com_range["x"]
            yrange = base_com_range["y"]
            zrange = base_com_range["z"]

            bias = torch.tensor(
                [
                    torch_rand_float(xrange[0], xrange[1], (1, 1), device=env.device).item(),
                    torch_rand_float(yrange[0], yrange[1], (1, 1), device=env.device).item(),
                    torch_rand_float(zrange[0], zrange[1], (1, 1), device=env.device).item(),
                ],
                dtype=torch.float,
                device=env.device,
            )
            simulator._base_com_bias[env_id] = bias
            body_props[body_index].com.x += bias[0].item()
            body_props[body_index].com.y += bias[1].item()
            body_props[body_index].com.z += bias[2].item()
            gym.set_actor_rigid_body_properties(env_ptr, actor, body_props, recomputeInertia=True)
    elif simulator.__class__.__name__ == "IsaacSim":
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise RuntimeError("IsaacSim base COM randomization requires isaaclab.") from exc
        from holosoma.simulator.isaacsim.events import randomize_body_com

        torso_name = env.robot_config.torso_name
        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return

        low = torch.tensor(
            [base_com_range["x"][0], base_com_range["y"][0], base_com_range["z"][0]],
            dtype=torch.float,
            device="cpu",
        )
        high = torch.tensor(
            [base_com_range["x"][1], base_com_range["y"][1], base_com_range["z"][1]],
            dtype=torch.float,
            device="cpu",
        )
        asset_cfg = SceneEntityCfg("robot", body_names=[torso_name])
        asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
        randomize_body_com(
            simulator,
            env_ids_cpu,
            asset_cfg,
            (low, high),
            operation="add",
            distribution="uniform",
            num_envs=simulator.training_config.num_envs,
        )
    elif simulator.simulator_config.mujoco_backend == MujocoBackend.WARP:
        from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

        # convert xyz to 012
        base_com_range_remapped = {}
        for key, value in base_com_range.items():
            assert len(value) == 2, f"Range for '{key}' must have exactly 2 elements, got {len(value)}"
            base_com_range_remapped["xyz".index(key)] = (value[0], value[1])
        randomize_field(
            simulator,
            field=getattr(randomize_base_com_startup, MUJOCO_FIELD_ATTR),
            ranges=base_com_range_remapped,
            env_ids=idx,
            entity_names=[env.robot_config.torso_name],
            entity_type="body",
            operation="add",
            distribution="uniform",
        )

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Unsupported simulator type '{type(simulator).__name__}' for base COM randomization."
        )


@mujoco_required_field("body_mass")
def randomize_mass_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    enable_link_mass: bool = True,
    link_mass_range: Sequence[float] = (1.0, 1.0),
    enable_base_mass: bool = True,
    added_mass_range: Sequence[float] = (0.0, 0.0),
    enabled: bool = True,
    **_,
) -> None:
    """Randomize link and base masses at startup.

    Note: link_mass_range uses SCALING (e.g., 0.9-1.2 = 90-120% of original),
          added_mass_range uses ADDITION (e.g., -1.0 to 3.0 kg offset).
    """
    if not enabled:
        return

    logger.info(
        f"[Randomization] Mass: "
        f"link_mass={link_mass_range} (operation=scale, enabled={enable_link_mass}), "
        f"base_mass={added_mass_range} (operation=add, enabled={enable_base_mass})"
    )

    simulator = env.simulator
    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    env._randomize_link_mass = bool(enable_link_mass)
    env._randomize_base_mass = bool(enable_base_mass)

    if hasattr(simulator, "gym"):
        gym = simulator.gym
        body_names = list(env.robot_config.randomize_link_body_names or [])
        torso_name = env.robot_config.torso_name
        if idx.numel() > 0:
            sample_env = idx[0].item()
            sample_env_ptr = simulator.envs[sample_env]
            sample_actor = simulator.robot_handles[sample_env]
            sample_props = gym.get_actor_rigid_body_properties(sample_env_ptr, sample_actor)
            if enable_link_mass and body_names:
                link_masses = [
                    float(sample_props[simulator._body_list.index(name)].mass)
                    for name in body_names
                    if name in simulator._body_list
                ]
                if link_masses:
                    logger.debug(
                        "[randomize_mass_startup][IsaacGym] default link mass range: "
                        f"min={min(link_masses):.6f}, max={max(link_masses):.6f}"
                    )
            if enable_base_mass and torso_name in simulator._body_list:
                base_mass = float(sample_props[simulator._body_list.index(torso_name)].mass)
                logger.debug(f"[randomize_mass_startup][IsaacGym] default torso mass: {base_mass:.6f}")
        for env_id in idx.tolist():
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
            if enable_link_mass and body_names:
                for body_name in body_names:
                    if body_name not in simulator._body_list:
                        continue
                    body_index = simulator._body_list.index(body_name)
                    scale = np.random.uniform(link_mass_range[0], link_mass_range[1])
                    body_props[body_index].mass *= scale  # Scale operation: multiply by factor
            if enable_base_mass and torso_name in simulator._body_list:
                base_index = simulator._body_list.index(torso_name)
                delta = np.random.uniform(added_mass_range[0], added_mass_range[1])
                body_props[base_index].mass += delta  # Add operation: offset by delta
            gym.set_actor_rigid_body_properties(env_ptr, actor, body_props, recomputeInertia=True)
    elif simulator.__class__.__name__ == "IsaacSim":
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("IsaacSim mass randomization requires isaaclab.") from exc

        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return

        if enable_link_mass:
            asset_cfg = SceneEntityCfg("robot", body_names=env.robot_config.randomize_link_body_names)
            asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
            _isaacsim_randomize_rigid_body_mass(
                simulator,
                env_ids_cpu,
                asset_cfg,
                (link_mass_range[0], link_mass_range[1]),
                operation="scale",
            )

        if enable_base_mass:
            asset_cfg = SceneEntityCfg("robot", body_names=[env.robot_config.torso_name])
            asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
            _isaacsim_randomize_rigid_body_mass(
                simulator,
                env_ids_cpu,
                asset_cfg,
                (added_mass_range[0], added_mass_range[1]),
                operation="add",
            )
    elif simulator.simulator_config.mujoco_backend == MujocoBackend.WARP:
        from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

        # randomize over the range (scale and/or shift)
        if idx.numel() == 0:
            return

        if enable_link_mass:
            assert len(link_mass_range) == 2, (
                f"link_mass_range must have exactly 2 elements, got {len(link_mass_range)}"
            )
            randomize_field(
                simulator,
                field=getattr(randomize_mass_startup, MUJOCO_FIELD_ATTR),
                ranges=(link_mass_range[0], link_mass_range[1]),
                env_ids=idx,
                entity_names=env.robot_config.randomize_link_body_names,
                entity_type="body",
                operation="scale",
            )

        if enable_base_mass:
            assert len(added_mass_range) == 2, (
                f"added_mass_range must have exactly 2 elements, got {len(added_mass_range)}"
            )
            randomize_field(
                simulator,
                field=getattr(randomize_mass_startup, MUJOCO_FIELD_ATTR),
                ranges=(added_mass_range[0], added_mass_range[1]),
                env_ids=idx,
                entity_names=[env.robot_config.torso_name],
                entity_type="body",
                operation="add",
            )

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Mass randomization not supported for simulator type '{type(simulator).__name__}'."
        )


@mujoco_required_field("geom_friction")
def randomize_friction_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    friction_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize contact friction coefficients for robot rigid shapes.

    Note: Uses ABSOLUTE operation to set friction values (e.g., [0.5, 1.5]).
    """
    env._randomize_friction = bool(enabled)
    env._friction_range = list(friction_range)
    if not enabled:
        return

    logger.info(f"[Randomization] Friction: range={friction_range} (operation=abs)")

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator

    num_buckets = 64
    buckets = torch_rand_float(
        friction_range[0],
        friction_range[1],
        (num_buckets, 1),
        device="cpu",
    )

    idx_cpu = idx.to(device="cpu", dtype=torch.long)
    bucket_ids = torch.randint(0, num_buckets, (idx_cpu.shape[0],), device="cpu")
    friction_samples_cpu = buckets[bucket_ids]

    if hasattr(simulator, "gym"):
        gym = simulator.gym
        for offset, env_id in enumerate(idx_cpu.tolist()):
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            shape_props = gym.get_actor_rigid_shape_properties(env_ptr, actor)
            friction_value = friction_samples_cpu[offset].item()
            for prop in shape_props:
                prop.friction = friction_value
            gym.set_actor_rigid_shape_properties(env_ptr, actor, shape_props)
    elif simulator.__class__.__name__ == "IsaacSim":
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("IsaacSim friction randomization requires isaaclab.") from exc
        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return

        asset_cfg = SceneEntityCfg("robot", body_names=".*")
        asset_cfg.resolve(simulator.scene)  # Not stricly required, but a good practice

        _isaacsim_randomize_rigid_body_material(
            simulator,
            env_ids_cpu,
            asset_cfg,
            static_friction_range=(friction_range[0], friction_range[1]),
            dynamic_friction_range=(friction_range[0], friction_range[1]),
            restitution_range=(0.0, 0.0),
            num_buckets=num_buckets,
        )

    elif simulator.simulator_config.mujoco_backend == MujocoBackend.WARP:
        from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

        assert len(friction_range) == 2, f"friction_range must have exactly 2 elements, got {len(friction_range)}"
        randomize_field(
            simulator,
            field=getattr(randomize_friction_startup, MUJOCO_FIELD_ATTR),
            ranges={0: (friction_range[0], friction_range[1])},
            env_ids=idx,
            operation="abs",
        )

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Unsupported simulator type '{type(simulator).__name__}' for friction randomization."
        )


def randomize_robot_rigid_body_material_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    static_friction_range: Sequence[float],
    dynamic_friction_range: Sequence[float],
    restitution_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize robot rigid body material properties (friction, restitution)."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_robot_rigid_body_material_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim material randomization requires isaaclab.") from exc

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    asset_cfg = SceneEntityCfg("robot", body_names=".*")
    asset_cfg.resolve(simulator.scene)

    num_buckets = 64
    _isaacsim_randomize_rigid_body_material(
        simulator,
        env_ids_cpu,
        asset_cfg,
        static_friction_range=(static_friction_range[0], static_friction_range[1]),
        dynamic_friction_range=(dynamic_friction_range[0], dynamic_friction_range[1]),
        restitution_range=(restitution_range[0], restitution_range[1]),
        num_buckets=num_buckets,
    )


def randomize_object_rigid_body_material_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    static_friction_range: Sequence[float],
    dynamic_friction_range: Sequence[float],
    restitution_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize object rigid body material properties (friction, restitution)."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_object_rigid_body_material_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim material randomization requires isaaclab.") from exc

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    object_names = _get_object_scene_entity_names(simulator)
    if not object_names:
        logger.warning("No object scene entities found for material randomization. Skipping.")
        return

    num_buckets = 64
    for object_name in object_names:
        asset_cfg = SceneEntityCfg(object_name, body_names=".*")
        asset_cfg.resolve(simulator.scene)

        _isaacsim_randomize_rigid_body_material(
            simulator,
            env_ids_cpu,
            asset_cfg,
            static_friction_range=(static_friction_range[0], static_friction_range[1]),
            dynamic_friction_range=(dynamic_friction_range[0], dynamic_friction_range[1]),
            restitution_range=(restitution_range[0], restitution_range[1]),
            num_buckets=num_buckets,
        )


def randomize_object_rigid_body_mass_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    mass_distribution_params: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize object rigid body mass."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_object_rigid_body_mass_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg

    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim mass randomization requires isaaclab.") from exc

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    object_names = _get_object_scene_entity_names(simulator)
    if not object_names:
        logger.warning("No object scene entities found for mass randomization. Skipping.")
        return

    for object_name in object_names:
        asset_cfg = SceneEntityCfg(object_name, body_names=".*")
        asset_cfg.resolve(simulator.scene)

        _isaacsim_randomize_rigid_body_mass(
            simulator,
            env_ids_cpu,
            asset_cfg,
            (mass_distribution_params[0], mass_distribution_params[1]),
            operation="add",
        )


def randomize_object_rigid_body_inertia_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    inertia_distribution_params_dict: dict[str, tuple[float, float]],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize object rigid body inertia."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_object_rigid_body_inertia_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim inertia randomization requires isaaclab.") from exc

    from holosoma.simulator.isaacsim.events import randomize_rigid_body_inertia

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    object_names = _get_object_scene_entity_names(simulator)
    if not object_names:
        logger.warning("No object scene entities found for inertia randomization. Skipping.")
        return

    ordering = ["Ixx", "Iyy", "Izz", "Ixy", "Iyz", "Ixz"]
    lower_bounds = [inertia_distribution_params_dict[key][0] for key in ordering]
    upper_bounds = [inertia_distribution_params_dict[key][1] for key in ordering]
    inertia_distribution_params = (torch.tensor(lower_bounds, device="cpu"), torch.tensor(upper_bounds, device="cpu"))

    for object_name in object_names:
        asset_cfg = SceneEntityCfg(object_name, body_names=".*")
        asset_cfg.resolve(simulator.scene)

        randomize_rigid_body_inertia(
            simulator,
            env_ids_cpu,
            asset_cfg,
            inertia_distribution_params,
            operation="scale",
            distribution="uniform",
        )


def randomize_object_scale_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    scale_values: Sequence[float] | Sequence[Sequence[float]] | None = None,
    scale_value: float | Sequence[float] | None = None,
    relative_child_path: str | None = None,
    object_height: float | None = None,
    enabled: bool = True,
    **_,
) -> None:
    """Randomize object USD scale before simulation starts (IsaacSim only)."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_object_scale_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Gf, Sdf, UsdGeom, Vt
        import isaaclab.utils.math as math_utils
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim scale randomization requires isaaclab.") from exc

    object_names = _get_object_scene_entity_names(simulator)
    if not object_names:
        logger.warning("No object scene entities found for scale randomization. Skipping.")
        return

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    stage = get_current_stage()
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    if not hasattr(env, "object_scale_factors"):
        env.object_scale_factors = torch.ones(env.num_envs, 3, device=env.device, dtype=torch.float32)
    if not hasattr(env, "object_scale_factors_z"):
        env.object_scale_factors_z = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    _setup_object_scale_reference_bounds(env, object_names, object_height)

    from holosoma.managers.command.terms.wbt import (
        get_scaled_object_support_delta,
        get_scaled_object_support_delta_from_points,
    )

    if getattr(simulator.scene.cfg, "replicate_physics", False):
        raise RuntimeError(
            "Object scale randomization is incompatible with replicate_physics=True. "
            "Per-environment USD scale edits are not reliably applied when physics is replicated across environments. "
            "Disable scale randomization or set --simulator.config.scene.replicate-physics False."
        )

    for object_name in object_names:
        prim_paths = _get_scene_entity_prim_paths(simulator, object_name)
        if not prim_paths:
            logger.warning(f"No prim paths found for object '{object_name}' scale randomization. Skipping.")
            continue

        if scale_value is not None:
            fixed_scale = torch.as_tensor(scale_value, device="cpu", dtype=torch.float32)
            if fixed_scale.ndim == 0:
                rand_samples = fixed_scale.repeat(len(env_ids_cpu), 3).view(len(env_ids_cpu), 3)
            else:
                fixed_scale = fixed_scale.flatten()
                if fixed_scale.numel() != 3:
                    raise ValueError(
                        "scale_value must be a scalar or a 3-element sequence, "
                        f"got shape {tuple(fixed_scale.shape)}"
                    )
                rand_samples = fixed_scale.unsqueeze(0).repeat(len(env_ids_cpu), 1)
            scale_z_samples = rand_samples[:, 2]
        elif scale_values is not None:
            choices = torch.as_tensor(scale_values, device="cpu", dtype=torch.float32)
            if choices.ndim == 1:
                if choices.numel() == 0:
                    raise ValueError("scale_values must contain at least one scale choice.")
                choice_ids = torch.randint(0, choices.numel(), (len(env_ids_cpu),), device="cpu")
                selected = choices[choice_ids]
                rand_samples = selected.unsqueeze(1).repeat(1, 3)
                scale_z_samples = selected
            elif choices.ndim == 2 and choices.shape[1] == 3:
                if choices.shape[0] == 0:
                    raise ValueError("scale_values must contain at least one xyz scale choice.")
                choice_ids = torch.randint(0, choices.shape[0], (len(env_ids_cpu),), device="cpu")
                rand_samples = choices[choice_ids]
                scale_z_samples = rand_samples[:, 2]
            else:
                raise ValueError(
                    "scale_values must be a 1-D sequence of uniform scale choices or an Nx3 sequence of xyz choices, "
                    f"got shape {tuple(choices.shape)}."
                )
        elif isinstance(scale_range, dict):
            range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
            ranges = torch.tensor(range_list, device="cpu")
            rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids_cpu), 3), device="cpu")
            scale_z_samples = rand_samples[:, 2]
        elif scale_range is not None:
            rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids_cpu), 1), device="cpu")
            scale_z_samples = rand_samples[:, 0]
            rand_samples = rand_samples.repeat(1, 3)
        else:
            raise ValueError("randomize_object_scale_startup requires scale_value, scale_values, or scale_range.")
        rand_samples_list = rand_samples.tolist()

        with Sdf.ChangeBlock():
            for i, env_id in enumerate(env_ids_cpu.tolist()):
                prim_path = prim_paths[env_id] + relative_child_path
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

                scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
                has_scale_attr = scale_spec is not None
                if not has_scale_attr:
                    scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

                scale_spec.default = Gf.Vec3f(*rand_samples_list[i])

                if not has_scale_attr:
                    op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                    if op_order_spec is None:
                        op_order_spec = Sdf.AttributeSpec(
                            prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                        )
                    op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

        env_ids_device = env_ids_cpu.to(device=env.device)
        env.object_scale_factors[env_ids_device] = rand_samples.to(device=env.device, dtype=torch.float32)
        env.object_scale_factors_z[env_ids_device] = scale_z_samples.to(device=env.device, dtype=torch.float32)

        rigid_object = simulator.scene.rigid_objects.get(object_name)
        if rigid_object is None:
            continue

        support_delta = None
        local_support_points = getattr(env, "object_local_support_points_by_actor", {}).get(object_name)
        local_center = getattr(env, "object_local_bbox_center_by_actor", {}).get(object_name)
        local_half_extent = getattr(env, "object_local_bbox_half_extent_by_actor", {}).get(object_name)

        try:
            current_root_state = rigid_object.data.root_state_w[env_ids_device].clone()
            quat_xyzw = current_root_state[:, [4, 5, 6, 3]]
            scales_device = rand_samples.to(device=env.device, dtype=torch.float32)

            if local_support_points is not None:
                support_delta = get_scaled_object_support_delta_from_points(
                    quat_xyzw=quat_xyzw,
                    scales_xyz=scales_device,
                    local_support_points=local_support_points,
                )
            elif local_center is not None and local_half_extent is not None:
                support_delta = get_scaled_object_support_delta(
                    quat_xyzw=quat_xyzw,
                    scales_xyz=scales_device,
                    local_bbox_center=local_center,
                    local_bbox_half_extent=local_half_extent,
                )

            if support_delta is None:
                continue

            current_root_state[:, 2] += support_delta
            rigid_object.data.default_root_state[env_ids_device, 2] += support_delta
            rigid_object.write_root_pose_to_sim(current_root_state[:, :7], env_ids_device)
            rigid_object.write_root_velocity_to_sim(current_root_state[:, 7:], env_ids_device)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to apply startup grounded z compensation for object "
                f"'{object_name}' after scale randomization: {exc}"
            )


def set_object_init_pose_noise(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    object_pos_noise: Sequence[float],
    overall_noise_scale: float | None = None,
    enabled: bool = True,
    **_,
) -> None:
    """Update MotionCommand object position noise scale (applies on reset)."""
    if not enabled:
        return

    motion_command = env.command_manager.get_state("motion_command")
    if motion_command is None:
        logger.warning("MotionCommand not found; object position noise update skipped.")
        return

    from dataclasses import replace

    init_pose_cfg = motion_command.init_pose_cfg
    new_cfg = replace(
        init_pose_cfg,
        object_pos=list(object_pos_noise),
        overall_noise_scale=init_pose_cfg.overall_noise_scale
        if overall_noise_scale is None
        else float(overall_noise_scale),
    )
    motion_command.init_pose_cfg = new_cfg


def configure_torque_rfi(
    env,
    env_ids,
    *,
    enabled: bool | None = None,
    rfi_lim: float | None = None,
    **_,
) -> None:
    """Toggle torque RFI injection flag."""
    prev_enabled, prev_lim = env._pending_torque_rfi
    enabled_flag = prev_enabled if enabled is None else bool(enabled)
    rfi_limit = prev_lim if rfi_lim is None else float(rfi_lim)
    env._pending_torque_rfi = (enabled_flag, rfi_limit)

    state = env.randomization_manager.get_state("actuator_randomizer_state")
    if state is not None:
        state.enable_rfi_lim = enabled_flag
    term = _get_joint_action_term(env)
    if term is not None:
        term.configure_torque_rfi(enabled=enabled_flag, rfi_lim=rfi_limit)


def apply_pushes(
    env,
    *,
    enabled: bool | None = None,
    push_interval_s: Sequence[float] | None = None,
    max_push_vel: Sequence[float] | None = None,
    **_,
) -> None:
    """Apply random pushes based on the current schedule."""
    state = env.randomization_manager.get_state("push_randomizer_state")
    if state is None:
        raise AttributeError("PushRandomizerState is not registered with the randomization manager.")

    state.configure(enabled=enabled, push_interval_s=push_interval_s, max_push_vel=max_push_vel)
    env._push_robots_enabled = state.enabled

    if env.is_evaluating or not state.enabled:
        return

    push_robot_env_ids = state.due_envs(env.dt)
    if push_robot_env_ids.numel() == 0:
        return

    state.zero_counters(push_robot_env_ids)
    state.resample(push_robot_env_ids)
    env._max_push_vel = state.max_push_vel.clone()
    env._push_robots(push_robot_env_ids)
