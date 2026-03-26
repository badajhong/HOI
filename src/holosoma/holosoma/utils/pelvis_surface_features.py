from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from holosoma.config_types.robot import ObjectConfig
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.safe_torch_import import torch

_EPS = 1e-8


def _normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < _EPS:
        if fallback is None:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm < _EPS:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return fallback / fallback_norm
    return vector / norm


def _parse_float_list(text: str | None, expected_len: int, default: Sequence[float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    values = [float(v) for v in text.replace(",", " ").split()]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {values}")
    return np.asarray(values, dtype=np.float64)


def _rotation_matrix_from_rpy(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _transform_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_matrix_from_rpy(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def _quat_xyzw_to_rotation_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return points @ rotation.T + translation


def _closest_point_on_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def _ray_intersects_triangle(origin: np.ndarray, direction: np.ndarray, triangle: np.ndarray) -> float | None:
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = float(np.dot(edge1, pvec))
    if abs(det) < _EPS:
        return None

    inv_det = 1.0 / det
    tvec = origin - v0
    u = float(np.dot(tvec, pvec)) * inv_det
    if u < 0.0 or u > 1.0:
        return None

    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec)) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return None

    t = float(np.dot(edge2, qvec)) * inv_det
    if t <= _EPS:
        return None
    return t


def _triangulate_face(indices: Sequence[int]) -> list[list[int]]:
    if len(indices) < 3:
        return []
    return [[indices[0], indices[i], indices[i + 1]] for i in range(1, len(indices) - 1)]


def _parse_obj_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with mesh_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("v "):
                _, x, y, z, *rest = stripped.split()
                vertices.append([float(x), float(y), float(z)])
            elif stripped.startswith("f "):
                raw_indices = []
                for token in stripped.split()[1:]:
                    index_str = token.split("/")[0]
                    index = int(index_str)
                    if index < 0:
                        index = len(vertices) + index
                    else:
                        index -= 1
                    raw_indices.append(index)
                faces.extend(_triangulate_face(raw_indices))
    if not vertices or not faces:
        raise ValueError(f"OBJ mesh '{mesh_path}' does not contain vertices/faces.")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _parse_ascii_stl_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_map: dict[tuple[float, float, float], int] = {}
    current_triangle: list[int] = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            stripped = line.strip()
            if not stripped.startswith("vertex "):
                continue
            _, x, y, z = stripped.split()
            vertex = (float(x), float(y), float(z))
            vertex_index = vertex_map.get(vertex)
            if vertex_index is None:
                vertex_index = len(vertices)
                vertex_map[vertex] = vertex_index
                vertices.append(list(vertex))
            current_triangle.append(vertex_index)
            if len(current_triangle) == 3:
                faces.append(current_triangle)
                current_triangle = []
    if not vertices or not faces:
        raise ValueError(f"ASCII STL mesh '{mesh_path}' does not contain vertices/faces.")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _parse_binary_stl_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_map: dict[tuple[float, float, float], int] = {}
    with mesh_path.open("rb") as file:
        file.read(80)
        triangle_count = struct.unpack("<I", file.read(4))[0]
        for _ in range(triangle_count):
            file.read(12)
            triangle_indices: list[int] = []
            for _ in range(3):
                vertex = struct.unpack("<fff", file.read(12))
                key = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
                vertex_index = vertex_map.get(key)
                if vertex_index is None:
                    vertex_index = len(vertices)
                    vertex_map[key] = vertex_index
                    vertices.append(list(key))
                triangle_indices.append(vertex_index)
            faces.append(triangle_indices)
            file.read(2)
    if not vertices or not faces:
        raise ValueError(f"Binary STL mesh '{mesh_path}' does not contain vertices/faces.")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _load_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        return _parse_obj_mesh(mesh_path)
    if suffix == ".stl":
        file_size = mesh_path.stat().st_size
        with mesh_path.open("rb") as file:
            header = file.read(84)
        if len(header) >= 84:
            triangle_count = struct.unpack("<I", header[80:84])[0]
            expected_size = 84 + triangle_count * 50
            if expected_size == file_size:
                return _parse_binary_stl_mesh(mesh_path)
        return _parse_ascii_stl_mesh(mesh_path)
    raise ValueError(f"Unsupported mesh format '{mesh_path.suffix}' for '{mesh_path}'.")


class CollisionGeometry:
    def closest_point_and_normal(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        raise NotImplementedError


@dataclass(frozen=True)
class BoxGeometry(CollisionGeometry):
    half_extents: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray

    def closest_point_and_normal(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        point_local = self.rotation.T @ (point - self.translation)
        clamped = np.clip(point_local, -self.half_extents, self.half_extents)
        inside = bool(np.all(np.abs(point_local) <= self.half_extents + _EPS))

        if inside:
            face_margin = self.half_extents - np.abs(point_local)
            axis = int(np.argmin(face_margin))
            direction = 1.0 if point_local[axis] >= 0.0 else -1.0
            closest_local = point_local.copy()
            closest_local[axis] = direction * self.half_extents[axis]
            normal_local = np.zeros(3, dtype=np.float64)
            normal_local[axis] = direction
        else:
            closest_local = clamped
            normal_local = _normalize(point_local - closest_local)

        closest_world = self.rotation @ closest_local + self.translation
        normal_world = _normalize(self.rotation @ normal_local)
        distance = float(np.linalg.norm(point - closest_world))
        return closest_world, normal_world, distance


@dataclass(frozen=True)
class SphereGeometry(CollisionGeometry):
    radius: float
    translation: np.ndarray

    def closest_point_and_normal(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        delta = point - self.translation
        normal = _normalize(delta, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
        closest = self.translation + normal * self.radius
        distance = abs(float(np.linalg.norm(delta)) - self.radius)
        return closest, normal, distance


@dataclass(frozen=True)
class CylinderGeometry(CollisionGeometry):
    radius: float
    half_length: float
    translation: np.ndarray
    rotation: np.ndarray

    def closest_point_and_normal(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        point_local = self.rotation.T @ (point - self.translation)
        radial = point_local[:2]
        radial_norm = float(np.linalg.norm(radial))
        z = float(np.clip(point_local[2], -self.half_length, self.half_length))

        if radial_norm < _EPS:
            radial_dir = np.array([1.0, 0.0], dtype=np.float64)
        else:
            radial_dir = radial / radial_norm

        side_point = np.array([radial_dir[0] * self.radius, radial_dir[1] * self.radius, z], dtype=np.float64)
        cap_point = np.array(
            [
                radial_dir[0] * min(radial_norm, self.radius),
                radial_dir[1] * min(radial_norm, self.radius),
                self.half_length if point_local[2] >= 0.0 else -self.half_length,
            ],
            dtype=np.float64,
        )

        side_distance = float(np.linalg.norm(point_local - side_point))
        cap_distance = float(np.linalg.norm(point_local - cap_point))
        closest_local = side_point if side_distance <= cap_distance else cap_point

        if abs(closest_local[2]) == self.half_length and radial_norm <= self.radius + _EPS:
            normal_local = np.array([0.0, 0.0, 1.0 if closest_local[2] > 0.0 else -1.0], dtype=np.float64)
        else:
            normal_local = _normalize(
                np.array([closest_local[0], closest_local[1], 0.0], dtype=np.float64),
                fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )

        closest_world = self.rotation @ closest_local + self.translation
        normal_world = _normalize(self.rotation @ normal_local)
        distance = float(np.linalg.norm(point - closest_world))
        return closest_world, normal_world, distance


def _mesh_as_box_geometry(mesh_path: Path, scale: np.ndarray, shape_transform: np.ndarray) -> BoxGeometry:
    vertices, _ = _load_mesh(mesh_path)
    scaled_vertices = vertices * scale
    bounds_min = scaled_vertices.min(axis=0)
    bounds_max = scaled_vertices.max(axis=0)
    local_center = 0.5 * (bounds_min + bounds_max)
    half_extents = 0.5 * np.maximum(bounds_max - bounds_min, _EPS)
    rotation = shape_transform[:3, :3].copy()
    translation = rotation @ local_center + shape_transform[:3, 3]
    return BoxGeometry(
        half_extents=half_extents,
        translation=translation,
        rotation=rotation,
    )


@dataclass(frozen=True)
class MeshGeometry(CollisionGeometry):
    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    centroid: np.ndarray

    def _contains(self, point: np.ndarray) -> bool:
        direction = _normalize(np.array([1.0, 0.371, 0.529], dtype=np.float64))
        intersections: list[float] = []
        triangles = self.vertices[self.faces]
        for triangle in triangles:
            hit = _ray_intersects_triangle(point, direction, triangle)
            if hit is None:
                continue
            if not any(abs(hit - existing) < 1e-6 for existing in intersections):
                intersections.append(hit)
        return len(intersections) % 2 == 1

    def closest_point_and_normal(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        triangles = self.vertices[self.faces]
        best_distance_sq = float("inf")
        best_point = triangles[0, 0]
        best_normal = self.face_normals[0]

        for face_index, triangle in enumerate(triangles):
            closest = _closest_point_on_triangle(point, triangle[0], triangle[1], triangle[2])
            distance_sq = float(np.dot(point - closest, point - closest))
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_point = closest
                best_normal = self.face_normals[face_index]

        inside = self._contains(point)
        direction = point - best_point
        if inside or np.linalg.norm(direction) < 1e-6:
            normal = best_normal
        else:
            normal = _normalize(direction, fallback=best_normal)
        return best_point, _normalize(normal, fallback=best_normal), math.sqrt(best_distance_sq)


def _mesh_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    centroid = vertices.mean(axis=0)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.asarray([_normalize(normal) for normal in normals], dtype=np.float64)
    face_centers = triangles.mean(axis=1)
    outward_mask = np.sum(normals * (face_centers - centroid), axis=1) < 0.0
    normals[outward_mask] *= -1.0
    return normals


def _node_path(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        filename = filename[len("package://") :]
    mesh_path = Path(filename)
    if mesh_path.is_absolute():
        return mesh_path
    return (urdf_path.parent / mesh_path).resolve()


def _geometry_nodes(link_node: ET.Element) -> Iterable[ET.Element]:
    collision_nodes = list(link_node.findall("collision"))
    if collision_nodes:
        return collision_nodes
    return link_node.findall("visual")


def _link_transforms(robot_root: ET.Element) -> tuple[dict[str, ET.Element], dict[str, np.ndarray]]:
    links = {link.attrib["name"]: link for link in robot_root.findall("link")}
    joints_by_parent: dict[str, list[ET.Element]] = {}
    child_links: set[str] = set()

    for joint in robot_root.findall("joint"):
        parent_node = joint.find("parent")
        child_node = joint.find("child")
        if parent_node is None or child_node is None:
            continue
        parent = parent_node.attrib["link"]
        child = child_node.attrib["link"]
        joints_by_parent.setdefault(parent, []).append(joint)
        child_links.add(child)

    root_links = [name for name in links if name not in child_links] or list(links)
    transforms = {root_link: np.eye(4, dtype=np.float64) for root_link in root_links}
    queue = list(root_links)

    while queue:
        parent_link = queue.pop()
        parent_transform = transforms[parent_link]
        for joint in joints_by_parent.get(parent_link, []):
            child_node = joint.find("child")
            if child_node is None:
                continue
            child_link = child_node.attrib["link"]
            origin_node = joint.find("origin")
            joint_xyz = _parse_float_list(origin_node.attrib.get("xyz") if origin_node is not None else None, 3, [0, 0, 0])
            joint_rpy = _parse_float_list(origin_node.attrib.get("rpy") if origin_node is not None else None, 3, [0, 0, 0])
            transforms[child_link] = parent_transform @ _transform_from_xyz_rpy(joint_xyz, joint_rpy)
            queue.append(child_link)

    for link_name in links:
        transforms.setdefault(link_name, np.eye(4, dtype=np.float64))

    return links, transforms


def _load_collision_geometries(urdf_path: str, mesh_mode: str = "full") -> list[CollisionGeometry]:
    mesh_mode = mesh_mode.lower().strip()
    if mesh_mode not in {"full", "box"}:
        raise ValueError(f"Unsupported mesh_mode '{mesh_mode}'. Expected one of: 'full', 'box'.")

    urdf_file = Path(resolve_data_file_path(urdf_path)).resolve()
    robot_root = ET.parse(urdf_file).getroot()
    links, link_transforms = _link_transforms(robot_root)
    geometries: list[CollisionGeometry] = []

    for link_name, link_node in links.items():
        link_transform = link_transforms[link_name]
        for geom_node in _geometry_nodes(link_node):
            origin_node = geom_node.find("origin")
            collision_xyz = _parse_float_list(
                origin_node.attrib.get("xyz") if origin_node is not None else None, 3, [0.0, 0.0, 0.0]
            )
            collision_rpy = _parse_float_list(
                origin_node.attrib.get("rpy") if origin_node is not None else None, 3, [0.0, 0.0, 0.0]
            )
            shape_transform = link_transform @ _transform_from_xyz_rpy(collision_xyz, collision_rpy)
            geometry_node = geom_node.find("geometry")
            if geometry_node is None:
                continue

            mesh_node = geometry_node.find("mesh")
            if mesh_node is not None:
                mesh_path = _node_path(urdf_file, mesh_node.attrib["filename"])
                scale = _parse_float_list(mesh_node.attrib.get("scale"), 3, [1.0, 1.0, 1.0])
                if mesh_mode == "box":
                    geometries.append(_mesh_as_box_geometry(mesh_path, scale, shape_transform))
                    continue
                vertices, faces = _load_mesh(mesh_path)
                vertices = vertices * scale
                vertices = _apply_transform(vertices, shape_transform)
                face_normals = _mesh_face_normals(vertices, faces)
                geometries.append(
                    MeshGeometry(vertices=vertices, faces=faces, face_normals=face_normals, centroid=vertices.mean(axis=0))
                )
                continue

            box_node = geometry_node.find("box")
            if box_node is not None:
                size = _parse_float_list(box_node.attrib.get("size"), 3, [1.0, 1.0, 1.0])
                geometries.append(
                    BoxGeometry(
                        half_extents=size / 2.0,
                        translation=shape_transform[:3, 3].copy(),
                        rotation=shape_transform[:3, :3].copy(),
                    )
                )
                continue

            sphere_node = geometry_node.find("sphere")
            if sphere_node is not None:
                geometries.append(
                    SphereGeometry(
                        radius=float(sphere_node.attrib["radius"]),
                        translation=shape_transform[:3, 3].copy(),
                    )
                )
                continue

            cylinder_node = geometry_node.find("cylinder")
            if cylinder_node is not None:
                geometries.append(
                    CylinderGeometry(
                        radius=float(cylinder_node.attrib["radius"]),
                        half_length=float(cylinder_node.attrib["length"]) / 2.0,
                        translation=shape_transform[:3, 3].copy(),
                        rotation=shape_transform[:3, :3].copy(),
                    )
                )

    if not geometries:
        raise ValueError(f"No collision/visual geometry could be loaded from URDF '{urdf_file}'.")
    return geometries


@dataclass(frozen=True)
class SurfaceFeatureResult:
    phi: np.ndarray
    grad_phi: np.ndarray
    v_norm: np.ndarray
    v_tan: np.ndarray

    @property
    def u_t(self) -> np.ndarray:
        return np.concatenate([self.phi.reshape(1), self.grad_phi, self.v_norm, self.v_tan], axis=0)


class CollisionScene:
    def __init__(self, geometries: Sequence[CollisionGeometry]):
        if not geometries:
            raise ValueError("CollisionScene requires at least one geometry.")
        self._geometries = list(geometries)

    def compute_surface_feature(
        self,
        pelvis_pos_w: np.ndarray,
        pelvis_lin_vel_w: np.ndarray,
        object_pos_w: np.ndarray,
        object_quat_xyzw: np.ndarray,
    ) -> SurfaceFeatureResult:
        object_rotation = _quat_xyzw_to_rotation_matrix(object_quat_xyzw)
        pelvis_pos_obj = object_rotation.T @ (pelvis_pos_w - object_pos_w)

        best_normal_obj: np.ndarray | None = None
        best_distance = float("inf")
        for geometry in self._geometries:
            _, normal_obj, distance = geometry.closest_point_and_normal(pelvis_pos_obj)
            if distance < best_distance:
                best_normal_obj = normal_obj
                best_distance = distance

        if best_normal_obj is None:
            raise RuntimeError("Failed to compute closest surface point for pelvis feature.")

        grad_phi_w = _normalize(object_rotation @ best_normal_obj)
        v_norm = np.dot(pelvis_lin_vel_w, grad_phi_w) * grad_phi_w
        v_tan = pelvis_lin_vel_w - v_norm
        return SurfaceFeatureResult(
            phi=np.asarray([best_distance], dtype=np.float64),
            grad_phi=grad_phi_w.astype(np.float64),
            v_norm=v_norm.astype(np.float64),
            v_tan=v_tan.astype(np.float64),
        )


class PelvisSurfaceFeatureComputer:
    def __init__(self, scenes_by_key: dict[str, CollisionScene]):
        if not scenes_by_key:
            raise ValueError("PelvisSurfaceFeatureComputer requires at least one object scene.")
        self._scenes_by_key = scenes_by_key
        self._default_key = next(iter(scenes_by_key))

    @classmethod
    def from_object_config(
        cls,
        object_cfg: ObjectConfig,
        mesh_mode: str = "full",
    ) -> PelvisSurfaceFeatureComputer:
        scene_paths = dict(object_cfg.object_urdf_name_to_path or {})
        if not scene_paths and object_cfg.object_urdf_path:
            scene_paths["default"] = resolve_data_file_path(object_cfg.object_urdf_path)
        if not scene_paths:
            raise ValueError("No object URDF path is configured for live pelvis surface features.")

        scenes = {key: CollisionScene(_load_collision_geometries(path, mesh_mode=mesh_mode)) for key, path in scene_paths.items()}
        return cls(scenes)

    def _scene_for_key(self, object_key: str | None) -> CollisionScene:
        if object_key is not None and object_key in self._scenes_by_key:
            return self._scenes_by_key[object_key]
        return self._scenes_by_key[self._default_key]

    def compute_batch(
        self,
        pelvis_pos_w: torch.Tensor,
        pelvis_lin_vel_w: torch.Tensor,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
        object_keys: Sequence[str | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        num_envs = int(pelvis_pos_w.shape[0])
        if object_keys is None:
            object_keys = [self._default_key] * num_envs
        if len(object_keys) != num_envs:
            raise ValueError(f"Expected {num_envs} object keys, got {len(object_keys)}")

        pelvis_pos_np = pelvis_pos_w.detach().cpu().numpy().astype(np.float64)
        pelvis_vel_np = pelvis_lin_vel_w.detach().cpu().numpy().astype(np.float64)
        object_pos_np = object_pos_w.detach().cpu().numpy().astype(np.float64)
        object_quat_np = object_quat_w.detach().cpu().numpy().astype(np.float64)

        phi_values: list[np.ndarray] = []
        grad_values: list[np.ndarray] = []
        v_norm_values: list[np.ndarray] = []
        v_tan_values: list[np.ndarray] = []

        for env_index in range(num_envs):
            scene = self._scene_for_key(object_keys[env_index])
            feature = scene.compute_surface_feature(
                pelvis_pos_w=pelvis_pos_np[env_index],
                pelvis_lin_vel_w=pelvis_vel_np[env_index],
                object_pos_w=object_pos_np[env_index],
                object_quat_xyzw=object_quat_np[env_index],
            )
            phi_values.append(feature.phi)
            grad_values.append(feature.grad_phi)
            v_norm_values.append(feature.v_norm)
            v_tan_values.append(feature.v_tan)

        dtype = pelvis_pos_w.dtype
        device = pelvis_pos_w.device
        phi = torch.as_tensor(np.stack(phi_values, axis=0), device=device, dtype=dtype)
        grad_phi = torch.as_tensor(np.stack(grad_values, axis=0), device=device, dtype=dtype)
        v_norm = torch.as_tensor(np.stack(v_norm_values, axis=0), device=device, dtype=dtype)
        v_tan = torch.as_tensor(np.stack(v_tan_values, axis=0), device=device, dtype=dtype)
        u_t = torch.cat([phi, grad_phi, v_norm, v_tan], dim=-1)
        return {
            "phi": phi,
            "grad_phi": grad_phi,
            "v_norm": v_norm,
            "v_tan": v_tan,
            "u_t": u_t,
        }
