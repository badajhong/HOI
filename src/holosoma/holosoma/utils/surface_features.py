from __future__ import annotations

"""Generic body-surface feature utilities.

This module keeps its historical filename for backward compatibility, but the
APIs below are intentionally body-agnostic and can be used for pelvis, hands,
or any other tracked body.
"""

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from holosoma.config_types.robot import ObjectConfig
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import quaternion_to_matrix
from holosoma.utils.safe_torch_import import torch

_EPS = 1e-8
_MESH_TRIANGLE_CHUNK_SIZE = 4096


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


def _constant_axis_like(reference: torch.Tensor, axis: int) -> torch.Tensor:
    basis = torch.zeros_like(reference)
    basis[..., axis] = 1.0
    return basis


def _safe_normalize_torch(vectors: torch.Tensor, eps: float = _EPS) -> tuple[torch.Tensor, torch.Tensor]:
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    normalized = torch.where(norms > eps, vectors / norms.clamp_min(eps), torch.zeros_like(vectors))
    return normalized, norms


def _normalize_torch_with_fallback(
    vectors: torch.Tensor,
    fallback: torch.Tensor,
    eps: float = _EPS,
    default_axis: int = 2,
) -> torch.Tensor:
    normalized, norms = _safe_normalize_torch(vectors, eps)
    fallback_normalized, fallback_norms = _safe_normalize_torch(fallback, eps)
    default_basis = _constant_axis_like(vectors, axis=default_axis)
    safe_fallback = torch.where(fallback_norms > eps, fallback_normalized, default_basis)
    return torch.where(norms > eps, normalized, safe_fallback)


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

    # URDF mesh paths in this dataset are not fully consistent.  Some are
    # relative to the URDF folder, while others are relative to an ancestor
    # package/model folder, e.g. objects/largebox/largebox.obj from models/.
    candidates = [(urdf_path.parent / mesh_path).resolve()]
    for ancestor in urdf_path.parents:
        candidate = (ancestor / mesh_path).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
    same_folder_candidate = (urdf_path.parent / mesh_path.name).resolve()
    if same_folder_candidate not in candidates:
        candidates.append(same_folder_candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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


class _TensorCollisionScene:
    def __init__(
        self,
        mesh_triangles: torch.Tensor | None,
        mesh_face_normals: torch.Tensor | None,
        sphere_centers: torch.Tensor | None,
        sphere_radii: torch.Tensor | None,
        box_half_extents: torch.Tensor | None,
        box_translations: torch.Tensor | None,
        box_rotations: torch.Tensor | None,
        cylinder_radii: torch.Tensor | None,
        cylinder_half_lengths: torch.Tensor | None,
        cylinder_translations: torch.Tensor | None,
        cylinder_rotations: torch.Tensor | None,
    ):
        self._cpu_tensors = {
            "mesh_triangles": mesh_triangles,
            "mesh_face_normals": mesh_face_normals,
            "sphere_centers": sphere_centers,
            "sphere_radii": sphere_radii,
            "box_half_extents": box_half_extents,
            "box_translations": box_translations,
            "box_rotations": box_rotations,
            "cylinder_radii": cylinder_radii,
            "cylinder_half_lengths": cylinder_half_lengths,
            "cylinder_translations": cylinder_translations,
            "cylinder_rotations": cylinder_rotations,
        }
        self._tensor_cache: dict[tuple[str, torch.dtype], dict[str, torch.Tensor | None]] = {}

    @staticmethod
    def _stack_optional(values: list[np.ndarray], dtype: np.dtype = np.float32) -> torch.Tensor | None:
        if not values:
            return None
        return torch.from_numpy(np.stack(values, axis=0).astype(dtype, copy=False))

    @classmethod
    def from_geometries(cls, geometries: Sequence[CollisionGeometry]) -> _TensorCollisionScene:
        mesh_triangles: list[np.ndarray] = []
        mesh_face_normals: list[np.ndarray] = []
        sphere_centers: list[np.ndarray] = []
        sphere_radii: list[np.ndarray] = []
        box_half_extents: list[np.ndarray] = []
        box_translations: list[np.ndarray] = []
        box_rotations: list[np.ndarray] = []
        cylinder_radii: list[np.ndarray] = []
        cylinder_half_lengths: list[np.ndarray] = []
        cylinder_translations: list[np.ndarray] = []
        cylinder_rotations: list[np.ndarray] = []

        for geometry in geometries:
            if isinstance(geometry, MeshGeometry):
                mesh_triangles.append(geometry.vertices[geometry.faces].astype(np.float32, copy=False))
                mesh_face_normals.append(geometry.face_normals.astype(np.float32, copy=False))
                continue
            if isinstance(geometry, SphereGeometry):
                sphere_centers.append(geometry.translation.astype(np.float32, copy=False))
                sphere_radii.append(np.asarray(geometry.radius, dtype=np.float32))
                continue
            if isinstance(geometry, BoxGeometry):
                box_half_extents.append(geometry.half_extents.astype(np.float32, copy=False))
                box_translations.append(geometry.translation.astype(np.float32, copy=False))
                box_rotations.append(geometry.rotation.astype(np.float32, copy=False))
                continue
            if isinstance(geometry, CylinderGeometry):
                cylinder_radii.append(np.asarray(geometry.radius, dtype=np.float32))
                cylinder_half_lengths.append(np.asarray(geometry.half_length, dtype=np.float32))
                cylinder_translations.append(geometry.translation.astype(np.float32, copy=False))
                cylinder_rotations.append(geometry.rotation.astype(np.float32, copy=False))

        mesh_triangles_tensor = None
        if mesh_triangles:
            mesh_triangles_tensor = torch.from_numpy(np.concatenate(mesh_triangles, axis=0).astype(np.float32, copy=False))

        mesh_face_normals_tensor = None
        if mesh_face_normals:
            mesh_face_normals_tensor = torch.from_numpy(
                np.concatenate(mesh_face_normals, axis=0).astype(np.float32, copy=False)
            )

        sphere_radii_tensor = None
        if sphere_radii:
            sphere_radii_tensor = torch.from_numpy(np.asarray(sphere_radii, dtype=np.float32))

        cylinder_radii_tensor = None
        if cylinder_radii:
            cylinder_radii_tensor = torch.from_numpy(np.asarray(cylinder_radii, dtype=np.float32))

        cylinder_half_lengths_tensor = None
        if cylinder_half_lengths:
            cylinder_half_lengths_tensor = torch.from_numpy(np.asarray(cylinder_half_lengths, dtype=np.float32))

        return cls(
            mesh_triangles=mesh_triangles_tensor,
            mesh_face_normals=mesh_face_normals_tensor,
            sphere_centers=cls._stack_optional(sphere_centers),
            sphere_radii=sphere_radii_tensor,
            box_half_extents=cls._stack_optional(box_half_extents),
            box_translations=cls._stack_optional(box_translations),
            box_rotations=cls._stack_optional(box_rotations),
            cylinder_radii=cylinder_radii_tensor,
            cylinder_half_lengths=cylinder_half_lengths_tensor,
            cylinder_translations=cls._stack_optional(cylinder_translations),
            cylinder_rotations=cls._stack_optional(cylinder_rotations),
        )

    def _cached_tensors(self, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor | None]:
        cache_key = (str(device), dtype)
        cached = self._tensor_cache.get(cache_key)
        if cached is not None:
            return cached

        cached = {
            name: tensor.to(device=device, dtype=dtype, non_blocking=True) if tensor is not None else None
            for name, tensor in self._cpu_tensors.items()
        }
        self._tensor_cache[cache_key] = cached
        return cached

    @staticmethod
    def _closest_points_on_triangles(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
        # points: [batch, 3], triangles: [num_triangles, 3, 3]
        point_count = points.shape[0]
        triangle_count = triangles.shape[0]
        p = points[:, None, :]
        a = triangles[None, :, 0, :]
        b = triangles[None, :, 1, :]
        c = triangles[None, :, 2, :]
        ab = b - a
        ac = c - a

        ap = p - a
        d1 = torch.sum(ab * ap, dim=-1)
        d2 = torch.sum(ac * ap, dim=-1)
        mask_a = (d1 <= 0.0) & (d2 <= 0.0)

        bp = p - b
        d3 = torch.sum(ab * bp, dim=-1)
        d4 = torch.sum(ac * bp, dim=-1)
        mask_b = (d3 >= 0.0) & (d4 <= d3)

        vc = d1 * d4 - d3 * d2
        denom_ab = torch.where(torch.abs(d1 - d3) > _EPS, d1 - d3, torch.ones_like(d1))
        edge_ab = a + (d1 / denom_ab).unsqueeze(-1) * ab
        mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)

        cp = p - c
        d5 = torch.sum(ab * cp, dim=-1)
        d6 = torch.sum(ac * cp, dim=-1)
        mask_c = (d6 >= 0.0) & (d5 <= d6)

        vb = d5 * d2 - d1 * d6
        denom_ac = torch.where(torch.abs(d2 - d6) > _EPS, d2 - d6, torch.ones_like(d2))
        edge_ac = a + (d2 / denom_ac).unsqueeze(-1) * ac
        mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)

        va = d3 * d6 - d5 * d4
        denom_bc = torch.where(torch.abs((d4 - d3) + (d5 - d6)) > _EPS, (d4 - d3) + (d5 - d6), torch.ones_like(d4))
        edge_bc = b + (((d4 - d3) / denom_bc).unsqueeze(-1) * (c - b))
        mask_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

        face_denom = torch.where(torch.abs(va + vb + vc) > _EPS, va + vb + vc, torch.ones_like(va))
        face_v = vb / face_denom
        face_w = vc / face_denom
        face_point = a + ab * face_v.unsqueeze(-1) + ac * face_w.unsqueeze(-1)

        closest = a.expand(point_count, triangle_count, 3).clone()
        closest = torch.where(mask_b.unsqueeze(-1), b.expand(point_count, triangle_count, 3), closest)
        closest = torch.where(mask_ab.unsqueeze(-1), edge_ab, closest)
        closest = torch.where(mask_c.unsqueeze(-1), c.expand(point_count, triangle_count, 3), closest)
        closest = torch.where(mask_ac.unsqueeze(-1), edge_ac, closest)
        closest = torch.where(mask_bc.unsqueeze(-1), edge_bc, closest)

        face_mask = ~(mask_a | mask_b | mask_ab | mask_c | mask_ac | mask_bc)
        closest = torch.where(face_mask.unsqueeze(-1), face_point, closest)
        return closest

    @staticmethod
    def _reduce_best(phi_candidates: torch.Tensor, grad_candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_ids = torch.arange(phi_candidates.shape[0], device=phi_candidates.device)
        best_indices = torch.argmin(phi_candidates, dim=1)
        best_phi = phi_candidates.gather(1, best_indices.unsqueeze(-1))
        best_grad = grad_candidates[batch_ids, best_indices]
        return best_phi, best_grad

    def _mesh_closest_distance_and_normal(
        self,
        points_obj: torch.Tensor,
        tensors: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        triangles = tensors["mesh_triangles"]
        face_normals = tensors["mesh_face_normals"]
        if triangles is None or face_normals is None or triangles.numel() == 0:
            return None

        batch_ids = torch.arange(points_obj.shape[0], device=points_obj.device)
        best_distance_sq = torch.full((points_obj.shape[0],), float("inf"), device=points_obj.device, dtype=points_obj.dtype)
        best_closest = torch.zeros((points_obj.shape[0], 3), device=points_obj.device, dtype=points_obj.dtype)
        best_normal = _constant_axis_like(best_closest, axis=2)

        for start in range(0, triangles.shape[0], _MESH_TRIANGLE_CHUNK_SIZE):
            end = min(start + _MESH_TRIANGLE_CHUNK_SIZE, triangles.shape[0])
            triangle_chunk = triangles[start:end]
            normal_chunk = face_normals[start:end]

            closest_chunk = self._closest_points_on_triangles(points_obj, triangle_chunk)
            diff_chunk = points_obj[:, None, :] - closest_chunk
            distance_sq_chunk = torch.sum(diff_chunk * diff_chunk, dim=-1)
            chunk_best_distance_sq, chunk_best_indices = torch.min(distance_sq_chunk, dim=1)
            chunk_best_closest = closest_chunk[batch_ids, chunk_best_indices]
            chunk_best_normal = normal_chunk.index_select(0, chunk_best_indices)

            update_mask = chunk_best_distance_sq < best_distance_sq
            best_distance_sq = torch.where(update_mask, chunk_best_distance_sq, best_distance_sq)
            best_closest = torch.where(update_mask.unsqueeze(-1), chunk_best_closest, best_closest)
            best_normal = torch.where(update_mask.unsqueeze(-1), chunk_best_normal, best_normal)

        direction_obj = points_obj - best_closest
        grad_obj = _normalize_torch_with_fallback(direction_obj, best_normal)
        return torch.sqrt(best_distance_sq.clamp_min(0.0)).unsqueeze(-1), grad_obj

    def _sphere_closest_distance_and_normal(
        self,
        points_obj: torch.Tensor,
        tensors: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        centers = tensors["sphere_centers"]
        radii = tensors["sphere_radii"]
        if centers is None or radii is None or centers.numel() == 0:
            return None

        delta = points_obj[:, None, :] - centers[None, :, :]
        radial_fallback = _constant_axis_like(delta, axis=0)
        outward_normal = _normalize_torch_with_fallback(delta, radial_fallback, default_axis=0)
        closest = centers[None, :, :] + outward_normal * radii.view(1, -1, 1)
        direction_obj = points_obj[:, None, :] - closest
        phi_candidates = torch.linalg.norm(direction_obj, dim=-1)
        grad_candidates = _normalize_torch_with_fallback(direction_obj, outward_normal, default_axis=0)
        return self._reduce_best(phi_candidates, grad_candidates)

    def _box_closest_distance_and_normal(
        self,
        points_obj: torch.Tensor,
        tensors: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        half_extents = tensors["box_half_extents"]
        translations = tensors["box_translations"]
        rotations = tensors["box_rotations"]
        if half_extents is None or translations is None or rotations is None or half_extents.numel() == 0:
            return None

        delta_obj = points_obj[:, None, :] - translations[None, :, :]
        point_local = torch.einsum("gji,bgj->bgi", rotations, delta_obj)
        half_extents_expanded = half_extents[None, :, :]
        clamped = torch.maximum(torch.minimum(point_local, half_extents_expanded), -half_extents_expanded)
        inside = torch.all(torch.abs(point_local) <= half_extents_expanded + _EPS, dim=-1)

        margin = half_extents_expanded - torch.abs(point_local)
        axis = torch.argmin(margin, dim=-1, keepdim=True)
        selected_half_extents = half_extents_expanded.expand(point_local.shape[0], -1, -1).gather(-1, axis)
        selected_coordinate = point_local.gather(-1, axis)
        sign = torch.where(selected_coordinate >= 0.0, torch.ones_like(selected_coordinate), -torch.ones_like(selected_coordinate))

        closest_inside = point_local.clone()
        closest_inside.scatter_(-1, axis, sign * selected_half_extents)

        normal_local_inside = torch.zeros_like(point_local)
        normal_local_inside.scatter_(-1, axis, sign)

        closest_local = torch.where(inside.unsqueeze(-1), closest_inside, clamped)
        direction_local = point_local - closest_local
        normal_local_outside = _normalize_torch_with_fallback(direction_local, normal_local_inside)
        normal_local = torch.where(inside.unsqueeze(-1), normal_local_inside, normal_local_outside)

        phi_candidates = torch.linalg.norm(direction_local, dim=-1)
        grad_candidates = torch.einsum("gij,bgj->bgi", rotations, normal_local)
        grad_candidates = _normalize_torch_with_fallback(grad_candidates, grad_candidates)
        return self._reduce_best(phi_candidates, grad_candidates)

    def _cylinder_closest_distance_and_normal(
        self,
        points_obj: torch.Tensor,
        tensors: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        radii = tensors["cylinder_radii"]
        half_lengths = tensors["cylinder_half_lengths"]
        translations = tensors["cylinder_translations"]
        rotations = tensors["cylinder_rotations"]
        if radii is None or half_lengths is None or translations is None or rotations is None or radii.numel() == 0:
            return None

        delta_obj = points_obj[:, None, :] - translations[None, :, :]
        point_local = torch.einsum("gji,bgj->bgi", rotations, delta_obj)
        radial = point_local[..., :2]
        radial_fallback = _constant_axis_like(radial, axis=0)
        radial_dir = _normalize_torch_with_fallback(radial, radial_fallback, default_axis=0)
        radial_norm = torch.linalg.norm(radial, dim=-1, keepdim=True)

        half_lengths_view = half_lengths.view(1, -1, 1)
        radii_view = radii.view(1, -1, 1)
        z_clamped = torch.maximum(torch.minimum(point_local[..., 2:3], half_lengths_view), -half_lengths_view)
        side_point = torch.cat([radial_dir * radii_view, z_clamped], dim=-1)

        cap_z = torch.where(point_local[..., 2:3] >= 0.0, half_lengths_view, -half_lengths_view)
        cap_radius = torch.minimum(radial_norm, radii_view)
        cap_point = torch.cat([radial_dir * cap_radius, cap_z], dim=-1)

        side_distance = torch.linalg.norm(point_local - side_point, dim=-1)
        cap_distance = torch.linalg.norm(point_local - cap_point, dim=-1)
        use_side = side_distance <= cap_distance
        closest_local = torch.where(use_side.unsqueeze(-1), side_point, cap_point)

        side_normal = torch.cat([radial_dir, torch.zeros_like(point_local[..., 2:3])], dim=-1)
        cap_normal = torch.zeros_like(point_local)
        cap_normal[..., 2] = torch.where(point_local[..., 2] >= 0.0, 1.0, -1.0)
        use_cap_normal = (~use_side) & (radial_norm.squeeze(-1) <= radii.view(1, -1) + _EPS)
        normal_local = torch.where(use_cap_normal.unsqueeze(-1), cap_normal, side_normal)

        phi_candidates = torch.linalg.norm(point_local - closest_local, dim=-1)
        grad_candidates = torch.einsum("gij,bgj->bgi", rotations, normal_local)
        grad_candidates = _normalize_torch_with_fallback(grad_candidates, grad_candidates)
        return self._reduce_best(phi_candidates, grad_candidates)

    def closest_distance_and_normal_obj(self, points_obj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = self._cached_tensors(points_obj.device, points_obj.dtype)
        candidates: list[tuple[torch.Tensor, torch.Tensor]] = []

        mesh_candidate = self._mesh_closest_distance_and_normal(points_obj, tensors)
        if mesh_candidate is not None:
            candidates.append(mesh_candidate)

        sphere_candidate = self._sphere_closest_distance_and_normal(points_obj, tensors)
        if sphere_candidate is not None:
            candidates.append(sphere_candidate)

        box_candidate = self._box_closest_distance_and_normal(points_obj, tensors)
        if box_candidate is not None:
            candidates.append(box_candidate)

        cylinder_candidate = self._cylinder_closest_distance_and_normal(points_obj, tensors)
        if cylinder_candidate is not None:
            candidates.append(cylinder_candidate)

        if not candidates:
            raise RuntimeError("Tensor collision scene has no geometry to query.")

        if len(candidates) == 1:
            return candidates[0]

        phi_candidates = torch.cat([candidate[0] for candidate in candidates], dim=1)
        grad_candidates = torch.stack([candidate[1] for candidate in candidates], dim=1)
        return self._reduce_best(phi_candidates, grad_candidates)


@dataclass(frozen=True)
class SurfaceFeatureResult:
    phi: np.ndarray
    grad_phi: np.ndarray
    v_t: np.ndarray
    v_norm: np.ndarray
    v_tan: np.ndarray

    @property
    def ir_t(self) -> np.ndarray:
        return np.concatenate([self.phi.reshape(1), self.grad_phi, self.v_t, self.v_norm, self.v_tan], axis=0)


class CollisionScene:
    def __init__(self, geometries: Sequence[CollisionGeometry]):
        if not geometries:
            raise ValueError("CollisionScene requires at least one geometry.")
        self._geometries = list(geometries)
        self._tensor_scene = _TensorCollisionScene.from_geometries(self._geometries)

    def compute_surface_feature(
        self,
        query_pos_w: np.ndarray,
        query_lin_vel_w: np.ndarray,
        object_pos_w: np.ndarray,
        object_quat_xyzw: np.ndarray,
    ) -> SurfaceFeatureResult:
        object_rotation = _quat_xyzw_to_rotation_matrix(object_quat_xyzw)
        query_pos_obj = object_rotation.T @ (query_pos_w - object_pos_w)

        best_normal_obj: np.ndarray | None = None
        best_distance = float("inf")
        for geometry in self._geometries:
            _, normal_obj, distance = geometry.closest_point_and_normal(query_pos_obj)
            if distance < best_distance:
                best_normal_obj = normal_obj
                best_distance = distance

        if best_normal_obj is None:
            raise RuntimeError("Failed to compute the closest surface point for the queried body feature.")

        grad_phi_w = _normalize(object_rotation @ best_normal_obj)
        v_t = query_lin_vel_w.astype(np.float64, copy=False)
        v_norm = np.dot(query_lin_vel_w, grad_phi_w) * grad_phi_w
        v_tan = query_lin_vel_w - v_norm
        return SurfaceFeatureResult(
            phi=np.asarray([best_distance], dtype=np.float64),
            grad_phi=grad_phi_w.astype(np.float64),
            v_t=v_t,
            v_norm=v_norm.astype(np.float64),
            v_tan=v_tan.astype(np.float64),
        )

    @torch.no_grad()
    def compute_surface_features_batch(
        self,
        query_pos_w: torch.Tensor,
        query_lin_vel_w: torch.Tensor,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        object_rotation = quaternion_to_matrix(object_quat_w, w_last=True)
        object_rotation_inv = object_rotation.transpose(1, 2)
        query_pos_obj = torch.bmm(object_rotation_inv, (query_pos_w - object_pos_w).unsqueeze(-1)).squeeze(-1)

        # phi: [batch, 1], grad_phi_obj: [batch, 3]
        phi, grad_phi_obj = self._tensor_scene.closest_distance_and_normal_obj(query_pos_obj)
        grad_phi_w = torch.bmm(object_rotation, grad_phi_obj.unsqueeze(-1)).squeeze(-1)
        grad_phi_w = _normalize_torch_with_fallback(grad_phi_w, grad_phi_w)

        # v_t / v_norm / v_tan: [batch, 3]
        v_t = query_lin_vel_w
        v_norm = torch.sum(v_t * grad_phi_w, dim=-1, keepdim=True) * grad_phi_w
        v_tan = v_t - v_norm

        # ir_t: [batch, 13] = [phi(1), grad_phi(3), v_t(3), v_norm(3), v_tan(3)]
        ir_t = torch.cat([phi, grad_phi_w, v_t, v_norm, v_tan], dim=-1)
        return {
            "phi": phi,
            "grad_phi": grad_phi_w,
            "v_t": v_t,
            "v_norm": v_norm,
            "v_tan": v_tan,
            "ir_t": ir_t,
        }


class SurfaceFeatureComputer:
    def __init__(self, scenes_by_key: dict[str, CollisionScene]):
        if not scenes_by_key:
            raise ValueError("SurfaceFeatureComputer requires at least one object scene.")
        self._scenes_by_key = scenes_by_key
        self._default_key = next(iter(scenes_by_key))

    @classmethod
    def from_object_config(
        cls,
        object_cfg: ObjectConfig,
        mesh_mode: str = "full",
    ) -> SurfaceFeatureComputer:
        scene_paths = dict(object_cfg.object_urdf_name_to_path or {})
        if not scene_paths and object_cfg.object_urdf_path:
            scene_paths["default"] = resolve_data_file_path(object_cfg.object_urdf_path)
        if not scene_paths:
            raise ValueError("No object URDF path is configured for live surface features.")

        scenes = {key: CollisionScene(_load_collision_geometries(path, mesh_mode=mesh_mode)) for key, path in scene_paths.items()}
        return cls(scenes)

    def _scene_for_key(self, object_key: str | None) -> CollisionScene:
        if object_key is not None and object_key in self._scenes_by_key:
            return self._scenes_by_key[object_key]
        return self._scenes_by_key[self._default_key]

    @torch.no_grad()
    def compute_batch(
        self,
        body_pos_w: torch.Tensor | None = None,
        body_lin_vel_w: torch.Tensor | None = None,
        object_pos_w: torch.Tensor | None = None,
        object_quat_w: torch.Tensor | None = None,
        object_keys: Sequence[str | None] | None = None,
        **legacy_kwargs,
    ) -> dict[str, torch.Tensor]:
        if "pelvis_pos_w" in legacy_kwargs:
            if body_pos_w is not None:
                raise TypeError("Pass only one of body_pos_w or pelvis_pos_w.")
            body_pos_w = legacy_kwargs.pop("pelvis_pos_w")
        if "pelvis_lin_vel_w" in legacy_kwargs:
            if body_lin_vel_w is not None:
                raise TypeError("Pass only one of body_lin_vel_w or pelvis_lin_vel_w.")
            body_lin_vel_w = legacy_kwargs.pop("pelvis_lin_vel_w")
        if legacy_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(legacy_kwargs)}")
        if body_pos_w is None or body_lin_vel_w is None or object_pos_w is None or object_quat_w is None:
            raise TypeError("body_pos_w, body_lin_vel_w, object_pos_w, and object_quat_w are required.")

        num_envs = int(body_pos_w.shape[0])
        if object_keys is None:
            object_keys = [self._default_key] * num_envs
        if len(object_keys) != num_envs:
            raise ValueError(f"Expected {num_envs} object keys, got {len(object_keys)}")
    
        if num_envs == 0:
            return {
                "phi": torch.empty((0, 1), device=body_pos_w.device, dtype=body_pos_w.dtype),
                "grad_phi": torch.empty((0, 3), device=body_pos_w.device, dtype=body_pos_w.dtype),
                "v_t": torch.empty((0, 3), device=body_pos_w.device, dtype=body_pos_w.dtype),
                "v_norm": torch.empty((0, 3), device=body_pos_w.device, dtype=body_pos_w.dtype),
                "v_tan": torch.empty((0, 3), device=body_pos_w.device, dtype=body_pos_w.dtype),
                "ir_t": torch.empty((0, 13), device=body_pos_w.device, dtype=body_pos_w.dtype),
            }

        device = body_pos_w.device
        dtype = body_pos_w.dtype

        # [num_envs, 1] / [num_envs, 3] / [num_envs, 13]
        phi = torch.empty((num_envs, 1), device=device, dtype=dtype)
        grad_phi = torch.empty((num_envs, 3), device=device, dtype=dtype)
        v_t = torch.empty((num_envs, 3), device=device, dtype=dtype)
        v_norm = torch.empty((num_envs, 3), device=device, dtype=dtype)
        v_tan = torch.empty((num_envs, 3), device=device, dtype=dtype)
        ir_t = torch.empty((num_envs, 13), device=device, dtype=dtype)

        grouped_indices: dict[str, list[int]] = {}
        for env_index, object_key in enumerate(object_keys):
            resolved_key = object_key if object_key is not None and object_key in self._scenes_by_key else self._default_key
            grouped_indices.setdefault(resolved_key, []).append(env_index)

        for object_key, env_indices in grouped_indices.items():
            scene = self._scene_for_key(object_key)
            env_index_tensor = torch.as_tensor(env_indices, device=device, dtype=torch.long)
            features = scene.compute_surface_features_batch(
                query_pos_w=body_pos_w.index_select(0, env_index_tensor),
                query_lin_vel_w=body_lin_vel_w.index_select(0, env_index_tensor),
                object_pos_w=object_pos_w.index_select(0, env_index_tensor),
                object_quat_w=object_quat_w.index_select(0, env_index_tensor),
            )
            phi.index_copy_(0, env_index_tensor, features["phi"])
            grad_phi.index_copy_(0, env_index_tensor, features["grad_phi"])
            v_t.index_copy_(0, env_index_tensor, features["v_t"])
            v_norm.index_copy_(0, env_index_tensor, features["v_norm"])
            v_tan.index_copy_(0, env_index_tensor, features["v_tan"])
            ir_t.index_copy_(0, env_index_tensor, features["ir_t"])

        return {
            "phi": phi,
            "grad_phi": grad_phi,
            "v_t": v_t,
            "v_norm": v_norm,
            "v_tan": v_tan,
            "ir_t": ir_t,
        }


# Legacy alias kept for compatibility with older imports.
PelvisSurfaceFeatureComputer = SurfaceFeatureComputer
