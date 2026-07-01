#!/usr/bin/env python3
"""Convert SUGAR n-OMOMO data into Holosoma train and train_r1 layouts.

The converter writes:

* train/motions/<object>/<name>_original.npz
* train/objects/<object>/<object>.obj
* train/objects/<object>/<object>.urdf
* train/objects/<object>/sample_points.npy
* train_r1/motions/<object>/<name>_original.npz
* train_r1/objects/<object>/<object>.obj
* train_r1/objects/<object>/<object>.urdf
* train_r1/objects/<object>/sample_points.npy

SUGAR robot files in this folder store G1 29-DoF joint positions.  The G1
output preserves that source robot in MuJoCo order:
[root xyz, root qwxyz, 29 G1 joints, object xyz, object qwxyz].

The optional R1 output maps common G1/R1 joints into MuJoCo order:
[root xyz, root qwxyz, 26 R1 joints, object xyz, object qwxyz].
R1-only head joints are left at zero.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is present in the project env.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]
SUGAR_ROOT = Path(__file__).resolve().parent / "SUGAR"
DEFAULT_MOTIONS_ROOT = REPO_ROOT / "train_r1" / "motions"
DEFAULT_OBJECTS_ROOT = REPO_ROOT / "train_r1" / "objects"
DEFAULT_G1_MOTIONS_ROOT = REPO_ROOT / "train" / "motions"
DEFAULT_G1_OBJECTS_ROOT = REPO_ROOT / "train" / "objects"
DEFAULT_RETARGETING_MODELS_ROOT = (
    REPO_ROOT / "src" / "holosoma_retargeting" / "holosoma_retargeting" / "models"
)
DEFAULT_R1_ROBOT_XML = DEFAULT_RETARGETING_MODELS_ROOT / "r1" / "r1_26dof.xml"
DEFAULT_SAMPLE_COUNT = 340

G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

R1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "head_pitch_joint",
    "head_yaw_joint",
]

OBJECT_PARM_DEFAULTS: dict[str, Any] = {
    "static_friction_range": [0.3, 1.6],
    "dynamic_friction_range": [0.3, 1.2],
    "restitution_range": [0.0, 0.5],
    "mass_distribution_params": [1.0, 4.0],
    "inertia_distribution_params_dict": {
        "Ixx": [0.5, 1.5],
        "Iyy": [0.5, 1.5],
        "Izz": [0.5, 1.5],
        "Ixy": [0.5, 1.5],
        "Iyz": [0.5, 1.5],
        "Ixz": [0.5, 1.5],
    },
}

OBJECT_PARM_OVERRIDES: dict[str, dict[str, list[float]]] = {
    "sugarbottle": {
        "static_friction_range": [0.3, 1.6],
        "dynamic_friction_range": [0.3, 1.2],
        "restitution_range": [0.0, 0.5],
        "mass_distribution_params": [0.3, 1.5],
    },
    "sugarchair": {
        "static_friction_range": [0.3, 2.0],
        "dynamic_friction_range": [0.3, 1.2],
        "restitution_range": [0.0, 0.5],
        "mass_distribution_params": [1.0, 2.0],
    },
}

SOURCE_TO_OUTPUT_OBJECT_NAME = {
    "bottle": "sugarbottle",
    "chair": "sugarchair",
}


@dataclass(frozen=True)
class ConvertedMotion:
    source: Path
    output: Path
    object_name: str
    frames: int
    fps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sugar-root", type=Path, default=SUGAR_ROOT)
    parser.add_argument("--motions-root", type=Path, default=DEFAULT_MOTIONS_ROOT)
    parser.add_argument("--objects-root", type=Path, default=DEFAULT_OBJECTS_ROOT)
    parser.add_argument("--g1-motions-root", type=Path, default=DEFAULT_G1_MOTIONS_ROOT)
    parser.add_argument("--g1-objects-root", type=Path, default=DEFAULT_G1_OBJECTS_ROOT)
    parser.add_argument("--retargeting-models-root", type=Path, default=DEFAULT_RETARGETING_MODELS_ROOT)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_R1_ROBOT_XML)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--ground-clearance", type=float, default=0.0)
    parser.add_argument("--ground-align-mode", choices=("global", "first", "per-frame"), default="per-frame")
    parser.add_argument("--no-ground-align", action="store_true")
    parser.add_argument("--root-body-index", type=int, default=0)
    parser.add_argument("--output-prefix", default="sugar")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-r1", action="store_true")
    parser.add_argument("--skip-g1", action="store_true")
    parser.add_argument("--skip-motions", action="store_true")
    parser.add_argument("--skip-objects", action="store_true")
    parser.add_argument("--skip-retargeting-models", action="store_true")
    parser.add_argument("--skip-object-parm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def discover_task_dirs(sugar_root: Path) -> list[Path]:
    return sorted(path for path in sugar_root.iterdir() if (path / "data_000" / "robot_50hz.npz").exists())


def discover_object_names(sugar_root: Path) -> list[str]:
    names: list[str] = []
    for path in sorted(sugar_root.iterdir()):
        if not path.is_dir() or (path / "data_000").exists():
            continue
        if (path / "config.yaml").exists() or (path / "obj_aligned.obj").exists() or (path / "obj_aligned.usd").exists():
            names.append(path.name)
    return names


def infer_object_name(task_dir: Path, object_names: list[str]) -> str:
    task_name = task_dir.name.lower()
    matches = [name for name in object_names if name.lower() in task_name]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return max(matches, key=len)
    raise ValueError(f"Could not infer object for task '{task_dir.name}' from objects {object_names}.")


def output_object_name(source_object_name: str) -> str:
    return SOURCE_TO_OUTPUT_OBJECT_NAME.get(source_object_name, source_object_name)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_object_motion(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = NumpyCompatUnpickler(f).load()
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a dict.")
    return data


class NumpyCompatUnpickler(pickle.Unpickler):
    """Load NumPy 2.x pickles in environments that only expose numpy.core."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core":
            module = "numpy.core"
        elif module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-12)


def matrix_to_quat_wxyz(rot_mats: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(np.asarray(rot_mats, dtype=np.float64)).as_quat()
    return normalize_quat_wxyz(quat_xyzw[:, [3, 0, 1, 2]])


def fps_value(robot_data: dict[str, np.ndarray]) -> int:
    fps = robot_data.get("fps")
    if fps is None:
        return 50
    return int(np.asarray(fps).reshape(-1)[0])


def map_g1_to_r1(g1_joint_pos: np.ndarray) -> np.ndarray:
    g1_index = {name: idx for idx, name in enumerate(G1_JOINT_NAMES)}
    r1 = np.zeros((g1_joint_pos.shape[0], len(R1_JOINT_NAMES)), dtype=np.float64)
    for r1_idx, name in enumerate(R1_JOINT_NAMES):
        if name in g1_index:
            r1[:, r1_idx] = g1_joint_pos[:, g1_index[name]]
    return r1


def ground_align_qpos(
    qpos: np.ndarray,
    robot_xml: Path,
    ground_clearance: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift root z so the lowest R1 robot body reaches ground_clearance."""
    try:
        import mujoco  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("Ground alignment requires mujoco. Pass --no-ground-align to skip it.") from exc

    if not robot_xml.exists():
        raise FileNotFoundError(f"Robot XML for ground alignment does not exist: {robot_xml}")

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    robot_nq = model.nq
    if qpos.shape[1] < robot_nq:
        raise ValueError(f"qpos width {qpos.shape[1]} is smaller than robot nq {robot_nq} from {robot_xml}")

    min_z_by_frame: list[float] = []
    for frame in qpos:
        data.qpos[:] = frame[:robot_nq]
        mujoco.mj_forward(model, data)
        # Exclude world body at index 0.
        min_z_by_frame.append(float(np.min(data.xpos[1:, 2])))

    min_z = np.asarray(min_z_by_frame, dtype=np.float64)
    if mode == "global":
        z_shift = np.full(qpos.shape[0], ground_clearance - float(np.min(min_z)), dtype=np.float64)
    elif mode == "first":
        z_shift = np.full(qpos.shape[0], ground_clearance - float(min_z[0]), dtype=np.float64)
    elif mode == "per-frame":
        z_shift = ground_clearance - min_z
    else:
        raise ValueError(f"Unsupported ground alignment mode: {mode}")

    aligned = qpos.copy()
    aligned[:, 2] += z_shift
    return aligned, z_shift


def extract_r1_root_and_dofs(
    robot_data: dict[str, np.ndarray],
    root_body_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    joint_pos = np.asarray(robot_data["joint_pos"], dtype=np.float64)
    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must be 2D, got {joint_pos.shape}.")

    if joint_pos.shape[1] == 7 + len(R1_JOINT_NAMES):
        return joint_pos[:, :7], joint_pos[:, 7:]
    if joint_pos.shape[1] == 7 + len(G1_JOINT_NAMES):
        return joint_pos[:, :7], map_g1_to_r1(joint_pos[:, 7:])

    if "body_pos_w" not in robot_data or "body_quat_w" not in robot_data:
        raise ValueError("joint_pos does not include root pose, and body_pos_w/body_quat_w are missing.")

    root_pos = np.asarray(robot_data["body_pos_w"], dtype=np.float64)[:, root_body_index]
    root_quat = normalize_quat_wxyz(np.asarray(robot_data["body_quat_w"], dtype=np.float64)[:, root_body_index])
    root_qpos = np.concatenate([root_pos, root_quat], axis=1)

    if joint_pos.shape[1] == len(R1_JOINT_NAMES):
        return root_qpos, joint_pos
    if joint_pos.shape[1] == len(G1_JOINT_NAMES):
        return root_qpos, map_g1_to_r1(joint_pos)

    raise ValueError(
        f"Unsupported joint_pos width {joint_pos.shape[1]}; expected "
        f"{len(G1_JOINT_NAMES)}, {len(R1_JOINT_NAMES)}, "
        f"{7 + len(G1_JOINT_NAMES)}, or {7 + len(R1_JOINT_NAMES)}."
    )


def extract_g1_root_and_dofs(
    robot_data: dict[str, np.ndarray],
    root_body_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    joint_pos = np.asarray(robot_data["joint_pos"], dtype=np.float64)
    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must be 2D, got {joint_pos.shape}.")

    if joint_pos.shape[1] == 7 + len(G1_JOINT_NAMES):
        root_qpos = joint_pos[:, :7].copy()
        root_qpos[:, 3:7] = normalize_quat_wxyz(root_qpos[:, 3:7])
        return root_qpos, joint_pos[:, 7:]

    if "body_pos_w" not in robot_data or "body_quat_w" not in robot_data:
        raise ValueError("joint_pos does not include root pose, and body_pos_w/body_quat_w are missing.")

    root_pos = np.asarray(robot_data["body_pos_w"], dtype=np.float64)[:, root_body_index]
    root_quat = normalize_quat_wxyz(np.asarray(robot_data["body_quat_w"], dtype=np.float64)[:, root_body_index])
    root_qpos = np.concatenate([root_pos, root_quat], axis=1)

    if joint_pos.shape[1] == len(G1_JOINT_NAMES):
        return root_qpos, joint_pos

    raise ValueError(
        f"Unsupported G1 joint_pos width {joint_pos.shape[1]}; expected "
        f"{len(G1_JOINT_NAMES)} or {7 + len(G1_JOINT_NAMES)}."
    )


def convert_motion(
    task_dir: Path,
    source_object_name: str,
    object_name: str,
    motions_root: Path,
    output_prefix: str,
    root_body_index: int,
    robot_xml: Path,
    ground_clearance: float,
    ground_align_mode: str,
    ground_align: bool,
    overwrite: bool,
    dry_run: bool,
) -> ConvertedMotion:
    data_dir = task_dir / "data_000"
    robot_data = load_npz(data_dir / "robot_50hz.npz")
    object_data = load_object_motion(data_dir / "obj_motion_global_50hz.pkl")

    root_qpos, r1_dofs = extract_r1_root_and_dofs(robot_data, root_body_index=root_body_index)
    object_pos = np.asarray(object_data["obj_trans"], dtype=np.float64)
    object_quat = matrix_to_quat_wxyz(np.asarray(object_data["obj_rot"], dtype=np.float64))

    frames = min(root_qpos.shape[0], r1_dofs.shape[0], object_pos.shape[0], object_quat.shape[0])
    qpos = np.concatenate(
        [
            root_qpos[:frames],
            r1_dofs[:frames],
            object_pos[:frames],
            object_quat[:frames],
        ],
        axis=1,
    )
    root_z_shift = np.zeros(qpos.shape[0], dtype=np.float64)
    if ground_align:
        qpos, root_z_shift = ground_align_qpos(qpos, robot_xml, ground_clearance, ground_align_mode)

    contact_path = data_dir / "contact_labels_50hz.npy"
    extra_arrays: dict[str, np.ndarray] = {}
    if contact_path.exists():
        extra_arrays["sugar_contact_label"] = np.asarray(np.load(contact_path, allow_pickle=True)[:frames], dtype=bool)
    if "obj_scale" in object_data:
        extra_arrays["object_scale"] = np.asarray(object_data["obj_scale"], dtype=np.float32)

    output_name = f"{output_prefix}_{object_name}_{sanitize_name(task_dir.name)}_data000_original.npz"
    output = motions_root / object_name / output_name
    if output.exists() and not overwrite:
        return ConvertedMotion(task_dir, output, object_name, frames, fps_value(robot_data))

    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            qpos=qpos,
            fps=fps_value(robot_data),
            cost=np.asarray(0.0, dtype=np.float64),
            source_dataset=np.asarray("SUGAR"),
            source_task=np.asarray(task_dir.name),
            source_object_name=np.asarray(source_object_name),
            target_object_name=np.asarray(object_name),
            source_robot_joint_names=np.asarray(G1_JOINT_NAMES),
            target_robot_joint_names=np.asarray(R1_JOINT_NAMES),
            root_z_shift=np.asarray(root_z_shift, dtype=np.float64),
            ground_clearance=np.asarray(ground_clearance, dtype=np.float64),
            ground_align_mode=np.asarray(ground_align_mode if ground_align else "none"),
            **extra_arrays,
        )
    return ConvertedMotion(task_dir, output, object_name, frames, fps_value(robot_data))


def convert_motion_g1(
    task_dir: Path,
    source_object_name: str,
    object_name: str,
    motions_root: Path,
    output_prefix: str,
    root_body_index: int,
    overwrite: bool,
    dry_run: bool,
) -> ConvertedMotion:
    data_dir = task_dir / "data_000"
    robot_data = load_npz(data_dir / "robot_50hz.npz")
    object_data = load_object_motion(data_dir / "obj_motion_global_50hz.pkl")

    root_qpos, g1_dofs = extract_g1_root_and_dofs(robot_data, root_body_index=root_body_index)
    object_pos = np.asarray(object_data["obj_trans"], dtype=np.float64)
    object_quat = matrix_to_quat_wxyz(np.asarray(object_data["obj_rot"], dtype=np.float64))

    frames = min(root_qpos.shape[0], g1_dofs.shape[0], object_pos.shape[0], object_quat.shape[0])
    qpos = np.concatenate(
        [
            root_qpos[:frames],
            g1_dofs[:frames],
            object_pos[:frames],
            object_quat[:frames],
        ],
        axis=1,
    )

    contact_path = data_dir / "contact_labels_50hz.npy"
    extra_arrays: dict[str, np.ndarray] = {}
    if contact_path.exists():
        extra_arrays["sugar_contact_label"] = np.asarray(np.load(contact_path, allow_pickle=True)[:frames], dtype=bool)
    if "obj_scale" in object_data:
        extra_arrays["object_scale"] = np.asarray(object_data["obj_scale"], dtype=np.float32)

    output_name = f"{output_prefix}_{object_name}_{sanitize_name(task_dir.name)}_data000_original.npz"
    output = motions_root / object_name / output_name
    if output.exists() and not overwrite:
        return ConvertedMotion(task_dir, output, object_name, frames, fps_value(robot_data))

    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            qpos=qpos,
            fps=fps_value(robot_data),
            cost=np.asarray(0.0, dtype=np.float64),
            source_dataset=np.asarray("SUGAR"),
            source_task=np.asarray(task_dir.name),
            source_object_name=np.asarray(source_object_name),
            target_object_name=np.asarray(object_name),
            source_robot_joint_names=np.asarray(G1_JOINT_NAMES),
            target_robot_joint_names=np.asarray(G1_JOINT_NAMES),
            target_robot=np.asarray("g1"),
            **extra_arrays,
        )
    return ConvertedMotion(task_dir, output, object_name, frames, fps_value(robot_data))


def resolve_source_obj(sugar_root: Path, object_name: str) -> Path:
    candidates = [
        sugar_root / object_name / "obj_aligned.obj",
        sugar_root / object_name / f"{object_name}.obj",
        sugar_root.parent / "RESMIMIC" / object_name / f"{object_name}.obj",
    ]
    if object_name == "chair":
        candidates.append(sugar_root.parent / "RESMIMIC" / "chair" / "chair.obj")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find an OBJ mesh for '{object_name}'. Searched:\n  {searched}")


def parse_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                face: list[int] = []
                for token in line.split()[1:]:
                    raw = token.split("/")[0]
                    if not raw:
                        continue
                    idx = int(raw)
                    face.append(idx - 1 if idx > 0 else len(vertices) + idx)
                for i in range(1, len(face) - 1):
                    faces.append([face[0], face[i], face[i + 1]])

    if not vertices:
        raise ValueError(f"No vertices found in OBJ: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def sample_surface_points(obj_path: Path, count: int, seed: int = 0) -> np.ndarray:
    vertices, faces = parse_obj_mesh(obj_path)
    rng = np.random.default_rng(seed)
    if faces.size == 0:
        indices = rng.choice(vertices.shape[0], size=count, replace=vertices.shape[0] < count)
        return vertices[indices]

    tris = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    valid = areas > 1e-12
    if not np.any(valid):
        indices = rng.choice(vertices.shape[0], size=count, replace=vertices.shape[0] < count)
        return vertices[indices]

    tris = tris[valid]
    areas = areas[valid]
    tri_indices = rng.choice(tris.shape[0], size=count, p=areas / areas.sum())
    chosen = tris[tri_indices]

    u = rng.random(count)
    v = rng.random(count)
    sqrt_u = np.sqrt(u)
    return (
        (1.0 - sqrt_u)[:, None] * chosen[:, 0]
        + (sqrt_u * (1.0 - v))[:, None] * chosen[:, 1]
        + (sqrt_u * v)[:, None] * chosen[:, 2]
    )


def _insert_attr_if_missing(tag: str, name: str, value: str) -> str:
    if re.search(rf"\b{re.escape(name)}=", tag):
        return tag
    return tag[:-2].rstrip() + f' {name}="{value}"/>'


def _add_ground_contact_properties(xml: str) -> str:
    match = re.search(r'<geom\s+name="ground"[^>]*/>', xml)
    if match is None:
        return xml

    tag = match.group(0)
    tag = _insert_attr_if_missing(tag, "quat", "1 0 0 0")
    tag = _insert_attr_if_missing(tag, "condim", "1")
    tag = _insert_attr_if_missing(tag, "conaffinity", "15")
    return xml[: match.start()] + tag + xml[match.end() :]


def _object_mesh_rel_path(robot_dir: Path, obj_file: Path, base_xml: str) -> str:
    meshdir_match = re.search(r'<compiler\b[^>]*\bmeshdir="([^"]+)"', base_xml)
    meshdir = meshdir_match.group(1) if meshdir_match else "."
    mesh_base_dir = (robot_dir / meshdir).resolve()
    return os.path.relpath(obj_file.resolve(), mesh_base_dir).replace(os.sep, "/")


def generate_object_xml(base_xml: str, object_name: str, object_rel_path: str) -> str:
    xml = base_xml
    mesh_name = f"{object_name}_mesh"
    if f'name="{mesh_name}"' not in xml:
        mesh_line = f'    <mesh name="{mesh_name}" file="{object_rel_path}" scale="1 1 1"/>\n'
        asset_close = xml.find("  </asset>")
        if asset_close == -1:
            raise ValueError("Could not find </asset> in base XML.")
        xml = xml[:asset_close] + "\n" + mesh_line + xml[asset_close:]

    xml = _add_ground_contact_properties(xml)
    body_name = f"{object_name}_link"
    if f'name="{body_name}"' not in xml:
        object_body = f"""
    <body name="{body_name}">
        <freejoint/>
        <inertial pos="0 0 0" mass="0.1" diaginertia="0.002 0.002 0.002"/>
        <geom name="{object_name}" type="mesh" mesh="{mesh_name}"
                contype="1" conaffinity="1"
                pos="0 0 0" quat="1 0 0 0"
                rgba="0.7 0.8 0.9 0.7"
                friction="0.9 0.5 0.5"
                solref="0.02 1"
                solimp="0.9 0.95 0.001"/>
    </body>

    <light name="sun" pos="0 0 5" dir="0 0 -1" directional="true"
         diffuse="1 1 1" ambient="0.2 0.2 0.2" specular="0.2 0.2 0.2"
         castshadow="true"/>
"""
        xml = xml.replace("  </worldbody>", object_body + "  </worldbody>")
    return xml


def write_object_assets(
    source_obj: Path,
    objects_root: Path,
    object_name: str,
    sample_count: int,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    object_dir = objects_root / object_name
    dest_obj = object_dir / f"{object_name}.obj"
    dest_urdf = object_dir / f"{object_name}.urdf"
    dest_samples = object_dir / "sample_points.npy"

    if dry_run:
        return object_dir

    object_dir.mkdir(parents=True, exist_ok=True)
    if overwrite or not dest_obj.exists():
        shutil.copy2(source_obj, dest_obj)
    if overwrite or not dest_urdf.exists():
        dest_urdf.write_text(urdf_text(object_name))
    if overwrite or not dest_samples.exists():
        sample_points = sample_surface_points(dest_obj, sample_count)
        np.save(dest_samples, sample_points)
    return object_dir


def write_robot_scene_xml(
    retargeting_models_root: Path,
    robot_dir_name: str,
    base_xml_name: str,
    output_xml_name: str,
    object_name: str,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    robot_dir = retargeting_models_root / robot_dir_name
    base_xml_path = robot_dir / base_xml_name
    obj_file = retargeting_models_root / "objects" / object_name / f"{object_name}.obj"
    output_path = robot_dir / output_xml_name

    if output_path.exists() and not overwrite:
        return output_path
    if dry_run:
        return output_path
    if not base_xml_path.exists():
        raise FileNotFoundError(f"Base {robot_dir_name} XML not found: {base_xml_path}")
    if not obj_file.exists():
        raise FileNotFoundError(f"Object mesh not found for {robot_dir_name} XML generation: {obj_file}")

    base_xml = base_xml_path.read_text()
    obj_rel_path = _object_mesh_rel_path(robot_dir, obj_file, base_xml)
    output_path.write_text(generate_object_xml(base_xml, object_name, obj_rel_path))
    return output_path


def write_r1_scene_xml(
    retargeting_models_root: Path,
    object_name: str,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    return write_robot_scene_xml(
        retargeting_models_root=retargeting_models_root,
        robot_dir_name="r1",
        base_xml_name="r1_26dof.xml",
        output_xml_name=f"r1_26dof_w_{object_name}.xml",
        object_name=object_name,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def write_g1_scene_xml(
    retargeting_models_root: Path,
    object_name: str,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    return write_robot_scene_xml(
        retargeting_models_root=retargeting_models_root,
        robot_dir_name="g1",
        base_xml_name="g1_29dof.xml",
        output_xml_name=f"g1_29dof_w_{object_name}.xml",
        object_name=object_name,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def urdf_text(object_name: str) -> str:
    mesh_path = f"objects/{object_name}/{object_name}.obj"
    return f"""<?xml version="1.0" ?>
<robot name="{object_name}.urdf">
  <dynamics damping="0.5" friction="0.9"/>
  <link name="baseLink">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
    <contact>
      <lateral_friction value="0.9"/>
      <rolling_friction value="0.5"/>
      <stiffness value="30000"/>
      <damping value="1000"/>
    </contact>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="1.0 1.0 1.0"/>
      </geometry>
      <material name="mat">
        <color rgba="0.7 0.8 0.9 1"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="1.0 1.0 1.0"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def convert_object(
    sugar_root: Path,
    source_object_name: str,
    object_name: str,
    objects_root: Path,
    g1_objects_root: Path,
    retargeting_models_root: Path,
    sample_count: int,
    overwrite: bool,
    dry_run: bool,
    skip_r1: bool,
    skip_g1: bool,
    skip_retargeting_models: bool,
) -> list[Path]:
    source_obj = resolve_source_obj(sugar_root, source_object_name)
    object_dirs: list[Path] = []
    if not skip_r1:
        object_dirs.append(write_object_assets(source_obj, objects_root, object_name, sample_count, overwrite, dry_run))
    if not skip_g1:
        object_dirs.append(write_object_assets(source_obj, g1_objects_root, object_name, sample_count, overwrite, dry_run))

    if not skip_retargeting_models:
        write_object_assets(
            source_obj,
            retargeting_models_root / "objects",
            object_name,
            sample_count,
            overwrite,
            dry_run,
        )
        if not skip_r1:
            write_r1_scene_xml(retargeting_models_root, object_name, overwrite, dry_run)
        if not skip_g1:
            write_g1_scene_xml(retargeting_models_root, object_name, overwrite, dry_run)
    return object_dirs


def update_object_parm(objects_root: Path, object_names: list[str], overwrite: bool, dry_run: bool) -> None:
    if yaml is None:
        print("[WARN] PyYAML is unavailable; skipping objects_parm.yaml update.")
        return

    path = objects_root / "objects_parm.yaml"
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
    else:
        data = {}

    data.setdefault("defaults", OBJECT_PARM_DEFAULTS)
    objects = data.setdefault("objects", {})
    changed = False
    for object_name in object_names:
        if object_name in objects and not overwrite:
            continue
        objects[object_name] = OBJECT_PARM_OVERRIDES.get(object_name, {})
        changed = True

    if changed and not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite and "objects:" in path.read_text():
            blocks = []
            for object_name in object_names:
                if object_name in (yaml.safe_load(path.read_text()) or {}).get("objects", {}):
                    continue
                params = OBJECT_PARM_OVERRIDES.get(object_name, {})
                lines = [f"  {object_name}:"]
                for key, value in params.items():
                    lines.append(f"    {key}: {value}")
                blocks.append("\n".join(lines))
            if blocks:
                original = path.read_text().rstrip()
                path.write_text(original + "\n\n" + "\n\n".join(blocks) + "\n")
            return
        path.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> None:
    args = parse_args()
    sugar_root = args.sugar_root.resolve()
    motions_root = args.motions_root.resolve()
    objects_root = args.objects_root.resolve()
    g1_motions_root = args.g1_motions_root.resolve()
    g1_objects_root = args.g1_objects_root.resolve()
    retargeting_models_root = args.retargeting_models_root.resolve()
    robot_xml = args.robot_xml.resolve()

    if args.skip_r1 and args.skip_g1:
        raise ValueError("Both --skip-r1 and --skip-g1 were passed; nothing would be written.")

    if not sugar_root.exists():
        raise FileNotFoundError(f"SUGAR root does not exist: {sugar_root}")

    object_names = discover_object_names(sugar_root)
    task_dirs = discover_task_dirs(sugar_root)
    if not task_dirs:
        raise FileNotFoundError(f"No SUGAR task folders with data_000/robot_50hz.npz found under {sugar_root}")

    converted_objects: set[str] = set()
    converted_motions: list[ConvertedMotion] = []
    for task_dir in task_dirs:
        source_object_name = infer_object_name(task_dir, object_names)
        object_name = output_object_name(source_object_name)
        converted_objects.add(object_name)
        if not args.skip_motions:
            if not args.skip_r1:
                converted_motions.append(
                    convert_motion(
                        task_dir=task_dir,
                        source_object_name=source_object_name,
                        object_name=object_name,
                        motions_root=motions_root,
                        output_prefix=args.output_prefix,
                        root_body_index=args.root_body_index,
                        robot_xml=robot_xml,
                        ground_clearance=args.ground_clearance,
                        ground_align_mode=args.ground_align_mode,
                        ground_align=not args.no_ground_align,
                        overwrite=args.overwrite,
                        dry_run=args.dry_run,
                    )
                )
            if not args.skip_g1:
                converted_motions.append(
                    convert_motion_g1(
                        task_dir=task_dir,
                        source_object_name=source_object_name,
                        object_name=object_name,
                        motions_root=g1_motions_root,
                        output_prefix=args.output_prefix,
                        root_body_index=args.root_body_index,
                        overwrite=args.overwrite,
                        dry_run=args.dry_run,
                    )
                )

    if not args.skip_objects:
        for source_object_name in sorted({infer_object_name(task_dir, object_names) for task_dir in task_dirs}):
            object_name = output_object_name(source_object_name)
            object_dirs = convert_object(
                sugar_root=sugar_root,
                source_object_name=source_object_name,
                object_name=object_name,
                objects_root=objects_root,
                g1_objects_root=g1_objects_root,
                retargeting_models_root=retargeting_models_root,
                sample_count=args.sample_count,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                skip_r1=args.skip_r1,
                skip_g1=args.skip_g1,
                skip_retargeting_models=args.skip_retargeting_models,
            )
            print(f"[object] {object_name}: {', '.join(str(path) for path in object_dirs)}")

    if not args.skip_object_parm:
        if not args.skip_r1:
            update_object_parm(objects_root, sorted(converted_objects), overwrite=args.overwrite, dry_run=args.dry_run)
        g1_object_parm = g1_objects_root / "objects_parm.yaml"
        if not args.skip_g1 and g1_object_parm.exists():
            update_object_parm(g1_objects_root, sorted(converted_objects), overwrite=args.overwrite, dry_run=args.dry_run)

    for item in converted_motions:
        print(f"[motion] {item.source.name} -> {item.output} ({item.frames} frames @ {item.fps} Hz)")

    print(
        f"Done. Converted {len(converted_motions)} motions and {len(converted_objects) if not args.skip_objects else 0} objects."
    )


if __name__ == "__main__":
    main()
