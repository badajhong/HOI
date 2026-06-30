#!/usr/bin/env python3
"""Generate frame-level object-contact labels for HOI motion npz files.

The default mode labels intended contact from SMPLH human joints, then maps the
contact to robot body names. This is usually the right target for training:
the label says where the robot should contact the object, independent of
whether the retargeted robot currently succeeds.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import (  # noqa: E402,I001
    DEMO_JOINTS_REGISTRY,
    MotionDataConfig,
)


DEFAULT_OBJECT_ROOTS = (
    Path("train_r1/objects"),
    Path("src/holosoma_retargeting/holosoma_retargeting/models/objects"),
    Path("src/holosoma_retargeting/holosoma_retargeting/models"),
)

DEFAULT_HUMAN_JOINT_REGEX = (
    r"^(L|R)_(Wrist|HandCenter|Index[123]|Middle[123]|Pinky[123]|Ring[123]|Thumb[123]|Toe|Ankle)$"
)

SMPLH_HAND_CENTER_JOINTS = {
    "left": (18, 21, 27, 24),  # Index1, Middle1, Ring1, Pinky1
    "right": (37, 40, 46, 43),  # Index1, Middle1, Ring1, Pinky1
}


def append_smplh_hand_centers_if_needed(human_joints: np.ndarray, data_format: str) -> np.ndarray:
    if data_format != "smplh" or human_joints.shape[1] >= 54:
        return human_joints
    if human_joints.shape[1] != 52:
        return human_joints

    left_center = human_joints[:, SMPLH_HAND_CENTER_JOINTS["left"]].mean(axis=1, keepdims=True)
    right_center = human_joints[:, SMPLH_HAND_CENTER_JOINTS["right"]].mean(axis=1, keepdims=True)
    return np.concatenate([human_joints, left_center, right_center], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add object contact labels to motion npz files. By default, labels "
            "are computed from SMPLH human_joints and mapped to robot bodies."
        )
    )
    parser.add_argument("--input", required=True, help="Input .npz file or directory containing .npz files.")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output .npz file or directory. If omitted, writes next to the input "
            "with a _contact suffix, or a sibling *_contact_labeled folder."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input files with the added label keys.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files.",
    )
    parser.add_argument(
        "--source",
        choices=("human", "robot"),
        default="human",
        help=(
            "human: compute intended contacts from SMPLH human_joints. "
            "robot: compute contacts from retargeted robot body_pos_w."
        ),
    )
    parser.add_argument(
        "--human-reference",
        default=None,
        help=(
            "Original SMPLH .npz to use when the input file does not contain "
            "human_joints. Most useful for labeling a single converted RL .npz."
        ),
    )
    parser.add_argument(
        "--human-reference-root",
        default="train_r1/motions",
        help=(
            "Root used to find matching *_original.npz files when labeling converted "
            "RL files that do not contain human_joints."
        ),
    )
    parser.add_argument(
        "--object-root",
        action="append",
        default=None,
        help=(
            "Directory containing <object_name>/sample_points.npy. Can be passed "
            "multiple times. Defaults cover train_r1 and retargeting object models."
        ),
    )
    parser.add_argument(
        "--object-name",
        default=None,
        help="Object name. If omitted, parsed from filenames like sub1_suitcase_001.npz.",
    )
    parser.add_argument(
        "--sample-points",
        default=None,
        help="Explicit object-local sample_points.npy file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Contact threshold in meters against object surface sample points.",
    )
    parser.add_argument(
        "--data-format",
        default="smplh",
        help="Human data format used for joint names and mapping.",
    )
    parser.add_argument(
        "--robot-type",
        default="r1",
        help="Robot type used to map human contact joints to robot body names. Use g1 for G1 data.",
    )
    parser.add_argument(
        "--human-joint-regex",
        default=DEFAULT_HUMAN_JOINT_REGEX,
        help="Regex selecting human joints for human-source labels.",
    )
    parser.add_argument(
        "--robot-body-regex",
        default=".*",
        help="Regex selecting robot bodies for robot-source diagnostic labels.",
    )
    parser.add_argument(
        "--exclude-regex",
        default=None,
        help="Optional regex to exclude selected human joints or robot bodies.",
    )
    parser.add_argument(
        "--dilate-frames",
        type=int,
        default=0,
        help="Expand positive labels by this many frames on each side.",
    )
    parser.add_argument(
        "--save-nearest-index",
        action="store_true",
        help="Also save the nearest object sample-point index for debugging.",
    )
    return parser.parse_args()


def object_name_from_motion_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_original"):
        stem = stem[: -len("_original")]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Could not infer object name from {path}. Expected name like sub1_suitcase_001.npz; "
            "pass --object-name explicitly."
        )
    return parts[1]


def resolve_sample_points(
    object_name: str,
    sample_points: str | None,
    object_roots: Iterable[str | Path] | None,
) -> Path:
    if sample_points:
        path = Path(sample_points)
        if not path.exists():
            raise FileNotFoundError(f"sample_points file does not exist: {path}")
        return path

    roots = [Path(root) for root in object_roots] if object_roots else list(DEFAULT_OBJECT_ROOTS)
    for root in roots:
        candidate = root / object_name / "sample_points.npy"
        if candidate.exists():
            return candidate
    searched = ", ".join(str(root / object_name / "sample_points.npy") for root in roots)
    raise FileNotFoundError(f"Could not find sample_points.npy for object '{object_name}'. Searched: {searched}")


def discover_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.npz"))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def output_path_for(input_file: Path, input_root: Path, output_arg: str | None, in_place: bool) -> Path:
    if in_place:
        return input_file

    if output_arg is None:
        if input_root.is_dir():
            output_root = input_root.with_name(input_root.name + "_contact_labeled")
            return output_root / input_file.relative_to(input_root)
        return input_file.with_name(input_file.stem + "_contact.npz")

    output = Path(output_arg)
    if input_root.is_dir():
        return output / input_file.relative_to(input_root)
    if output.suffix == ".npz":
        return output
    return output / input_file.name


def find_human_reference(input_file: Path, explicit_reference: str | None, reference_root: str | Path) -> Path | None:
    if explicit_reference is not None:
        return Path(explicit_reference)

    stem = input_file.stem
    if stem.endswith("_original"):
        return input_file

    root = Path(reference_root)
    if not root.exists():
        return None

    candidates = list(root.rglob(stem + "_original.npz"))
    if candidates:
        return sorted(candidates)[0]

    candidates = list(root.rglob(stem + ".npz"))
    if candidates:
        return sorted(candidates)[0]

    return None


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_npz_dict(path: Path, arrays: dict[str, np.ndarray], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return quat / norm


def rotation_from_wxyz(quat_wxyz: np.ndarray) -> Rotation:
    quat_wxyz = normalize_quat_wxyz(quat_wxyz)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(quat_xyzw)


def object_pose_from_arrays(arrays: dict[str, np.ndarray], source_name: str) -> tuple[np.ndarray, np.ndarray]:
    if "object_pos_w" in arrays and "object_quat_w" in arrays:
        return (
            np.asarray(arrays["object_pos_w"], dtype=np.float64),
            np.asarray(arrays["object_quat_w"], dtype=np.float64),
        )

    if "qpos" in arrays:
        qpos = np.asarray(arrays["qpos"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] < 14:
            raise ValueError(f"{source_name}: qpos does not look like it contains a dynamic object pose.")
        return qpos[:, -7:-4], qpos[:, -4:]

    raise ValueError(f"{source_name}: could not find object_pos_w/object_quat_w or qpos object pose.")


def resample_linear(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values)
    source_len = values.shape[0]
    if source_len == target_len:
        return values.copy()
    if source_len < 2:
        raise ValueError("Cannot resample a sequence with fewer than 2 frames.")

    flat = values.reshape(source_len, -1)
    src_t = np.linspace(0.0, 1.0, source_len)
    dst_t = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float64)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, flat[:, i])
    return out.reshape((target_len,) + values.shape[1:])


def select_names(names: list[str], include_regex: str, exclude_regex: str | None) -> tuple[list[str], np.ndarray]:
    include = re.compile(include_regex)
    exclude = re.compile(exclude_regex) if exclude_regex else None

    selected_names: list[str] = []
    selected_indices: list[int] = []
    for idx, name in enumerate(names):
        if not include.search(name):
            continue
        if exclude is not None and exclude.search(name):
            continue
        selected_names.append(name)
        selected_indices.append(idx)
    if not selected_names:
        raise ValueError(f"No names matched include regex {include_regex!r}.")
    return selected_names, np.asarray(selected_indices, dtype=np.int32)


def compute_distances_to_object(
    query_points_w: np.ndarray,
    object_pos_w: np.ndarray,
    object_quat_wxyz: np.ndarray,
    object_sample_points_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query_points_w = np.asarray(query_points_w, dtype=np.float64)
    object_pos_w = np.asarray(object_pos_w, dtype=np.float64)
    object_quat_wxyz = np.asarray(object_quat_wxyz, dtype=np.float64)
    object_sample_points_local = np.asarray(object_sample_points_local, dtype=np.float64)

    if query_points_w.ndim != 3 or query_points_w.shape[-1] != 3:
        raise ValueError(f"query_points_w must have shape (T, N, 3), got {query_points_w.shape}.")
    if object_pos_w.shape != (query_points_w.shape[0], 3):
        raise ValueError(
            f"object_pos_w shape {object_pos_w.shape} does not match query time dimension {query_points_w.shape[0]}."
        )
    if object_quat_wxyz.shape != (query_points_w.shape[0], 4):
        raise ValueError(
            f"object_quat_w shape {object_quat_wxyz.shape} does not match query time dimension "
            f"{query_points_w.shape[0]}."
        )

    rotations = rotation_from_wxyz(object_quat_wxyz)
    centered = query_points_w - object_pos_w[:, None, :]
    rot_w_from_obj = rotations.as_matrix()
    local = np.einsum("tij,tnj->tni", np.swapaxes(rot_w_from_obj, 1, 2), centered)

    tree = cKDTree(object_sample_points_local)
    distances, nearest_indices = tree.query(local.reshape(-1, 3), k=1)
    return (
        distances.reshape(query_points_w.shape[:2]).astype(np.float32),
        nearest_indices.reshape(query_points_w.shape[:2]).astype(np.int32),
    )


def dilate_labels(labels: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return labels
    out = labels.copy()
    for offset in range(1, radius + 1):
        out[offset:] |= labels[:-offset]
        out[:-offset] |= labels[offset:]
    return out


def human_joint_to_robot_body(joint_name: str, mapping: dict[str, str]) -> str | None:
    if joint_name in mapping:
        return mapping[joint_name]

    # SMPLH hand contacts are often detected on finger joints. Collapse them to
    # the mapped wrist/hand body so the label can drive a robot contact reward.
    left_hand_tokens = ("L_Index", "L_Middle", "L_Pinky", "L_Ring", "L_Thumb")
    right_hand_tokens = ("R_Index", "R_Middle", "R_Pinky", "R_Ring", "R_Thumb")
    if joint_name.startswith(left_hand_tokens):
        return mapping.get("L_HandCenter", mapping.get("L_Wrist"))
    if joint_name.startswith(right_hand_tokens):
        return mapping.get("R_HandCenter", mapping.get("R_Wrist"))
    return None


def aggregate_human_labels_to_robot(
    human_labels: np.ndarray,
    human_distances: np.ndarray,
    human_names: list[str],
    data_format: str,
    robot_type: str,
    available_body_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    motion_cfg = MotionDataConfig(data_format=data_format, robot_type=robot_type)
    mapping = motion_cfg.resolved_joints_mapping

    robot_names: list[str] = []
    robot_name_to_index: dict[str, int] = {}
    pairs: list[tuple[int, int]] = []
    skipped: list[str] = []
    available = set(available_body_names) if available_body_names is not None else None

    for human_col, human_name in enumerate(human_names):
        robot_name = human_joint_to_robot_body(human_name, mapping)
        if robot_name is None:
            skipped.append(human_name)
            continue
        if available is not None and robot_name not in available:
            skipped.append(f"{human_name}->{robot_name}")
            continue
        if robot_name not in robot_name_to_index:
            robot_name_to_index[robot_name] = len(robot_names)
            robot_names.append(robot_name)
        pairs.append((human_col, robot_name_to_index[robot_name]))

    if not pairs:
        raise ValueError(
            "No selected human joints could be mapped to robot bodies. "
            f"Check --robot-type ({robot_type}) and --human-joint-regex."
        )

    robot_labels = np.zeros((human_labels.shape[0], len(robot_names)), dtype=bool)
    robot_distances = np.full((human_distances.shape[0], len(robot_names)), np.inf, dtype=np.float32)
    for human_col, robot_col in pairs:
        robot_labels[:, robot_col] |= human_labels[:, human_col]
        robot_distances[:, robot_col] = np.minimum(robot_distances[:, robot_col], human_distances[:, human_col])

    if available_body_names is None:
        body_indices = np.full((len(robot_names),), -1, dtype=np.int32)
    else:
        body_indices = np.asarray([available_body_names.index(name) for name in robot_names], dtype=np.int32)

    return (
        robot_labels,
        robot_distances,
        np.asarray(robot_names),
        body_indices,
        skipped,
    )


def make_human_contact_labels(
    target_arrays: dict[str, np.ndarray],
    input_file: Path,
    args: argparse.Namespace,
    object_samples: np.ndarray,
) -> dict[str, np.ndarray]:
    human_arrays = target_arrays
    if "human_joints" not in human_arrays:
        human_ref = find_human_reference(input_file, args.human_reference, args.human_reference_root)
        if human_ref is None or not human_ref.exists():
            raise FileNotFoundError(
                f"{input_file}: no human_joints in input and no matching human reference found. "
                "Pass --human-reference or --human-reference-root."
            )
        human_arrays = load_npz_dict(human_ref)
        if "human_joints" not in human_arrays:
            raise ValueError(f"{human_ref}: expected human_joints key.")
    else:
        human_ref = input_file

    demo_joints = DEMO_JOINTS_REGISTRY[args.data_format]
    selected_names, selected_indices = select_names(demo_joints, args.human_joint_regex, args.exclude_regex)

    object_pos_w, object_quat_w = object_pose_from_arrays(target_arrays, str(input_file))
    target_len = int(object_pos_w.shape[0])

    human_joints = np.asarray(human_arrays["human_joints"], dtype=np.float64)
    human_joints = resample_linear(human_joints, target_len)
    human_joints = append_smplh_hand_centers_if_needed(human_joints, args.data_format)
    if len(selected_indices) > 0 and human_joints.shape[1] <= int(np.max(selected_indices)):
        raise ValueError(
            f"{human_ref}: selected joint index {int(np.max(selected_indices))} is unavailable "
            f"for human_joints with {human_joints.shape[1]} joints."
        )
    query_points = human_joints[:, selected_indices]

    distances, nearest_indices = compute_distances_to_object(query_points, object_pos_w, object_quat_w, object_samples)
    human_labels = distances <= args.threshold
    human_labels = dilate_labels(human_labels, args.dilate_frames)

    out: dict[str, np.ndarray] = {
        "contact_human_object_label": human_labels,
        "contact_human_object_distance": distances,
        "contact_human_joint_names": np.asarray(selected_names),
        "contact_human_joint_indices": selected_indices,
        "contact_human_reference_file": np.asarray(str(human_ref)),
    }
    if args.save_nearest_index:
        out["contact_human_object_nearest_sample_index"] = nearest_indices

    body_names = [str(name) for name in target_arrays["body_names"].tolist()] if "body_names" in target_arrays else None
    try:
        robot_labels, robot_distances, robot_names, robot_body_indices, skipped = aggregate_human_labels_to_robot(
            human_labels,
            distances,
            selected_names,
            args.data_format,
            args.robot_type,
            body_names,
        )
    except ValueError:
        robot_labels = np.zeros((0, 0), dtype=bool)
        robot_distances = np.zeros((0, 0), dtype=np.float32)
        robot_names = np.asarray([], dtype=str)
        robot_body_indices = np.asarray([], dtype=np.int32)
        skipped = selected_names

    if robot_names.size > 0:
        out.update(
            {
                "contact_robot_body_label_from_human": robot_labels,
                "contact_robot_body_distance_from_human": robot_distances,
                "contact_robot_body_names_from_human": robot_names,
                "contact_robot_body_indices_from_human": robot_body_indices,
                "contact_object_label": robot_labels,
                "contact_object_distance": robot_distances,
                "contact_object_names": robot_names,
                "contact_object_indices": robot_body_indices,
                "contact_object_target": np.asarray("robot_body_from_human"),
            }
        )
    else:
        out.update(
            {
                "contact_object_label": human_labels,
                "contact_object_distance": distances,
                "contact_object_names": np.asarray(selected_names),
                "contact_object_indices": selected_indices,
                "contact_object_target": np.asarray("human_joint"),
            }
        )

    out["contact_human_to_robot_skipped"] = np.asarray(skipped)
    return out


def make_robot_contact_labels(
    arrays: dict[str, np.ndarray],
    input_file: Path,
    args: argparse.Namespace,
    object_samples: np.ndarray,
    object_name: str,
) -> dict[str, np.ndarray]:
    if "body_pos_w" not in arrays or "body_names" not in arrays:
        raise ValueError(f"{input_file}: robot-source labels require body_pos_w and body_names.")

    body_names = [str(name) for name in arrays["body_names"].tolist()]
    exclude_parts = [r"^world$"]
    if object_name:
        exclude_parts.append(re.escape(object_name))
    user_exclude = args.exclude_regex
    exclude_regex = "|".join(exclude_parts + ([user_exclude] if user_exclude else []))

    selected_names, selected_indices = select_names(body_names, args.robot_body_regex, exclude_regex)
    object_pos_w, object_quat_w = object_pose_from_arrays(arrays, str(input_file))
    query_points = np.asarray(arrays["body_pos_w"], dtype=np.float64)[:, selected_indices]

    distances, nearest_indices = compute_distances_to_object(query_points, object_pos_w, object_quat_w, object_samples)
    labels = distances <= args.threshold
    labels = dilate_labels(labels, args.dilate_frames)

    out: dict[str, np.ndarray] = {
        "contact_robot_object_label": labels,
        "contact_robot_object_distance": distances,
        "contact_robot_body_names": np.asarray(selected_names),
        "contact_robot_body_indices": selected_indices,
        "contact_object_label": labels,
        "contact_object_distance": distances,
        "contact_object_names": np.asarray(selected_names),
        "contact_object_indices": selected_indices,
        "contact_object_target": np.asarray("robot_body"),
    }
    if args.save_nearest_index:
        out["contact_robot_object_nearest_sample_index"] = nearest_indices
    return out


def label_file(input_file: Path, output_file: Path, args: argparse.Namespace, input_root: Path) -> dict[str, object]:
    arrays = load_npz_dict(input_file)
    object_name = args.object_name or object_name_from_motion_path(input_file)
    sample_path = resolve_sample_points(object_name, args.sample_points, args.object_root)
    object_samples = np.load(sample_path)

    if args.source == "human":
        label_arrays = make_human_contact_labels(arrays, input_file, args, object_samples)
    else:
        label_arrays = make_robot_contact_labels(arrays, input_file, args, object_samples, object_name)

    label_arrays.update(
        {
            "contact_object_source": np.asarray("human_smplh" if args.source == "human" else "retarget_robot"),
            "contact_object_name": np.asarray(object_name),
            "contact_object_threshold_m": np.asarray(args.threshold, dtype=np.float32),
            "contact_object_sample_points_file": np.asarray(str(sample_path)),
            "contact_object_dilate_frames": np.asarray(args.dilate_frames, dtype=np.int32),
        }
    )

    output_arrays = dict(arrays)
    output_arrays.update(label_arrays)
    save_npz_dict(output_file, output_arrays, overwrite=args.overwrite or args.in_place)

    labels = label_arrays["contact_object_label"]
    names = label_arrays["contact_object_names"]
    any_contact = np.any(labels, axis=1) if labels.size else np.zeros((0,), dtype=bool)
    return {
        "input": input_file,
        "output": output_file,
        "object": object_name,
        "target": str(label_arrays["contact_object_target"]),
        "num_frames": int(labels.shape[0]),
        "num_targets": int(labels.shape[1]) if labels.ndim == 2 else 0,
        "num_contact_frames": int(any_contact.sum()),
        "names": [str(name) for name in names.tolist()],
        "relative": input_file.relative_to(input_root) if input_root.is_dir() else input_file.name,
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    input_files = discover_input_files(input_root)
    if not input_files:
        raise ValueError(f"No .npz files found under {input_root}")
    if args.in_place and args.output is not None:
        raise ValueError("--output cannot be used with --in-place")

    summaries = []
    for input_file in input_files:
        output_file = output_path_for(input_file, input_root, args.output, args.in_place)
        summary = label_file(input_file, output_file, args, input_root)
        summaries.append(summary)
        print(
            f"[contact] {summary['relative']} -> {output_file} | "
            f"object={summary['object']} target={summary['target']} "
            f"frames_with_contact={summary['num_contact_frames']}/{summary['num_frames']} "
            f"targets={summary['num_targets']}"
        )

    total_frames = sum(int(s["num_frames"]) for s in summaries)
    total_contact_frames = sum(int(s["num_contact_frames"]) for s in summaries)
    print(
        f"[contact] done: files={len(summaries)}, "
        f"frames_with_contact={total_contact_frames}/{total_frames}, source={args.source}"
    )


if __name__ == "__main__":
    main()
