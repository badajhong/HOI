# viser_utils.py
from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any
from typing import List, Tuple

import numpy as np
import viser  # type: ignore[import-not-found]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]


def create_motion_control_sliders(
    server: viser.ViserServer,
    viser_robot: ViserUrdf,
    robot_base_frame: viser.FrameHandle,
    motion_sequence: np.ndarray,
    *,
    robot_dof: int,
    viser_object: ViserUrdf | None = None,
    object_base_frame: viser.FrameHandle | None = None,
    contains_object_in_qpos: bool = True,
    initial_fps: int = 30,
    initial_interp_mult: int = 2,
    loop: bool = True,
    contact_urdf: ViserUrdf | None = None,
    robot_mesh_handles_by_link: Mapping[str, Sequence[Any]] | None = None,
    contact_mesh_handles_by_link: Mapping[str, Sequence[Any]] | None = None,
    contact_labels: np.ndarray | None = None,
    contact_body_names: Sequence[str] | None = None,
    contact_visual_link_aliases: Mapping[str, Sequence[str]] | None = None,
    debug_hand_target_points_obj: np.ndarray | None = None,
    debug_hand_target_valid: np.ndarray | None = None,
    debug_hand_target_body_names: Sequence[str] | None = None,
    debug_hand_target_body_filter: Sequence[str] = ("left_hand_contact_link", "right_hand_contact_link"),
) -> Tuple[List[viser.GuiInputHandle[int]], List[float]]:
    """
    Create a slider + play/pause controls and a background player thread with smooth, slerp-based interpolation.

    Assumed qpos layout per frame (MuJoCo order):
        [0:3]   robot base position   (xyz)
        [3:7]   robot base quaternion (wxyz)
        [7:7+R] robot joints          (R = robot_dof)
        [-7:-4] object position  (xyz)            # only if contains_object_in_qpos and viser_object provided
        [-4:]   object quaternion (wxyz)          # only if contains_object_in_qpos and viser_object provided

    Args:
        server: Viser server.
        viser_robot: ViserUrdf for the robot.
        robot_base_frame: server.scene.add_frame(...) return for the robot root frame (we set wxyz/position here).
        motion_sequence: np.ndarray with shape [T, D], sequence of qpos frames.
        robot_dof: number of actuated joints expected by viser_robot.
        viser_object: optional ViserUrdf for an object.
        object_base_frame: optional frame handle for the object root.
        contains_object_in_qpos: set True if motion_sequence includes the object 7D pose at the end.
        initial_fps: base FPS for playback.
        initial_interp_mult: visual upsampling multiplier.
        loop: whether to wrap around at the end.

    Returns:
        (controls, initial_values) — currently returns the [frame_slider] and [0.0]
    """
    qpos = motion_sequence
    n_frames = int(qpos.shape[0])
    if n_frames == 0:
        raise ValueError("motion_sequence is empty.")

    has_object_input = (
        viser_object is not None
        and object_base_frame is not None
        and contains_object_in_qpos
        and qpos.shape[1] >= (7 + robot_dof + 7)
    )
    has_contact_overlay = (
        contact_urdf is not None
        and contact_mesh_handles_by_link is not None
        and contact_labels is not None
        and contact_body_names is not None
    )
    if has_contact_overlay:
        contact_labels = np.asarray(contact_labels, dtype=bool)
        if contact_labels.shape[0] != n_frames:
            raise ValueError(
                f"contact_labels frame count {contact_labels.shape[0]} does not match motion_sequence frames {n_frames}."
            )
        if contact_labels.ndim != 2 or contact_labels.shape[1] != len(contact_body_names):
            raise ValueError("contact_labels must have shape (T, len(contact_body_names)).")

    debug_hand_targets_obj: np.ndarray | None = None
    debug_hand_targets_valid: np.ndarray | None = None
    debug_hand_target_columns: list[int] = []
    debug_hand_target_names: list[str] = []
    has_debug_hand_targets = (
        debug_hand_target_points_obj is not None
        and debug_hand_target_valid is not None
        and debug_hand_target_body_names is not None
    )
    if has_debug_hand_targets:
        debug_hand_targets_obj = np.asarray(debug_hand_target_points_obj, dtype=float)
        debug_hand_targets_valid = np.asarray(debug_hand_target_valid, dtype=bool)
        if debug_hand_targets_obj.ndim == 3:
            debug_hand_targets_obj = debug_hand_targets_obj[:, :, None, :]
        if debug_hand_targets_obj.ndim != 4 or debug_hand_targets_obj.shape[0] != n_frames:
            raise ValueError(
                "debug_hand_target_points_obj must have shape (T, N, K, 3) "
                f"or (T, N, 3), got {debug_hand_targets_obj.shape}."
            )
        if debug_hand_targets_valid.shape[:2] != debug_hand_targets_obj.shape[:2]:
            raise ValueError("debug_hand_target_valid must have shape (T, N).")
        if len(debug_hand_target_body_names) != debug_hand_targets_obj.shape[1]:
            raise ValueError("debug_hand_target_body_names length must match target point body dimension.")
        wanted = {str(name) for name in debug_hand_target_body_filter}
        debug_hand_target_names = [str(name) for name in debug_hand_target_body_names]
        debug_hand_target_columns = [
            idx for idx, name in enumerate(debug_hand_target_names) if name in wanted
        ]
        has_debug_hand_targets = bool(debug_hand_target_columns)

    def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
        q = _quat_normalize(q)
        w, x, y, z = q
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    debug_hand_target_handles: dict[tuple[int, int], Any] = {}
    if has_debug_hand_targets:
        assert debug_hand_targets_obj is not None
        left_rank_colors = (
            (255.0, 230.0, 0.0),
            (255.0, 190.0, 0.0),
            (255.0, 150.0, 0.0),
            (255.0, 110.0, 0.0),
            (255.0, 70.0, 0.0),
        )
        right_rank_colors = (
            (0.0, 230.0, 255.0),
            (0.0, 190.0, 255.0),
            (0.0, 150.0, 255.0),
            (0.0, 110.0, 255.0),
            (0.0, 70.0, 255.0),
        )
        hand_colors = {
            "left_hand_contact_link": left_rank_colors,
            "right_hand_contact_link": right_rank_colors,
        }
        for col in debug_hand_target_columns:
            body_name = debug_hand_target_names[col]
            body_colors = hand_colors.get(body_name, left_rank_colors)
            for rank in range(debug_hand_targets_obj.shape[2]):
                color = np.array([body_colors[min(rank, len(body_colors) - 1)]], dtype=np.float32)
                handle = server.scene.add_point_cloud(
                    f"/debug/hand_contact_target/{body_name}/{rank}",
                    points=np.zeros((1, 3), dtype=np.float32),
                    colors=color,
                    point_size=max(0.018, 0.03 - 0.002 * rank),
                    point_shape="circle",
                )
                handle.visible = False
                debug_hand_target_handles[(col, rank)] = handle

    # ---------------- GUI ----------------
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, n_frames - 1), step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=int(initial_fps), min=1, max=240, step=1)
    with server.gui.add_folder("Smoothing"):
        interp_mult_in = server.gui.add_number(
            "Visual FPS multiplier", initial_value=int(initial_interp_mult), min=1, max=8, step=1
        )

    # ---------------- helpers ----------------
    def _quat_normalize(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float)
        n = float(np.linalg.norm(q))
        return q if n == 0.0 else q / n

    def _quat_continuous(prev_q: np.ndarray | None, curr_q: np.ndarray) -> np.ndarray:
        q = _quat_normalize(curr_q)
        if prev_q is None:
            return q
        return -q if float(np.dot(prev_q, q)) < 0.0 else q

    def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
        q0 = _quat_normalize(q0)
        q1 = _quat_normalize(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            q = q0 + u * (q1 - q0)
            return _quat_normalize(q)
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        s = np.sin(theta)
        return (np.sin((1.0 - u) * theta) * q0 + np.sin(u * theta) * q1) / s

    def _interp_frame(qpos_arr: np.ndarray, i0: int, i1: int, u: float) -> np.ndarray:
        """SLERP for base & (optional) object quats; linear for positions and joints."""
        q0 = qpos_arr[i0]
        q1 = qpos_arr[i1]
        out = q0.copy()

        # Robot base (MuJoCo order: pos first, then quat)
        out[0:3] = (1.0 - u) * q0[0:3] + u * q1[0:3]  # pos (xyz)
        out[3:7] = _slerp(q0[3:7], q1[3:7], u)  # quat (wxyz)

        # Joints
        j0 = q0[7 : 7 + robot_dof]
        j1 = q1[7 : 7 + robot_dof]
        out[7 : 7 + robot_dof] = (1.0 - u) * j0 + u * j1

        # Object (optional) (MuJoCo order: pos first, then quat)
        if has_object_input:
            out[-7:-4] = (1.0 - u) * q0[-7:-4] + u * q1[-7:-4]  # obj pos (xyz)
            out[-4:] = _slerp(q0[-4:], q1[-4:], u)  # obj quat (wxyz)
        return out

    # ---------------- state ----------------
    playing = {"flag": False}
    tick = {"next": time.perf_counter()}  # absolute time for next draw
    prev: dict[str, np.ndarray | None] = {"robot_q": None, "obj_q": None}  # for continuity
    nonlocal_f = {"f": float(frame_slider.value)}  # fractional frame cursor
    updating_programmatically = {"flag": False}  # flag to prevent callback from pausing during programmatic updates

    # ---------------- draw ----------------
    def _apply_contact_overlay(joints: np.ndarray, frame_index: int) -> None:
        if not has_contact_overlay:
            return
        assert contact_urdf is not None
        assert contact_mesh_handles_by_link is not None
        assert contact_labels is not None
        assert contact_body_names is not None

        contact_urdf.update_cfg(joints)
        robot_visual_enabled = bool(getattr(viser_robot, "show_visual", True))
        contact_visual_enabled = bool(getattr(contact_urdf, "show_visual", True))
        frame_index = int(np.clip(frame_index, 0, n_frames - 1))
        active_links = {
            str(name)
            for name, is_active in zip(contact_body_names, contact_labels[frame_index], strict=False)
            if bool(is_active)
        }
        if contact_visual_link_aliases is not None:
            for link_name in tuple(active_links):
                active_links.update(contact_visual_link_aliases.get(link_name, ()))
        for link_name, handles in contact_mesh_handles_by_link.items():
            visible = link_name in active_links
            for handle in handles:
                handle.visible = visible and contact_visual_enabled

        if robot_mesh_handles_by_link is not None:
            contact_link_names = set(contact_mesh_handles_by_link)
            for link_name, handles in robot_mesh_handles_by_link.items():
                if link_name not in contact_link_names:
                    continue
                visible = link_name not in active_links
                for handle in handles:
                    handle.visible = visible and robot_visual_enabled

    def _apply_debug_hand_targets(q: np.ndarray, frame_index: int) -> None:
        if not has_debug_hand_targets:
            return
        assert debug_hand_targets_obj is not None
        assert debug_hand_targets_valid is not None

        if not has_object_input:
            for handle in debug_hand_target_handles.values():
                handle.visible = False
            return

        frame_index = int(np.clip(frame_index, 0, n_frames - 1))
        object_pos = q[-7:-4]
        object_quat = q[-4:]
        object_rot = _quat_wxyz_to_matrix(object_quat)
        for (col, rank), handle in debug_hand_target_handles.items():
            valid = bool(debug_hand_targets_valid[frame_index, col])
            if valid:
                point_obj = debug_hand_targets_obj[frame_index, col, rank]
                point_w = object_pos + object_rot @ point_obj
                handle.points = point_w[None].astype(np.float32)
            handle.visible = valid

    def _apply_frame_from_q(q: np.ndarray, frame_index: int) -> None:
        # joints -> ensure length
        joints = q[7 : 7 + robot_dof]
        if joints.shape[0] != robot_dof:
            joints = (
                joints[:robot_dof] if joints.shape[0] > robot_dof else np.pad(joints, (0, robot_dof - joints.shape[0]))
            )
        viser_robot.update_cfg(joints)
        _apply_contact_overlay(joints, frame_index)

        # robot base (MuJoCo order: pos first, then quat)
        robot_base_frame.position = q[0:3]  # pos (xyz)
        r_q = _quat_continuous(prev["robot_q"], q[3:7])
        prev["robot_q"] = r_q
        robot_base_frame.wxyz = r_q

        # object (optional) (MuJoCo order: pos first, then quat)
        if has_object_input and object_base_frame is not None:
            object_base_frame.position = q[-7:-4]  # obj pos (xyz)
            o_q = _quat_continuous(prev["obj_q"], q[-4:])
            prev["obj_q"] = o_q
            object_base_frame.wxyz = o_q
        elif object_base_frame is not None and viser_object is not None:
            # fallback static pose
            object_base_frame.position = np.zeros(3)
            object_base_frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

        _apply_debug_hand_targets(q, frame_index)

    def _apply_discrete_frame(i: int) -> None:
        i = int(np.clip(i, 0, n_frames - 1))
        _apply_frame_from_q(qpos[i], i)

    # ---------------- controls ----------------
    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]
        # reset timing & continuity starting from the current slider frame
        tick["next"] = time.perf_counter()
        prev["robot_q"] = None
        prev["obj_q"] = None
        nonlocal_f["f"] = float(frame_slider.value)

    @fps_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    @interp_mult_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    @frame_slider.on_update
    def _(_evt) -> None:
        # Only pause if this is a user interaction, not a programmatic update
        if not updating_programmatically["flag"]:
            # Pause when scrubbing so the background loop doesn't overwrite immediately
            playing["flag"] = False
            tick["next"] = time.perf_counter()
            frame_val = int(frame_slider.value)
            _apply_discrete_frame(frame_val)
            prev["robot_q"] = None
            prev["obj_q"] = None
            nonlocal_f["f"] = float(frame_val)

    # ---------------- player loop ----------------
    def _player_loop() -> None:
        if n_frames <= 1:
            return
        while True:
            if playing["flag"]:
                now = time.perf_counter()
                fps_val = max(1, int(fps_in.value))
                mult = max(1, int(interp_mult_in.value))
                dt = 1.0 / (fps_val * mult)

                if now >= tick["next"]:
                    # advance by one visual step
                    f = nonlocal_f["f"] + 1.0 / mult
                    if loop:
                        f = f % max(1, n_frames)
                    else:
                        f = min(f, float(n_frames - 1))
                    nonlocal_f["f"] = f

                    k0 = int(np.floor(f))
                    k1 = (k0 + 1) % max(1, n_frames) if loop else min(k0 + 1, n_frames - 1)
                    u = float(f - k0)

                    q_interp = _interp_frame(qpos, k0, k1, u)
                    _apply_frame_from_q(q_interp, k0)

                    # Update slider to show current frame number in real-time
                    # Use flag to prevent callback from pausing playback
                    updating_programmatically["flag"] = True
                    frame_slider.value = k0
                    updating_programmatically["flag"] = False

                    tick["next"] = now + dt
                else:
                    time.sleep(min(0.002, max(0.0, tick["next"] - now)))
            else:
                time.sleep(0.02)

    threading.Thread(target=_player_loop, daemon=True).start()

    # initial draw
    _apply_discrete_frame(0)

    # keep consistent with your previous return convention
    return [frame_slider], [0.0]
