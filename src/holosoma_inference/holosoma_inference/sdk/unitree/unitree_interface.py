"""Unitree robot interface using C++/pybind11 binding."""

import time
from typing import Any

import numpy as np
from loguru import logger

from holosoma_inference.camera.realsense_depth_wh_8060 import RealSenseDepthCamera
from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface


class UnitreeInterface(BaseInterface):
    """Interface for Unitree robots using C++/pybind11 binding."""

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)
        self._unitree_motor_order = None
        self._kp_level = 1.0
        self._kd_level = 1.0
        self._depth_expected_shape: tuple[int, int, int] | None = None
        self._depth_window: np.ndarray | None = None
        self._depth_camera: Any | None = None
        self._depth_show_window = False
        self._depth_display_scale = 4
        self._depth_logged_first_frame = False
        self._depth_cv2: Any | None = None
        self._depth_display_disabled = False
        self._init_binding()

    def _init_binding(self):
        """Initialize C++/pybind11 binding."""
        try:
            import unitree_interface
        except ImportError as e:
            raise ImportError("unitree_interface python binding not found.") from e

        robot_type_map = {
            "G1": unitree_interface.RobotType.G1,
            "H1": unitree_interface.RobotType.H1,
            "H1_2": unitree_interface.RobotType.H1_2,
            "GO2": unitree_interface.RobotType.GO2,
        }
        message_type_map = {"HG": unitree_interface.MessageType.HG, "GO2": unitree_interface.MessageType.GO2}

        self.unitree_interface = unitree_interface.create_robot(
            self.interface_str,
            robot_type_map[self.robot_config.robot.upper()],
            message_type_map[self.robot_config.message_type.upper()],
        )
        self.unitree_interface.set_control_mode(unitree_interface.ControlMode.PR)

        # GO2 SDK motor order differs from joint order
        if self.robot_config.robot.lower() == "go2":
            self._unitree_motor_order = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

    def get_low_state(self) -> np.ndarray:
        """Get robot state as numpy array."""
        state = self.unitree_interface.read_low_state()
        base_pos = np.zeros(3)
        quat = np.array(state.imu.quat)
        motor_pos = np.array(state.motor.q)
        base_lin_vel = np.zeros(3)
        base_ang_vel = np.array(state.imu.omega)
        motor_vel = np.array(state.motor.dq)

        joint_pos = np.zeros(self.robot_config.num_joints)
        joint_vel = np.zeros(self.robot_config.num_joints)
        motor_order = self._unitree_motor_order or self.robot_config.joint2motor

        for j_id in range(self.robot_config.num_joints):
            m_id = motor_order[j_id]
            joint_pos[j_id] = float(motor_pos[m_id])
            joint_vel[j_id] = float(motor_vel[m_id])

        return np.concatenate([base_pos, quat, joint_pos, base_lin_vel, base_ang_vel, joint_vel]).reshape(1, -1)

    def configure_depth_window(
        self,
        expected_shape: tuple[int, int, int] = (5, 60, 80),
        show_window: bool = False,
        display_scale: int = 4,
    ) -> None:
        """Configure lazy RealSense depth capture for fused WBT policies."""
        self._depth_expected_shape = tuple(int(value) for value in expected_shape)
        if len(self._depth_expected_shape) != 3:
            raise ValueError(f"Expected depth window shape [T, H, W], got {expected_shape}.")
        self._depth_window = None
        self._depth_show_window = bool(show_window)
        self._depth_display_scale = max(int(display_scale), 1)
        self._depth_logged_first_frame = False
        self._depth_display_disabled = False

    def _start_depth_camera_if_needed(self) -> None:
        if self._depth_camera is not None:
            return
        if self._depth_expected_shape is None:
            raise RuntimeError("RealSense depth window was requested before configure_depth_window() was called.")

        assert self._depth_expected_shape is not None
        _, expected_height, expected_width = self._depth_expected_shape

        camera = RealSenseDepthCamera(
            width=640,
            height=480,
            fps=90,
            output_width=expected_width,
            output_height=expected_height,
            min_depth_m=0.05,
            max_depth_m=20.0,
            queue_size=1,
            discard_initial_frames=30,
        )
        self._depth_camera = camera.__enter__()
        logger.info(
            "RealSense depth provider started: "
            f"{self._depth_camera.stream_description()}, depth_scale={self._depth_camera.depth_scale:.8f} m/unit, "
            f"window_shape={self._depth_expected_shape}"
        )

    def _show_depth_frame(self, frame: np.ndarray) -> None:
        if not self._depth_show_window or self._depth_display_disabled:
            return

        try:
            if self._depth_cv2 is None:
                import cv2  # noqa: PLC0415

                self._depth_cv2 = cv2

            cv2 = self._depth_cv2
            finite = np.isfinite(frame) & (frame > 0.0)
            preview = np.zeros(frame.shape, dtype=np.uint8)
            if np.any(finite):
                valid = frame[finite]
                lo = float(valid.min())
                hi = float(np.quantile(valid, 0.99))
                if hi <= lo:
                    hi = lo + 1e-6
                normalized = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
                preview = (normalized * 255.0).astype(np.uint8)
                preview[~finite] = 0

            color_preview = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
            color_preview[~finite] = 0
            if self._depth_display_scale != 1:
                color_preview = cv2.resize(
                    color_preview,
                    None,
                    fx=self._depth_display_scale,
                    fy=self._depth_display_scale,
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow("Holosoma RealSense depth", color_preview)
            cv2.waitKey(1)
        except Exception as exc:
            logger.warning(f"Disabling RealSense depth debug window after display failure: {type(exc).__name__}: {exc}")
            self._depth_display_disabled = True

    def get_depth_window(self) -> np.ndarray | None:
        """Return a RealSense depth window shaped [T, H, W] in meters."""
        if self._depth_expected_shape is None:
            return None
        self._start_depth_camera_if_needed()
        assert self._depth_expected_shape is not None

        frame = self._depth_camera.read()
        _, expected_height, expected_width = self._depth_expected_shape
        if frame.shape == (expected_width, expected_height):
            frame = frame.T
        if frame.shape != (expected_height, expected_width):
            raise RuntimeError(
                f"Unexpected RealSense depth frame shape {frame.shape}; expected {(expected_height, expected_width)}."
            )

        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        if not self._depth_logged_first_frame:
            valid = frame[np.isfinite(frame) & (frame > 0.0)]
            if valid.size:
                logger.info(
                    "Received first RealSense depth frame: "
                    f"shape={frame.shape}, min={float(valid.min()):.3f}, "
                    f"max={float(valid.max()):.3f}, mean={float(valid.mean()):.3f}, "
                    f"valid_ratio={valid.size / frame.size:.3f}"
                )
            else:
                logger.warning(
                    f"Received first RealSense depth frame with no positive finite pixels: shape={frame.shape}"
                )
            self._depth_logged_first_frame = True

        self._show_depth_frame(frame)

        window_size, _, _ = self._depth_expected_shape
        if self._depth_window is None:
            self._depth_window = np.repeat(frame[None, :, :], window_size, axis=0)
        else:
            self._depth_window[:-1] = self._depth_window[1:].copy()
            self._depth_window[-1] = frame

        return self._depth_window.copy()

    def warmup_depth_window(self, duration_s: float, rate_hz: float = 50.0) -> None:
        """Pre-fill the RealSense depth window before policy execution."""
        duration_s = max(float(duration_s), 0.0)
        if duration_s <= 0.0:
            return
        if self._depth_expected_shape is None:
            raise RuntimeError("Cannot warm up RealSense depth before configure_depth_window() is called.")

        logger.info(f"Warming up RealSense depth for {duration_s:.1f}s before policy start.")
        start_time = time.perf_counter()
        deadline = start_time + duration_s
        period = 1.0 / max(float(rate_hz), 1.0)
        frames = 0

        while time.perf_counter() < deadline:
            frame_start = time.perf_counter()
            self.get_depth_window()
            frames += 1
            remaining = deadline - time.perf_counter()
            sleep_s = min(period - (time.perf_counter() - frame_start), remaining)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        elapsed = time.perf_counter() - start_time
        logger.info(f"RealSense depth warmup complete: frames={frames}, elapsed={elapsed:.2f}s")

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ):
        """Send low-level command to robot."""
        cmd_q_target = np.zeros(self.robot_config.num_motors)
        cmd_dq_target = np.zeros(self.robot_config.num_motors)
        cmd_tau_target = np.zeros(self.robot_config.num_motors)
        cmd_kp = np.zeros(self.robot_config.num_motors) if kp_override is not None else None
        cmd_kd = np.zeros(self.robot_config.num_motors) if kd_override is not None else None

        motor_order = self._unitree_motor_order or self.robot_config.joint2motor
        for j_id in range(self.robot_config.num_joints):
            m_id = motor_order[j_id]
            cmd_q_target[m_id] = float(cmd_q[j_id])
            cmd_dq_target[m_id] = float(cmd_dq[j_id])
            cmd_tau_target[m_id] = float(cmd_tau[j_id])
            if cmd_kp is not None:
                cmd_kp[m_id] = float(kp_override[j_id])
            if cmd_kd is not None:
                cmd_kd[m_id] = float(kd_override[j_id])

        cmd = self.unitree_interface.create_zero_command()
        cmd.q_target = list(cmd_q_target)
        cmd.dq_target = list(cmd_dq_target)
        cmd.tau_ff = list(cmd_tau_target)

        motor_kp = np.array(cmd_kp if cmd_kp is not None else self.robot_config.motor_kp)
        motor_kd = np.array(cmd_kd if cmd_kd is not None else self.robot_config.motor_kd)
        cmd.kp = list(motor_kp * self._kp_level)
        cmd.kd = list(motor_kd * self._kd_level)

        self.unitree_interface.write_low_command(cmd)

    def get_joystick_msg(self):
        """Get wireless controller message."""
        return self.unitree_interface.read_wireless_controller()

    def get_joystick_key(self, wc_msg=None):
        """Get current key from joystick message."""
        if wc_msg is None:
            wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return None
        return self._wc_key_map.get(getattr(wc_msg, "keys", 0), None)

    @property
    def kp_level(self):
        """Get proportional gain level."""
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        """Set proportional gain level."""
        self._kp_level = value

    @property
    def kd_level(self):
        """Get derivative gain level."""
        return self._kd_level

    @kd_level.setter
    def kd_level(self, value):
        """Set derivative gain level."""
        self._kd_level = value
