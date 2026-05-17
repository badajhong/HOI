from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.observation import ObservationConfig
from holosoma_inference.policies import BasePolicy
from holosoma_inference.policies.wbt_utils import MotionClockUtil, PinocchioRobot, TimestepUtil
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.depth_stream import DepthWindowSub
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_mul,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._default_observation_config = config.observation
        self._residual_fused_export_spec: dict | None = None
        self._uses_residual_fused_export = False
        self._di_onnx_session = None
        self._di_onnx_input_names: list[str] = []
        self._di_onnx_output_names: list[str] = []
        self._di_model_path: str | None = None
        self._static_di_latent: np.ndarray | None = None
        self._static_depth_window: np.ndarray | None = None
        self._depth_window_sub: DepthWindowSub | None = None
        self._missing_depth_window_warned = False
        self._proprioception_window: np.ndarray | None = None
        self._proprioception_initialized = False
        self._waiting_for_motion_clip_logged = False
        self._policy_armed = False
        self._logged_fused_input_stats = False
        self._saved_debug_depth_window = False

        # initialize motion state
        self.motion_clip_progressing = False
        self.curr_motion_timestep = config.task.motion_start_timestep
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None

        # Initialize clock for sim-time synchronization
        clock_sub = ClockSub()
        clock_sub.start()
        clock_util = MotionClockUtil(clock_sub)
        self.timestep_util = TimestepUtil(
            clock=clock_util,
            interval_ms=1000.0 / config.task.rl_rate,
            start_timestep=config.task.motion_start_timestep,
        )

        # Read use_sim_time from config
        self.use_sim_time = config.task.use_sim_time

        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

        super().__init__(config)

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            # Fallback to default_dof_angles if not specified
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

        if config.robot.stiff_startup_kp is not None:
            self._stiff_hold_kp = np.array(config.robot.stiff_startup_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kp for WBT policy")

        if config.robot.stiff_startup_kd is not None:
            self._stiff_hold_kd = np.array(config.robot.stiff_startup_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kd for WBT policy")

        if self._stiff_hold_q.shape[1] != self.num_dofs:
            raise ValueError("Stiff startup pose dimension mismatch with robot DOFs")

        # Prompt user before entering stiff mode (only if stdin is available)
        def _show_warning():
            logger.warning(
                colored(
                    "⚠️  Non-interactive mode detected - cannot prompt for stiff mode confirmation!",
                    "red",
                    attrs=["bold"],
                )
            )

        if hasattr(self, "_shared_hardware_source"):
            logger.info(colored("Skipping stiff hold prompt (secondary policy)", "yellow"))
        elif sys.stdin.isatty():
            logger.info(colored("\n⚠️  Ready to enter stiff hold mode", "yellow", attrs=["bold"]))
            logger.info(colored("Press Enter to continue...", "yellow"))
            try:
                input()
                logger.info(colored("✓ Entering stiff hold mode", "green"))
            except EOFError:
                # [drockyd] seems like in some cases, input() will raise EOFError even in interactive mode.
                _show_warning()
        else:
            _show_warning()

    def _get_ref_body_orientation_in_world(self, robot_state_data):
        # Create configuration for pinocchio robot
        # Note:
        # 1. pinocchio quaternion is in xyzw format, robot_state_data is in wxyz format
        # 2. joint sequences in pinocchio robot and real robot are different

        # free base pos, does not matter
        root_pos = robot_state_data[0, :3]

        # free base ori, wxyz -> xyzw
        root_ori_xyzw = wxyz_to_xyzw(robot_state_data[:, 3:7])[0]

        # dof pos in real robot -> pinocchio robot
        num_dofs = self.num_dofs
        dof_pos_in_real = robot_state_data[0, 7 : 7 + num_dofs]
        dof_pos_in_pinocchio = dof_pos_in_real[self.pinocchio_robot.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)

        ref_ori_xyzw = self.pinocchio_robot.fk_and_get_ref_body_orientation_in_world(configuration)
        return xyzw_to_wxyz(ref_ori_xyzw)

    def _apply_observation_config(self, obs_config: ObservationConfig) -> None:
        self.obs_config = obs_config
        self.obs_scales = self.obs_config.obs_scales
        self.obs_dims = self.obs_config.obs_dims
        self.obs_dict = self.obs_config.obs_dict
        self.obs_dim_dict = self._calculate_obs_dim_dict()
        self.history_length_dict = self.obs_config.history_length_dict
        self._initialize_history_state()

    def _build_residual_fused_observation_config(self, export_spec: dict) -> ObservationConfig:
        shared_terms = list(export_spec["shared_obs_terms"])
        shared_dims = {str(key): int(value) for key, value in export_spec["shared_obs_dims"].items()}
        shared_scales = dict(export_spec["shared_obs_scales"])
        shared_history = int(export_spec.get("shared_obs_history_length", 1))
        di_mode = str(export_spec.get("di_encoder_mode", "fused"))

        if di_mode == "fused":
            return ObservationConfig(
                obs_dict={"shared_obs": shared_terms},
                obs_dims=dict(shared_dims),
                obs_scales={str(key): float(value) for key, value in shared_scales.items()},
                history_length_dict={"shared_obs": shared_history},
            )

        obs_dict = {
            "shared_obs": shared_terms,
            "di_ae_latent": [str(export_spec.get("latent_term", "di_ae_latent"))],
        }
        obs_dims = dict(shared_dims)
        obs_dims[str(export_spec.get("latent_term", "di_ae_latent"))] = int(export_spec["latent_dim"])
        obs_scales = {str(key): float(value) for key, value in shared_scales.items()}
        obs_scales[str(export_spec.get("latent_term", "di_ae_latent"))] = 1.0
        history_length_dict = {
            "shared_obs": shared_history,
            "di_ae_latent": int(export_spec.get("latent_history_length", 1)),
        }
        return ObservationConfig(
            obs_dict=obs_dict,
            obs_dims=obs_dims,
            obs_scales=obs_scales,
            history_length_dict=history_length_dict,
        )

    def _residual_di_mode(self) -> str:
        if self._residual_fused_export_spec is None:
            return ""
        return str(self._residual_fused_export_spec.get("di_encoder_mode", "fused"))

    def _uses_fused_proprioception_window(self) -> bool:
        if self._residual_fused_export_spec is None:
            return False
        return bool(self._residual_fused_export_spec.get("uses_proprioception_window", False))

    def _reset_proprioception_window(self) -> None:
        self._proprioception_window = None
        self._proprioception_initialized = False

    def _close_depth_window_subscriber(self) -> None:
        if self._depth_window_sub is not None:
            self._depth_window_sub.close()
            self._depth_window_sub = None

    def _start_depth_window_subscriber(self, export_spec: dict) -> None:
        depth_shape = tuple(int(value) for value in export_spec["depth_window_shape"])
        if self.config.task.depth_window_zmq_port <= 0:
            return

        self._depth_window_sub = DepthWindowSub(
            port=self.config.task.depth_window_zmq_port,
            host=self.config.task.depth_window_zmq_host,
            timeout_ms=self.config.task.depth_window_zmq_timeout_ms,
            expected_shape=depth_shape,
            show_window=self.config.task.show_depth_window,
            display_scale=self.config.task.depth_window_display_scale,
        )
        self._depth_window_sub.start()

    def _configure_interface_depth_provider(self, export_spec: dict) -> bool:
        configure_depth_window = getattr(self.interface, "configure_depth_window", None)
        if not callable(configure_depth_window):
            return False

        depth_shape = tuple(int(value) for value in export_spec["depth_window_shape"])
        configure_depth_window(
            expected_shape=depth_shape,
            show_window=self.config.task.show_depth_window,
            display_scale=self.config.task.depth_window_display_scale,
        )
        logger.info(f"Configured interface depth provider with window_shape={depth_shape}")

        warmup_seconds = float(self.config.task.depth_window_warmup_seconds)
        if warmup_seconds > 0.0:
            warmup_depth_window = getattr(self.interface, "warmup_depth_window", None)
            if not callable(warmup_depth_window):
                raise RuntimeError(
                    "task.depth_window_warmup_seconds was requested, but the runtime interface does not expose "
                    "warmup_depth_window()."
                )
            warmup_depth_window(warmup_seconds, rate_hz=self.config.task.rl_rate)
        return True

    def _setup_live_depth_provider(self, export_spec: dict) -> None:
        depth_source = str(self.config.task.depth_window_source).lower()
        if depth_source not in {"zmq", "realsense", "auto"}:
            raise ValueError(
                f"Unsupported task.depth_window_source={self.config.task.depth_window_source!r}; "
                "expected 'zmq', 'realsense', or 'auto'."
            )

        if depth_source in {"realsense", "auto"} and self._configure_interface_depth_provider(export_spec):
            logger.info("Using RealSense/interface depth provider; no external depth publisher is required.")
            return

        if depth_source == "realsense":
            raise RuntimeError(
                "task.depth_window_source='realsense' was requested, but the runtime interface does not expose "
                "configure_depth_window()/get_depth_window()."
            )

        self._start_depth_window_subscriber(export_spec)

    def _maybe_resolve_di_model_path(self, model_path: str, export_spec: dict) -> str | None:
        if self.config.task.di_model_path:
            return self.config.task.di_model_path

        artifact_name = export_spec.get("di_encoder_artifact")
        if not artifact_name:
            return None

        candidate = Path(model_path).with_name(str(artifact_name))
        if candidate.exists():
            return str(candidate)
        return None

    def _setup_separate_di_provider(self, model_path: str, export_spec: dict) -> None:
        self._di_onnx_session = None
        self._di_onnx_input_names = []
        self._di_onnx_output_names = []
        self._di_model_path = self._maybe_resolve_di_model_path(model_path, export_spec)
        self._static_di_latent = None
        self._static_depth_window = None
        self._depth_window_sub = None
        self._missing_depth_window_warned = False
        self._reset_proprioception_window()

        latent_dim = int(export_spec["latent_dim"])
        if self.config.task.di_latent_path:
            latent = np.asarray(np.load(self.config.task.di_latent_path), dtype=np.float32)
            if latent.ndim == 1:
                latent = latent.reshape(1, -1)
            if latent.shape != (1, latent_dim):
                raise ValueError(
                    f"Expected DI latent override shape (1, {latent_dim}), got {latent.shape} from "
                    f"{self.config.task.di_latent_path}."
                )
            self._static_di_latent = latent
            logger.info(f"Loaded static DI latent override: {self.config.task.di_latent_path}")

        if self.config.task.depth_window_npy_path:
            depth_window = np.asarray(np.load(self.config.task.depth_window_npy_path), dtype=np.float32)
            self._static_depth_window = self._prepare_depth_window_input(depth_window)
            logger.info(f"Loaded static depth window override: {self.config.task.depth_window_npy_path}")
        elif self._static_di_latent is None:
            self._setup_live_depth_provider(export_spec)

        if self._di_model_path:
            self._di_onnx_session = onnxruntime.InferenceSession(self._di_model_path)
            self._di_onnx_input_names = [inp.name for inp in self._di_onnx_session.get_inputs()]
            self._di_onnx_output_names = [out.name for out in self._di_onnx_session.get_outputs()]
            logger.info(f"Loaded separate DI encoder ONNX: {self._di_model_path}")
        elif self._static_di_latent is None:
            logger.warning(
                "This older residual WBT export expects a separate DI latent source. "
                "Provide `--task.di-model-path` plus a depth provider, or use "
                "`--task.di-latent-path` for a precomputed latent."
            )

    def _setup_fused_depth_provider(self) -> None:
        self._di_onnx_session = None
        self._di_onnx_input_names = []
        self._di_onnx_output_names = []
        self._di_model_path = None
        self._static_di_latent = None
        self._static_depth_window = None
        self._depth_window_sub = None
        self._missing_depth_window_warned = False
        self._reset_proprioception_window()
        if self.config.task.depth_window_npy_path:
            depth_window = np.asarray(np.load(self.config.task.depth_window_npy_path), dtype=np.float32)
            self._static_depth_window = self._prepare_depth_window_input(depth_window)
            logger.info(f"Loaded static depth window override: {self.config.task.depth_window_npy_path}")
            return

        self._setup_live_depth_provider(self._residual_fused_export_spec)

    def _prepare_depth_window_input(self, depth_window: np.ndarray) -> np.ndarray:
        depth_window_np = np.asarray(depth_window, dtype=np.float32)
        if depth_window_np.ndim == 3:
            depth_window_np = depth_window_np.reshape(1, *depth_window_np.shape)
        if depth_window_np.ndim != 4 or depth_window_np.shape[0] != 1:
            raise ValueError(
                f"Expected depth window shape [T, H, W] or [1, T, H, W], got {depth_window_np.shape}."
            )
        if self._residual_fused_export_spec is not None:
            expected_shape = tuple(int(value) for value in self._residual_fused_export_spec["depth_window_shape"])
            actual_shape = tuple(int(value) for value in depth_window_np.shape[1:])
            if actual_shape == (expected_shape[0], expected_shape[2], expected_shape[1]):
                depth_window_np = depth_window_np.transpose(0, 1, 3, 2)
                actual_shape = tuple(int(value) for value in depth_window_np.shape[1:])
            if actual_shape != expected_shape:
                raise ValueError(f"Expected depth window shape [1, {expected_shape}], got {depth_window_np.shape}.")
        return np.ascontiguousarray(depth_window_np, dtype=np.float32)

    def _get_current_depth_window(self) -> np.ndarray | None:
        if self._static_depth_window is not None:
            return self._static_depth_window.copy()

        get_depth_window = getattr(self.interface, "get_depth_window", None)
        if callable(get_depth_window):
            depth_window = get_depth_window()
            if depth_window is None:
                pass
            else:
                return self._prepare_depth_window_input(depth_window)
        if self._depth_window_sub is not None:
            depth_window = self._depth_window_sub.get_depth_window()
            if depth_window is None:
                return None
            return self._prepare_depth_window_input(depth_window)
        return None

    def _maybe_log_fused_model_input(
        self,
        shared_obs: np.ndarray,
        depth_window: np.ndarray,
        proprioception_window: np.ndarray | None,
        actor_obs: np.ndarray,
    ) -> None:
        if not self.config.task.log_depth_window_input_stats or self._logged_fused_input_stats:
            return

        finite_depth = depth_window[np.isfinite(depth_window) & (depth_window > 0.0)]
        if finite_depth.size:
            depth_stats = (
                f"shape={depth_window.shape}, min={float(finite_depth.min()):.4f}, "
                f"max={float(finite_depth.max()):.4f}, mean={float(finite_depth.mean()):.4f}, "
                f"valid_ratio={finite_depth.size / depth_window.size:.4f}"
            )
        else:
            depth_stats = f"shape={depth_window.shape}, no positive finite pixels"

        proprio_stats = "none"
        if proprioception_window is not None:
            proprio_stats = (
                f"shape={proprioception_window.shape}, min={float(np.min(proprioception_window)):.4f}, "
                f"max={float(np.max(proprioception_window)):.4f}, mean={float(np.mean(proprioception_window)):.4f}"
            )

        logger.info(
            "Fused ONNX input check: "
            f"shared_obs_shape={shared_obs.shape}, depth=({depth_stats}), "
            f"proprio=({proprio_stats}), actor_obs_shape={actor_obs.shape}"
        )

        debug_path = self.config.task.depth_window_debug_npy_path
        if debug_path and not self._saved_debug_depth_window:
            np.save(debug_path, depth_window)
            logger.info(f"Saved first fused ONNX depth window to {debug_path}")
            self._saved_debug_depth_window = True

        self._logged_fused_input_stats = True

    def _get_fused_depth_window_or_zeros(self) -> np.ndarray:
        depth_window = self._get_current_depth_window()
        if depth_window is not None:
            return depth_window

        if self._residual_fused_export_spec is None:
            raise RuntimeError("Residual fused export spec is missing.")

        depth_shape = tuple(int(value) for value in self._residual_fused_export_spec["depth_window_shape"])
        if not self.config.task.allow_missing_depth_window:
            if self._depth_window_sub is None:
                raise RuntimeError(
                    "Fused residual WBT ONNX requires a depth window, but no depth provider is configured. "
                    "Start a simulator bridge that publishes depth, pass `--task.depth-window-npy-path`, or set "
                    "`--task.allow-missing-depth-window` only for unsafe debugging with zero depth."
                )
            raise RuntimeError(
                "Fused residual WBT ONNX requires a live depth window, but no simulator depth frame has arrived on "
                f"ZMQ port {self.config.task.depth_window_zmq_port}. Start the simulator bridge depth "
                "publisher before starting the policy, or set `--task.allow-missing-depth-window` only for unsafe "
                "debugging with zero depth."
            )

        if not self._missing_depth_window_warned:
            if self._depth_window_sub is None:
                logger.warning(
                    "Fused residual WBT ONNX needs a depth window, but no depth provider was found. "
                    "Using a zero depth window so inference can start. For meaningful HOI behavior, provide "
                    "`--task.depth-window-npy-path`, a simulator depth ZMQ publisher, or an interface exposing "
                    "`get_depth_window()`."
                )
            else:
                logger.warning(
                    "Fused residual WBT ONNX needs a depth window, but no simulator depth frame has arrived on "
                    f"ZMQ port {self.config.task.depth_window_zmq_port} yet. Using zeros until frames arrive."
                )
            self._missing_depth_window_warned = True
        return np.zeros((1, *depth_shape), dtype=np.float32)

    def _get_current_proprioception(self, robot_state_data: np.ndarray) -> np.ndarray:
        if self.config.task.debug.force_zero_angular_velocity:
            base_ang_vel = np.zeros((1, 3), dtype=np.float32)
        else:
            base_ang_vel = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles
        dof_vel = robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]
        return np.concatenate([base_ang_vel, dof_pos, dof_vel], axis=1).astype(np.float32, copy=False)

    def _get_current_proprioception_window(self, robot_state_data: np.ndarray) -> np.ndarray:
        if self._residual_fused_export_spec is None:
            raise RuntimeError("Residual fused export spec is missing.")

        proprioception_shape = self._residual_fused_export_spec.get("proprioception_window_shape")
        if proprioception_shape is None:
            raise RuntimeError("Fused export does not declare a proprioception_window_shape.")
        window_size, feature_dim = (int(value) for value in proprioception_shape)

        proprioception = self._get_current_proprioception(robot_state_data)
        if proprioception.shape != (1, feature_dim):
            raise ValueError(f"Expected proprioception shape (1, {feature_dim}), got {proprioception.shape}.")

        if self._proprioception_window is None:
            self._proprioception_window = np.zeros((1, window_size, feature_dim), dtype=np.float32)

        if not self._proprioception_initialized:
            self._proprioception_window[:] = np.repeat(proprioception[:, None, :], window_size, axis=1)
            self._proprioception_initialized = True
        else:
            self._proprioception_window[:, :-1, :] = self._proprioception_window[:, 1:, :].copy()
            self._proprioception_window[:, -1, :] = proprioception

        return self._proprioception_window.copy()

    def _get_di_latent(self) -> np.ndarray:
        assert self._residual_fused_export_spec is not None
        latent_dim = int(self._residual_fused_export_spec["latent_dim"])
        if self._static_di_latent is not None:
            return self._static_di_latent.copy()

        if self._di_onnx_session is None:
            raise RuntimeError(
                "This older residual WBT export needs a separate DI latent source, but no DI encoder is configured. "
                "Use `--task.di-model-path` together with an interface that exposes `get_depth_window()`, "
                "or pass `--task.di-latent-path`."
            )

        depth_window = self._get_current_depth_window()
        if depth_window is None:
            raise RuntimeError(
                "Residual fused WBT inference could not obtain a depth window. "
                "Provide `--task.depth-window-npy-path` for offline testing or expose "
                "`interface.get_depth_window()` in the runtime interface."
            )

        input_name = self._di_onnx_input_names[0]
        output_name = self._di_onnx_output_names[0]
        latent = self._di_onnx_session.run([output_name], {input_name: depth_window})[0]
        latent = np.asarray(latent, dtype=np.float32)
        if latent.ndim == 1:
            latent = latent.reshape(1, -1)
        if latent.shape != (1, latent_dim):
            raise ValueError(f"Expected DI encoder output shape (1, {latent_dim}), got {latent.shape}.")
        return latent

    def _get_object_type_one_hot(self) -> np.ndarray:
        if self._residual_fused_export_spec is None:
            raise RuntimeError("Residual fused export spec is missing.")
        shared_dims = self._residual_fused_export_spec["shared_obs_dims"]
        if "obj_type_one_hot" not in shared_dims:
            raise RuntimeError("Residual fused export expects obj_type_one_hot, but the export spec omitted it.")
        num_classes = int(shared_dims["obj_type_one_hot"])
        if num_classes <= 0:
            raise ValueError(f"obj_type_one_hot dimension must be positive, got {num_classes}.")
        object_type_id = int(np.clip(self.config.task.object_type_id, 0, num_classes - 1))
        one_hot = np.zeros((1, num_classes), dtype=np.float32)
        one_hot[0, object_type_id] = 1.0
        return one_hot

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Extract KP/KD from ONNX metadata (same as base class)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None
        self.policy_action_scale = self._resolve_policy_action_scale_from_metadata(metadata, model_path)
        self._residual_fused_export_spec = None
        self._uses_residual_fused_export = False
        self._apply_observation_config(self._default_observation_config)

        export_spec = metadata.get("policy_export_spec")
        if isinstance(export_spec, dict) and export_spec.get("kind") == "wbt_student_residual_fused":
            self._residual_fused_export_spec = export_spec
            self._uses_residual_fused_export = True
            self._apply_observation_config(self._build_residual_fused_observation_config(export_spec))
            if self._residual_di_mode() == "fused":
                self._setup_fused_depth_provider()
            else:
                self._setup_separate_di_provider(model_path, export_spec)

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")
        if self._uses_residual_fused_export:
            if self._residual_di_mode() == "fused":
                logger.info("Configured WBT inference for one-file fused DI+student+residual policy.")
            else:
                logger.info("Configured WBT inference for legacy fused student+residual policy with separate DI.")

        # get initial command and ref quat xyzw at the configured start timestep
        time_step = np.array([[self.config.task.motion_start_timestep]], dtype=np.float32)

        # Use configured observation dimensions (including history) instead of a hard-coded value.
        if self._uses_residual_fused_export:
            shared_obs_template = self.obs_buf_dict.get("shared_obs")
            if shared_obs_template is None:
                raise ValueError("Residual fused WBT policy requires 'shared_obs' group.")
            if self._residual_di_mode() == "fused":
                assert self._residual_fused_export_spec is not None
                depth_flat_dim = int(self._residual_fused_export_spec["depth_window_flat_dim"])
                depth_flat = np.zeros((1, depth_flat_dim), dtype=np.float32)
                obs_parts = [shared_obs_template, depth_flat]
                if self._uses_fused_proprioception_window():
                    proprio_flat_dim = int(self._residual_fused_export_spec["proprioception_window_flat_dim"])
                    obs_parts.append(np.zeros((1, proprio_flat_dim), dtype=np.float32))
                obs = np.concatenate(obs_parts, axis=1).astype(np.float32, copy=False)
            else:
                latent_template = self.obs_buf_dict.get("di_ae_latent")
                if latent_template is None:
                    raise ValueError("Residual fused WBT policy requires 'di_ae_latent' group in separate mode.")
                obs = np.concatenate([shared_obs_template, latent_template], axis=1).astype(np.float32, copy=False)
        else:
            actor_obs_template = self.obs_buf_dict.get("actor_obs")
            if actor_obs_template is None:
                raise ValueError("Observation group 'actor_obs' must be configured for WBT policy.")
            obs = actor_obs_template.copy()
        input_feed = {"obs": obs, "time_step": time_step}
        outputs = self.onnx_policy_session.run(["joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)

        # motion_command_t/ref_quat_xyzw_t will be used in get_current_obs_buffer_dict
        self.motion_command_t = np.concatenate(outputs[0:2], axis=1)  # (1, 58)
        self.ref_quat_xyzw_t = outputs[2]
        # duplicate, will be used in _get_init_target and _handle_stop_policy
        self.motion_command_0 = self.motion_command_t.copy()
        self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()

        def policy_act(input_feed):
            output = self.onnx_policy_session.run(["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)
            action = output[0]
            motion_command = np.concatenate(output[1:3], axis=1)
            ref_quat_xyzw = output[3]
            return action, motion_command, ref_quat_xyzw

        self.policy = policy_act

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
                "obs_config": self.obs_config,
                "residual_fused_export_spec": self._residual_fused_export_spec,
                "uses_residual_fused_export": self._uses_residual_fused_export,
                "di_onnx_session": self._di_onnx_session,
                "di_onnx_input_names": list(self._di_onnx_input_names),
                "di_onnx_output_names": list(self._di_onnx_output_names),
                "di_model_path": self._di_model_path,
                "static_di_latent": None if self._static_di_latent is None else self._static_di_latent.copy(),
                "static_depth_window": None if self._static_depth_window is None else self._static_depth_window.copy(),
                "depth_window_sub": self._depth_window_sub,
                "missing_depth_window_warned": self._missing_depth_window_warned,
                "proprioception_window": (
                    None if self._proprioception_window is None else self._proprioception_window.copy()
                ),
                "proprioception_initialized": self._proprioception_initialized,
                "policy_armed": self._policy_armed,
                "logged_fused_input_stats": self._logged_fused_input_stats,
                "saved_debug_depth_window": self._saved_debug_depth_window,
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self._apply_observation_config(state.get("obs_config", self._default_observation_config))
        self._residual_fused_export_spec = state.get("residual_fused_export_spec")
        self._uses_residual_fused_export = bool(state.get("uses_residual_fused_export", False))
        self._di_onnx_session = state.get("di_onnx_session")
        self._di_onnx_input_names = list(state.get("di_onnx_input_names", []))
        self._di_onnx_output_names = list(state.get("di_onnx_output_names", []))
        self._di_model_path = state.get("di_model_path")
        static_di_latent = state.get("static_di_latent")
        self._static_di_latent = None if static_di_latent is None else static_di_latent.copy()
        static_depth_window = state.get("static_depth_window")
        self._static_depth_window = None if static_depth_window is None else static_depth_window.copy()
        self._depth_window_sub = state.get("depth_window_sub")
        self._missing_depth_window_warned = bool(state.get("missing_depth_window_warned", False))
        proprioception_window = state.get("proprioception_window")
        self._proprioception_window = None if proprioception_window is None else proprioception_window.copy()
        self._proprioception_initialized = bool(state.get("proprioception_initialized", False))
        self.motion_command_0 = state["motion_command_0"].copy()
        self.ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.robot_yaw_offset = 0.0
        self._waiting_for_motion_clip_logged = False
        self._policy_armed = bool(state.get("policy_armed", False))
        self._logged_fused_input_stats = bool(state.get("logged_fused_input_stats", False))
        self._saved_debug_depth_window = bool(state.get("saved_debug_depth_window", False))

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self._stiff_hold_active = True
        self._policy_armed = False
        self.robot_yaw_offset = 0.0
        self._waiting_for_motion_clip_logged = False
        self._logged_fused_input_stats = False
        self._reset_proprioception_window()
        if self._depth_window_sub is not None:
            self._depth_window_sub.reset()

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to first pose in motion command
            target_dof_pos = self.motion_command_0[:, : self.num_dofs]

            q_target = dof_pos + (target_dof_pos - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = {}

        if self._uses_residual_fused_export:
            current_obs_buffer_dict["motion_command_joint_pos"] = self.motion_command_t[:, : self.num_dofs]
            current_obs_buffer_dict["obj_type_one_hot"] = self._get_object_type_one_hot()
            if self._residual_di_mode() != "fused":
                current_obs_buffer_dict["di_ae_latent"] = self._get_di_latent()
        else:
            # motion_command
            current_obs_buffer_dict["motion_command"] = self.motion_command_t

        # motion_ref_ori_b
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)  # wxyz
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        # robot_ref_ori
        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  #  wxyz
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)

        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

        # base_ang_vel
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]

        # dof_pos
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles

        # dof_vel
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # actions
        current_obs_buffer_dict["actions"] = self.last_policy_action

        return current_obs_buffer_dict

    def prepare_obs_for_rl(self, robot_state_data):
        if not self._uses_residual_fused_export:
            return super().prepare_obs_for_rl(robot_state_data)

        group_outputs = self._prepare_group_observations(robot_state_data)
        if "shared_obs" not in group_outputs:
            raise KeyError("Residual fused WBT policy requires 'shared_obs' observation group.")
        shared_obs = group_outputs["shared_obs"].astype(np.float32, copy=False)
        depth_window = None
        proprioception_window = None
        if self._residual_di_mode() == "fused":
            depth_window = self._get_fused_depth_window_or_zeros()
            obs_parts = [shared_obs, depth_window.reshape(depth_window.shape[0], -1)]
            if self._uses_fused_proprioception_window():
                proprioception_window = self._get_current_proprioception_window(robot_state_data)
                obs_parts.append(proprioception_window.reshape(proprioception_window.shape[0], -1))
            actor_obs = np.concatenate(obs_parts, axis=1).astype(np.float32, copy=False)
        else:
            if "di_ae_latent" not in group_outputs:
                raise KeyError("Residual fused WBT policy requires 'di_ae_latent' group in separate mode.")
            latent = group_outputs["di_ae_latent"].astype(np.float32, copy=False)
            actor_obs = np.concatenate([shared_obs, latent], axis=1).astype(np.float32, copy=False)

        obs_dict = dict(group_outputs)
        obs_dict["actor_obs"] = actor_obs
        if depth_window is not None:
            self._maybe_log_fused_model_input(shared_obs, depth_window, proprioception_window, actor_obs)
        return obs_dict

    def rl_inference(self, robot_state_data):
        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the configured start while waiting to trigger the clip.
            self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
            self.curr_motion_timestep = self.timestep_util.timestep
            if not self._waiting_for_motion_clip_logged:
                self.logger.info("Policy running at fixed start timestep; press `s` to play the motion clip.")
                self._waiting_for_motion_clip_logged = True

        obs = self.prepare_obs_for_rl(robot_state_data)
        if self.config.task.print_observations:
            self._print_observations(obs)

        input_feed = {"time_step": np.array([[self.curr_motion_timestep]], dtype=np.float32), "obs": obs["actor_obs"]}
        policy_action, self.motion_command_t, self.ref_quat_xyzw_t = self.policy(input_feed)

        # clip policy action
        policy_action = np.clip(policy_action, -100, 100)
        # store last policy action
        self.last_policy_action = policy_action.copy()
        # scale policy action
        self.scaled_policy_action = policy_action * self.policy_action_scale
        # update motion timestep
        self._set_motion_timestep()

        return self.scaled_policy_action

    def _get_manual_command(self, robot_state_data):
        # TODO: instead of adding kp/kd_override in def _set_motor_command,
        # just use the motor_kp/motor_kd when calling it in _fill_motor_commands
        if self.config.task.show_depth_window and self._policy_armed:
            self._get_current_depth_window()
        if not self._stiff_hold_active:
            return None
        return {
            "q": self._stiff_hold_q.copy(),
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._policy_armed = True
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self._waiting_for_motion_clip_logged = False
        self._logged_fused_input_stats = False
        self._reset_proprioception_window()
        if self._depth_window_sub is not None:
            self._depth_window_sub.reset()
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)

    def _set_motion_timestep(self):
        if self.motion_clip_progressing:
            prev = self.curr_motion_timestep

            if self.use_sim_time:
                self.curr_motion_timestep = self.timestep_util.get_timestep(log=self.logger)
            else:
                self.curr_motion_timestep += 1

            if self.curr_motion_timestep != prev:
                self.logger.info(f"Motion timestep: {prev} → {self.curr_motion_timestep}")  # noqa: G004

            # Stop motion clip at configured end timestep (keep policy running at final pose)
            if (end := self.config.task.motion_end_timestep) and self.curr_motion_timestep >= end:
                self.logger.info(colored(f"Reached end timestep {end}, stopping motion clip", "yellow"))
                self.motion_clip_progressing = False
                self.curr_motion_timestep = end

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self._policy_armed = False
        self.logger.info("Actions set to stiff startup command")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self.robot_yaw_offset = 0.0
        self._waiting_for_motion_clip_logged = False
        self._logged_fused_input_stats = False
        self._reset_proprioception_window()
        if self._depth_window_sub is not None:
            self._depth_window_sub.reset()

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.use_policy_action = True
        self.get_ready_state = False
        self._stiff_hold_active = False
        self._policy_armed = True
        self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.motion_clip_progressing = True
        self._waiting_for_motion_clip_logged = False
        self._logged_fused_input_stats = False
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)
        self._reset_proprioception_window()
        if self._depth_window_sub is not None:
            self._depth_window_sub.reset()

        if self.config.task.motion_start_timestep > 0 or self.config.task.motion_end_timestep is not None:
            start_str = str(self.config.task.motion_start_timestep)
            end_str = str(self.config.task.motion_end_timestep) if self.config.task.motion_end_timestep else "end"
            self.logger.info(colored(f"Starting motion clip from timestep {start_str} to {end_str}", "blue"))
        else:
            self.logger.info(colored("Starting motion clip", "blue"))

    def handle_keyboard_button(self, keycode):
        """Add new keyboard button to start and end the motion clips"""
        if keycode == "s":
            self._handle_start_motion_clip()
        else:
            super().handle_keyboard_button(keycode)

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses for WBT-specific controls."""
        if cur_key == "start":
            # Start playing motion clip
            self._handle_start_motion_clip()
        else:
            # Delegate all other buttons to base class
            super().handle_joystick_button(cur_key)
        super()._print_control_status()

    def _capture_robot_yaw_offset(self):
        """Capture robot yaw when policy starts to use as reference offset."""
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            self.logger.warning("Unable to capture robot yaw offset - missing robot state.")
            return

        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  # wxyz
        yaw = self._quat_yaw(robot_ref_ori)
        self.robot_yaw_offset = yaw
        self.logger.info(colored(f"Robot yaw offset captured at {np.degrees(yaw):.1f} deg", "blue"))

    def _capture_motion_yaw_offset(self, ref_quat_xyzw_0: np.ndarray) -> float:
        """Capture motion yaw when policy starts to use as reference offset."""
        self.motion_yaw_offset = self._quat_yaw(xyzw_to_wxyz(ref_quat_xyzw_0))
        self.logger.info(colored(f"Motion yaw offset captured at {np.degrees(self.motion_yaw_offset):.1f} deg", "blue"))

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        """Remove stored yaw offset from robot orientation quaternion."""
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        """Extract yaw angle from quaternion array of shape (1, 4)."""
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)
