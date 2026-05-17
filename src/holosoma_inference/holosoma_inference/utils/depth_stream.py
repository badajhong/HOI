"""ZMQ depth-window subscriber for fused WBT inference."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import zmq
from loguru import logger


class DepthWindowSub:
    """Subscribe to simulator depth frames and build a [1, T, H, W] window."""

    def __init__(
        self,
        port: int = 5556,
        expected_shape: tuple[int, int, int] = (5, 60, 80),
        host: str = "localhost",
        timeout_ms: int = 1000,
        show_window: bool = False,
        display_scale: int = 4,
        window_name: str = "Holosoma simulator depth",
    ) -> None:
        self.port = int(port)
        self.expected_shape = tuple(int(value) for value in expected_shape)
        self.host = str(host)
        self.timeout_ms = int(timeout_ms)
        self.show_window = bool(show_window)
        self.display_scale = max(int(display_scale), 1)
        self.window_name = str(window_name)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.window: np.ndarray | None = None
        self.enabled = False
        self._logged_first_frame = False
        self._cv2: Any | None = None
        self._display_disabled = False

    def start(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.enabled = True
        logger.info(
            f"Depth frame subscriber started, connecting to {self.host}:{self.port}, "
            f"window_shape={self.expected_shape}"
        )
        if self.show_window and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            logger.warning("Depth debug window requested, but no display server was detected. Disabling window.")
            self._display_disabled = True

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        _, expected_height, expected_width = self.expected_shape
        frame = np.asarray(frame, dtype=np.float32)
        frame = np.squeeze(frame)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]

        if frame.shape == (expected_width, expected_height):
            frame = frame.T
        if frame.shape != (expected_height, expected_width):
            raise ValueError(
                f"Unexpected depth frame shape {frame.shape}; expected {(expected_height, expected_width)}."
            )

        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(frame, dtype=np.float32)

    def _decode_frame(self, header_bytes: bytes, payload: bytes) -> np.ndarray:
        header = json.loads(header_bytes.decode("utf-8"))
        shape = tuple(int(value) for value in header["shape"])
        dtype = np.dtype(header.get("dtype", "float32"))
        return np.frombuffer(payload, dtype=dtype).reshape(shape).astype(np.float32, copy=False)

    def _show_depth_frame(self, frame: np.ndarray) -> None:
        if not self.show_window or self._display_disabled:
            return

        try:
            if self._cv2 is None:
                import cv2  # noqa: PLC0415

                self._cv2 = cv2

            cv2 = self._cv2
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
            if self.display_scale != 1:
                color_preview = cv2.resize(
                    color_preview,
                    None,
                    fx=self.display_scale,
                    fy=self.display_scale,
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow(self.window_name, color_preview)
            cv2.waitKey(1)
        except Exception as exc:
            logger.warning(f"Disabling depth debug window after display failure: {type(exc).__name__}: {exc}")
            self._display_disabled = True

    def _receive_latest_frame(self) -> np.ndarray | None:
        if not self.enabled or self.socket is None:
            return None

        latest_frame: np.ndarray | None = None
        if self.window is None:
            try:
                header_bytes, payload = self.socket.recv_multipart()
                latest_frame = self._decode_frame(header_bytes, payload)
            except zmq.Again:
                return None

        while True:
            try:
                header_bytes, payload = self.socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

            latest_frame = self._decode_frame(header_bytes, payload)

        return latest_frame

    def get_depth_window(self) -> np.ndarray | None:
        latest_frame = self._receive_latest_frame()
        if latest_frame is None:
            return None if self.window is None else self.window[None, ...].copy()

        frame = self._prepare_frame(latest_frame)
        if not self._logged_first_frame:
            valid = frame[np.isfinite(frame) & (frame > 0.0)]
            if valid.size:
                logger.info(
                    "Received first simulator depth frame: "
                    f"shape={frame.shape}, min={float(valid.min()):.3f}, "
                    f"max={float(valid.max()):.3f}, mean={float(valid.mean()):.3f}, "
                    f"valid_ratio={valid.size / frame.size:.3f}"
                )
            else:
                logger.warning(
                    "Received first simulator depth frame with no positive finite pixels: "
                    f"shape={frame.shape}"
                )
            self._logged_first_frame = True

        self._show_depth_frame(frame)

        window_size, _, _ = self.expected_shape
        if self.window is None:
            self.window = np.repeat(frame[None, :, :], window_size, axis=0)
        else:
            self.window[:-1] = self.window[1:].copy()
            self.window[-1] = frame

        return self.window[None, ...].copy()

    def reset(self) -> None:
        self.window = None
        self._logged_first_frame = False

    def close(self) -> None:
        if self._cv2 is not None and self.show_window:
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception as exc:
                logger.debug(f"Ignoring depth debug window cleanup failure: {type(exc).__name__}: {exc}")
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()
        self.enabled = False
