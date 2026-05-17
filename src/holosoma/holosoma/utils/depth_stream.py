"""Small ZMQ helpers for streaming robot depth frames from simulation."""

from __future__ import annotations

import json

import numpy as np
import zmq
from loguru import logger


def depth_frame_to_numpy(depth_frame) -> np.ndarray:
    """Convert a simulator depth frame to a contiguous float32 [H, W] array."""
    if hasattr(depth_frame, "detach"):
        depth_frame = depth_frame.detach().cpu().numpy()

    frame = np.asarray(depth_frame, dtype=np.float32)
    frame = np.squeeze(frame)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    if frame.ndim != 2:
        raise ValueError(f"Expected depth frame shape [H, W], got {frame.shape}.")

    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(frame, dtype=np.float32)


class DepthFramePub:
    """Publish latest simulator depth frame as a ZMQ multipart message."""

    def __init__(self, port: int = 5556) -> None:
        self.port = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.enabled = False

    def start(self) -> None:
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.port}")
            self.enabled = True
            logger.info(f"Depth frame publisher started on port {self.port}")
        except Exception as exc:
            logger.error(f"Failed to start depth frame publisher: {exc}")
            self.enabled = False

    def publish(self, depth_frame, sim_time_ms: int | None = None) -> None:
        if not self.enabled or self.socket is None:
            return

        try:
            frame = depth_frame_to_numpy(depth_frame)
            header = {
                "shape": list(frame.shape),
                "dtype": "float32",
                "sim_time_ms": None if sim_time_ms is None else int(sim_time_ms),
            }
            self.socket.send_multipart(
                [json.dumps(header).encode("utf-8"), frame.tobytes()],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass
        except Exception as exc:
            logger.warning(f"Depth frame publish failed: {exc}")

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()
        self.enabled = False
