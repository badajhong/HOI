"""Shared robot depth-camera acquisition and preprocessing defaults."""

from __future__ import annotations

import numpy as np
import torch

ROBOT_DEPTH_RAW_RESOLUTION_WH = (640, 480)
ROBOT_DEPTH_OUTPUT_RESOLUTION_WH = (64, 48)
ROBOT_DEPTH_MIN_M = 0.07
ROBOT_DEPTH_MAX_M = 5.0
ROBOT_DEPTH_HORIZONTAL_FOV_DEG = 75.0
ROBOT_DEPTH_VERTICAL_FOV_DEG = 62.0
ROBOT_DEPTH_FOCAL_LENGTH = 24.0
ROBOT_DEPTH_HORIZONTAL_APERTURE = 36.831695422990094
ROBOT_DEPTH_VERTICAL_APERTURE = 28.8413097133229


def _uniform_indices(source_size: int, output_size: int, *, device: torch.device) -> torch.Tensor:
    if source_size < output_size:
        raise ValueError(f"Cannot uniformly sample {output_size} values from source size {source_size}.")
    return torch.linspace(0, source_size - 1, output_size, device=device).round().to(dtype=torch.long)


def preprocess_robot_depth_tensor(depth_frames: torch.Tensor) -> torch.Tensor:
    """Uniformly sample raw depth frames to 64x48 and apply metric depth bounds."""
    squeeze_batch = depth_frames.ndim == 2
    if squeeze_batch:
        depth_frames = depth_frames.unsqueeze(0)
    if depth_frames.ndim == 4 and depth_frames.shape[-1] == 1:
        depth_frames = depth_frames[..., 0]
    if depth_frames.ndim != 3:
        raise ValueError(f"Expected depth shape [N,H,W], [N,H,W,1], or [H,W], got {tuple(depth_frames.shape)}.")

    output_width, output_height = ROBOT_DEPTH_OUTPUT_RESOLUTION_WH
    source_height, source_width = depth_frames.shape[-2:]
    if (source_width, source_height) != ROBOT_DEPTH_OUTPUT_RESOLUTION_WH:
        row_indices = _uniform_indices(source_height, output_height, device=depth_frames.device)
        column_indices = _uniform_indices(source_width, output_width, device=depth_frames.device)
        depth_frames = depth_frames.index_select(-2, row_indices).index_select(-1, column_indices)

    valid = torch.isfinite(depth_frames) & (depth_frames > 0.0)
    bounded = torch.clamp(depth_frames, min=ROBOT_DEPTH_MIN_M, max=ROBOT_DEPTH_MAX_M)
    output = torch.where(valid, bounded, torch.zeros_like(bounded))
    return output[0] if squeeze_batch else output


def preprocess_robot_depth_array(depth_frame: np.ndarray) -> np.ndarray:
    """NumPy wrapper for the shared robot depth preprocessing convention."""
    tensor = torch.as_tensor(np.asarray(depth_frame), dtype=torch.float32)
    return preprocess_robot_depth_tensor(tensor).cpu().numpy()
