from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from holosoma.config_types.video import FixedCameraConfig, VideoConfig
from holosoma.simulator.shared.video_recorder import VideoRecorderInterface, _encode_video_array_worker


class _DummySimulator:
    def __init__(self) -> None:
        self.num_envs = 1
        self.device = "cpu"
        self.robot_root_states = torch.zeros(1, 13)
        self.simulator_config = SimpleNamespace(sim=SimpleNamespace(fps=60.0, control_decimation=4))


class _DummyRecorder(VideoRecorderInterface):
    def _capture_frame_impl(self) -> None:
        self._add_frame(np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8))


def test_video_encoding_runs_off_reset_path(monkeypatch, tmp_path):
    encoded_episodes: list[int] = []

    def fake_create_video(*, video_frames, fps, save_dir, output_format, wandb_logging, episode_id):
        encoded_episodes.append(episode_id)

    monkeypatch.setattr("holosoma.simulator.shared.video_recorder.create_video", fake_create_video)
    config = VideoConfig(
        camera=FixedCameraConfig(),
        save_dir=str(tmp_path),
        async_encoding=True,
        async_encoding_backend="thread",
        upload_to_wandb=True,
    )
    recorder = _DummyRecorder(config, _DummySimulator())

    recorder.start_recording(episode_id=42)
    for _ in range(4):
        recorder.capture_frame()
    recorder.stop_recording()

    assert recorder._get_frame_count() == 0
    assert recorder._encoding_futures

    recorder.cleanup()
    assert encoded_episodes == [42]


def test_process_worker_encodes_video_without_wandb(tmp_path):
    video_array = np.zeros((2, 16, 16, 3), dtype=np.uint8)

    video_path = _encode_video_array_worker(
        video_array,
        fps=15.0,
        save_dir=str(tmp_path),
        output_format="mp4",
        episode_id=7,
    )

    assert video_path is not None
    assert video_path.endswith(".mp4")
    assert (tmp_path / video_path.split("/")[-1]).exists()
