from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.observation.terms.r1_fastsac import interaction_progress
from holosoma.utils.task_phase import TwoPhaseSchedule


def test_two_phase_schedule_uses_per_clip_local_boundaries(tmp_path):
    annotation = tmp_path / "phases.yaml"
    annotation.write_text(
        """
motions:
  object/clip_a.npz:
    phase_1_start_timestep: 3
  object/clip_b.npz:
    phase_1_start_timestep: 2
""".lstrip()
    )
    command = SimpleNamespace(
        device="cpu",
        motion=SimpleNamespace(
            clip_files=["/data/object/clip_a.npz", "/data/object/clip_b.npz"],
            clip_ranges=[(0, 10), (10, 20)],
        ),
        clip_ids=torch.tensor([0, 0, 1, 1]),
        time_steps=torch.tensor([2, 3, 11, 12]),
    )

    schedule = TwoPhaseSchedule(command, annotation_path=str(annotation))

    assert schedule.phase(command).tolist() == [0, 1, 0, 1]
    assert schedule.phase_1_start_steps_for_envs(command).tolist() == [3, 3, 12, 12]
    assert schedule.phase_1_end_steps_for_envs(command).tolist() == [10, 10, 20, 20]


def test_two_phase_schedule_anchors_annotations_after_prepended_frames(tmp_path):
    annotation = tmp_path / "phases.yaml"
    annotation.write_text(
        """
motions:
  object/clip_a.npz:
    phase_1_start_timestep: 3
  object/clip_b.npz:
    phase_1_start_timestep: 2
""".lstrip()
    )
    command = SimpleNamespace(
        device="cpu",
        motion=SimpleNamespace(
            clip_files=["/data/object/clip_a.npz", "/data/object/clip_b.npz"],
            # Each expanded clip has ten synthetic prepend frames and ten
            # synthetic append frames around ten frames of real motion.
            clip_ranges=[(0, 30), (30, 60)],
            real_motion_clip_ranges=[(10, 20), (40, 50)],
        ),
        clip_ids=torch.tensor([0, 0, 1, 1]),
        time_steps=torch.tensor([12, 13, 41, 42]),
    )

    schedule = TwoPhaseSchedule(command, annotation_path=str(annotation))

    assert schedule.phase(command).tolist() == [0, 1, 0, 1]
    assert schedule.phase_1_start_steps_for_envs(command).tolist() == [13, 13, 42, 42]
    assert schedule.phase_1_end_steps_for_envs(command).tolist() == [20, 20, 50, 50]


def test_interaction_progress_reaches_one_at_real_motion_end_and_stays_clamped(tmp_path):
    annotation = tmp_path / "phases.yaml"
    annotation.write_text(
        """
motions:
  object/clip_a.npz:
    phase_1_start_timestep: 3
""".lstrip()
    )
    command = SimpleNamespace(
        device="cpu",
        motion=SimpleNamespace(
            clip_files=["/data/object/clip_a.npz"],
            clip_ranges=[(0, 30)],
            real_motion_clip_ranges=[(10, 20)],
        ),
        clip_ids=torch.zeros(5, dtype=torch.long),
        time_steps=torch.tensor([12, 13, 19, 20, 25]),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_state=lambda _name: command),
    )

    progress = interaction_progress(env, annotation_path=str(annotation)).squeeze(-1)

    torch.testing.assert_close(progress, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))


def test_two_phase_schedule_validates_annotation_against_real_motion_length(tmp_path):
    annotation = tmp_path / "phases.yaml"
    annotation.write_text(
        """
motions:
  object/clip_a.npz:
    phase_1_start_timestep: 11
""".lstrip()
    )
    command = SimpleNamespace(
        device="cpu",
        motion=SimpleNamespace(
            clip_files=["/data/object/clip_a.npz"],
            clip_ranges=[(0, 30)],
            real_motion_clip_ranges=[(10, 20)],
        ),
        clip_ids=torch.tensor([0]),
        time_steps=torch.tensor([0]),
    )

    with pytest.raises(ValueError, match="10 real motion frames"):
        TwoPhaseSchedule(command, annotation_path=str(annotation))
