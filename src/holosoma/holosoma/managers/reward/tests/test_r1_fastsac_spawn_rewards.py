from types import SimpleNamespace

import torch

from holosoma.managers.reward.terms.r1_fastsac import _spawn_adjusted_object_position


def test_spawn_adjusted_object_position_preserves_reference_delta_and_xyz_offset() -> None:
    command = SimpleNamespace(
        object_pos_w=torch.tensor(
            [
                [1.0, 2.0, 0.50],
                [-1.0, 4.0, 0.25],
            ]
        ),
        # Includes XY sector randomization and scale-dependent grounding in Z.
        object_pos_reward_offset=torch.tensor(
            [
                [0.40, -0.20, 0.15],
                [-0.10, 0.30, -0.05],
            ]
        ),
    )

    actual = _spawn_adjusted_object_position(command)

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                [1.40, 1.80, 0.65],
                [-1.10, 4.30, 0.20],
            ]
        ),
    )


def test_spawn_adjusted_object_position_is_reference_without_randomization() -> None:
    reference = torch.tensor([[0.2, -0.1, 0.4]])
    command = SimpleNamespace(object_pos_w=reference)

    actual = _spawn_adjusted_object_position(command)

    torch.testing.assert_close(actual, reference)
