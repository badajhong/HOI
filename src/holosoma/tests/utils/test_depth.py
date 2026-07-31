import torch

from holosoma.utils.depth import preprocess_robot_depth_tensor


def test_preprocess_robot_depth_uniformly_samples_and_bounds_values():
    rows = torch.arange(480, dtype=torch.float32).view(480, 1)
    columns = torch.arange(640, dtype=torch.float32).view(1, 640)
    depth = 0.07 + rows * 0.001 + columns * 0.001
    depth[0, 0] = float("nan")
    depth[-1, -1] = 10.0

    output = preprocess_robot_depth_tensor(depth)

    expected_rows = torch.linspace(0, 479, 48).round().long()
    expected_columns = torch.linspace(0, 639, 64).round().long()
    expected = depth.index_select(0, expected_rows).index_select(1, expected_columns)
    expected = torch.where(
        torch.isfinite(expected) & (expected > 0.0),
        expected.clamp(0.07, 5.0),
        torch.zeros_like(expected),
    )
    assert output.shape == (48, 64)
    torch.testing.assert_close(output, expected)
