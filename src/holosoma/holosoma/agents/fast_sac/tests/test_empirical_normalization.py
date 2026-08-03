from __future__ import annotations

import torch

from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization


def test_empirical_normalization_merges_between_batch_variance() -> None:
    normalizer = EmpiricalNormalization(shape=1, device="cpu", eps=0.0)
    normalizer.train()

    normalizer(torch.zeros(100, 1), update=True)
    normalizer(torch.full((100, 1), 10.0), update=True)

    torch.testing.assert_close(normalizer.mean, torch.tensor([5.0]))
    torch.testing.assert_close(normalizer.std, torch.tensor([5.0]))
    assert int(normalizer.count.item()) == 200


def test_empirical_normalization_matches_concatenated_population_stats() -> None:
    generator = torch.Generator().manual_seed(17)
    batches = [
        torch.randn(31, 4, generator=generator) * 0.5 - 3.0,
        torch.randn(47, 4, generator=generator) * 2.0 + 5.0,
        torch.randn(19, 4, generator=generator) * 1.5 + 1.0,
    ]
    expected = torch.cat(batches, dim=0)

    normalizer = EmpiricalNormalization(shape=4, device="cpu", eps=0.0)
    normalizer.train()
    for batch in batches:
        normalizer(batch, update=True)

    torch.testing.assert_close(normalizer.mean, expected.mean(dim=0), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        normalizer.std,
        expected.var(dim=0, unbiased=False).sqrt(),
        rtol=1e-5,
        atol=1e-6,
    )

