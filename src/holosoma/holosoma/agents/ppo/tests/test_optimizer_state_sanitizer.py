from __future__ import annotations

import torch
from torch import nn

from holosoma.agents.ppo.ppo import PPO


def test_sanitize_optimizer_state_resets_non_dict_state() -> None:
    module = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-5)
    param = next(module.parameters())
    optimizer.state[param] = dict.__getitem__

    ppo = object.__new__(PPO)
    ppo._sanitize_optimizer_state(optimizer, module, "actor_optimizer")

    assert optimizer.state[param] == {}


def test_sanitize_optimizer_state_keeps_valid_adamw_state() -> None:
    module = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-5)
    loss = module(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()

    param = next(module.parameters())
    state_before = optimizer.state[param]

    ppo = object.__new__(PPO)
    ppo._sanitize_optimizer_state(optimizer, module, "actor_optimizer")

    assert optimizer.state[param] is state_before
    assert set(("step", "exp_avg", "exp_avg_sq")).issubset(state_before)
