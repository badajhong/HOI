from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from holosoma.config_values.wbt.r1.experiment import r1_teacher
from holosoma.eval_teacher_agent import (
    TeacherEvalConfig,
    _apply_motion_overrides,
    _preserve_checkpoint_object_type_space,
)


def _config_with_adaptive_motion(*, as_dict: bool):
    setup_term = r1_teacher.command.setup_terms["motion_command"]
    motion_config = dataclasses.replace(
        setup_term.params["motion_config"],
        start_at_timestep_zero_prob=0.4,
        use_adaptive_timesteps_sampler=True,
        adaptive_timestep_sampling_ratio=0.3,
    )
    if as_dict:
        motion_config = dataclasses.asdict(motion_config)

    params = dict(setup_term.params)
    params["motion_config"] = motion_config
    setup_terms = dict(r1_teacher.command.setup_terms)
    setup_terms["motion_command"] = dataclasses.replace(setup_term, params=params)
    return dataclasses.replace(
        r1_teacher,
        command=dataclasses.replace(r1_teacher.command, setup_terms=setup_terms),
    )


@pytest.mark.parametrize("as_dict", [False, True])
def test_teacher_eval_start_zero_disables_training_only_adaptive_sampling(as_dict: bool) -> None:
    config = _config_with_adaptive_motion(as_dict=as_dict)
    cli_config = TeacherEvalConfig(
        motion_file="motion.npz",
        object_urdf_path=None,
        start_at_timestep_zero_prob=1.0,
    )

    updated = _apply_motion_overrides(config, cli_config)
    motion_config = updated.command.setup_terms["motion_command"].params["motion_config"]

    if isinstance(motion_config, dict):
        assert motion_config["start_at_timestep_zero_prob"] == 1.0
        assert motion_config["use_adaptive_timesteps_sampler"] is False
        assert motion_config["adaptive_timestep_sampling_ratio"] == 0.0
    else:
        assert motion_config.start_at_timestep_zero_prob == 1.0
        assert motion_config.use_adaptive_timesteps_sampler is False
        assert motion_config.adaptive_timestep_sampling_ratio == 0.0


def test_preserve_checkpoint_object_space_keeps_only_active_runtime_keys() -> None:
    motion_command = SimpleNamespace(
        motion=SimpleNamespace(has_object=True, clip_object_keys=["whitechair"]),
        num_object_types=1,
        device=torch.device("cpu"),
        clip_ids=torch.zeros(4, dtype=torch.long),
        object_type_ids=torch.zeros(4, dtype=torch.long),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    )
    checkpoint_object_keys = ["largebox", "plasticbox", "smalltable", "suitcase", "whitechair"]
    saved_config = SimpleNamespace(
        robot=SimpleNamespace(
            object=SimpleNamespace(
                object_urdf_name_to_path={object_key: f"{object_key}.urdf" for object_key in checkpoint_object_keys}
            )
        )
    )

    _preserve_checkpoint_object_type_space(env, saved_config)

    assert motion_command.num_object_types == 5
    assert motion_command.object_key_to_id == {"whitechair": 4}
    assert motion_command.object_type_id_per_clip.tolist() == [4]
    assert motion_command.object_type_ids.tolist() == [4, 4, 4, 4]
    assert torch.nn.functional.one_hot(
        motion_command.object_type_ids,
        num_classes=motion_command.num_object_types,
    ).shape == (4, 5)
