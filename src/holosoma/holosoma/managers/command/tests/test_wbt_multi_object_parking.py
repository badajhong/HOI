from __future__ import annotations

from types import MethodType
from types import SimpleNamespace

import torch

from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler, INACTIVE_OBJECT_PARK_DEPTH_M, MotionCommand
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


class FakeSimulator:
    def __init__(self, env_origins: torch.Tensor):
        self.scene = SimpleNamespace(env_origins=env_origins)
        self.actor_state_calls: list[dict[str, object]] = []
        self.write_state_update_count = 0

    def set_actor_states(
        self,
        actor_names: list[str],
        env_ids: torch.Tensor,
        states: torch.Tensor,
        write_updates: bool = True,
    ) -> None:
        self.actor_state_calls.append(
            {
                "actor_names": tuple(actor_names),
                "env_ids": env_ids.detach().cpu().clone(),
                "states": states.detach().cpu().clone(),
                "write_updates": write_updates,
            }
        )

    def write_state_updates(self) -> None:
        self.write_state_update_count += 1


def _make_multi_object_command(
    object_type_ids: list[int],
    simulator_active_object_type_ids: list[int],
) -> tuple[MotionCommand, FakeSimulator]:
    num_envs = len(object_type_ids)
    env_origins = torch.tensor(
        [[float(env_id) * 3.0, 0.0, 0.0] for env_id in range(num_envs)],
        dtype=torch.float32,
    )
    simulator = FakeSimulator(env_origins)
    resmimicchair_scales = torch.full((num_envs, 3), 0.5, dtype=torch.float32)
    whitechair_scales = torch.full((num_envs, 3), 1.5, dtype=torch.float32)

    command = object.__new__(MotionCommand)
    command.motion = SimpleNamespace(has_object=True)
    command.device = "cpu"
    command.num_envs = num_envs
    command._env = SimpleNamespace(
        simulator=simulator,
        object_scale_factors=torch.ones(num_envs, 3, dtype=torch.float32),
        object_scale_factors_z=torch.ones(num_envs, dtype=torch.float32),
        object_scale_factors_by_actor={
            "object_resmimicchair": resmimicchair_scales,
            "object_whitechair": whitechair_scales,
        },
        object_scale_factors_z_by_actor={
            "object_resmimicchair": resmimicchair_scales[:, 2],
            "object_whitechair": whitechair_scales[:, 2],
        },
    )
    command.object_name_to_indices = {
        "resmimicchair": torch.tensor([10, 11, 12], dtype=torch.long),
        "whitechair": torch.tensor([20, 21, 22], dtype=torch.long),
    }
    command.object_key_to_id = {"resmimicchair": 0, "whitechair": 1}
    command.object_type_ids = torch.tensor(object_type_ids, dtype=torch.long)
    command.simulator_active_object_type_ids = torch.tensor(simulator_active_object_type_ids, dtype=torch.long)
    command.active_object_indices = torch.full((num_envs,), -1, dtype=torch.long)
    return command, simulator


def _object_states(num_envs: int) -> torch.Tensor:
    states = torch.zeros((num_envs, 13), dtype=torch.float32)
    states[:, 0] = torch.arange(num_envs, dtype=torch.float32)
    states[:, 6] = 1.0
    return states


def _calls_for_actor(simulator: FakeSimulator, actor_name: str) -> list[dict[str, object]]:
    return [call for call in simulator.actor_state_calls if call["actor_names"] == (actor_name,)]


def test_multi_object_sync_reparks_inactive_chair_even_without_type_change() -> None:
    command, simulator = _make_multi_object_command(
        object_type_ids=[1, 1, 1],
        simulator_active_object_type_ids=[1, 1, 1],
    )
    env_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    object_states = _object_states(env_ids.numel())

    command.set_simulator_object_states(env_ids, object_states)

    whitechair_calls = _calls_for_actor(simulator, "object_whitechair")
    resmimicchair_calls = _calls_for_actor(simulator, "object_resmimicchair")

    assert len(whitechair_calls) == 1
    assert whitechair_calls[0]["env_ids"].tolist() == [0, 1, 2]
    assert torch.allclose(whitechair_calls[0]["states"], object_states)
    assert whitechair_calls[0]["write_updates"] is False

    assert len(resmimicchair_calls) == 1
    assert resmimicchair_calls[0]["env_ids"].tolist() == [0, 1, 2]
    parked_states = resmimicchair_calls[0]["states"]
    expected_parked_z = simulator.scene.env_origins[env_ids, 2] - INACTIVE_OBJECT_PARK_DEPTH_M
    assert torch.allclose(parked_states[:, 2], expected_parked_z)
    assert torch.allclose(parked_states[:, 3:7], torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1))
    assert resmimicchair_calls[0]["write_updates"] is False

    assert command.active_object_indices.tolist() == [20, 21, 22]
    assert command.simulator_active_object_type_ids.tolist() == [1, 1, 1]
    torch.testing.assert_close(
        command._env.object_scale_factors,
        command._env.object_scale_factors_by_actor["object_whitechair"],
    )
    torch.testing.assert_close(
        command._env.object_scale_factors_z,
        command._env.object_scale_factors_z_by_actor["object_whitechair"],
    )
    assert simulator.write_state_update_count == 1


def _make_transition_command() -> MotionCommand:
    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command._body_indexes_in_motion = torch.tensor([0], dtype=torch.long)
    command._joint_indexes_in_motion = torch.tensor([0], dtype=torch.long)

    class FakeMotion(SimpleNamespace):
        @property
        def has_multiple_clips(self) -> bool:
            return len(self.clip_ranges) > 1

        def validate_time_major_tensor_lengths(self) -> None:
            expected_len = int(self.time_step_total)
            for attr_name in (
                "_joint_pos",
                "_joint_vel",
                "_body_pos_w",
                "_body_quat_w",
                "_body_lin_vel_w",
                "_body_ang_vel_w",
                "_object_pos_w",
                "_object_quat_w",
                "_object_lin_vel_w",
                "_object_ang_vel_w",
                "_contact_object_label",
                "_contact_object_distance",
                "_contact_object_target_points_obj",
                "_contact_object_target_valid",
            ):
                assert int(getattr(self, attr_name).shape[0]) == expected_len

    motion = FakeMotion(
        has_object=True,
        has_contact_labels=True,
        clip_ranges=[(0, 2), (2, 4)],
        real_motion_clip_ranges=[(0, 2), (2, 4)],
        clip_files=["clip_a.npz", "clip_b.npz"],
        clip_object_keys=["box", "chair"],
        contact_body_names=["hand"],
        _joint_pos=torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
        _joint_vel=torch.zeros(4, 1),
        _body_pos_w=torch.zeros(4, 1, 3),
        _body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * 4),
        _body_lin_vel_w=torch.zeros(4, 1, 3),
        _body_ang_vel_w=torch.zeros(4, 1, 3),
        _object_pos_w=torch.zeros(4, 3),
        _object_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 4),
        _object_lin_vel_w=torch.zeros(4, 3),
        _object_ang_vel_w=torch.zeros(4, 3),
        _contact_object_label=torch.tensor([[True], [False], [False], [True]]),
        _contact_object_distance=torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
        _contact_object_target_points_obj=torch.tensor([10.0, 20.0, 30.0, 40.0])
        .view(4, 1, 1, 1)
        .expand(4, 1, 2, 3)
        .clone(),
        _contact_object_target_valid=torch.tensor([[True], [True], [False], [True]]),
        time_step_total=4,
    )
    command.motion = motion

    def build_default_pose_state(self, *, use_motion_end: bool = False, motion_idx: int | None = None):
        del use_motion_end
        return {"anchor_idx": int(motion_idx)}

    def default_motion_state(self, default_state, dtype, device):
        del dtype, device
        return default_state

    def motion_state(self, idx, dtype, device):
        del dtype, device
        return {"anchor_idx": int(idx)}

    def build_transition_segment(
        self,
        *,
        start_state,
        target_state,
        num_steps,
        drop_first,
        drop_last,
        dtype,
        device,
    ):
        del self, drop_first, drop_last
        value = float(1000 + start_state["anchor_idx"] * 10 + target_state["anchor_idx"])
        return {
            "joint_pos": torch.full((num_steps, 1), value, dtype=dtype, device=device),
            "joint_vel": torch.full((num_steps, 1), value, dtype=dtype, device=device),
            "body_pos": torch.full((num_steps, 1, 3), value, dtype=dtype, device=device),
            "body_quat": torch.full((num_steps, 1, 4), value, dtype=dtype, device=device),
            "body_lin_vel": torch.full((num_steps, 1, 3), value, dtype=dtype, device=device),
            "body_ang_vel": torch.full((num_steps, 1, 3), value, dtype=dtype, device=device),
            "object_pos": torch.full((num_steps, 3), value, dtype=dtype, device=device),
            "object_quat": torch.full((num_steps, 4), value, dtype=dtype, device=device),
            "object_lin_vel": torch.full((num_steps, 3), value, dtype=dtype, device=device),
            "object_ang_vel": torch.full((num_steps, 3), value, dtype=dtype, device=device),
        }

    command._build_default_pose_state = MethodType(build_default_pose_state, command)
    command._default_motion_state = MethodType(default_motion_state, command)
    command._motion_state = MethodType(motion_state, command)
    command._build_transition_segment = MethodType(build_transition_segment, command)
    return command


def test_default_pose_transitions_are_added_to_every_motion_clip() -> None:
    command = _make_transition_command()

    command._add_transition_to_each_clip(num_steps=2, prepend=True)

    assert command.motion.clip_ranges == [(0, 4), (4, 8)]
    assert command.motion.real_motion_clip_ranges == [(2, 4), (6, 8)]
    assert command.motion.time_step_total == 8
    torch.testing.assert_close(
        command.motion._joint_pos.flatten(),
        torch.tensor([1000.0, 1000.0, 10.0, 20.0, 1022.0, 1022.0, 30.0, 40.0]),
    )
    torch.testing.assert_close(
        command.motion._contact_object_label.flatten(),
        torch.tensor([False, False, True, False, False, False, False, True]),
    )
    torch.testing.assert_close(
        command.motion._contact_object_distance.flatten(),
        torch.tensor([float("inf"), float("inf"), 0.1, 0.2, float("inf"), float("inf"), 0.3, 0.4]),
    )
    torch.testing.assert_close(
        command.motion._contact_object_target_valid.flatten(),
        torch.tensor([False, False, True, True, False, False, False, True]),
    )
    torch.testing.assert_close(
        command.motion._contact_object_target_points_obj[:, 0, 0, 0],
        torch.tensor([0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 30.0, 40.0]),
    )

    command._add_transition_to_each_clip(num_steps=2, prepend=False)

    assert command.motion.clip_ranges == [(0, 6), (6, 12)]
    assert command.motion.real_motion_clip_ranges == [(2, 4), (8, 10)]
    assert command.motion.time_step_total == 12
    torch.testing.assert_close(
        command.motion._joint_pos.flatten(),
        torch.tensor(
            [
                1000.0,
                1000.0,
                10.0,
                20.0,
                1033.0,
                1033.0,
                1022.0,
                1022.0,
                30.0,
                40.0,
                1077.0,
                1077.0,
            ]
        ),
    )
    torch.testing.assert_close(
        command.motion._contact_object_target_valid.flatten(),
        torch.tensor([False, False, True, True, False, False, False, False, False, True, False, False]),
    )
    torch.testing.assert_close(
        command.motion._contact_object_target_points_obj[:, 0, 0, 0],
        torch.tensor([0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 30.0, 40.0, 0.0, 0.0]),
    )


def test_transition_segment_helper_is_available_and_interpolates() -> None:
    command = object.__new__(MotionCommand)
    command.motion = SimpleNamespace(has_object=True)

    quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    start_state = {
        "joint_pos": torch.tensor([0.0]),
        "joint_vel": torch.tensor([0.0]),
        "body_pos": torch.zeros(1, 3),
        "body_quat": quat.unsqueeze(0),
        "body_lin_vel": torch.zeros(1, 3),
        "body_ang_vel": torch.zeros(1, 3),
        "object_pos": torch.zeros(3),
        "object_quat": quat,
        "object_lin_vel": torch.zeros(3),
        "object_ang_vel": torch.zeros(3),
    }
    target_state = {
        "joint_pos": torch.tensor([2.0]),
        "joint_vel": torch.tensor([4.0]),
        "body_pos": torch.ones(1, 3) * 2.0,
        "body_quat": quat.unsqueeze(0),
        "body_lin_vel": torch.ones(1, 3) * 4.0,
        "body_ang_vel": torch.ones(1, 3) * 6.0,
        "object_pos": torch.ones(3) * 8.0,
        "object_quat": quat,
        "object_lin_vel": torch.ones(3) * 10.0,
        "object_ang_vel": torch.ones(3) * 12.0,
    }

    segment = command._build_transition_segment(
        start_state=start_state,
        target_state=target_state,
        num_steps=2,
        drop_first=False,
        drop_last=True,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(segment["joint_pos"].flatten(), torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(segment["joint_vel"].flatten(), torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(segment["body_pos"][:, 0, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(segment["object_pos"][:, 0], torch.tensor([0.0, 4.0]))


def test_completion_percent_start0_uses_real_motion_range_without_renaming_metric() -> None:
    env = object.__new__(WholeBodyTrackingManager)
    env.device = torch.device("cpu")
    env.reset_buf = torch.ones(4, dtype=torch.bool)
    env.log_dict = {}

    motion = SimpleNamespace(
        real_motion_clip_ranges=[(10, 20)],
        clip_files=["clip_a.npz"],
        clip_object_keys=["box"],
    )
    motion_command = SimpleNamespace(
        motion=motion,
        started_at_timestep_zero=torch.ones(4, dtype=torch.bool),
        clip_ids=torch.zeros(4, dtype=torch.long),
        clip_start_steps=torch.zeros(4, dtype=torch.long),
        clip_end_steps=torch.full((4,), 30, dtype=torch.long),
        time_steps=torch.tensor([5, 10, 14, 25], dtype=torch.long),
    )

    env._update_motion_start0_log_dict(motion_command)

    metric = env.log_dict["Motion/box/clip_a/completion_percent_start0"]
    torch.testing.assert_close(metric, torch.tensor([0.0, 0.0, 50.0, 100.0]))


def test_adaptive_timestep_sampler_respects_clip_ranges() -> None:
    sampler = AdaptiveTimestepsSampler(motion_time_step_total=100, device="cpu", env_fps=10)
    sampler.bin_failed_count[:] = 1.0
    sampler.bin_failed_count[8] = 100.0

    starts = torch.tensor([0, 20, 80, 80], dtype=torch.long)
    ends_exclusive = torch.tensor([10, 40, 90, 90], dtype=torch.long)

    sampled = sampler.sample_time_steps_in_ranges(starts, ends_exclusive)

    assert sampled.shape == starts.shape
    assert torch.all(sampled >= starts)
    assert torch.all(sampled < ends_exclusive)
