from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.dagger_student.teacher_transition_h5 import (
    TEACHER_TRANSITION_FIELDS,
    TeacherTransitionH5Writer,
    infer_observation_mode,
)
from holosoma.agents.fast_sac.fast_sac import Critic
from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.module_utils import setup_ppo_actor_module, setup_ppo_critic_module
from holosoma.config_types.algo import DaggerStudentConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.eval_utils import CheckpointConfig, load_checkpoint, load_saved_experiment_config
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)


def _valid_teacher_action_rows(
    teacher_actions: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Identify finite teacher rows whose every raw action is within bounds."""

    return torch.isfinite(teacher_actions).all(dim=-1) & (
        teacher_actions.abs().amax(dim=-1) <= float(threshold)
    )


class StackDaggerBuffer:
    _TENSOR_NAMES = (
        "obs",
        "actions",
        "critic_obs",
        "executed_actions",
        "is_student_action",
        "teacher_action_valid",
        "rewards",
        "next_obs",
        "next_critic_obs",
        "terminals",
    )

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        storage_device: str | torch.device,
    ):
        if capacity <= 0:
            raise ValueError(f"Stack buffer capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.storage_device = torch.device(storage_device)
        self.pin_memory = self.storage_device.type == "cpu"
        self.obs = torch.empty(
            (self.capacity, obs_dim),
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=self.pin_memory,
        )
        self.actions = torch.empty(
            (self.capacity, action_dim),
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=self.pin_memory,
        )
        self.critic_obs = torch.empty(
            (self.capacity, critic_obs_dim), dtype=torch.float32, device=self.storage_device,
            pin_memory=self.pin_memory,
        )
        self.next_obs = torch.empty_like(self.obs)
        self.next_critic_obs = torch.empty_like(self.critic_obs)
        self.executed_actions = torch.empty_like(self.actions)
        self.is_student_action = torch.empty(
            (self.capacity, 1), dtype=torch.float32, device=self.storage_device, pin_memory=self.pin_memory
        )
        self.teacher_action_valid = torch.empty_like(self.is_student_action)
        self.rewards = torch.empty(
            (self.capacity, 1), dtype=torch.float32, device=self.storage_device, pin_memory=self.pin_memory
        )
        self.terminals = torch.empty(
            (self.capacity, 1), dtype=torch.float32, device=self.storage_device, pin_memory=self.pin_memory
        )
        self.size = 0
        self.write_idx = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: torch.Tensor,
        teacher_actions: torch.Tensor,
        critic_obs: torch.Tensor,
        executed_actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        terminals: torch.Tensor,
        is_student_action: torch.Tensor | None = None,
        teacher_action_valid: torch.Tensor | None = None,
    ) -> None:
        actions = teacher_actions
        if obs.shape[0] != actions.shape[0]:
            raise ValueError(f"Buffer add mismatch: obs batch {obs.shape[0]} vs actions batch {actions.shape[0]}")

        obs_storage = obs.detach().to(
            device=self.storage_device,
            dtype=torch.float32,
            # The buffer can be sampled immediately after collection.  An
            # asynchronous CUDA-to-CPU copy would expose partially written
            # pinned memory to the first optimization batch.
            non_blocking=False,
        )
        actions_storage = actions.detach().to(
            device=self.storage_device,
            dtype=torch.float32,
            non_blocking=False,
        )
        if is_student_action is None:
            # Keep the public buffer API compatible with checkpoints/tests
            # created before execution-source tracking was added. Unknown
            # transitions are conservatively excluded from V learning.
            is_student_action = torch.zeros_like(rewards, dtype=torch.float32)
        if teacher_action_valid is None:
            teacher_action_valid = torch.ones_like(rewards, dtype=torch.float32)
        extras = [
            critic_obs,
            executed_actions,
            is_student_action.view(-1, 1),
            teacher_action_valid.view(-1, 1),
            rewards.view(-1, 1),
            next_obs,
            next_critic_obs,
            terminals.view(-1, 1),
        ]
        extra_storage = [
            value.detach().to(device=self.storage_device, dtype=torch.float32, non_blocking=False)
            for value in extras
        ]
        batch_size = int(obs_storage.shape[0])
        if batch_size == 0:
            return

        if batch_size >= self.capacity:
            obs_storage = obs_storage[-self.capacity :]
            actions_storage = actions_storage[-self.capacity :]
            extra_storage = [value[-self.capacity :] for value in extra_storage]
            batch_size = self.capacity

        destinations = [
            self.obs, self.actions, self.critic_obs, self.executed_actions,
            self.is_student_action, self.teacher_action_valid, self.rewards,
            self.next_obs, self.next_critic_obs, self.terminals,
        ]
        sources = [obs_storage, actions_storage, *extra_storage]

        end_idx = self.write_idx + batch_size
        if end_idx <= self.capacity:
            for destination, source in zip(destinations, sources):
                destination[self.write_idx : end_idx].copy_(source, non_blocking=False)
        else:
            first_chunk = self.capacity - self.write_idx
            second_chunk = batch_size - first_chunk
            for destination, source in zip(destinations, sources):
                destination[self.write_idx :].copy_(source[:first_chunk], non_blocking=False)
                destination[:second_chunk].copy_(source[first_chunk:], non_blocking=False)

        self.write_idx = (self.write_idx + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, batch_size: int, device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty DAgger buffer.")

        indices = torch.randint(0, self.size, (batch_size,), device=self.storage_device)
        batch = {
            "obs": self.obs.index_select(0, indices),
            "teacher_actions": self.actions.index_select(0, indices),
            "critic_obs": self.critic_obs.index_select(0, indices),
            "actions": self.executed_actions.index_select(0, indices),
            "is_student_action": self.is_student_action.index_select(0, indices),
            "teacher_action_valid": self.teacher_action_valid.index_select(0, indices),
            "rewards": self.rewards.index_select(0, indices),
            "next_obs": self.next_obs.index_select(0, indices),
            "next_critic_obs": self.next_critic_obs.index_select(0, indices),
            "terminals": self.terminals.index_select(0, indices),
        }

        if device is not None:
            target_device = torch.device(device)
            if target_device != self.storage_device:
                batch = {
                    key: value.to(device=target_device, dtype=torch.float32, non_blocking=self.pin_memory)
                    for key, value in batch.items()
                }
        return batch

    def valid_actor_indices(self) -> torch.Tensor:
        """Return storage-device indices whose raw teacher labels were valid."""

        return torch.nonzero(
            self.teacher_action_valid[: self.size, 0] > 0.5,
            as_tuple=False,
        ).squeeze(-1)

    def sample_actor(
        self,
        batch_size: int,
        valid_indices: torch.Tensor,
        device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample actor observations and valid teacher labels with replacement."""

        if batch_size <= 0:
            raise ValueError(f"Actor batch size must be positive, got {batch_size}.")
        if valid_indices.numel() == 0:
            raise RuntimeError("Cannot sample actor labels: recent DAgger buffer has no valid teacher labels.")
        valid_indices = valid_indices.to(device=self.storage_device, dtype=torch.long)
        sampled = torch.randint(
            0,
            valid_indices.numel(),
            (batch_size,),
            device=self.storage_device,
        )
        indices = valid_indices.index_select(0, sampled)
        batch = {
            "obs": self.obs.index_select(0, indices),
            "teacher_actions": self.actions.index_select(0, indices),
        }
        if device is not None:
            target_device = torch.device(device)
            if target_device != self.storage_device:
                batch = {
                    key: value.to(
                        device=target_device,
                        dtype=torch.float32,
                        non_blocking=self.pin_memory,
                    )
                    for key, value in batch.items()
                }
        return batch

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return the valid replay contents and circular-buffer cursor.

        Only initialized rows are serialized while the buffer is still filling.
        Once full, preserving the physical ring layout together with
        ``write_idx`` makes the restored buffer exactly equivalent.
        """

        return {
            "version": 3,
            "capacity": self.capacity,
            "size": self.size,
            "write_idx": self.write_idx,
            "tensors": {
                # A sliced tensor still references the full backing storage in
                # torch serialization. Clone partial buffers so checkpoints do
                # not contain uninitialized capacity; a full buffer can be
                # serialized directly without doubling its memory footprint.
                name: (
                    getattr(self, name).detach()
                    if self.size == self.capacity
                    else getattr(self, name)[: self.size].detach().clone()
                )
                for name in self._TENSOR_NAMES
            },
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`get_checkpoint_state`."""

        version = int(state.get("version", 1))
        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported DAgger buffer checkpoint version: {version}")

        saved_capacity = int(state["capacity"])
        if saved_capacity != self.capacity:
            raise ValueError(
                "DAgger buffer capacity must match for an exact resume: "
                f"checkpoint={saved_capacity}, configured={self.capacity}."
            )

        size = int(state["size"])
        write_idx = int(state["write_idx"])
        if not 0 <= size <= self.capacity:
            raise ValueError(f"Invalid saved DAgger buffer size {size} for capacity {self.capacity}.")
        if not 0 <= write_idx < self.capacity:
            raise ValueError(f"Invalid saved DAgger buffer write_idx {write_idx} for capacity {self.capacity}.")

        tensors = state.get("tensors")
        if not isinstance(tensors, dict):
            raise ValueError("DAgger buffer checkpoint is missing its tensor payload.")
        for name in self._TENSOR_NAMES:
            if name not in tensors:
                if version == 1 and name == "is_student_action":
                    # Version 1 did not record which policy generated the
                    # executed action. Conservatively exclude these unknown
                    # rows from V learning; Q can still use all of them.
                    self.is_student_action[:size].zero_()
                    continue
                if version <= 2 and name == "teacher_action_valid":
                    # Older buffers retained only clipped teacher labels, so
                    # raw outliers cannot be identified safely. Keep their
                    # transitions for V/Q but exclude them from actor BC.
                    self.teacher_action_valid[:size].zero_()
                    continue
                raise ValueError(f"DAgger buffer checkpoint is missing tensor '{name}'.")
            destination = getattr(self, name)
            source = tensors[name]
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"Saved DAgger buffer field '{name}' must be a tensor.")
            expected_shape = (size, *destination.shape[1:])
            if tuple(source.shape) != expected_shape:
                raise ValueError(
                    f"Saved DAgger buffer field '{name}' has shape {tuple(source.shape)}, "
                    f"expected {expected_shape}."
                )
            if size > 0:
                destination[:size].copy_(
                    source.to(device=self.storage_device, dtype=destination.dtype),
                    non_blocking=False,
                )

        self.size = size
        self.write_idx = write_idx


class FixedActorAnchorBuffer:
    """Non-overwriting valid-teacher anchor used only for actor supervision."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        storage_device: str | torch.device,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"Teacher anchor capacity must be positive, got {capacity}.")
        self.capacity = int(capacity)
        self.storage_device = torch.device(storage_device)
        self.pin_memory = self.storage_device.type == "cpu"
        self.obs = torch.empty(
            (self.capacity, obs_dim),
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=self.pin_memory,
        )
        self.actions = torch.empty(
            (self.capacity, action_dim),
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=self.pin_memory,
        )
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs: torch.Tensor, teacher_actions: torch.Tensor) -> int:
        """Append up to the remaining capacity and never overwrite old rows."""

        if obs.shape[0] != teacher_actions.shape[0]:
            raise ValueError(
                f"Anchor add mismatch: obs batch {obs.shape[0]} vs actions batch "
                f"{teacher_actions.shape[0]}."
            )
        remaining = self.capacity - self.size
        accepted = min(int(obs.shape[0]), remaining)
        if accepted <= 0:
            return 0
        end = self.size + accepted
        self.obs[self.size : end].copy_(
            obs[:accepted].detach().to(self.storage_device, dtype=torch.float32),
            non_blocking=False,
        )
        self.actions[self.size : end].copy_(
            teacher_actions[:accepted].detach().to(self.storage_device, dtype=torch.float32),
            non_blocking=False,
        )
        self.size = end
        return accepted

    def sample(
        self,
        batch_size: int,
        device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty teacher anchor buffer.")
        indices = torch.randint(0, self.size, (batch_size,), device=self.storage_device)
        batch = {
            "obs": self.obs.index_select(0, indices),
            "teacher_actions": self.actions.index_select(0, indices),
        }
        if device is not None:
            target_device = torch.device(device)
            if target_device != self.storage_device:
                batch = {
                    key: value.to(
                        device=target_device,
                        dtype=torch.float32,
                        non_blocking=self.pin_memory,
                    )
                    for key, value in batch.items()
                }
        return batch

    def get_checkpoint_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "capacity": self.capacity,
            "size": self.size,
            "obs": self.obs[: self.size].detach().clone(),
            "actions": self.actions[: self.size].detach().clone(),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 1)) != 1:
            raise ValueError(f"Unsupported teacher anchor checkpoint version: {state.get('version')}.")
        saved_capacity = int(state["capacity"])
        if saved_capacity != self.capacity:
            raise ValueError(
                "Teacher anchor capacity must match for an exact resume: "
                f"checkpoint={saved_capacity}, configured={self.capacity}."
            )
        size = int(state["size"])
        if not 0 <= size <= self.capacity:
            raise ValueError(f"Invalid teacher anchor size {size} for capacity {self.capacity}.")
        for name in ("obs", "actions"):
            destination = getattr(self, name)
            source = state.get(name)
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"Teacher anchor checkpoint field '{name}' must be a tensor.")
            expected_shape = (size, *destination.shape[1:])
            if tuple(source.shape) != expected_shape:
                raise ValueError(
                    f"Teacher anchor field '{name}' has shape {tuple(source.shape)}, "
                    f"expected {expected_shape}."
                )
            if size:
                destination[:size].copy_(
                    source.to(self.storage_device, dtype=destination.dtype),
                    non_blocking=False,
                )
        self.size = size


def _capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    """Capture process RNG state without depending on pickle-only NumPy arrays."""

    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if include_cuda and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by :func:`_capture_rng_state`."""

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if isinstance(numpy_state, dict):
        keys = numpy_state["state"]
        if isinstance(keys, torch.Tensor):
            keys = keys.detach().cpu().numpy().astype(np.uint32, copy=False)
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                keys,
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )

    torch_cpu_state = state.get("torch_cpu")
    if isinstance(torch_cpu_state, torch.Tensor):
        torch.set_rng_state(torch_cpu_state.detach().cpu())

    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        available_devices = torch.cuda.device_count()
        for device_idx, cuda_state in enumerate(cuda_states[:available_devices]):
            torch.cuda.set_rng_state(cuda_state.detach().cpu(), device=device_idx)


class DaggerStudent(BaseAlgo):
    config: DaggerStudentConfig

    def __init__(
        self,
        env: BaseTask,
        config: DaggerStudentConfig,
        log_dir,
        device="cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(env, config, device, multi_gpu_cfg)
        self.log_dir = str(log_dir)
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=self.config.num_steps_per_env,
            num_learning_iterations=self.config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )

        self.current_learning_iteration = 0
        self.teacher_actor = None
        self._init_config()
        _ = self.env.reset_all()

    def _init_config(self) -> None:
        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()
        self.algo_history_length_dict = {
            group_name: group_cfg.history_length
            for group_name, group_cfg in self.env.observation_manager.cfg.groups.items()
        }
        self.actor_obs_keys = list(self.config.module_dict.actor.input_dim)
        self.teacher_obs_group = self.config.teacher_obs_group
        if self.teacher_obs_group not in self.algo_obs_dim_dict:
            raise ValueError(
                f"Teacher observation group '{self.teacher_obs_group}' is missing from observation manager dims: "
                f"{list(self.algo_obs_dim_dict.keys())}"
            )

        self.num_act = self.env.robot_config.actions_dim
        self.actor_learning_rate = self.config.actor_learning_rate
        self.value_learning_rate = float(
            self.config.value_learning_rate
            if self.config.value_learning_rate is not None
            else self.config.critic_learning_rate
        )
        self.q_learning_rate = float(
            self.config.q_learning_rate
            if self.config.q_learning_rate is not None
            else self.config.critic_learning_rate
        )
        self.value_target_tau = float(
            self.config.value_target_tau
            if self.config.value_target_tau is not None
            else self.config.target_tau
        )
        self.q_target_tau = float(
            self.config.q_target_tau
            if self.config.q_target_tau is not None
            else self.config.target_tau
        )
        if self.value_learning_rate <= 0.0 or self.q_learning_rate <= 0.0:
            raise ValueError("value_learning_rate and q_learning_rate must be positive.")
        if not 0.0 <= self.value_target_tau <= 1.0:
            raise ValueError("value_target_tau must be in [0, 1].")
        if not 0.0 <= self.q_target_tau <= 1.0:
            raise ValueError("q_target_tau must be in [0, 1].")
        self.max_actor_learning_rate = self.config.max_actor_learning_rate or max(self.actor_learning_rate, 1e-2)
        self.min_actor_learning_rate = self.config.min_actor_learning_rate or min(self.actor_learning_rate, 1e-5)
        if not 0.0 <= float(self.config.teacher_mixture_start) <= 1.0:
            raise ValueError("teacher_mixture_start must be in [0, 1].")
        if not 0.0 <= float(self.config.teacher_mixture_end) <= 1.0:
            raise ValueError("teacher_mixture_end must be in [0, 1].")
        if int(self.config.teacher_mixture_decay_iterations) < 0:
            raise ValueError("teacher_mixture_decay_iterations must be non-negative.")
        if int(self.config.teacher_anchor_capacity) <= 0:
            raise ValueError("teacher_anchor_capacity must be positive.")
        if not 0.0 <= float(self.config.teacher_anchor_sampling_ratio) <= 1.0:
            raise ValueError("teacher_anchor_sampling_ratio must be in [0, 1].")
        if float(self.config.teacher_action_outlier_threshold) <= 0.0:
            raise ValueError("teacher_action_outlier_threshold must be positive.")
        if float(self.config.actor_huber_delta) <= 0.0:
            raise ValueError("actor_huber_delta must be positive.")
        if float(self.config.student_action_clip) <= 0.0:
            raise ValueError("student_action_clip must be positive.")
        if float(self.config.teacher_action_outlier_threshold) > float(
            self.config.student_action_clip
        ):
            raise ValueError(
                "teacher_action_outlier_threshold must not exceed student_action_clip; "
                "otherwise clipped teacher actions could still become actor labels."
            )
        teacher_buffer_sampling_probability = self.config.teacher_buffer_sampling_probability
        if teacher_buffer_sampling_probability is not None and not (
            0.0 < float(teacher_buffer_sampling_probability) <= 1.0
        ):
            raise ValueError("teacher_buffer_sampling_probability must be in (0, 1].")
        self._last_teacher_mixture_ratio = float(self.config.teacher_mixture_start)
        self._last_teacher_execution_ratio = 0.0
        self._last_student_action_clip_fraction = 0.0
        self._last_teacher_action_clip_fraction = 0.0
        self._last_student_action_abs_max = 0.0
        self._last_teacher_action_abs_max = 0.0
        self._last_teacher_action_outlier_row_fraction = 0.0
        self._last_actor_anchor_sample_ratio = 0.0
        self._last_recent_teacher_label_valid_fraction = 0.0
        self._last_teacher_buffer_accepted = 0
        self._last_teacher_buffer_candidates = 0
        self._last_teacher_buffer_seen = 0
        self._last_teacher_buffer_saved = 0
        self._teacher_buffer_sampling_probability = 1.0

    def setup(self):
        logger.info("Setting up DAgger student")
        self._setup_student_actor()
        self._setup_critics()
        if getattr(self.env, "teacher", None):
            self._setup_teacher_actor()
        self._setup_buffer()
        self._setup_teacher_transition_writer()

    def _setup_student_actor(self) -> None:
        self.actor = setup_ppo_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        self.actor.std.requires_grad_(False)
        if self.is_multi_gpu:
            self._synchronize_actor_weights()
        self.actor_optimizer = instantiate(
            self.config.actor_optimizer,
            params=self.actor.parameters(),
            lr=self.actor_learning_rate,
        )

    def _setup_teacher_actor(self) -> None:
        teacher_reference = getattr(self.env, "teacher", None)
        if not teacher_reference:
            raise ValueError(
                "DAgger student training requires a teacher checkpoint. "
                "Pass `--teacher=/path/to/model.pt`."
            )

        resolved_teacher_path = load_checkpoint(str(teacher_reference), self.log_dir)
        teacher_config, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(resolved_teacher_path)))
        teacher_payload = torch.load(resolved_teacher_path, map_location=self.device)

        teacher_algo_config = getattr(teacher_config.algo, "config", None)
        if teacher_algo_config is None or not hasattr(teacher_algo_config, "module_dict"):
            raise ValueError("Teacher checkpoint must come from a PPO-style actor/critic experiment.")

        teacher_actor_cfg = teacher_algo_config.module_dict.actor
        teacher_obs_dim = self.algo_obs_dim_dict[self.teacher_obs_group]
        if not isinstance(teacher_obs_dim, int):
            raise ValueError(f"Teacher observation group '{self.teacher_obs_group}' must be concatenated.")

        self.teacher_actor = setup_ppo_actor_module(
            obs_dim_dict={"actor_obs": teacher_obs_dim},
            module_config=teacher_actor_cfg,
            num_actions=self.num_act,
            init_noise_std=getattr(teacher_algo_config, "init_noise_std", 1.0),
            device=self.device,
            history_length={"actor_obs": 1},
        )
        self.teacher_actor.load_state_dict(teacher_payload["actor_model_state_dict"])
        self.teacher_actor.eval()
        for parameter in self.teacher_actor.parameters():
            parameter.requires_grad_(False)

        logger.info(f"Loaded frozen teacher actor from {resolved_teacher_path}")

    def _setup_critics(self) -> None:
        self.critic_obs_keys = list(self.config.value_critic.input_dim)
        self.value_critic = setup_ppo_critic_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.value_critic,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        critic_obs_indices = {
            "critic_obs": {"start": 0, "end": critic_obs_dim, "size": critic_obs_dim}
        }
        self.qnet = Critic(
            obs_indices=critic_obs_indices,
            obs_keys=["critic_obs"],
            n_act=self.num_act,
            num_atoms=self.config.num_atoms,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
            hidden_dim=self.config.critic_hidden_dim,
            use_layer_norm=self.config.q_use_layer_norm,
            num_q_networks=self.config.num_q_networks,
            device=self.device,
        )
        self.target_value_critic = setup_ppo_critic_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.value_critic,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        self.qnet_target = Critic(
            obs_indices=critic_obs_indices,
            obs_keys=["critic_obs"],
            n_act=self.num_act,
            num_atoms=self.config.num_atoms,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
            hidden_dim=self.config.critic_hidden_dim,
            use_layer_norm=self.config.q_use_layer_norm,
            num_q_networks=self.config.num_q_networks,
            device=self.device,
        )
        self.critic_obs_normalizer: nn.Module
        if self.config.critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(shape=critic_obs_dim, device=self.device)
        else:
            self.critic_obs_normalizer = nn.Identity()
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        if self.is_multi_gpu:
            self._synchronize_model(self.value_critic)
            self._synchronize_model(self.qnet)
            self.target_value_critic.load_state_dict(self.value_critic.state_dict())
            self.qnet_target.load_state_dict(self.qnet.state_dict())
        for target in (self.target_value_critic, self.qnet_target):
            target.requires_grad_(False)
        self.value_optimizer = instantiate(
            self.config.critic_optimizer,
            params=self.value_critic.parameters(),
            lr=self.value_learning_rate,
        )
        self.q_optimizer = torch.optim.AdamW(
            self.qnet.parameters(),
            lr=self.q_learning_rate,
            weight_decay=self.config.q_weight_decay,
            fused=str(self.device).startswith("cuda"),
            betas=(0.9, 0.95),
        )

    def _setup_buffer(self) -> None:
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        buffer_device = self._resolve_buffer_device()
        self.buffer = StackDaggerBuffer(
            self.config.stack_buffer, actor_obs_dim, critic_obs_dim, self.num_act, buffer_device
        )
        self.teacher_anchor_buffer = FixedActorAnchorBuffer(
            self.config.teacher_anchor_capacity,
            actor_obs_dim,
            self.num_act,
            buffer_device,
        )
        scalars_per_transition = 2 * actor_obs_dim + 2 * critic_obs_dim + 2 * self.num_act + 4
        buffer_bytes = self.config.stack_buffer * scalars_per_transition * torch.finfo(torch.float32).bits // 8
        anchor_bytes = (
            self.config.teacher_anchor_capacity
            * (actor_obs_dim + self.num_act)
            * torch.finfo(torch.float32).bits
            // 8
        )
        logger.info(
            f"Allocated DAgger stack buffer with capacity={self.config.stack_buffer}, "
            f"actor_obs_dim={actor_obs_dim}, critic_obs_dim={critic_obs_dim}, action_dim={self.num_act}, "
            f"storage_device={self.buffer.storage_device}, footprint={buffer_bytes / (1024**3):.2f} GiB; "
            f"fixed actor anchor capacity={self.config.teacher_anchor_capacity}, "
            f"footprint={anchor_bytes / (1024**3):.2f} GiB"
        )

    def _resolve_teacher_buffer_sampling_probability(self) -> float:
        configured = self.config.teacher_buffer_sampling_probability
        if configured is not None:
            return float(configured)

        # Pre-sample uniformly before the on-disk reservoir so a large DAgger
        # rollout does not repeatedly rewrite random compressed H5 chunks.  At
        # the default schedule this targets one reservoir-capacity in
        # expectation; the reservoir still handles statistical overflow.
        iterations = max(int(self.config.num_learning_iterations), 0)
        decay = max(int(self.config.teacher_mixture_decay_iterations), 0)
        start = float(self.config.teacher_mixture_start)
        end = float(self.config.teacher_mixture_end)
        if decay == 0:
            probability_sum = iterations * end
        else:
            decay_iterations = min(iterations, decay)
            probability_sum = (
                decay_iterations * start
                + (end - start)
                * decay_iterations
                * (decay_iterations - 1)
                / (2.0 * decay)
                + max(iterations - decay_iterations, 0) * end
            )
        expected_teacher_rows = (
            float(self.env.num_envs)
            * float(self.config.num_steps_per_env)
            * max(probability_sum, 0.0)
        )
        if expected_teacher_rows <= 0.0:
            return 1.0
        return min(
            1.0,
            float(self.config.teacher_buffer_max_transitions) / expected_teacher_rows,
        )

    def _setup_teacher_transition_writer(self) -> None:
        self.teacher_transition_writer: TeacherTransitionH5Writer | None = None
        configured_output = self.config.teacher_buffer_output
        if configured_output is None or not str(configured_output).strip():
            return

        relative_path = Path(str(configured_output))
        if relative_path.is_absolute():
            raise ValueError(
                "teacher_buffer_output must be relative to the student run directory, "
                f"got absolute path: {relative_path}"
            )
        run_dir = Path(self.log_dir).expanduser().resolve()
        output_path = (run_dir / relative_path).resolve()
        if output_path != run_dir and run_dir not in output_path.parents:
            raise ValueError(
                "teacher_buffer_output must stay inside the student run directory, "
                f"got: {configured_output}"
            )
        if self.is_multi_gpu:
            output_path = output_path.with_name(
                f"{output_path.stem}_rank{self.gpu_global_rank:02d}{output_path.suffix or '.h5'}"
            )
            logger.warning(
                "Multi-GPU DAgger writes one independently consumable teacher reservoir per rank: "
                f"{output_path.name}"
            )

        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        observation_mode = infer_observation_mode(self.env, self.actor_obs_keys)
        self._teacher_buffer_sampling_probability = (
            self._resolve_teacher_buffer_sampling_probability()
        )
        motion_command = self.env.command_manager.get_state("motion_command")
        source_metadata: dict[str, Any] = {
            "rank": self.gpu_global_rank,
            "world_size": self.gpu_world_size,
            "teacher_row_sampling_probability": self._teacher_buffer_sampling_probability,
        }
        if motion_command is not None:
            task_names = getattr(motion_command, "_task_index_names", None)
            object_key_to_id = getattr(motion_command, "object_key_to_id", None)
            if task_names is not None:
                source_metadata["task_index_names"] = list(task_names)
            if object_key_to_id is not None:
                source_metadata["object_key_to_id"] = dict(object_key_to_id)

        self.teacher_transition_writer = TeacherTransitionH5Writer(
            output_path,
            max_transitions=int(self.config.teacher_buffer_max_transitions),
            actor_obs_dim=actor_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=self.num_act,
            actor_obs_keys=self.actor_obs_keys,
            critic_obs_keys=self.critic_obs_keys,
            observation_mode=observation_mode,
            seed=int(self.config.teacher_buffer_reservoir_seed) + self.gpu_global_rank,
            metadata=source_metadata,
        )
        self._last_teacher_buffer_seen = self.teacher_transition_writer.seen
        self._last_teacher_buffer_saved = self.teacher_transition_writer.saved
        logger.info(
            "Teacher-executed transition export enabled: "
            f"path={output_path}, mode={observation_mode}, "
            f"capacity={self.config.teacher_buffer_max_transitions:,}, "
            f"row_sampling_probability={self._teacher_buffer_sampling_probability:.6f}."
        )

    def _get_obs_dim(self, obs_keys: list[str]) -> int:
        obs_dim = 0
        for obs_key in obs_keys:
            key_dim = self.algo_obs_dim_dict[obs_key]
            if not isinstance(key_dim, int):
                raise ValueError(f"Observation dimension for {obs_key} is not concatenated: {key_dim}")
            obs_dim += key_dim
        return obs_dim

    def _get_zero_input(self) -> torch.Tensor:
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        return torch.zeros(1, actor_obs_dim, device=self.device)

    def _resolve_buffer_device(self) -> str:
        requested = str(self.config.buffer_device).strip()
        requested_lower = requested.lower()
        if requested_lower == "auto":
            return self.device if str(self.device).startswith("cuda") else "cpu"
        if requested_lower == "gpu":
            if not str(self.device).startswith("cuda"):
                raise ValueError("buffer_device='gpu' requires a CUDA training device.")
            return self.device
        return requested

    def _train_mode(self) -> None:
        self.actor.train()
        self.value_critic.train()
        self.qnet.train()
        self.critic_obs_normalizer.train()

    def _eval_mode(self) -> None:
        self.actor.eval()
        self.value_critic.eval()
        self.qnet.eval()
        self.critic_obs_normalizer.eval()

    def _synchronize_actor_weights(self) -> None:
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)
        logger.info(f"Synchronized student actor weights across {self.gpu_world_size} GPUs")

    def _synchronize_model(self, model: nn.Module) -> None:
        for parameter in model.parameters():
            torch.distributed.broadcast(parameter.data, src=0)

    def _reduce_model_gradients(self, model: nn.Module) -> None:
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
            parameter.grad.div_(self.gpu_world_size)

    def _reduce_actor_gradients(self) -> None:
        grads = [param.grad.view(-1) for param in self.actor.parameters() if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.actor.parameters():
            if param.grad is None:
                continue
            numel = param.grad.numel()
            param.grad.copy_(all_grads[offset : offset + numel].view_as(param.grad))
            offset += numel

    def learn(self):
        if self.teacher_actor is None:
            raise RuntimeError(
                "DAgger student training requires a teacher actor, but none was configured. "
                "Pass `--teacher=/path/to/model.pt` when training."
            )

        self._train_mode()

        obs_dict = self.env.reset_all()
        for obs_key, value in obs_dict.items():
            obs_dict[obs_key] = value.to(self.device)

        for it in range(
            self.current_learning_iteration,
            self.current_learning_iteration + self.config.num_learning_iterations,
        ):
            self.current_learning_iteration = it

            with self.logging_helper.record_collection_time():
                obs_dict = self._collect_rollout(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)

            if it % self.config.save_interval == 0:
                self._save_iteration_checkpoint(it)

        self._save_iteration_checkpoint(self.current_learning_iteration)

    def _save_iteration_checkpoint(self, iteration: int) -> None:
        """Save compact history plus atomically replaced per-rank replay state."""

        if self.is_main_process:
            model_path = os.path.join(self.log_dir, f"model_{iteration:05d}.pt")
            self.save(model_path, include_replay=False)
        resume_name = (
            "resume_latest.pt"
            if not self.is_multi_gpu or self.gpu_global_rank == 0
            else f"resume_latest_rank{self.gpu_global_rank:02d}.pt"
        )
        self.save(
            os.path.join(self.log_dir, resume_name),
            include_replay=True,
            upload_to_wandb=False,
            atomic=True,
        )
        if self.is_main_process:
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{iteration:05d}.onnx"))

    @torch.no_grad()
    def _collect_rollout(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.actor.eval()
        self.teacher_actor.eval()

        teacher_probability = self._teacher_mixture_probability()
        teacher_execution_ratios: list[float] = []
        student_clip_fractions: list[float] = []
        teacher_clip_fractions: list[float] = []
        teacher_outlier_row_fractions: list[float] = []
        student_action_abs_maxima: list[float] = []
        teacher_action_abs_maxima: list[float] = []
        action_clip = float(self.config.student_action_clip)
        teacher_outlier_threshold = float(self.config.teacher_action_outlier_threshold)
        teacher_transition_batches: dict[str, list[torch.Tensor]] | None = None
        self._last_teacher_buffer_candidates = 0
        if self.teacher_transition_writer is not None:
            teacher_transition_batches = {name: [] for name in TEACHER_TRANSITION_FIELDS}

        for _ in range(self.config.num_steps_per_env):
            actor_obs = torch.cat([obs_dict[key] for key in self.actor_obs_keys], dim=1)
            critic_obs = torch.cat([obs_dict[key] for key in self.critic_obs_keys], dim=1)
            if self.config.critic_obs_normalization:
                self.critic_obs_normalizer(critic_obs, update=True)
            teacher_obs = obs_dict[self.teacher_obs_group]

            student_actions = self.actor.act_inference({"actor_obs": actor_obs})
            teacher_actions = self.teacher_actor.act_inference({"actor_obs": teacher_obs})

            clipped_student_actions = student_actions.clamp(-action_clip, action_clip)
            finite_teacher_actions = torch.nan_to_num(
                teacher_actions,
                nan=0.0,
                posinf=action_clip,
                neginf=-action_clip,
            )
            clipped_teacher_actions = finite_teacher_actions.clamp(-action_clip, action_clip)
            teacher_action_valid = _valid_teacher_action_rows(
                teacher_actions,
                teacher_outlier_threshold,
            )
            requested_teacher = (
                torch.rand(self.env.num_envs, device=self.device) < teacher_probability
            )
            # A frozen teacher can extrapolate catastrophically on learner-OOD
            # states. Never execute or supervise from such an output; fall back
            # to the bounded student action while retaining the transition for
            # V/Q learning.
            use_teacher = requested_teacher & teacher_action_valid
            executed_actions = torch.where(
                use_teacher.unsqueeze(-1),
                clipped_teacher_actions,
                clipped_student_actions,
            )

            if use_teacher.any() and len(self.teacher_anchor_buffer) < self.teacher_anchor_buffer.capacity:
                anchor_indices = torch.nonzero(use_teacher, as_tuple=False).squeeze(-1)
                remaining = self.teacher_anchor_buffer.capacity - len(self.teacher_anchor_buffer)
                if anchor_indices.numel() > remaining:
                    permutation = torch.randperm(anchor_indices.numel(), device=anchor_indices.device)
                    anchor_indices = anchor_indices.index_select(0, permutation[:remaining])
                self.teacher_anchor_buffer.add(
                    actor_obs.index_select(0, anchor_indices),
                    clipped_teacher_actions.index_select(0, anchor_indices),
                )

            teacher_execution_ratios.append(float(use_teacher.float().mean().item()))
            student_clip_fractions.append(
                float((student_actions.abs() > action_clip).float().mean().item())
            )
            teacher_clip_fractions.append(
                float((~torch.isfinite(teacher_actions) | (teacher_actions.abs() > action_clip)).float().mean().item())
            )
            teacher_outlier_row_fractions.append(
                float((~teacher_action_valid).float().mean().item())
            )
            student_action_abs_maxima.append(float(student_actions.abs().max().item()))
            teacher_action_abs_maxima.append(
                float(
                    torch.nan_to_num(
                        teacher_actions.abs(),
                        nan=float("inf"),
                        posinf=float("inf"),
                        neginf=float("inf"),
                    ).max().item()
                )
            )

            # Keep autoregressive observations and Q replay aligned with the
            # teacher/student mixture action actually applied this step.
            self.env.student_prev_actions[:] = executed_actions
            self.env.student_base_actions[:] = executed_actions
            next_obs_dict, rewards, dones, infos = self.env.step({"actions": executed_actions})
            for obs_key, value in next_obs_dict.items():
                next_obs_dict[obs_key] = value.to(self.device)
            rewards = rewards.to(self.device)
            # Environments may expose done flags as integer 0/1 tensors.
            # Normalize once before using them as masks or terminal flags.
            dones = dones.to(self.device).bool()
            timeouts = infos["time_outs"].to(self.device).bool()
            next_actor_obs = torch.cat([next_obs_dict[key] for key in self.actor_obs_keys], dim=1)
            next_critic_obs = torch.cat([next_obs_dict[key] for key in self.critic_obs_keys], dim=1)
            if dones.any():
                final_actor_obs = torch.cat(
                    [infos["final_observations"][key] for key in self.actor_obs_keys], dim=1
                ).to(self.device)
                final_critic_obs = torch.cat(
                    [infos["final_observations"][key] for key in self.critic_obs_keys], dim=1
                ).to(self.device)
                next_actor_obs = torch.where(dones.unsqueeze(1), final_actor_obs, next_actor_obs)
                next_critic_obs = torch.where(dones.unsqueeze(1), final_critic_obs, next_critic_obs)
            # Time limits bootstrap; motion end, distance failure, and fallen do not.
            terminals = dones & ~timeouts

            if teacher_transition_batches is not None and use_teacher.any():
                self._last_teacher_buffer_candidates += int(use_teacher.sum().item())
                teacher_mask = use_teacher
                if self._teacher_buffer_sampling_probability < 1.0:
                    teacher_mask = teacher_mask & (
                        torch.rand(self.env.num_envs, device=self.device)
                        < self._teacher_buffer_sampling_probability
                    )
                if teacher_mask.any():
                    teacher_rows = {
                        "observations": actor_obs[teacher_mask],
                        "critic_observations": critic_obs[teacher_mask],
                        "actions": executed_actions[teacher_mask],
                        "rewards": rewards[teacher_mask],
                        "dones": dones[teacher_mask],
                        "truncations": timeouts[teacher_mask],
                        "next_observations": next_actor_obs[teacher_mask],
                        "next_critic_observations": next_critic_obs[teacher_mask],
                    }
                    for name, value in teacher_rows.items():
                        teacher_transition_batches[name].append(value.detach())

            self.buffer.add(
                obs=actor_obs,
                teacher_actions=clipped_teacher_actions,
                critic_obs=critic_obs,
                executed_actions=executed_actions,
                rewards=rewards,
                next_obs=next_actor_obs,
                next_critic_obs=next_critic_obs,
                terminals=terminals,
                is_student_action=~use_teacher,
                teacher_action_valid=teacher_action_valid,
            )
            self.logging_helper.update_episode_stats(rewards, dones, infos)
            obs_dict = next_obs_dict

        self._last_teacher_buffer_accepted = 0
        if teacher_transition_batches is not None and teacher_transition_batches["observations"]:
            stats = self.teacher_transition_writer.append(
                {
                    name: torch.cat(values, dim=0)
                    for name, values in teacher_transition_batches.items()
                }
            )
            self._last_teacher_buffer_accepted = stats.accepted
            self._last_teacher_buffer_seen = stats.seen
            self._last_teacher_buffer_saved = stats.saved

        self.actor.train()
        self._last_teacher_mixture_ratio = teacher_probability
        self._last_teacher_execution_ratio = sum(teacher_execution_ratios) / len(teacher_execution_ratios)
        self._last_student_action_clip_fraction = sum(student_clip_fractions) / len(student_clip_fractions)
        self._last_teacher_action_clip_fraction = sum(teacher_clip_fractions) / len(teacher_clip_fractions)
        self._last_teacher_action_outlier_row_fraction = (
            sum(teacher_outlier_row_fractions) / len(teacher_outlier_row_fractions)
        )
        self._last_student_action_abs_max = max(student_action_abs_maxima)
        self._last_teacher_action_abs_max = max(teacher_action_abs_maxima)
        return obs_dict

    def _teacher_mixture_probability(self) -> float:
        decay_iterations = int(self.config.teacher_mixture_decay_iterations)
        if decay_iterations == 0:
            return float(self.config.teacher_mixture_end)
        progress = min(max(float(self.current_learning_iteration) / decay_iterations, 0.0), 1.0)
        start = float(self.config.teacher_mixture_start)
        end = float(self.config.teacher_mixture_end)
        return start + progress * (end - start)

    def _sample_actor_supervision_batch(
        self,
        batch_size: int,
        valid_recent_indices: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor] | None, float]:
        """Sample fixed-anchor and recent valid labels at the configured ratio."""

        anchor = getattr(self, "teacher_anchor_buffer", None)
        anchor_available = anchor is not None and len(anchor) > 0
        recent_available = valid_recent_indices.numel() > 0
        if not anchor_available and not recent_available:
            return None, 0.0

        ratio = float(self.config.teacher_anchor_sampling_ratio)
        if anchor_available and recent_available:
            anchor_count = int(batch_size * ratio)
            if batch_size >= 2 and 0.0 < ratio < 1.0:
                anchor_count = min(max(anchor_count, 1), batch_size - 1)
            recent_count = batch_size - anchor_count
        elif anchor_available:
            anchor_count, recent_count = batch_size, 0
        else:
            anchor_count, recent_count = 0, batch_size

        pieces: list[dict[str, torch.Tensor]] = []
        if anchor_count:
            pieces.append(anchor.sample(anchor_count, device=self.device))
        if recent_count:
            pieces.append(
                self.buffer.sample_actor(
                    recent_count,
                    valid_recent_indices,
                    device=self.device,
                )
            )
        actor_batch = {
            key: torch.cat([piece[key] for piece in pieces], dim=0)
            for key in ("obs", "teacher_actions")
        }
        return actor_batch, float(anchor_count / batch_size)

    def _training_step(self) -> dict[str, float]:
        if len(self.buffer) == 0:
            return {
                "mse_loss": 0.0,
                "actor_huber_loss": 0.0,
                "action_mae": 0.0,
                "value_loss": 0.0,
                "value_mean": 0.0,
                "value_target_mean": 0.0,
                "value_target_std": 0.0,
                "value_td_abs_mean": 0.0,
                "student_transition_ratio": 0.0,
                "q_loss": 0.0,
                "q1_loss": 0.0,
                "q2_loss": 0.0,
                "buffer_size": 0.0,
                "buffer_fill_ratio": 0.0,
            }

        num_updates = max(int(self.config.num_updates_per_iteration), 1)
        batch_size = min(int(self.config.batch_size), len(self.buffer))
        valid_recent_indices = self.buffer.valid_actor_indices()
        self._last_recent_teacher_label_valid_fraction = float(
            valid_recent_indices.numel() / len(self.buffer)
        )
        anchor = getattr(self, "teacher_anchor_buffer", None)
        local_actor_labels_available = bool(
            valid_recent_indices.numel() > 0 or (anchor is not None and len(anchor) > 0)
        )
        actor_label_ranks = int(local_actor_labels_available)
        if self.is_multi_gpu:
            actor_label_rank_count = torch.tensor(
                actor_label_ranks,
                device=self.device,
                dtype=torch.long,
            )
            torch.distributed.all_reduce(
                actor_label_rank_count,
                op=torch.distributed.ReduceOp.SUM,
            )
            actor_label_ranks = int(actor_label_rank_count.item())

        actor_huber_losses: list[float] = []
        mse_losses: list[float] = []
        mae_losses: list[float] = []
        value_losses: list[float] = []
        value_means: list[float] = []
        value_target_means: list[float] = []
        value_target_stds: list[float] = []
        value_td_abs_means: list[float] = []
        student_transition_ratios: list[float] = []
        q_losses: list[float] = []
        q1_losses: list[float] = []
        q2_losses: list[float] = []
        actor_anchor_sample_ratios: list[float] = []
        for _ in range(num_updates):
            batch = self.buffer.sample(batch_size=batch_size, device=self.device)
            actor_batch, anchor_sample_ratio = self._sample_actor_supervision_batch(
                batch_size,
                valid_recent_indices,
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            if actor_label_ranks > 0:
                actor_loss_for_backward: torch.Tensor
                if actor_batch is not None:
                    pred_actions = self.actor.act_inference({"actor_obs": actor_batch["obs"]})
                    actor_huber_loss = F.smooth_l1_loss(
                        pred_actions,
                        actor_batch["teacher_actions"],
                        beta=float(self.config.actor_huber_delta),
                    )
                    # Keep MSE/MAE as diagnostics while optimizing only the robust
                    # Huber objective.
                    mse_loss = F.mse_loss(pred_actions, actor_batch["teacher_actions"])
                    mae_loss = F.l1_loss(pred_actions, actor_batch["teacher_actions"])
                    actor_loss_for_backward = actor_huber_loss
                    if self.is_multi_gpu:
                        actor_loss_for_backward = actor_loss_for_backward * (
                            self.gpu_world_size / actor_label_ranks
                        )

                    actor_huber_losses.append(float(actor_huber_loss.item()))
                    mse_losses.append(float(mse_loss.item()))
                    mae_losses.append(float(mae_loss.item()))
                    actor_anchor_sample_ratios.append(anchor_sample_ratio)
                else:
                    # Other ranks have valid labels. Participate in the same
                    # gradient collective so all actor replicas remain aligned.
                    actor_loss_for_backward = sum(
                        (
                            parameter.sum() * 0.0
                            for parameter in self.actor.parameters()
                            if parameter.requires_grad
                        ),
                        torch.zeros((), device=self.device),
                    )

                actor_loss_for_backward.backward()
                if self.is_multi_gpu:
                    self._reduce_actor_gradients()
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

            with torch.no_grad():
                next_actions = self.actor.act_inference({"actor_obs": batch["next_obs"]}).clamp(
                    -float(self.config.student_action_clip),
                    float(self.config.student_action_clip),
                )
                normalized_next_critic_obs = (
                    self.critic_obs_normalizer(batch["next_critic_obs"], update=False)
                    if self.config.critic_obs_normalization
                    else batch["next_critic_obs"]
                )
                q_target_distribution = self.qnet_target.projection(
                    normalized_next_critic_obs,
                    next_actions,
                    batch["rewards"].squeeze(-1),
                    1.0 - batch["terminals"].squeeze(-1),
                    torch.full_like(batch["rewards"].squeeze(-1), float(self.config.gamma)),
                )

            # Q learns from every teacher/student transition. V(s), however,
            # represents the current student policy and must not bootstrap on
            # states whose action was actually supplied by the teacher.
            student_mask = batch["is_student_action"].squeeze(-1) > 0.5
            local_student_count = int(student_mask.sum().item())
            global_student_count = local_student_count
            if self.is_multi_gpu:
                count_tensor = torch.tensor(local_student_count, device=self.device, dtype=torch.long)
                torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
                global_student_count = int(count_tensor.item())
            student_transition_ratios.append(
                float(global_student_count / (student_mask.numel() * self.gpu_world_size))
            )

            self.value_optimizer.zero_grad(set_to_none=True)
            if global_student_count > 0:
                if local_student_count > 0:
                    student_critic_obs = batch["critic_obs"][student_mask]
                    student_next_critic_obs = batch["next_critic_obs"][student_mask]
                    normalized_student_critic_obs = (
                        self.critic_obs_normalizer(student_critic_obs, update=False)
                        if self.config.critic_obs_normalization
                        else student_critic_obs
                    )
                    normalized_student_next_critic_obs = (
                        self.critic_obs_normalizer(student_next_critic_obs, update=False)
                        if self.config.critic_obs_normalization
                        else student_next_critic_obs
                    )
                    with torch.no_grad():
                        next_values = self.target_value_critic.evaluate(
                            {"critic_obs": normalized_student_next_critic_obs}
                        )
                        value_target = batch["rewards"][student_mask] + float(self.config.gamma) * (
                            1.0 - batch["terminals"][student_mask]
                        ) * next_values
                    values = self.value_critic.evaluate(
                        {"critic_obs": normalized_student_critic_obs}
                    )
                    local_value_loss = F.smooth_l1_loss(
                        values,
                        value_target,
                        beta=1.0,
                    ) * float(self.config.value_loss_coef)
                    value_loss = local_value_loss
                    if self.is_multi_gpu:
                        # Preserve a true global student-transition mean even
                        # when ranks contain different numbers of student rows.
                        value_loss = value_loss * (
                            local_student_count * self.gpu_world_size / global_student_count
                        )

                    value_losses.append(float(local_value_loss.item()))
                    value_means.append(float(values.detach().mean().item()))
                    value_target_means.append(float(value_target.mean().item()))
                    value_target_stds.append(float(value_target.std(unbiased=False).item()))
                    value_td_abs_means.append(
                        float((values.detach() - value_target).abs().mean().item())
                    )
                else:
                    # Other ranks have student transitions. Participate in
                    # gradient collectives with an explicit zero gradient.
                    value_loss = sum(
                        (parameter.sum() * 0.0 for parameter in self.value_critic.parameters()),
                        torch.zeros((), device=self.device),
                    )

                value_loss.backward()
                if self.is_multi_gpu:
                    self._reduce_model_gradients(self.value_critic)
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.value_critic.parameters(), self.config.max_grad_norm)
                self.value_optimizer.step()
                self._soft_update_model(
                    self.value_critic,
                    self.target_value_critic,
                    self.value_target_tau,
                )

            normalized_critic_obs = (
                self.critic_obs_normalizer(batch["critic_obs"], update=False)
                if self.config.critic_obs_normalization
                else batch["critic_obs"]
            )
            q_logits = self.qnet(normalized_critic_obs, batch["actions"])
            q_log_probs = F.log_softmax(q_logits, dim=-1)
            per_q_losses = -torch.sum(q_target_distribution * q_log_probs, dim=-1).mean(dim=1)
            q_loss = per_q_losses.sum() * float(self.config.q_loss_coef)
            self.q_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            if self.is_multi_gpu:
                self._reduce_model_gradients(self.qnet)
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.qnet.parameters(), self.config.max_grad_norm
                )
            self.q_optimizer.step()
            self._soft_update_model(self.qnet, self.qnet_target, self.q_target_tau)

            q_losses.append(float(q_loss.item()))
            q1_losses.append(float(per_q_losses[0].item()))
            q2_losses.append(float(per_q_losses[1].item()))

        self._last_actor_anchor_sample_ratio = (
            sum(actor_anchor_sample_ratios) / len(actor_anchor_sample_ratios)
            if actor_anchor_sample_ratios
            else 0.0
        )
        return {
            "actor_huber_loss": (
                sum(actor_huber_losses) / len(actor_huber_losses)
                if actor_huber_losses
                else 0.0
            ),
            "mse_loss": sum(mse_losses) / len(mse_losses) if mse_losses else 0.0,
            "action_mae": sum(mae_losses) / len(mae_losses) if mae_losses else 0.0,
            "value_loss": sum(value_losses) / len(value_losses) if value_losses else 0.0,
            "value_mean": sum(value_means) / len(value_means) if value_means else 0.0,
            "value_target_mean": (
                sum(value_target_means) / len(value_target_means) if value_target_means else 0.0
            ),
            "value_target_std": (
                sum(value_target_stds) / len(value_target_stds) if value_target_stds else 0.0
            ),
            "value_td_abs_mean": (
                sum(value_td_abs_means) / len(value_td_abs_means) if value_td_abs_means else 0.0
            ),
            "student_transition_ratio": (
                sum(student_transition_ratios) / len(student_transition_ratios)
            ),
            "q_loss": sum(q_losses) / len(q_losses),
            "q1_loss": sum(q1_losses) / len(q1_losses),
            "q2_loss": sum(q2_losses) / len(q2_losses),
            "buffer_size": float(len(self.buffer)),
            "buffer_fill_ratio": float(len(self.buffer) / self.buffer.capacity),
        }

    @torch.no_grad()
    def _soft_update_model(self, source: nn.Module, target: nn.Module, tau: float) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.lerp_(source_param, float(tau))

    def _post_epoch_logging(self, it: int, loss_dict: dict[str, float]) -> None:
        extra_log_dicts = {
            "Buffer": {
                "size": float(len(self.buffer)),
                "fill_ratio": float(len(self.buffer) / self.buffer.capacity),
                "teacher_anchor_size": float(len(self.teacher_anchor_buffer)),
                "teacher_anchor_fill_ratio": float(
                    len(self.teacher_anchor_buffer) / self.teacher_anchor_buffer.capacity
                ),
                "recent_teacher_label_valid_fraction": float(
                    self._last_recent_teacher_label_valid_fraction
                ),
                "teacher_h5_candidates": float(self._last_teacher_buffer_candidates),
                "teacher_h5_accepted": float(self._last_teacher_buffer_accepted),
                "teacher_h5_seen": float(self._last_teacher_buffer_seen),
                "teacher_h5_saved": float(self._last_teacher_buffer_saved),
                "teacher_h5_sampling_probability": float(
                    self._teacher_buffer_sampling_probability
                ),
            },
            "Policy": {
                "actor_learning_rate": float(self.actor_learning_rate),
                "value_learning_rate": float(self.value_learning_rate),
                "q_learning_rate": float(self.q_learning_rate),
                "teacher_mixture_probability": self._last_teacher_mixture_ratio,
                "teacher_execution_ratio": self._last_teacher_execution_ratio,
                "student_action_clip_fraction": self._last_student_action_clip_fraction,
                "teacher_action_clip_fraction": self._last_teacher_action_clip_fraction,
                "teacher_action_outlier_row_fraction": (
                    self._last_teacher_action_outlier_row_fraction
                ),
                "actor_anchor_sample_ratio": self._last_actor_anchor_sample_ratio,
                "student_action_abs_max": self._last_student_action_abs_max,
                "teacher_action_abs_max": self._last_teacher_action_abs_max,
            },
        }
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

    def load(self, ckpt_path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
        if ckpt_path is None:
            return None
        ckpt_path = Path(ckpt_path)
        if self.is_multi_gpu and self.gpu_global_rank > 0 and ckpt_path.name == "resume_latest.pt":
            rank_path = ckpt_path.with_name(
                f"resume_latest_rank{self.gpu_global_rank:02d}.pt"
            )
            if not rank_path.is_file():
                raise FileNotFoundError(
                    "Distributed exact replay resume requires one checkpoint per rank; "
                    f"missing {rank_path}."
                )
            ckpt_path = rank_path
        logger.info(f"Loading DAgger student checkpoint from {ckpt_path}")
        loaded_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        distributed_info = loaded_dict.get("distributed_checkpoint")
        checkpoint_rank = (
            int(distributed_info.get("rank", -1))
            if isinstance(distributed_info, dict)
            else None
        )
        restore_local_state = not self.is_multi_gpu or checkpoint_rank == self.gpu_global_rank
        if self.is_multi_gpu and checkpoint_rank is None and loaded_dict.get(
            "replay_state_included", False
        ):
            raise ValueError(
                "This replay checkpoint predates per-rank distributed state and cannot be "
                "resumed safely with multiple GPUs."
            )
        if self.is_multi_gpu and loaded_dict.get("replay_state_included", False):
            checkpoint_world_size = int(distributed_info.get("world_size", -1))
            if checkpoint_world_size != self.gpu_world_size:
                raise ValueError(
                    "Distributed replay checkpoint world-size mismatch: "
                    f"checkpoint={checkpoint_world_size}, current={self.gpu_world_size}."
                )
        normalization_info = loaded_dict.get("critic_normalization")
        checkpoint_value_normalized = bool(
            isinstance(normalization_info, dict)
            and normalization_info.get("value_critic_obs_normalized", False)
        )
        expected_value_normalized = bool(self.config.critic_obs_normalization)
        if "critic_model_state_dict" in loaded_dict and (
            checkpoint_value_normalized != expected_value_normalized
        ):
            raise ValueError(
                "Cannot resume V critic across different observation-normalization semantics: "
                f"checkpoint_normalized={checkpoint_value_normalized}, "
                f"configured_normalized={expected_value_normalized}. Start a new student run "
                "or use a checkpoint created with the same normalization schema."
            )
        if expected_value_normalized:
            if not isinstance(normalization_info, dict):
                raise ValueError(
                    "Checkpoint predates normalized V learning and cannot be resumed exactly."
                )
            checkpoint_eps = float(normalization_info.get("eps", float("nan")))
            configured_eps = float(getattr(self.critic_obs_normalizer, "eps", 0.0))
            if checkpoint_eps != configured_eps:
                raise ValueError(
                    "Critic normalizer epsilon mismatch: "
                    f"checkpoint={checkpoint_eps}, configured={configured_eps}."
                )
        self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
        if "critic_model_state_dict" in loaded_dict:
            self.value_critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            self.target_value_critic.load_state_dict(
                loaded_dict.get(
                    "target_critic_model_state_dict",
                    loaded_dict["critic_model_state_dict"],
                )
            )
        if "qnet_state_dict" in loaded_dict:
            self.qnet.load_state_dict(loaded_dict["qnet_state_dict"])
            self.qnet_target.load_state_dict(
                loaded_dict.get("qnet_target_state_dict", loaded_dict["qnet_state_dict"])
            )
        if expected_value_normalized and "critic_obs_normalizer_state" not in loaded_dict:
            raise KeyError("Normalized V/Q checkpoint is missing critic_obs_normalizer_state.")
        if "critic_obs_normalizer_state" in loaded_dict:
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_normalizer_state"])
        if self.config.load_optimizer and "actor_optimizer_state_dict" in loaded_dict:
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.actor_learning_rate = loaded_dict["actor_optimizer_state_dict"]["param_groups"][0]["lr"]
        if self.config.load_optimizer and "value_optimizer_state_dict" in loaded_dict:
            self.value_optimizer.load_state_dict(loaded_dict["value_optimizer_state_dict"])
            self.value_learning_rate = loaded_dict["value_optimizer_state_dict"]["param_groups"][0]["lr"]
        if self.config.load_optimizer and "q_optimizer_state_dict" in loaded_dict:
            self.q_optimizer.load_state_dict(loaded_dict["q_optimizer_state_dict"])
            self.q_learning_rate = loaded_dict["q_optimizer_state_dict"]["param_groups"][0]["lr"]
        buffer_state = loaded_dict.get("dagger_buffer_state") if restore_local_state else None
        if buffer_state is not None:
            self.buffer.load_checkpoint_state(buffer_state)
            logger.info(
                f"Restored DAgger replay buffer with size={len(self.buffer)}, "
                f"write_idx={self.buffer.write_idx}."
            )
        else:
            logger.warning(
                "Checkpoint has no DAgger replay-buffer state; resuming with an empty buffer "
                "for backward compatibility."
            )
        anchor_state = (
            loaded_dict.get("teacher_anchor_buffer_state") if restore_local_state else None
        )
        if anchor_state is not None:
            self.teacher_anchor_buffer.load_checkpoint_state(anchor_state)
            logger.info(
                "Restored fixed teacher actor anchor with "
                f"size={len(self.teacher_anchor_buffer)}."
            )
        else:
            logger.warning(
                "Checkpoint has no fixed teacher actor anchor; it will be refilled from new "
                "valid teacher-executed rows."
            )
        # New checkpoints record the next iteration explicitly so a resumed
        # run does not repeat the already-completed update.  Retain the legacy
        # ``iter`` fallback for older checkpoints.
        self.current_learning_iteration = int(
            loaded_dict.get("next_iter", loaded_dict.get("iter", 0))
        )
        if restore_local_state:
            self._restore_env_state(loaded_dict.get("env_state"))
            rng_state = loaded_dict.get("rng_state")
            if isinstance(rng_state, dict):
                _restore_rng_state(rng_state)
            else:
                logger.warning(
                    "Checkpoint has no RNG state; continuing with the current process RNG state "
                    "for backward compatibility."
                )
        else:
            logger.info(
                "Loaded shared model/optimizer state without rank-local replay, environment, "
                "or RNG state."
            )
        if loaded_dict.get("replay_state_included", False) and restore_local_state:
            logger.info(
                "Restored optimizer, replay, anchor, and rank-local RNG state. Physical simulator "
                "episodes are intentionally restarted by learn(); Isaac simulation state is not "
                "serialized."
            )
        return loaded_dict.get("infos")

    def _build_checkpoint_dict(
        self,
        infos: Any = None,
        *,
        include_replay: bool,
    ) -> dict[str, Any]:
        checkpoint_dict = {
            "actor_model_state_dict": self.actor.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_model_state_dict": self.value_critic.state_dict(),
            "target_critic_model_state_dict": self.target_value_critic.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "qnet_state_dict": self.qnet.state_dict(),
            "qnet_target_state_dict": self.qnet_target.state_dict(),
            "critic_obs_normalizer_state": self.critic_obs_normalizer.state_dict(),
            "critic_normalization": {
                "schema": 2,
                "enabled": bool(self.config.critic_obs_normalization),
                "value_critic_obs_normalized": bool(self.config.critic_obs_normalization),
                "eps": float(getattr(self.critic_obs_normalizer, "eps", 0.0)),
                "critic_obs_dim": self._get_obs_dim(self.critic_obs_keys),
            },
            "distributional_q_config": {
                "num_q_networks": self.config.num_q_networks,
                "num_atoms": self.config.num_atoms,
                "v_min": self.config.v_min,
                "v_max": self.config.v_max,
                "critic_hidden_dim": self.config.critic_hidden_dim,
                "use_layer_norm": self.config.q_use_layer_norm,
                "critic_obs_keys": list(self.critic_obs_keys),
                "critic_obs_dim": self._get_obs_dim(self.critic_obs_keys),
            },
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "rng_state": _capture_rng_state(),
            "iter": self.current_learning_iteration,
            "next_iter": self.current_learning_iteration + 1,
            "replay_state_included": bool(include_replay),
            "distributed_checkpoint": {
                "rank": int(self.gpu_global_rank),
                "world_size": int(self.gpu_world_size),
            },
            "infos": infos,
        }
        checkpoint_dict.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            checkpoint_dict["env_state"] = env_state
        if include_replay:
            checkpoint_dict["dagger_buffer_state"] = self.buffer.get_checkpoint_state()
            checkpoint_dict[
                "teacher_anchor_buffer_state"
            ] = self.teacher_anchor_buffer.get_checkpoint_state()
        return checkpoint_dict

    def save(
        self,
        path,
        infos=None,
        *,
        include_replay: bool = True,
        upload_to_wandb: bool = True,
        atomic: bool = False,
    ):
        checkpoint_dict = self._build_checkpoint_dict(
            infos,
            include_replay=include_replay,
        )
        path = Path(path)
        if atomic:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_name(f".{path.name}.tmp")
            logger.info(f"Saving atomically replaced full replay-resume checkpoint to {path}")
            try:
                torch.save(checkpoint_dict, temporary_path)
                os.replace(temporary_path, path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
        elif upload_to_wandb:
            self.logging_helper.save_checkpoint_artifact(checkpoint_dict, str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_dict, path)

    @property
    def inference_model(self):
        return self.actor

    @property
    def actor_onnx_wrapper(self):
        class ActorWrapper(nn.Module):
            def __init__(self, actor, action_clip: float):
                super().__init__()
                self.actor = actor
                self.action_clip = float(action_clip)

            def forward(self, actor_obs):
                return self.actor.act_inference({"actor_obs": actor_obs}).clamp(
                    -self.action_clip, self.action_clip
                )

        return ActorWrapper(self.actor, self.config.student_action_clip)

    def export(self, onnx_file_path: str):
        was_training = self.actor.training
        self._eval_mode()

        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is not None:
            export_motion_and_policy_as_onnx(
                self.actor_onnx_wrapper,
                motion_command,
                onnx_file_path,
                self.device,
            )
        else:
            export_policy_as_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=onnx_file_path,
                example_obs_dict={"actor_obs": self._get_zero_input()},
            )

        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.env)
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        attach_onnx_metadata(onnx_path=onnx_file_path, metadata=metadata)
        self.logging_helper.save_to_wandb(onnx_file_path)

        if was_training:
            self._train_mode()

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        actor = self.actor
        actor.eval()

        def policy(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            target_device = device or self.device
            actor_obs = torch.cat([obs_dict[key] for key in self.actor_obs_keys], dim=1).to(target_device)
            actions = actor.act_inference({"actor_obs": actor_obs}).clamp(
                -float(self.config.student_action_clip),
                float(self.config.student_action_clip),
            )
            self._sync_student_action_history(actions)
            return actions

        return policy

    def _sync_student_action_history(self, actions: torch.Tensor) -> None:
        """Keep student autoregressive action observations aligned with executed actions."""
        if hasattr(self.env, "student_prev_actions"):
            if tuple(actions.shape) == tuple(self.env.student_prev_actions.shape):
                self.env.student_prev_actions[:] = actions.detach().to(self.env.student_prev_actions.device)
        if hasattr(self.env, "student_base_actions"):
            if tuple(actions.shape) == tuple(self.env.student_base_actions.shape):
                self.env.student_base_actions[:] = actions.detach().to(self.env.student_base_actions.device)

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        self._eval_mode()
        obs_dict = self.env.reset_all()
        for obs_key, value in obs_dict.items():
            obs_dict[obs_key] = value.to(self.device)

        if max_eval_steps is None:
            max_eval_steps = int(self.env.max_episode_length)

        for _ in range(max_eval_steps):
            actor_obs = torch.cat([obs_dict[key] for key in self.actor_obs_keys], dim=1)
            actions = self.actor.act_inference({"actor_obs": actor_obs}).clamp(
                -float(self.config.student_action_clip),
                float(self.config.student_action_clip),
            )
            self._sync_student_action_history(actions)
            obs_dict, _, _, _ = self.env.step({"actions": actions})
            for obs_key, value in obs_dict.items():
                obs_dict[obs_key] = value.to(self.device)
