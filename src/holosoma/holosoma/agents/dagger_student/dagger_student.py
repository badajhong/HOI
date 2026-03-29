from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.module_utils import setup_ppo_actor_module
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


class FIFODaggerBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        if capacity <= 0:
            raise ValueError(f"FIFO buffer capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device="cpu")
        self.actions = torch.empty((self.capacity, action_dim), dtype=torch.float32, device="cpu")
        self.size = 0
        self.write_idx = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs: torch.Tensor, actions: torch.Tensor) -> None:
        if obs.shape[0] != actions.shape[0]:
            raise ValueError(f"Buffer add mismatch: obs batch {obs.shape[0]} vs actions batch {actions.shape[0]}")

        obs_cpu = obs.detach().to(device="cpu", dtype=torch.float32)
        actions_cpu = actions.detach().to(device="cpu", dtype=torch.float32)
        batch_size = int(obs_cpu.shape[0])
        if batch_size == 0:
            return

        if batch_size >= self.capacity:
            obs_cpu = obs_cpu[-self.capacity :]
            actions_cpu = actions_cpu[-self.capacity :]
            batch_size = self.capacity

        end_idx = self.write_idx + batch_size
        if end_idx <= self.capacity:
            self.obs[self.write_idx : end_idx].copy_(obs_cpu)
            self.actions[self.write_idx : end_idx].copy_(actions_cpu)
        else:
            first_chunk = self.capacity - self.write_idx
            second_chunk = batch_size - first_chunk
            self.obs[self.write_idx :].copy_(obs_cpu[:first_chunk])
            self.actions[self.write_idx :].copy_(actions_cpu[:first_chunk])
            self.obs[:second_chunk].copy_(obs_cpu[first_chunk:])
            self.actions[:second_chunk].copy_(actions_cpu[first_chunk:])

        self.write_idx = (self.write_idx + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty DAgger buffer.")

        indices = torch.randint(0, self.size, (batch_size,), device="cpu")
        obs = self.obs.index_select(0, indices).to(device=device, dtype=torch.float32, non_blocking=False)
        actions = self.actions.index_select(0, indices).to(device=device, dtype=torch.float32, non_blocking=False)
        return obs, actions


class DaggerStudent(BaseAlgo):
    config: DaggerStudentConfig

    def __init__(self, env: BaseTask, config: DaggerStudentConfig, log_dir, device="cpu", multi_gpu_cfg: dict | None = None):
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
        self.max_actor_learning_rate = self.config.max_actor_learning_rate or max(self.actor_learning_rate, 1e-2)
        self.min_actor_learning_rate = self.config.min_actor_learning_rate or min(self.actor_learning_rate, 1e-5)

    def setup(self):
        logger.info("Setting up DAgger student")
        self._setup_student_actor()
        if getattr(self.env, "teacher", None):
            self._setup_teacher_actor()
        self._setup_buffer()

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

    def _setup_buffer(self) -> None:
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        self.buffer = FIFODaggerBuffer(self.config.fifo_buffer, actor_obs_dim, self.num_act)
        logger.info(
            f"Allocated FIFO DAgger buffer with capacity={self.config.fifo_buffer}, "
            f"actor_obs_dim={actor_obs_dim}, action_dim={self.num_act}"
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

    def _train_mode(self) -> None:
        self.actor.train()

    def _eval_mode(self) -> None:
        self.actor.eval()

    def _synchronize_actor_weights(self) -> None:
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)
        logger.info(f"Synchronized student actor weights across {self.gpu_world_size} GPUs")

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

            if it % self.config.save_interval == 0 and self.is_main_process:
                self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))
                self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{it:05d}.onnx"))

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.onnx"))

    @torch.no_grad()
    def _collect_rollout(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.actor.eval()
        self.teacher_actor.eval()

        for _ in range(self.config.num_steps_per_env):
            actor_obs = torch.cat([obs_dict[key] for key in self.actor_obs_keys], dim=1)
            teacher_obs = obs_dict[self.teacher_obs_group]

            student_actions = self.actor.act_inference({"actor_obs": actor_obs})
            teacher_actions = self.teacher_actor.act_inference({"actor_obs": teacher_obs})

            next_obs_dict, rewards, dones, infos = self.env.step({"actions": student_actions})
            for obs_key, value in next_obs_dict.items():
                next_obs_dict[obs_key] = value.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            self.buffer.add(actor_obs, teacher_actions)
            self.logging_helper.update_episode_stats(rewards, dones, infos)
            obs_dict = next_obs_dict

        self.actor.train()
        return obs_dict

    def _training_step(self) -> dict[str, float]:
        if len(self.buffer) == 0:
            return {
                "mse_loss": 0.0,
                "action_mae": 0.0,
                "buffer_size": 0.0,
                "buffer_fill_ratio": 0.0,
            }

        num_updates = max(int(self.config.num_updates_per_iteration), 1)
        batch_size = min(int(self.config.batch_size), len(self.buffer))

        mse_losses: list[float] = []
        mae_losses: list[float] = []
        for _ in range(num_updates):
            batch_obs, batch_teacher_actions = self.buffer.sample(batch_size=batch_size, device=self.device)
            pred_actions = self.actor.act_inference({"actor_obs": batch_obs})
            mse_loss = F.mse_loss(pred_actions, batch_teacher_actions)
            mae_loss = F.l1_loss(pred_actions, batch_teacher_actions)

            self.actor_optimizer.zero_grad(set_to_none=True)
            mse_loss.backward()
            if self.is_multi_gpu:
                self._reduce_actor_gradients()
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()

            mse_losses.append(float(mse_loss.item()))
            mae_losses.append(float(mae_loss.item()))

        return {
            "mse_loss": sum(mse_losses) / len(mse_losses),
            "action_mae": sum(mae_losses) / len(mae_losses),
            "buffer_size": float(len(self.buffer)),
            "buffer_fill_ratio": float(len(self.buffer) / self.buffer.capacity),
        }

    def _post_epoch_logging(self, it: int, loss_dict: dict[str, float]) -> None:
        extra_log_dicts = {
            "Buffer": {
                "size": float(len(self.buffer)),
                "fill_ratio": float(len(self.buffer) / self.buffer.capacity),
            },
            "Policy": {
                "actor_learning_rate": float(self.actor_learning_rate),
            },
        }
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

    def load(self, ckpt_path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
        if ckpt_path is None:
            return None
        logger.info(f"Loading DAgger student checkpoint from {ckpt_path}")
        loaded_dict = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
        if self.config.load_optimizer and "actor_optimizer_state_dict" in loaded_dict:
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.actor_learning_rate = loaded_dict["actor_optimizer_state_dict"]["param_groups"][0]["lr"]
        self.current_learning_iteration = int(loaded_dict.get("iter", 0))
        self._restore_env_state(loaded_dict.get("env_state"))
        return loaded_dict.get("infos")

    def save(self, path, infos=None):
        checkpoint_dict = {
            "actor_model_state_dict": self.actor.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        checkpoint_dict.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            checkpoint_dict["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(checkpoint_dict, str(path))

    @property
    def inference_model(self):
        return self.actor

    @property
    def actor_onnx_wrapper(self):
        class ActorWrapper(nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, actor_obs):
                return self.actor.act_inference({"actor_obs": actor_obs})

        return ActorWrapper(self.actor)

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
            return actor.act_inference({"actor_obs": actor_obs})

        return policy

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
            actions = self.actor.act_inference({"actor_obs": actor_obs})
            obs_dict, _, _, _ = self.env.step({"actions": actions})
            for obs_key, value in obs_dict.items():
                obs_dict[obs_key] = value.to(self.device)
