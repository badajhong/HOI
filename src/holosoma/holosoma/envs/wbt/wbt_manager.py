from __future__ import annotations

import time
from pathlib import Path

import torch

from holosoma.envs.base_task.base_task import BaseTask

# from holosoma.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from holosoma.utils.simulator_config import SimulatorType


class WholeBodyTrackingManager(BaseTask):
    def __init__(self, tyro_config, *, device):
        super().__init__(tyro_config, device=device)

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()

        # -------------------------------- terms same with locomotion_manager.py [start]--------------------------------
        self.base_quat = self.simulator.base_quat
        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self._configure_default_dof_pos()
        self._init_domain_rand_buffers()

    def _configure_default_dof_pos(self):
        self.default_dof_pos_base = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if name not in self.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")
            angle = self.robot_config.init_state.default_joint_angles[name]
            self.default_dof_pos_base[i] = angle

        self.default_dof_pos_base = self.default_dof_pos_base.unsqueeze(0)  # (1, num_dof)
        self.default_dof_pos = self.default_dof_pos_base.repeat(self.num_envs, 1).clone()  # (num_envs, num_dof)

    def _pre_compute_observations_callback(self):
        self.base_quat[:] = self.simulator.base_quat[:]

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        contact_cache = getattr(self, "_object_contact_surface_cache", None)
        if contact_cache is not None:
            contact_cache.clear()
        self._object_contact_surface_cache_generation = getattr(self, "_object_contact_surface_cache_generation", 0) + 1

        self.need_to_refresh_envs[env_ids] = True
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # pending_episode_update_mask is only used in curriculum_term::AverageEpisodeLengthTracker.
        self._pending_episode_update_mask[env_ids] = True

    def _get_envs_to_refresh(self):
        return self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()

    def _refresh_envs_after_reset(self, env_ids):
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self.simulator.write_state_updates()
        self.simulator.clear_contact_forces_history(env_ids)
        self.need_to_refresh_envs[env_ids] = False
        self.simulator.refresh_sim_tensors()
        self._pre_compute_observations_callback()

    def _get_average_episode_tracker(self):
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return tracker

    # -------------------------------- terms same with locomotion_manager.py [end]--------------------------------

    def _update_log_dict(self):
        # _update_log_dict happens before reset_envs_idx
        for key in list(self.log_dict.keys()):
            if key.startswith(("Object/", "Motion/", "Termination/", "TerminationFailRate/")):
                del self.log_dict[key]

        # -------------------------------- terms same with locomotion_manager.py [start]--------------------------------
        avg = self._get_average_episode_tracker().get_average()
        self.log_dict["average_episode_length"] = avg.detach().cpu()
        # -------------------------------- terms same with locomotion_manager.py [end]--------------------------------
        # Add tracking metrics to log_dict
        motion_command = self.command_manager.get_state("motion_command")
        motion_command.update_metrics()
        self.log_dict.update(motion_command.metrics)
        self._update_motion_start0_log_dict(motion_command)
        self._update_motion_termination_log_dict(motion_command)

    def _update_motion_start0_log_dict(self, motion_command):
        if motion_command is None:
            return
        required_attrs = (
            "started_at_timestep_zero",
            "clip_ids",
            "clip_start_steps",
            "clip_end_steps",
            "time_steps",
        )
        if not all(hasattr(motion_command, attr) for attr in required_attrs):
            return

        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if done_env_ids.numel() == 0:
            return

        start0_mask = motion_command.started_at_timestep_zero[done_env_ids]
        env_ids = done_env_ids[start0_mask]
        if env_ids.numel() == 0:
            return

        clip_ids = motion_command.clip_ids[env_ids]
        real_motion_clip_ranges = getattr(motion_command.motion, "real_motion_clip_ranges", None)
        if real_motion_clip_ranges is not None and len(real_motion_clip_ranges) > 0:
            real_starts = torch.tensor(
                [clip_range[0] for clip_range in real_motion_clip_ranges],
                dtype=torch.long,
                device=self.device,
            )
            real_ends = torch.tensor(
                [clip_range[1] for clip_range in real_motion_clip_ranges],
                dtype=torch.long,
                device=self.device,
            )
            completion_start_steps = real_starts[clip_ids]
            completion_end_steps = real_ends[clip_ids]
        else:
            completion_start_steps = motion_command.clip_start_steps[env_ids]
            completion_end_steps = motion_command.clip_end_steps[env_ids]

        clip_progress = (motion_command.time_steps[env_ids] - completion_start_steps).to(dtype=torch.float32)
        terminal_progress = (completion_end_steps - completion_start_steps - 2).to(dtype=torch.float32)
        terminal_progress = torch.clamp(terminal_progress, min=1.0)
        completion_percent = torch.clamp(clip_progress / terminal_progress, min=0.0, max=1.0) * 100.0

        if hasattr(motion_command, "update_hard_motion_sampling_stats"):
            motion_command.update_hard_motion_sampling_stats(clip_ids, completion_percent)

        for clip_id in torch.unique(clip_ids).tolist():
            clip_id_int = int(clip_id)
            clip_mask = clip_ids == clip_id_int
            if not clip_mask.any():
                continue
            metric_prefix = self._get_motion_metric_prefix(motion_command, clip_id_int)
            self.log_dict[f"{metric_prefix}/completion_percent_start0"] = completion_percent[clip_mask].detach()

    def _update_motion_termination_log_dict(self, motion_command):
        if motion_command is None or self.termination_manager is None:
            return
        if not hasattr(motion_command, "clip_ids"):
            return

        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if done_env_ids.numel() == 0:
            return

        outcome_masks, detail_masks = self._get_termination_masks()
        if not outcome_masks:
            return

        clip_ids = motion_command.clip_ids[done_env_ids]
        for clip_id in torch.unique(clip_ids).tolist():
            clip_id_int = int(clip_id)
            clip_mask = clip_ids == clip_id_int
            if not clip_mask.any():
                continue

            env_ids_for_clip = done_env_ids[clip_mask]
            metric_prefix = self._get_termination_metric_prefix(motion_command, clip_id_int)
            episode_count = torch.as_tensor(float(env_ids_for_clip.numel()), device=self.device)
            self.log_dict[f"{metric_prefix}/episode"] = episode_count.detach()

            for outcome_name, outcome_mask in outcome_masks.items():
                outcome_count = outcome_mask[env_ids_for_clip].to(dtype=torch.float32).sum()
                self.log_dict[f"{metric_prefix}/{outcome_name}"] = outcome_count.detach()

            bad_tracking_mask = outcome_masks["bad_tracking"][env_ids_for_clip]
            bad_tracking_count = bad_tracking_mask.to(dtype=torch.float32).sum()
            fail_rate_prefix = self._get_termination_fail_rate_metric_prefix(motion_command, clip_id_int)
            for detail_name, detail_mask in detail_masks.items():
                detail_count = (detail_mask[env_ids_for_clip] & bad_tracking_mask).to(dtype=torch.float32).sum()
                rate_parts = torch.stack((detail_count, bad_tracking_count))
                self.log_dict[f"{fail_rate_prefix}/{detail_name}"] = rate_parts.detach()

    def _get_termination_masks(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        term_manager = self.termination_manager
        if term_manager is None:
            return {}, {}

        last_term_results = getattr(term_manager, "last_term_results", {}) or {}

        def _term_mask(term_name: str) -> torch.Tensor:
            mask = last_term_results.get(term_name)
            if mask is None:
                return zero
            return mask.to(device=self.device, dtype=torch.bool)

        outcome_masks: dict[str, torch.Tensor] = {
            "motion_ends": _term_mask("motion_ends"),
            "bad_tracking": _term_mask("bad_tracking"),
            "timeout": getattr(term_manager, "last_timeout_flags", zero).to(device=self.device, dtype=torch.bool),
        }
        for term_name in last_term_results:
            if term_name not in outcome_masks:
                outcome_masks[term_name] = _term_mask(term_name)

        bad_tracking_term = getattr(term_manager, "_term_instances", {}).get("bad_tracking")
        bad_tracking_reasons = getattr(bad_tracking_term, "last_reason_results", {}) or {}
        detail_masks: dict[str, torch.Tensor] = {}
        for detail_name in (
            "bad_object_pos",
            "bad_object_ori",
            "bad_motion_body_pos",
            "bad_ref_pos",
            "bad_ref_ori",
        ):
            mask = bad_tracking_reasons.get(detail_name)
            if mask is None:
                mask = zero
            detail_masks[detail_name] = mask.to(device=self.device, dtype=torch.bool)

        return outcome_masks, detail_masks

    @staticmethod
    def _sanitize_metric_component(value: str) -> str:
        return value.replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _get_motion_metric_prefix(self, motion_command, clip_id: int) -> str:
        clip_files = getattr(motion_command.motion, "clip_files", [])
        clip_object_keys = getattr(motion_command.motion, "clip_object_keys", [])

        if 0 <= clip_id < len(clip_files):
            motion_name = Path(str(clip_files[clip_id])).stem
        else:
            motion_name = f"clip_{clip_id:03d}"
        motion_name = self._sanitize_metric_component(motion_name)

        object_key = clip_object_keys[clip_id] if 0 <= clip_id < len(clip_object_keys) else None
        if object_key is None:
            return f"Motion/{motion_name}"
        object_name = self._sanitize_metric_component(str(object_key))
        return f"Motion/{object_name}/{motion_name}"

    def _get_termination_metric_prefix(self, motion_command, clip_id: int) -> str:
        motion_prefix = self._get_motion_metric_prefix(motion_command, clip_id)
        if motion_prefix.startswith("Motion/"):
            return f"Termination/{motion_prefix[len('Motion/') :]}"
        return f"Termination/{motion_prefix}"

    def _get_termination_fail_rate_metric_prefix(self, motion_command, clip_id: int) -> str:
        motion_prefix = self._get_motion_metric_prefix(motion_command, clip_id)
        if motion_prefix.startswith("Motion/"):
            return f"TerminationFailRate/{motion_prefix[len('Motion/') :]}"
        return f"TerminationFailRate/{motion_prefix}"

    def reset_all(self):
        # If reset_all is called several times, clear buffer in motion_command
        motion_command = self.command_manager.get_state("motion_command")
        motion_command.init_buffers()
        return super().reset_all()

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # TODO(jchen): Now,reset robot/object states is implemented in command/terms/wbt.MotionCommand.reset
        # discuss whether to move to here in the future.
        pass

    ########################################################### Push robots #########################################
    # TODO: This should be moved to the randomization manager.
    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        # Action delay buffers are now initialized by randomization manager's setup_action_delay_buffers term

        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.record_push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._randomize_push_robots = False
        self._max_push_vel = torch.zeros(6, dtype=torch.float32, device=self.device)

    def _push_robots(self, env_ids):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = self.randomization_manager.get_state("push_randomizer_state")
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 6:
            raise ValueError("WholeBodyTracking push velocity vector must have exactly 6 components.")

        rand = torch.rand(len(env_ids), 6, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        self.simulator.robot_root_states[env_ids, 7:13] = self.push_robot_vel_buf[env_ids]
        # Push impulses only take effect in the simulator once we write the mutated root state tensor back.
        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        self._max_push_vel = max_vel_tensor.clone()

    #########################################################################################################
    ## Debug visualization
    #########################################################################################################

    def _draw_debug_vis_isaacsim(self):
        motion_command = self.command_manager.get_state("motion_command")
        # torso link
        real_robot_pos_xyz = motion_command.robot_ref_pos_w.clone()
        real_robot_quat_xyzw = motion_command.robot_ref_quat_w.clone()
        real_robot_quat_wxyz = real_robot_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["real_robot"].visualize(real_robot_pos_xyz, real_robot_quat_wxyz)

        motion_robot_pos_xyz = motion_command.ref_pos_w.clone()
        motion_robot_quat_xyzw = motion_command.ref_quat_w.clone()
        motion_robot_quat_wxyz = motion_robot_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["motion_robot"].visualize(motion_robot_pos_xyz, motion_robot_quat_wxyz)

        for body_idx, body_names in enumerate(motion_command.motion_cfg.body_names_to_track):
            motion_robot_body_pos_xyz = motion_command.body_pos_w[0, body_idx].clone()
            motion_command.visualization_markers[f"motion_{body_names}"].visualize(
                motion_robot_body_pos_xyz.unsqueeze(0)
            )

        # object
        if motion_command.motion.has_object:
            real_object_pos_xyz = motion_command.simulator_object_pos_w.clone()
            real_object_quat_xyzw = motion_command.simulator_object_quat_w.clone()
            real_object_quat_wxyz = real_object_quat_xyzw[:, [3, 0, 1, 2]]
            object_scales = None
            if hasattr(self, "object_scale_factors"):
                # Frame markers look distorted under anisotropic scaling, so visualize
                # only the overall scale magnitude with a uniform marker scale.
                uniform_scale = self.object_scale_factors.amax(dim=1, keepdim=True)
                object_scales = uniform_scale.repeat(1, 3).clone()
            motion_command.visualization_markers["real_object"].visualize(
                real_object_pos_xyz,
                real_object_quat_wxyz,
                scales=object_scales,
            )

            motion_object_pos_xyz = motion_command.object_pos_w.clone()
            if hasattr(motion_command, "object_pos_reward_offset"):
                motion_object_pos_xyz = motion_object_pos_xyz + motion_command.object_pos_reward_offset
            motion_object_quat_xyzw = motion_command.object_quat_w.clone()
            motion_object_quat_wxyz = motion_object_quat_xyzw[:, [3, 0, 1, 2]]
            motion_command.visualization_markers["motion_object"].visualize(
                motion_object_pos_xyz,
                motion_object_quat_wxyz,
                scales=object_scales,
            )

    def _draw_debug_vis_isaacgym(self):
        self.simulator.clear_lines()
        n_bodies = len(self.motion_command.motion_cfg.body_names_to_track)
        for env_id in range(self.num_envs):
            for body_idx in range(n_bodies):
                color = (0.0, 1.0, 0.0)
                self.simulator.draw_sphere(
                    self.motion_command.body_pos_relative_w[env_id, body_idx], 0.03, color, env_id, body_idx
                )

                color = (0.0, 0.0, 1.0)
                self.simulator.draw_sphere(
                    self.motion_command.robot_body_pos_w[env_id, body_idx], 0.03, color, env_id, n_bodies + body_idx
                )

            color = (0.0, 1.0, 0.0)
            self.simulator.draw_sphere(self.motion_command.ref_pos_w[env_id], 0.05, color, env_id, n_bodies * 2 + 0)
            color = (0.0, 0.0, 1.0)
            self.simulator.draw_sphere(
                self.motion_command.robot_ref_pos_w[env_id], 0.05, color, env_id, n_bodies * 2 + 1
            )

    def _draw_debug_vis(self):
        if self.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._draw_debug_vis_isaacsim()
        elif self.simulator.get_simulator_type() == SimulatorType.ISAACGYM:
            self._draw_debug_vis_isaacgym()

    def step_visualize_motion(self, actions):
        motion_command = self.command_manager.get_state("motion_command")
        dt = 1.0 / float(motion_command.motion.fps)
        motion_command.step()
        print("time_steps: ", motion_command.time_steps[0].item())
        self._draw_debug_vis()

        # set root_states_from_motion_command
        root_pos = motion_command.root_pos_w.clone()
        root_ori = motion_command.root_quat_w.clone()  # wxyz
        root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
        root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()

        joint_pos = motion_command.joint_pos.clone()
        joint_vel = motion_command.joint_vel.clone()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.dof_pos[env_ids] = joint_pos
        self.simulator.dof_vel[env_ids] = joint_vel

        self.simulator.robot_root_states[env_ids, :3] = root_pos
        self.simulator.robot_root_states[env_ids, 3:7] = root_ori
        self.simulator.robot_root_states[env_ids, 7:10] = root_lin_vel
        self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel

        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)

        if motion_command.motion.has_object:
            # set object root_states from motion command
            object_pos = motion_command.object_pos_w.clone()
            object_ori = motion_command.object_quat_w.clone()
            object_lin_vel = motion_command.object_lin_vel_w.clone()

            object_states = torch.zeros(len(env_ids), 13, device=self.device)
            object_states[:, :3] = object_pos[:]
            object_states[:, 3:7] = object_ori[:]
            object_states[:, 7:10] = object_lin_vel[:]
            object_states[:, 10:13] = torch.zeros_like(object_lin_vel[:])
            motion_command.set_simulator_object_states(env_ids, object_states)

        self.simulator.scene.write_data_to_sim()
        self.simulator.sim.forward()
        self.simulator.sim.render()
        self.simulator.refresh_sim_tensors()

        time.sleep(dt)

        clip_end_steps = getattr(motion_command, "clip_end_steps", None)
        if clip_end_steps is not None:
            return motion_command.time_steps[0].item() >= clip_end_steps[0].item() - 2
        return motion_command.time_steps[0].item() >= motion_command.motion.time_step_total - 2
