# Holosoma Configuration Structure & Motion Data Handling

## 1. EXPERIMENT CONFIGURATION FILES & STRUCTURE

### Location of Experiment Definitions

**Primary Configuration Files:**
- [holosoma/config_values/experiment.py](holosoma/config_values/experiment.py) - Main experiment presets registry
- [holosoma/config_types/experiment.py](holosoma/config_types/experiment.py) - Config dataclass definitions
- Robot/Task-specific configs:
  - [holosoma/config_values/wbt/g1/experiment.py](holosoma/config_values/wbt/g1/experiment.py) - Whole-body tracking for G1
  - [holosoma/config_values/loco/g1/experiment.py](holosoma/config_values/loco/g1/experiment.py) - Locomotion for G1
  - Similar pattern for T1 robot

**Important:** Holosoma uses **Tyro** (Python dataclasses) instead of traditional YAML configs. All configuration is code-based and type-checked.

### Available Experiments (from DEFAULTS dict)

```python
{
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,           # ← WITH OBJECT
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
}
```

These are accessed via CLI as: `exp:g1-29dof-wbt-w-object` (underscores converted to hyphens)

### Experiment Configuration Structure

From `ExperimentConfig` in [holosoma/config_types/experiment.py](holosoma/config_types/experiment.py#L105-L155):

```python
@dataclass(frozen=True)
class ExperimentConfig:
    # Environment & Architecture
    env_class: str                                    # e.g., "holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager"
    
    # Components (all using Tyro subcommands)
    training: TrainingConfig                          # Headless, num_envs, seed, etc.
    algo: AlgoConfig                                  # PPO, SAC algorithms
    simulator: SimulatorConfig                        # IsaacGym, IsaacSim, MuJoCo
    robot: RobotConfig                                # G1, T1 specifications
    terrain: TerrainManagerCfg                        # Plane, mix, etc.
    observation: ObservationManagerCfg                # What observations to feed
    action: ActionManagerCfg                          # Joint position control, etc.
    reward: RewardManagerCfg                          # Reward function
    termination: TerminationManagerCfg                # Episode termination
    randomization: RandomizationManagerCfg            # Domain randomization
    command: CommandManagerCfg                        # Command/Motion handling
    curriculum: CurriculumManagerCfg                  # Curriculum learning
    logger: LoggerConfig                              # W&B, tensorboard
```

### G1 WBT Experiment Example

From [holosoma/config_values/wbt/g1/experiment.py](holosoma/config_values/wbt/g1/experiment.py#L18-L45):

```python
g1_29dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_manager",
        num_envs=8192,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=40000,
            save_interval=4000,
            entropy_coef=0.005,
        ),
    ),
    simulator=replace(simulator.isaacsim, ...),
    robot=replace(robot.g1_29dof, ...),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    command=command.g1_29dof_wbt_command,    # ← Motion handling here
    reward=reward.g1_29dof_wbt_reward,
)
```

---

## 2. MOTION CONFIGURATION PARAMETERS

### How Motion Parameters Are Handled

**Parameter Path:** `--command.setup_terms.motion_command.params.motion_config.motion_file=<path>`

This follows the hierarchical structure:
- `command` → CommandManagerCfg (top-level command manager config)
- `setup_terms` → dict of CommandTermCfg that run during environment setup
- `motion_command` → specific motion command term
- `params` → additional parameters passed to the motion_command
- `motion_config` → MotionConfig object with motion specifications
- `motion_file` → path to .npz file

### MotionConfig Definition

From [holosoma/config_types/command.py](holosoma/config_types/command.py#L69-L130):

```python
@dataclass(frozen=True)
class MotionConfig:
    """Motion-related configuration for Whole Body Tracking."""
    
    # Core motion data
    motion_file: str                                  # Path to .npz file
    body_name_ref: list[str]                          # Reference frame (e.g., ["torso_link"])
    body_names_to_track: list[str]                    # Bodies used for tracking
    
    # Sampling behavior
    use_adaptive_timesteps_sampler: bool = False      # Prioritize hard samples
    start_at_timestep_zero_prob: float = 0.2          # Prob of starting at 0
    freeze_at_timestep_zero_prob: float = 0.95        # Prob of freezing at 0
    
    # Pose interpolation
    enable_default_pose_prepend: bool = True          # Interpolate from default pose at start
    default_pose_prepend_duration_s: float = 2.0      # Duration in seconds
    enable_default_pose_append: bool = True           # Interpolate back to default pose at end
    default_pose_append_duration_s: float = 2.0       # Duration in seconds
    
    # Noise to initial pose
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(...)
```

### Default G1 WBT Motion Config

From [holosoma/config_values/wbt/g1/command.py](holosoma/config_values/wbt/g1/command.py#L17-L44):

```python
motion_config = MotionConfig(
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    body_names_to_track=[
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
        "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    ],
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=False,
    noise_to_initial_pose=init_pose_config,
)

motion_config_w_object = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
)
```

### Command Manager Configuration

From [holosoma/config_values/wbt/g1/command.py](holosoma/config_values/wbt/g1/command.py#L46-L75):

```python
g1_29dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config,  # ← Passed here
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)
```

### How to Override Motion File at Runtime

From demo scripts:

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file=$CONVERTED_FILE
```

---

## 3. NPZ FILE FORMAT & LOADING

### Expected NPZ Structure

From [holosoma/config_types/command.py](holosoma/config_types/command.py#L73-L90):

```
Motion file (.npz) must contain:
    
REQUIRED:
    joint_pos:        (T, 7+J) - [root_xyz(3), root_quat_wxyz(4), joint_angles(J)]
    joint_vel:        (T, 6+J) - [root_vel(3), root_ang_vel(3), joint_vels(J)]
    body_pos_w:       (T, B, 3) - world positions of B bodies
    body_quat_w:      (T, B, 4) - world quaternions (wxyz format) of B bodies
    body_lin_vel_w:   (T, B, 3) - linear velocities of B bodies
    body_ang_vel_w:   (T, B, 3) - angular velocities of B bodies
    body_names:       (B,)      - names of bodies as strings
    joint_names:      (J,)      - names of robot joints (excluding root)
    fps:              scalar    - frames per second

OPTIONAL:
    object_pos_w:     (T, 3)    - world position of object (if present)
    object_quat_w:    (T, 4)    - world quaternion of object (wxyz)
    object_lin_vel_w: (T, 3)    - linear velocity of object
    object_ang_vel_w: (T, 3)    - angular velocity of object
```

### NPZ Loading Implementation

From [holosoma/managers/command/terms/wbt.py](holosoma/managers/command/terms/wbt.py#L33-L120):

```python
class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
    ):
        motion_file = resolve_data_file_path(motion_file)  # Resolve package paths
        logger.info(f"Loading motion file: {motion_file}")
        body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(
            motion_file, device
        )
        # Index mapping: map robot's bodies/joints to motion file's bodies/joints
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, device)
        
        self._joint_indexes = joint_indexes
        self._body_indexes = body_indexes
        self.time_step_total = self._joint_pos.shape[0]

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        with cached_open(motion_file, "rb") as f, np.load(f) as data:
            self.fps = data["fps"]
            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()

            # Extract joint positions (skip first 7 which are root xyz+quat)
            self._joint_pos = torch.tensor(data["joint_pos"][:, 7:], dtype=torch.float32, device=device)
            # Extract joint velocities (skip first 6 which are root vel+ang_vel)
            self._joint_vel = torch.tensor(data["joint_vel"][:, 6:], dtype=torch.float32, device=device)
            
            # Body data
            self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            body_quat_w_wxyz = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Convert wxyz to xyzw
            self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

            # Optional object data
            self.has_object = "object_pos_w" in data
            if self.has_object:
                self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Convert wxyz to xyzw
                self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)

        return body_names, joint_names
```

### Property Access (Index Mapping)

```python
@property
def joint_pos(self) -> torch.Tensor:
    return self._joint_pos[:, self._joint_indexes]  # Filter to robot's joints

@property
def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_w[:, self._body_indexes]  # Filter to tracked bodies
    
# Similar for: joint_vel, body_quat_w, body_lin_vel_w, body_ang_vel_w
```

### Key Implementation Details

1. **File Path Resolution:** Uses `resolve_data_file_path()` to support both absolute paths and package-relative paths
2. **File Caching:** Uses `cached_open()` for efficient file handling
3. **Quaternion Conversion:** Converts quaternions from wxyz (stored) to xyzw (internal representation)
4. **Root Extraction:** First 7 columns of joint_pos are root (xyz + quat), extracted separately
5. **Index Mapping:** Maps robot's body/joint names to motion file's ordering
6. **Optional Object Support:** Checks for "object_pos_w" key and loads if present
7. **GPU Support:** Loads data directly to specified device (CPU/CUDA)

---

## 4. SETUP IN ENVIRONMENT

### Motion Command Setup in MotionCommand Class

From [holosoma/managers/command/terms/wbt.py](holosoma/managers/command/terms/wbt.py#L268-L330):

```python
class MotionCommand(CommandTermBase):
    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self._env = env
        
        # Handle motion_config being dict (after tyro.cli parsing)
        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = MotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg = self.motion_cfg.noise_to_initial_pose

    def setup(self) -> None:
        # Get robot's body/joint names from environment
        robot_body_names = self._env.simulator._body_list
        robot_joint_names = self._env.simulator.dof_names

        # Load motion data
        self.motion: MotionLoader = MotionLoader(
            self.motion_cfg.motion_file,
            robot_body_names,
            robot_joint_names,
            device=self.device,
        )

        # Store body and joint indexes for interpolation
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes

        # Maybe prepend/append default pose transitions
        self._maybe_add_default_pose_transition(prepend=True)
        self._maybe_add_default_pose_transition(prepend=False)

        # Setup adaptive sampler if enabled
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                self.motion.time_step_total, self.device, int(1 / (self._env.dt))
            )
```

---

## 5. CONFIG SAVING & LOADING

### Saving Configuration

From [holosoma/config_types/experiment.py](holosoma/config_types/experiment.py#L218-L230):

```python
def save_config(self, path: str) -> None:
    with open(path, "w") as file:
        yaml.safe_dump(self.to_serializable_dict(), file)

def to_serializable_dict(self) -> dict:
    """Return a JSON-friendly representation of the config."""
    return json.loads(json.dumps(dataclasses.asdict(self)))
```

**Saved to:** `experiment_dir/holosoma_config.yaml` (CONFIG_NAME = "holosoma_config.yaml")

### Loading Configuration

From [holosoma/utils/eval_utils.py](holosoma/utils/eval_utils.py#L92-L102):

```python
def load_saved_experiment_config(checkpoint_cfg: CheckpointConfig) -> tuple[ExperimentConfig, str | None]:
    # Load from W&B or local cache
    config_uri = f"{_WANDB_PREFIX}{wandb_run_path}/{CONFIG_NAME}"
    cached_config_path = get_cached_file_path(config_uri)
    
    with open(cached_config_path) as f:
        return ExperimentConfig(**yaml.safe_load(f)), wandb_run_path
```

---

## 6. COMMAND CONFIGURATION TYPE HIERARCHY

From [holosoma/config_types/command.py](holosoma/config_types/command.py):

```python
@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""
    func: str                                        # Import path (e.g., "holosoma.managers.command.terms.wbt:MotionCommand")
    params: dict[str, Any] = field(default_factory=dict)  # Parameters passed to function

@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""
    params: dict[str, Any] = field(default_factory=dict)        # Global parameters
    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)  # Hooks on env setup
    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)  # Hooks on reset
    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)   # Hooks on step
```

---

## 7. QUICK REFERENCE: KEY PATHS

| Component | File | Purpose |
|-----------|------|---------|
| Experiment Registry | [holosoma/config_values/experiment.py](holosoma/config_values/experiment.py) | List all available experiments |
| WBT Experiment | [holosoma/config_values/wbt/g1/experiment.py](holosoma/config_values/wbt/g1/experiment.py) | WBT-specific experiment configs |
| WBT Command Config | [holosoma/config_values/wbt/g1/command.py](holosoma/config_values/wbt/g1/command.py) | Motion configs for WBT |
| Config Types | [holosoma/config_types/command.py](holosoma/config_types/command.py) | MotionConfig dataclass |
| Config Types | [holosoma/config_types/experiment.py](holosoma/config_types/experiment.py) | ExperimentConfig dataclass |
| NPZ Loading | [holosoma/managers/command/terms/wbt.py](holosoma/managers/command/terms/wbt.py) | MotionLoader class |
| Training Entry | [holosoma/train_agent.py](holosoma/train_agent.py) | Main training script |
| Training Utilities | [holosoma/utils/experiment_paths.py](holosoma/utils/experiment_paths.py) | Path management |
| Config Utils | [holosoma/utils/config_utils.py](holosoma/utils/config_utils.py) | CONFIG_NAME constant |

