# Multi-Motion Training Setup

This document describes the implementation of multi-file motion training support for the holosoma project.

## What Was Implemented

### 1. **Motion Configuration Enhancement** (`config_types/command.py`)
   - Updated `MotionConfig` to support both `motion_file` and `motion_folder` parameters
   - Users can now specify **either**:
     - A single `.npz` file via `motion_file`
     - A folder containing multiple `.npz` files via `motion_folder`
   - Files in the folder are automatically discovered, sorted, and concatenated

### 2. **Motion Loader Update** (`managers/command/terms/wbt.py`)
   - Added `_load_and_concat_motions_from_folder()` method to concatenate multiple NPZ files
   - Automatically detects all `.npz` files in the specified folder
   - Concatenates data along the time dimension (T)
   - Preserves all motion metadata (FPS, body names, joint names)
   - Supports both object-less and object-aware motion data

### 3. **Command Configuration** (`config_values/wbt/g1/command.py`)
   - Created `motion_config_w_object_multi` configuration template
   - Added `g1_29dof_wbt_command_w_object_multi` command manager

### 4. **Experiment Configuration** (`config_values/wbt/g1/experiment.py`)
   - Added `g1_29dof_wbt_w_object_multi` experiment preset
   - Based on `g1_29dof_wbt_w_object` but uses multi-motion command

### 5. **Registration** (`config_values/experiment.py` and `config_values/command.py`)
   - Registered new experiment in DEFAULTS dictionary
   - Available as `exp:g1-29dof-wbt-w-object-multi`

## Usage

### Training with Multiple Motion Files

Use the following command structure:

```bash
source scripts/source_isaacsim_setup.sh

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-w-object-multi \
    logger:wandb \
    --command.setup-terms.motion-command.params.motion-config.motion-folder="/path/to/motion/folder/" \
    --robot.object.object-urdf-path="/path/to/object.urdf" \
    --training.num-envs 4096 \
    --algo.config.save-interval 1000 \
    --simulator.config.scene.env-spacing 5.0 \
    --logger.video.camera.offset '[3.0, -3.0, 1.5]' \
    --logger.video.camera.target-offset '[0.0, 0.0, 0.5]'
```

### Key Points

1. **Motion Folder Parameter**: Use underscores, not hyphens (pydantic uses snake_case)
   - ✅ `--command.setup-terms.motion-command.params.motion-config.motion-folder="/path/"`
   - ❌ `--command.setup-terms.motion-command.params.motion-config.motion_folder="/path/"` (tyro converts this)

2. **File Discovery**: All `.npz` files in the folder are automatically discovered and sorted alphabetically
   
3. **Concatenation**: Files are concatenated along the time dimension (T axis)
   - File 1: (T1, ...) 
   - File 2: (T2, ...)
   - Result: (T1 + T2, ...)

4. **FPS Validation**: A warning is logged if files have different FPS values (but training continues)

## Example Workflow

```bash
# 1. Create a folder with your motion files
mkdir -p ~/data/motions/suitcase/
cp motion_clip_1.npz ~/data/motions/suitcase/
cp motion_clip_2.npz ~/data/motions/suitcase/
cp motion_clip_3.npz ~/data/motions/suitcase/

# 2. Run training with the multi-motion experiment
source scripts/source_isaacsim_setup.sh

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-w-object-multi \
    logger:wandb \
    --command.setup-terms.motion-command.params.motion-config.motion-folder="~/data/motions/suitcase/" \
    --robot.object.object-urdf-path="~/data/objects/suitcase.urdf" \
    --training.num-envs 4096
```

## NPZ File Format Requirements

Each `.npz` file must contain:
- `fps`: Frames per second (int or float)
- `joint_pos`: Shape (T, 7+J) - root pose (xyz, wxyz) + joint positions
- `joint_vel`: Shape (T, 6+J) - root velocity + joint velocities
- `body_pos_w`: Shape (T, B, 3) - body positions in world frame
- `body_quat_w`: Shape (T, B, 4) - body quaternions (wxyz format)
- `body_lin_vel_w`: Shape (T, B, 3) - body linear velocities
- `body_ang_vel_w`: Shape (T, B, 3) - body angular velocities
- `body_names`: List of body names
- `joint_names`: List of joint names

### Optional (for object manipulation):
- `object_pos_w`: Shape (T, 3)
- `object_quat_w`: Shape (T, 4) - wxyz format
- `object_lin_vel_w`: Shape (T, 3)
- `object_ang_vel_w`: Shape (T, 3)

## Notes

- The concatenated motion sequence can be arbitrarily long, limited only by GPU memory
- Motion transitions are smooth when using the default pose interpolation settings
- Each clip is treated as a continuous sequence; there's no automatic gap-closing between clips
- For GPU training with large motion sequences, consider using the `adaptive_timesteps_sampler` if certain segments are more important

## Backward Compatibility

- Existing training commands using `motion_file` continue to work unchanged
- The default experiment `g1-29dof-wbt-w-object` still uses single-file training
- No changes to observation, reward, or termination configurations
