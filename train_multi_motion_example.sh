#!/bin/bash
# Multi-Motion Training Example for G1 Robot with Object (Suitcase)

# Source the IsaacSim environment
source scripts/source_isaacsim_setup.sh

# Example 1: Train with multiple motion clips from a folder
# The motion folder should contain .npz files (motion_clip_1.npz, motion_clip_2.npz, etc.)
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-w-object-multi \
    logger:wandb \
    --command.setup-terms.motion-command.params.motion-config.motion-folder="/home/rllab/haechan/holosoma/train/motions/suitcase_select" \
    --robot.object.object-urdf-path="/home/rllab/haechan/holosoma/train/objects/suitcase/suitcase.urdf" \
    --training.num-envs 16384 \
    --algo.config.save-interval 1000 \
    --simulator.config.scene.env-spacing 5.0 \
    --logger.video.camera.offset '[3.0, -3.0, 1.5]' \
    --logger.video.camera.target-offset '[0.0, 0.0, 0.5]'

# Example 2: If you want to override the motion folder at runtime
# python src/holosoma/holosoma/train_agent.py \
#     exp:g1-29dof-wbt-w-object-multi \
#     logger:wandb \
#     --command.setup-terms.motion-command.params.motion-config.motion-folder="/custom/path/to/motions/" \
#     --robot.object.object-urdf-path="/path/to/object.urdf" \
#     --training.num-envs 4096

# Example 3: Train with different environment spacing
# python src/holosoma/holosoma/train_agent.py \
#     exp:g1-29dof-wbt-w-object-multi \
#     --command.setup-terms.motion-command.params.motion-config.motion-folder="/home/rllab/haechan/holosoma/train/suitcase/" \
#     --training.num-envs 8192 \
#     --simulator.config.scene.env-spacing 3.0
