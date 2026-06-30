# Holosoma Motion Retargeting

This repository provides tools for retargeting human motion data to humanoid robots. It supports multiple data formats (smplh, mocap, lafan) and task types including robot-only motion, object interaction, scaled object interaction, and climbing.

**Data Requirements**: The retargeting pipeline requires motion data in world joint positions. For custom data, you need to prepare world joint positions in shape `(T, J, 3)` where T is the number of frames and J is the number of joints, and modify `demo_joints` and `joints_mapping` defined in `config_types/data_type.py`.

## Single Sequence Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type robot_only --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Object interaction (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type object_interaction --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Object interaction with SMPL-scaled object assets (matches the red demo points)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type object_interaction_scaled --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Climbing
python examples/robot_retarget.py --data_path demo_data/climb --task-type climbing --task-name mocap_climb_seq_0 --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --retargeter.debug --retargeter.visualize
```

**Note**: `object_interaction_scaled` creates scaled object URDF/XML assets automatically and applies a z-offset so compressed objects stay on the floor instead of floating. Add `--augmentation` to run sequences with augmentation. You must first run the original sequence before adding augmentation.

## Batch Processing for Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type robot_only --data_format smplh --save_dir demo_results_parallel/g1/robot_only/omomo --task-config.object-name ground

# Object interaction (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type object_interaction --data_format smplh --save_dir demo_results_parallel/g1/object_interaction/omomo --task-config.object-name largebox

# Object interaction with SMPL-scaled object assets
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type object_interaction_scaled --data_format smplh --save_dir demo_results_parallel/g1/object_interaction_scaled/omomo --task-config.object-name largebox

# Climbing
python examples/parallel_robot_retarget.py --data-dir demo_data/climb --task-type climbing --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --task-config.object-name multi_boxes --save_dir demo_results_parallel/g1/climbing/mocap_climb
```

**Note**: Add `--augmentation` to run original sequences and sequences with augmentation (for object interaction and climbing tasks).

## Data Preparation

We provide `demo_data/` for fast testing. To test on more motion sequences, please follow the instructions below to download and prepare the data.

### OMOMO

Our pipeline uses the processed dataset by InterMimic. The data format differs from the original OMOMO dataset.

1. Download the processed OMOMO data from [this link](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view)
2. Extract the downloaded folder to `demo_data/OMOMO_new`

The data should contain `.pt` files.

### LAFAN

#### Download the Original LAFAN Data

1. Download [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) by clicking "View Raw"
2. Put `lafan1.zip` in your designated data folder and uncompress it to `DATA_FOLDER_PATH/lafan`
3. The file structure should be `demo_data/lafan/*.bvh`

#### Convert the Original LAFAN Data Format for Motion Retargeting

We need some data processing files from the [LAFAN GitHub repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

```bash
cd holosoma_retargeting/data_utils/
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
mv ubisoft-laforge-animation-dataset/lafan1 .
python extract_global_positions.py --input_dir DATA_FOLDER_PATH/lafan --output_dir ../demo_data/lafan
```

This will convert the BVH files to `.npy` format with global joint positions.

**Note**: For LAFAN data, you need to relax the foot sticking constraint by setting `--retargeter.foot-sticking-tolerance` (default is stricter). You can adjust this tolerance number based on your data quality and retargeting results.

#### Single Sequence Retargeting on LAFAN

```bash
python examples/robot_retarget.py --data_path demo_data/lafan --task-type robot_only --task-name dance2_subject1 --data_format lafan --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/lafan --retargeter.debug --retargeter.visualize --retargeter.foot-sticking-tolerance 0.02
```

#### Batch Processing for Motion Retargeting on LAFAN

```bash
python examples/parallel_robot_retarget.py --data-dir demo_data/lafan --task-type robot_only --data_format lafan --save_dir demo_results_parallel/g1/robot_only/lafan --task-config.object-name ground --task-config.ground-range -10 10 --retargeter.foot-sticking-tolerance 0.02
```

### AMASS SMPL-X

#### Download the Original AMASS Data

1. Follow the [AMASS](https://amass.is.tue.mpg.de/) instructions to download the original AMASS data
2. The AMASS data structure should be `/path/to/amass/dataset_name/subject_name/*.npz`

#### Download SMPL-X Models

1. Follow the [SMPL-X](https://smpl-x.is.tue.mpg.de/index.html) instructions to download SMPL-X models
2. For AMASS data, we tested on SMPL-X N (neutral) format
3. The SMPL-X models structure should be `/path/to/models/smplx/SMPLX_NEUTRAL.npz`

#### Convert the Original AMASS SMPL-X Data Format for Motion Retargeting

We provide `data_utils/prep_amass_smplx_for_rt.py` for converting AMASS SMPLX data to the format required for motion retargeting.

```bash
# Install dependencies
cd holosoma_retargeting/data_utils/
git clone https://github.com/nghorbani/human_body_prior.git
pip install tqdm dotmap PyYAML omegaconf loguru
cd human_body_prior/
python setup.py develop
cd ../

# Run data processing
python prep_amass_smplx_for_rt.py \
  --amass-root-folder /path/to/amass \
  --output-folder /path/to/output \
  --model-root-folder /path/to/models
```

This will convert the AMASS `.npz` files to `.npz` format with global joint positions and height information.

**Note**: You can optionally specify `--subdataset-folder` to process only a specific subdataset (e.g., `HumanEva`). If not specified, it will process all datasets recursively.

#### Single Sequence Retargeting on AMASS SMPL-X

```bash
python examples/robot_retarget.py --data_path demo_data/amass_smplx_processed --task-type robot_only --task-name HumanEva_S3_Jog_1_stageii --data_format smplx --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/amass_smplx --retargeter.debug --retargeter.visualize
```

#### Batch Processing for Motion Retargeting on AMASS SMPL-X

```bash
python examples/parallel_robot_retarget.py --data-dir demo_data/amass_smplx_processed --task-type robot_only --data_format smplx --save_dir demo_results_parallel/g1/robot_only/amass_smplx --task-config.object-name ground --task-config.ground-range -10 10
```

## Check Visualizations of Saved Retargeting Results

```bash
# Visualize object-interaction results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --object_urdf models/largebox/largebox.urdf \
    --qpos_npz demo_results_parallel/g1/object_interaction/omomo/sub3_largebox_003_original.npz

# Visualize climbing results
python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_original.npz

python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes_scaled_0.74_0.74_0.89.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_z_scale_1.2.npz

# Visualize robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results_parallel/g1/robot_only/omomo/sub3_largebox_003_original.npz

# Visualize LAFAN robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/lafan/dance2_subject1.npz

# Visualize AMASS results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/amass_smplx/HumanEva_S3_Jog_1_stageii.npz

# Visualize AMASS results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results_parallel/g1/robot_only/amass_smplx/HumanEva_S1_Box_1_stageii_original.npz
```

## Quantitative Evaluation

```bash
# Evaluate robot-object interaction
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/object_interaction/omomo --data_dir demo_data/OMOMO_new --data_type "robot_object"

# Evaluate climbing sequence
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/climbing/mocap_climb --data_dir demo_data/climb --data_type "robot_terrain" --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf

# Evaluate robot only (OMOMO)
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/robot_only/omomo --data_dir demo_data/OMOMO_new --data_type "robot_only"
```

## Prepare Data for Training RL Whole-Body Tracking Policy

To prepare data for training RL whole-body tracking policies, you need to follow a two-step process:

1. **First, run retargeting** to obtain `.npz` files containing the retargeted robot motion. Use the retargeting commands shown in the sections above (Single Sequence Motion Retargeting or Batch Processing for Motion Retargeting).

2. **Then, run the data conversion code** below to convert the retargeted `.npz` files into the format required for RL training. The conversion script takes the retargeted `.npz` files as input and outputs converted files with the specified frame rate and format.

**Note**: If you run this code on Mac, please use `mjpython` instead of `python`.

### Mac (using mjpython)

```bash
mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox" --has_dynamic_object --once
```

### Robot-Only Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/lafan/dance2_subject1.npz --output_fps 50 --output_name converted_res/robot_only/dance2_subject1_mj_fps50.npz --data_format lafan --object_name "ground" --once
```

### Robot-Object Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox" --has_dynamic_object --once
```

### Object Contact Labeling for RL Rewards

You can add frame-level object contact labels after converting object-interaction motions for RL training. The default labeling mode uses the original SMPLH human motion, not the retargeted robot motion. This makes the label an intended-contact target:

```text
SMPLH human_joints + object pose + object sample_points
-> human/object contact
-> aggregate contact to mapped robot body links
```

The output `.npz` keeps all original arrays and adds contact keys:

```text
contact_object_label
contact_object_distance
contact_object_names
contact_object_indices
contact_object_source
contact_object_target
```

To preview intended contacts in Viser, use `object_interaction_contact`. The robot visual mesh is white by default; selected robot links turn red on frames where the corresponding SMPLH joints are within the contact threshold of the object surface. This is the same human-source contact definition used by `data_utils/label_object_contacts.py`.

```bash
python examples/robot_retarget.py \
  --data_path demo_data/OMOMO_new \
  --task-type object_interaction_contact \
  --task-name sub4_whitechair_030 \
  --data_format smplh \
  --task-config.object-name whitechair \
  --retargeter.debug \
  --retargeter.visualize \
  --robot r1 \
  --retargeter.penetration-tolerance 0.02
```

Use `--retargeter.contact-source robot` only when you want a diagnostic view of actual retargeted robot geometry proximity to the object.

For R1 SMPLH retargeting, two virtual hand-center joints are appended to the loaded SMPLH motion:
`L_HandCenter = mean(L_Index1, L_Middle1, L_Ring1, L_Pinky1)` and
`R_HandCenter = mean(R_Index1, R_Middle1, R_Ring1, R_Pinky1)`.
These map to fixed R1 proxy links `left_hand_contact_link` and `right_hand_contact_link`, which are children of the wrist roll links.
The default contact regex includes both these virtual hand-center joints and the SMPLH finger joints, so the code path stays explicit.
In Viser, R1 foot proxy contacts such as `left_ankle_constraint_A_link` also highlight the corresponding `left_ankle_roll_link` foot mesh so toe contacts are visible on the foot body.

For a single converted RL motion, pass the matching original retargeting result with `--human-reference`:

```bash
python data_utils/label_object_contacts.py \
  --input converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz \
  --output converted_res/object_interaction/sub3_largebox_003_mj_w_obj_contact.npz \
  --human-reference demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
  --object-root models/objects \
  --object-name largebox \
  --robot-type g1 \
  --threshold 0.05 \
  --overwrite
```

For batch labeling, keep converted RL filenames aligned with the original motion stem, then pass a root containing the original SMPLH `.npz` files:

```bash
python data_utils/label_object_contacts.py \
  --input converted_res/object_interaction \
  --output converted_res/object_interaction_contact_labeled \
  --human-reference-root demo_results/g1/object_interaction/omomo \
  --object-root models/objects \
  --robot-type g1 \
  --threshold 0.05 \
  --overwrite
```

For the R1 training folders in the main Holosoma workspace, run from the repository root:

```bash
R1_CONTACT_JOINT_REGEX='^(Pelvis|L_Hip|R_Hip|L_Knee|R_Knee|L_Shoulder|R_Shoulder|L_Elbow|R_Elbow|L_Ankle|R_Ankle|L_Toe|R_Toe|L_Wrist|R_Wrist|L_HandCenter|R_HandCenter|L_Index[123]|L_Middle[123]|L_Pinky[123]|L_Ring[123]|L_Thumb[123]|R_Index[123]|R_Middle[123]|R_Pinky[123]|R_Ring[123]|R_Thumb[123])$'

python src/holosoma_retargeting/holosoma_retargeting/data_utils/label_object_contacts.py \
  --input train_r1/rl \
  --output train_r1/rl_contact_labeled \
  --human-reference-root train_r1/motions \
  --object-root train_r1/objects \
  --robot-type r1 \
  --human-joint-regex "$R1_CONTACT_JOINT_REGEX" \
  --threshold 0.05 \
  --overwrite
```

This R1 regex selects the full SMPLH-to-R1 mapped major body set from `config_types/data_type.py`, plus SMPLH fingers and the virtual hand centers. With `--robot-type r1`, hand-center contacts map directly to `left_hand_contact_link` / `right_hand_contact_link`; finger contacts also collapse to those hand-contact proxy links when available.

When training the R1 teacher with these contact labels, keep the runtime threshold in the motion command config so the same value is shared by the object-contact reward and actor/critic current-contact observations:

```bash
--command.setup-terms.motion-command.params.motion-config.object-contact-threshold 0.05
```

Use `--source robot` only for diagnostics when you want to measure contact from the retargeted robot trajectory itself.

### OmniRetarget Data

For OmniRetarget data downloaded from HuggingFace, please add `--use_omniretarget_data` for data conversion.

```bash
python data_conversion/convert_data_format_mj.py --input_file OmniRetarget/robot-object/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj_omnirt.npz --data_format smplh --object_name "largebox" --has_dynamic_object --use_omniretarget_data --once
```

## Custom Human Motion Data Format
Please see the instructions for custom human motion data formats: [ADD_MOTION_FORMAT_README.md](ADD_MOTION_FORMAT_README.md)

## Custom Robot Type
Please see the instructions for retargeting custom robot types: [ADD_ROBOT_TYPE_README.md](ADD_ROBOT_TYPE_README.md)
