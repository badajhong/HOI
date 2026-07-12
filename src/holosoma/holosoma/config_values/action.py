"""Default action manager configurations."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_values.loco.g1.action import g1_29dof_joint_pos
from holosoma.config_values.loco.t1.action import t1_29dof_joint_pos

none = None

R1_24DOF_ACTION_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

r1_24dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={"action_dof_names": R1_24DOF_ACTION_DOF_NAMES},
            scale=1.0,
            clip=None,
        ),
    }
)

# The R1 asset still has 26 simulator DOFs; this legacy preset name now maps
# to the 24 controlled DOF action space that leaves the head locked to default.
r1_26dof_joint_pos = r1_24dof_joint_pos

DEFAULTS = {
    "none": none,
    "t1_29dof_joint_pos": t1_29dof_joint_pos,
    "g1_29dof_joint_pos": g1_29dof_joint_pos,
    "r1_24dof_joint_pos": r1_24dof_joint_pos,
    "r1_26dof_joint_pos": r1_26dof_joint_pos,
}
