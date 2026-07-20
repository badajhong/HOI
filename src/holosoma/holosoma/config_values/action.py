"""Default action manager configurations."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_values.loco.g1.action import g1_29dof_joint_pos
from holosoma.config_values.loco.t1.action import t1_29dof_joint_pos

none = None
r1_26dof_joint_pos = g1_29dof_joint_pos
r1_24dof_joint_pos_fixed_head = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func=(
                "holosoma.managers.action.terms.joint_control:"
                "JointPositionActionTermWithFixedJoints"
            ),
            params={"fixed_joint_names": ["head_pitch_joint", "head_yaw_joint"]},
            scale=1.0,
            clip=None,
        ),
    }
)

DEFAULTS = {
    "none": none,
    "t1_29dof_joint_pos": t1_29dof_joint_pos,
    "g1_29dof_joint_pos": g1_29dof_joint_pos,
    "r1_26dof_joint_pos": r1_26dof_joint_pos,
    "r1_24dof_joint_pos_fixed_head": r1_24dof_joint_pos_fixed_head,
}
