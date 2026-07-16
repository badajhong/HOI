"""Shared R1 object-contact settings."""

R1_OBJECT_CONTACT_THRESHOLD = 0.1
R1_OBJECT_CONTACT_LABEL_DISTANCE_SCALE = 0.1
R1_OBJECT_CONTACT_DISTANCE_CLIP = 0.5
R1_OBJECT_CONTACT_TARGET_DISTANCE_SCALE = 0.06
R1_OBJECT_CONTACT_TARGET_RELATIVE_CLIP = 0.25
R1_OBJECT_CONTACT_TARGET_TOPK = 1
R1_OBJECT_CONTACT_TARGET_COVERAGE_THRESHOLD = 0.1
R1_OBJECT_CONTACT_TARGET_COVERAGE_TEMPERATURE = 0.02
R1_OBJECT_CONTACT_TARGET_SURFACE_MISMATCH_THRESHOLD = 0.08
R1_OBJECT_CONTACT_TARGET_SURFACE_MISMATCH_SCALE = 0.04
R1_OBJECT_CONTACT_TARGET_SURFACE_DISTANCE_CUTOFF = 0.1
R1_OBJECT_HEAD_PROXIMITY_DISTANCE_SCALE = 0.05
R1_OBJECT_HEAD_PROXIMITY_DISTANCE_CUTOFF = 0.15
R1_OBJECT_HEAD_PROXIMITY_BODY_NAMES = ("head_pitch_link", "head_yaw_link")
R1_OBJECT_HEAD_PROXIMITY_BODY_POINTS = {
    # Local proxy points around the two R1 head collision meshes. The reward term
    # uses the nearest head proxy, so mesh-surface contacts are not missed just
    # because the body origin is still far from the object.
    "head_pitch_link": (
        (0.001, -0.032, 0.045),
        (0.070, -0.032, 0.045),
        (-0.068, -0.032, 0.045),
        (0.001, 0.009, 0.045),
        (0.001, -0.072, 0.045),
        (0.001, -0.032, 0.130),
        (0.001, -0.032, -0.040),
    ),
    "head_yaw_link": (
        (0.018, 0.000, -0.008),
        (0.081, 0.000, -0.008),
        (-0.045, 0.000, -0.008),
        (0.018, 0.073, -0.008),
        (0.018, -0.073, -0.008),
        (0.018, 0.000, 0.082),
        (0.018, 0.000, -0.098),
    ),
}

R1_OBJECT_CONTACT_BODY_NAMES = (
    "pelvis_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "left_ankle_constraint_A_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "right_ankle_constraint_A_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_hand_contact_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_hand_contact_link",
)

R1_OBJECT_CONTACT_REWARD_EXCLUDED_BODY_NAMES = {
    "head_pitch_link",
    "head_yaw_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
}
R1_OBJECT_CONTACT_REWARD_BODY_NAMES = tuple(
    body_name
    for body_name in R1_OBJECT_CONTACT_BODY_NAMES
    if body_name not in R1_OBJECT_CONTACT_REWARD_EXCLUDED_BODY_NAMES
)
R1_OBJECT_CONTACT_REWARD_BODY_NAMES_REGEX = (
    r"^(?!head_pitch_link$)(?!head_yaw_link$)"
    r"(?!left_wrist_roll_link$)(?!right_wrist_roll_link$).*"
)
