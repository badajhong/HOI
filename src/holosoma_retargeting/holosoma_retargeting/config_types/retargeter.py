"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


DEFAULT_CONTACT_HUMAN_JOINT_REGEX = (
    r"^(Pelvis|L_Hip|R_Hip|L_Knee|R_Knee|L_Shoulder|R_Shoulder|L_Elbow|R_Elbow|"
    r"L_Ankle|R_Ankle|L_Toe|R_Toe|L_Wrist|R_Wrist|L_HandCenter|R_HandCenter|"
    r"L_Index[123]|L_Middle[123]|L_Pinky[123]|L_Ring[123]|L_Thumb[123]|"
    r"R_Index[123]|R_Middle[123]|R_Pinky[123]|R_Ring[123]|R_Thumb[123])$"
)


@dataclass(frozen=True)
class FootLockConfig:
    """Configuration for explicit frame-range based foot locking constraints."""

    enable: bool = False
    """Whether to enforce explicit frame-range based foot locking constraints."""

    windows: dict[str, list[tuple[int, int]]] | None = None
    """Per-foot inclusive frame windows for locking.
    Example: {"L_Toe": [(30, 60)], "R_Toe": [(10, 20), (80, 95)]}"""

    z_floor: float = 0.0
    """Floor height used by Z pinning constraints."""

    tolerance: float = 5e-3
    """Tolerance for Z floor pinning constraints."""


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce object non-penetration constraints."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    surface_penetration_tolerance: float | None = None
    """Optional penetration tolerance for ground/surface contacts.

    If None, uses penetration_tolerance.
    """

    object_penetration_tolerance: float | None = None
    """Optional penetration tolerance for robot-object contacts.

    If None, uses penetration_tolerance.
    """

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    contact_visualization: bool = False
    """Whether to color robot links by object contact in the Viser playback."""

    contact_source: Literal["robot", "human"] = "human"
    """Source for contact visualization.
    robot: color links whose retargeted robot geometry is near the object geometry.
    human: color mapped robot links from selected human joints near the object surface.
    """

    contact_threshold: float = 0.05
    """Object-contact distance threshold in meters for contact visualization."""

    contact_human_joint_regex: str = DEFAULT_CONTACT_HUMAN_JOINT_REGEX
    """Human joint regex used to select contact targets and map them to robot links."""

    debug: bool = False
    """Whether to enable debug mode."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""
