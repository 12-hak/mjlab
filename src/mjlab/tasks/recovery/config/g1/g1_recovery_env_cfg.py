"""Unitree G1 recovery environment configuration."""

from mjlab.asset_zoo.robots import (
    G1_ACTION_SCALE,
    HOME_KEYFRAME,
    get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.recovery import mdp
from mjlab.tasks.recovery.recovery_env_cfg import make_recovery_env_cfg


def unitree_g1_recovery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 fall recovery configuration.

    Features:
    - Random fall positions (on back, front, sides)
    - Rewards for getting upright quickly and stably
    - Full 29-DOF control
    - Flat terrain to start
    - 10-second episodes
    """
    cfg = make_recovery_env_cfg()

    # Get the robot config (Full 29-DOF)
    robot_cfg = get_g1_robot_cfg()
    # Start from HOME state (will be overridden by fall reset)
    robot_cfg.init_state = HOME_KEYFRAME

    cfg.scene.entities = {"robot": robot_cfg}

    # Configure contact sensors
    feet_contact_cfg = ContactSensorCfg(
        name="feet_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r".*ankle_roll_link",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Self collision sensor
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg.scene.sensors = (
        feet_contact_cfg,
        self_collision_cfg,
    )

    # Configure action scale
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    cfg.viewer.body_name = "torso_link"

    # Configure foot friction randomization
    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

    # Update reward parameters for G1
    cfg.rewards["torso_height"].params["target_height"] = 0.75  # G1 standing height
    cfg.rewards["feet_contact"].params["sensor_name"] = feet_contact_cfg.name

    # Add self-collision penalty
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Add joint velocity penalty for smoother motion
    cfg.rewards["joint_vel_penalty"] = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=r".*"),
        },
    )

    # Apply play mode overrides
    if play:
        # Effectively infinite episode length
        cfg.episode_length_s = int(1e9)
        cfg.observations["policy"].enable_corruption = False

    return cfg
