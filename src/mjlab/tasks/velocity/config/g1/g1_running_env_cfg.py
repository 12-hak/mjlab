from mjlab.asset_zoo.robots import (
    G1_ACTION_SCALE,
    HOME_KEYFRAME,
    get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_importer import TerrainImporterCfg


def unitree_g1_running_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 high-speed running configuration.

    Features:
    - Velocity commands up to 3.0 m/s
    - Random uneven ground terrain
    - Arm swing rewards for natural running
    - Toe-based running gait encouragement
    - No state estimation (uses proprioceptive observations only)
    - Full 29-DOF control with heavy wrist suppression
    """
    cfg = make_velocity_env_cfg()

    # Get the robot config (Full 29-DOF)
    robot_cfg = get_g1_robot_cfg()
    # Use the taller Home state instead of the crouching one
    robot_cfg.init_state = HOME_KEYFRAME

    cfg.scene.entities = {"robot": robot_cfg}
    cfg.sim.nconmax = 512  # Reduced (was 1024) to save VRAM
    cfg.sim.njmax = 1600  # Reduced (was 4096) to fit in 12GB VRAM
    cfg.sim.contact_sensor_maxmatch = 512  # Sufficient for the 468 reported
    cfg.sim.mujoco.iterations = 80  # Moderate (was 100)
    cfg.sim.mujoco.ls_iterations = 80
    cfg.sim.mujoco.ccd_iterations = 60  # Moderate (was 100)

    # Configure rough terrain with random uneven ground
    cfg.scene.terrain = TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            curriculum=True,
            sub_terrains={
                "random_rough": HfRandomUniformTerrainCfg(
                    proportion=1.0,  # 100% random uneven ground
                    noise_range=(0.0, 0.08),  # 0-8cm height variation
                    noise_step=0.005,
                    horizontal_scale=0.1,
                    downsampled_scale=0.5,  # Smoother ripples
                    border_width=1.0,
                ),
            },
        ),
    )

    site_names = ("left_foot", "right_foot")
    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
    )

    # Foot contact sensors
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
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

    # Toe contact sensors - ONLY the unique front toe geoms (foot1 and foot7)
    toe_contact_cfg = ContactSensorCfg(
        name="toe_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r".*foot[17]_collision",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
    )

    # Heel contact sensors - The long geoms that reach the back (foot2 to foot6)
    heel_contact_cfg = ContactSensorCfg(
        name="heel_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r".*foot[2-6]_collision",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
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
        feet_ground_cfg,
        toe_contact_cfg,
        heel_contact_cfg,
        self_collision_cfg,
    )

    # Configure action scale
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    cfg.viewer.body_name = "torso_link"

    # Configure velocity commands for high-speed running
    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15
    # Increase max velocity to 3.0 m/s for sprinting
    twist_cmd.ranges.lin_vel_x = (-0.5, 3.0)
    twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
    twist_cmd.ranges.ang_vel_z = (-2.5, 2.5)  # Increased for much sharper turns

    # Configure observations (Default inherited from velocity_env_cfg)

    # Configure critic observations
    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    # Configure events
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

    # Configure posture rewards with running-specific tolerances

    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        # Waist
        r".*waist_yaw.*": 0.2,
        r".*waist_roll.*": 0.08,
        r".*waist_pitch.*": 0.1,
        # Arms - allow more movement for natural swing
        r".*shoulder_pitch.*": 0.3,
        r".*shoulder_roll.*": 0.15,
        r".*shoulder_yaw.*": 0.1,  # Added to prevent flailing
        r".*elbow.*": 0.25,
        r".*wrist.*": 0.3,
    }

    cfg.rewards["pose"].params["std_running"] = {
        # Lower body - significantly relaxed for high-speed sprinting
        r".*hip_pitch.*": 0.8,
        r".*hip_roll.*": 0.15,  # Relaxed to allow whole-leg sweep to center
        r".*hip_yaw.*": 0.05,
        r".*knee.*": 0.4,
        r".*ankle_pitch.*": 0.5,
        r".*ankle_roll.*": 0.1,  # Relaxed to keep feet flat while legs sweep
        # Waist - relaxed for agility, yaw especially for turning
        r".*waist_yaw.*": 0.05,  # Total lockdown on torso twist
        r".*waist_roll.*": 0.05,
        r".*waist_pitch.*": 0.1,
        # Arms - relaxed for fluid athletics swings
        r".*shoulder_pitch.*": 1.2,
        r".*shoulder_roll.*": 0.05,
        r".*shoulder_yaw.*": 0.05,
        r".*elbow.*": 0.4,
        r".*wrist.*": 0.1,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # High-speed forward lean for balance
    # Set to 0.52 rad (~30 degrees) as requested
    cfg.rewards["upright"] = RewardTermCfg(
        func=mdp.speed_dependent_lean,
        weight=40.0,
        params={
            "max_lean": 0.52,  # ~30 degrees
            "speed_threshold": 0.1,  # Gentle start
            "max_speed": 3.0,
            "std": 0.25,  # Smoother, more gradual ramp up
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "torso_link"]),
        },
    )
    cfg.rewards["body_ang_vel"].weight = -0.1
    cfg.rewards["angular_momentum"].weight = -0.01
    cfg.rewards["air_time"].weight = 2.0  # Increased for stride length

    # Penalize knees from touching (Knee distance fix)
    cfg.rewards["knee_distance"] = RewardTermCfg(
        func=mdp.knee_distance_penalty,
        weight=-20.0,  # Doubled (was -10.0)
        params={
            "min_distance": 0.18,  # Tightened (was 0.15)
            "asset_cfg": SceneEntityCfg("robot", body_names=r".*knee_link"),
        },
    )

    # Aggressive Velocity Tracking to fix high-speed drift/lag
    cfg.rewards["track_linear_velocity"].weight = 20.0  # Increased
    cfg.rewards["track_linear_velocity"].params["std"] = 0.5
    cfg.rewards["track_angular_velocity"].weight = (
        30.0  # Slightly reduced (was 40.0) for stability
    )
    cfg.rewards["track_angular_velocity"].params[
        "std"
    ] = 0.5  # Further softened (was 0.3)

    # Tracking Penalty (Linear instead of Exponential)
    # This keeps things moving forward even when the error is > 1.0 m/s
    cfg.rewards["track_lin_vel_l2"] = RewardTermCfg(
        func=mdp.track_lin_vel_l2_penalty,
        weight=-5.0,
        params={"command_name": "twist"},
    )

    cfg.rewards["lin_vel_y_l2"] = RewardTermCfg(
        func=mdp.lin_vel_y_l2,
        weight=-5.0,  # Relaxed (was -10.0) to allow natural lateral shift in banking
    )
    cfg.rewards["lin_vel_z_l2"] = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0)

    cfg.rewards["feet_roll_penalty"] = RewardTermCfg(
        func=mdp.feet_roll_penalty,
        weight=-50.0,  # Tightened back up (was -30.0) to force straight feet
        params={
            "sensor_name": feet_ground_cfg.name,
            "asset_cfg": SceneEntityCfg("robot", body_names=r".*ankle_roll_link"),
        },
    )

    cfg.rewards["feet_yaw_penalty"] = RewardTermCfg(
        func=mdp.feet_yaw_penalty,
        weight=-20.0,  # Penalty for pigeon-toed drift
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=r".*ankle_roll_link"),
        },
    )

    # NEW: Athletic Gait Mastery (Banking and Striding)
    cfg.rewards["waist_stability"] = RewardTermCfg(
        func=mdp.waist_stability,
        weight=-5.0,  # Reduced (was -10.0) for stability
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_roll_joint", "waist_pitch_joint"]
            ),
        },
    )
    cfg.rewards["banking"] = RewardTermCfg(
        func=mdp.banking_reward,
        weight=5.0,  # Reduced (was 10.0) so it doesn't overpower tracking
        params={
            "command_name": "twist",
            "lean_scale": -0.15,  # Lean into turns
        },
    )
    cfg.rewards["stride_length"] = RewardTermCfg(
        func=mdp.stride_length_reward,
        weight=2.0,  # Softened (was 3.0) to prevent explosive jumping
        params={
            "sensor_name": feet_ground_cfg.name,
            "command_name": "twist",
            "speed_threshold": 0.5,
        },
    )

    # Foot slip - critical for smooth running, no shuffling
    cfg.rewards["foot_slip"].weight = -1.0  # Relaxed slightly to allow push-off

    # Soft landing - penalize stomping/hard strikes
    cfg.rewards["soft_landing"].weight = -1e-4

    # Encourage leg lift/wing height for "lift"
    cfg.rewards["foot_swing_height"].weight = 2.0

    # Stay Tall Rewards - Squash the "squat"
    cfg.rewards["base_height"] = RewardTermCfg(
        func=mdp.base_height_l2,
        weight=-2.0,  # Penalty for sinking too low
        params={"target_height": 0.74},
    )
    cfg.rewards["knee_straightness"] = RewardTermCfg(
        func=mdp.knee_straightness_l2,
        weight=-1.0,  # Penalty for excessively bent knees
        params={"target_angle": 0.1},
    )

    # Toe-based running gait - Corrected sensors and stabilized weight
    cfg.rewards["toe_run"] = RewardTermCfg(
        func=mdp.toe_based_run,
        weight=4.0,  # Increased to enforce the transition
        params={
            "toe_sensor_name": toe_contact_cfg.name,
            "heel_sensor_name": heel_contact_cfg.name,
            "command_name": "twist",
            "speed_threshold": 0.4,  # Encouraging toe-strike even during walk
        },
    )

    # General heel strike penalty - strict
    cfg.rewards["heel_strike_penalty"] = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1.0,  # Increased
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=r".*ankle_pitch.*",
            ),
        },
    )
    # Anti-heel contact - strict
    cfg.rewards["anti_heel"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,  # Increased
        params={"sensor_name": heel_contact_cfg.name},
    )

    # Action smoothness
    cfg.rewards["action_rate_l2"].weight = -0.5

    # Add action acceleration penalty for even smoother motion
    cfg.rewards["action_acc_l2"] = RewardTermCfg(
        func=mdp.action_acc_l2,
        weight=-0.02,
    )

    # Leg-specific acceleration penalty - reduced to prevent paralysis
    cfg.rewards["smooth_legs"] = RewardTermCfg(
        func=mdp.smooth_leg_motion,
        weight=-2e-6,  # 50x reduction
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=r".*(hip|knee|ankle).*",
            ),
        },
    )

    # Add self collision penalty
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Allowed Arm Swing (Forward/Back only)
    cfg.rewards["arm_pitch_swing"] = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-5e-5,  # Even lower penalty to reward any forward motion
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=r".*(shoulder_pitch|elbow).*",
            ),
        },
    )

    # Bilateral Arm Symmetry (Softened to avoid over-constraint)
    cfg.rewards["arm_joint_symmetry_roll_yaw"] = RewardTermCfg(
        func=mdp.bilateral_joint_l2,
        weight=-0.5,  # Increased (was -0.2)
        params={
            "left_joint_names": r"left_shoulder_(roll|yaw).*",
            "right_joint_names": r"right_shoulder_(roll|yaw).*",
            "multiplier": -1.0,
        },
    )
    cfg.rewards["arm_joint_symmetry_pitch"] = RewardTermCfg(
        func=mdp.bilateral_joint_l2,
        weight=-0.5,  # Increased (was -0.2)
        params={
            "left_joint_names": r"(left_shoulder_pitch|left_elbow).*",
            "right_joint_names": r"(right_shoulder_pitch|right_elbow).*",
            "multiplier": -1.0,
        },
    )

    # Forbidden Arm Flailing - RELAXED to allow "gentle side swing"
    cfg.rewards["arm_lateral_stability"] = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=r".*(shoulder_roll|shoulder_yaw).*",
            ),
        },
    )

    # Wrist following - RELAXED to follow arm motion
    cfg.rewards["wrist_stability"] = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.01,  # Almost free
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=r".*wrist.*",
            ),
        },
    )

    # Wrist pose penalty - soft pull back to neutral
    cfg.rewards["wrist_pose_l2"] = RewardTermCfg(
        func=mdp.joint_pos_l2,
        weight=-0.05,  # Soft pull
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=r".*wrist.*",
            ),
        },
    )

    # Pop the generic arm control to avoid double counting
    cfg.rewards.pop("arm_control", None)

    # Sprinting Curriculum - push speed aggressively now that stability is high
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-1.0, 1.5)},
                {"step": 500, "lin_vel_x": (-1.5, 2.5)},
                {"step": 1500, "lin_vel_x": (-2.0, 3.0)},
            ],
        },
    )

    # Apply play mode overrides
    if play:
        # Effectively infinite episode length
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # Simplify terrain for play mode
        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg
