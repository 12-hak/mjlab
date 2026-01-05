"""Recovery task configuration.

This module provides a factory function to create a base recovery task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.recovery import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_recovery_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base recovery task configuration."""

    ##
    # Observations
    ##

    policy_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    critic_terms = policy_terms

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        )
    }

    ##
    # Events
    ##

    events = {
        "reset_to_fall": EventTermCfg(
            func=mdp.reset_to_random_fall_position,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "operation": "abs",
                "field": "geom_friction",
                "ranges": (0.5, 1.2),
            },
        ),
    }

    ##
    # Rewards
    ##

    rewards = {
        # Primary recovery objectives
        "torso_height": RewardTermCfg(
            func=mdp.torso_height_reward,
            weight=10.0,
            params={
                "target_height": 0.75,
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            },
        ),
        "torso_upright": RewardTermCfg(
            func=mdp.torso_upright_reward,
            weight=10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            },
        ),
        "standing_success": RewardTermCfg(
            func=mdp.standing_success_reward,
            weight=25.0,
            params={
                "height_threshold": 0.65,
                "upright_threshold": 0.9,
                "stability_threshold": 0.5,
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            },
        ),
        # Stability and efficiency
        "base_stability": RewardTermCfg(
            func=mdp.base_stability_reward,
            weight=5.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Contact rewards
        "feet_contact": RewardTermCfg(
            func=mdp.feet_contact_reward,
            weight=2.0,
            params={
                "sensor_name": "feet_contact",
            },
        ),
        # Regularization
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "dof_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-5),
    }

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        # Don't terminate on bad orientation - that's what we're recovering from!
    }

    ##
    # Assemble and return
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(
                terrain_type="plane",  # Start with flat terrain
                terrain_generator=None,
            ),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=None,  # No velocity commands for recovery
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=None,  # Can add curriculum later
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            asset_name="robot",
            body_name="",  # Set per-robot.
            distance=4.0,
            elevation=-20.0,
            azimuth=45.0,
        ),
        sim=SimulationCfg(
            nconmax=512,
            njmax=1600,
            contact_sensor_maxmatch=512,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=80,
                ls_iterations=80,
                ccd_iterations=60,
            ),
        ),
        decimation=4,
        episode_length_s=10.0,  # 10 seconds to recover
    )
