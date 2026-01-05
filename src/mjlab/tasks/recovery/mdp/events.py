"""Recovery-specific event functions for random fall initialization."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_to_random_fall_position(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot to random fallen positions (on back, front, sides).

    This event randomizes:
    - Base position (slightly above ground to prevent penetration)
    - Base orientation (random roll/pitch to simulate falls)
    - Joint positions (random within limits)
    - All velocities set to zero

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        asset_cfg: Scene entity configuration for the robot.
    """
    # Get the robot asset
    asset = env.scene[asset_cfg.name]

    # Sample random fall orientations
    # Roll: -180 to 180 degrees (can be on either side or upside down)
    # Pitch: -90 to 90 degrees (forward/backward tilt)
    # Yaw: -180 to 180 degrees (any rotation)
    roll = torch.rand(len(env_ids), device=env.device) * 2 * torch.pi - torch.pi
    pitch = torch.rand(len(env_ids), device=env.device) * torch.pi - torch.pi / 2
    yaw = torch.rand(len(env_ids), device=env.device) * 2 * torch.pi - torch.pi

    # Convert to quaternion
    quat = quat_from_euler_xyz(roll, pitch, yaw)

    # Set base position slightly above ground (0.3-0.5m to account for various orientations)
    base_pos = asset.data.default_root_state[env_ids, :3].clone()
    base_pos[:, 2] = torch.rand(len(env_ids), device=env.device) * 0.2 + 0.3

    # Randomize joint positions within a reasonable range
    # Get joint limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos_lower = joint_pos_limits[..., 0]
    joint_pos_upper = joint_pos_limits[..., 1]

    # Sample random joint positions (50% range around default to avoid extreme poses)
    default_joint_pos = asset.data.default_joint_pos[env_ids]
    joint_range = (joint_pos_upper - joint_pos_lower) * 0.5
    random_offset = (torch.rand_like(default_joint_pos) - 0.5) * joint_range
    joint_pos = default_joint_pos + random_offset
    joint_pos = torch.clamp(joint_pos, joint_pos_lower, joint_pos_upper)

    # Set the root state
    asset.write_root_link_pose_to_sim(
        torch.cat([base_pos, quat], dim=-1), env_ids=env_ids
    )

    # Set joint positions
    asset.write_joint_state_to_sim(
        joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
    )

    # Zero out all velocities
    asset.write_root_link_velocity_to_sim(
        torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids
    )


def reset_to_specific_fall_type(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    fall_type: str = "back",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot to specific fall position for curriculum learning.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        fall_type: Type of fall - "back", "front", "left_side", "right_side", "sitting"
        asset_cfg: Scene entity configuration for the robot.
    """
    asset = env.scene[asset_cfg.name]

    # Define fall orientations (roll, pitch, yaw in radians)
    fall_orientations = {
        "back": (torch.pi, 0.0, 0.0),  # On back
        "front": (0.0, torch.pi, 0.0),  # On front
        "left_side": (torch.pi / 2, 0.0, 0.0),  # On left side
        "right_side": (-torch.pi / 2, 0.0, 0.0),  # On right side
        "sitting": (0.0, torch.pi / 4, 0.0),  # Sitting position
    }

    if fall_type not in fall_orientations:
        fall_type = "back"

    roll, pitch, yaw = fall_orientations[fall_type]

    # Add small random variation
    roll += (torch.rand(len(env_ids), device=env.device) - 0.5) * 0.2
    pitch += (torch.rand(len(env_ids), device=env.device) - 0.5) * 0.2
    yaw += (torch.rand(len(env_ids), device=env.device) - 0.5) * 0.4

    quat = quat_from_euler_xyz(roll, pitch, yaw)

    # Set base position
    base_pos = asset.data.default_root_state[env_ids, :3].clone()
    base_pos[:, 2] = 0.4  # Fixed height for specific fall types

    # Randomize joint positions
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos_lower = joint_pos_limits[..., 0]
    joint_pos_upper = joint_pos_limits[..., 1]
    default_joint_pos = asset.data.default_joint_pos[env_ids]
    joint_range = (joint_pos_upper - joint_pos_lower) * 0.3
    random_offset = (torch.rand_like(default_joint_pos) - 0.5) * joint_range
    joint_pos = default_joint_pos + random_offset
    joint_pos = torch.clamp(joint_pos, joint_pos_lower, joint_pos_upper)

    # Set states
    asset.write_root_link_pose_to_sim(
        torch.cat([base_pos, quat], dim=-1), env_ids=env_ids
    )
    asset.write_joint_state_to_sim(
        joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(
        torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids
    )
