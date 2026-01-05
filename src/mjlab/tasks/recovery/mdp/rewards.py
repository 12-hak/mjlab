"""Recovery-specific reward functions."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalize self-collisions and contacts.

    Returns the total count of detections across all slots of the specified contact sensor.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    assert sensor.data.found is not None
    # Sum over all slots to return a [B] tensor
    return torch.sum(sensor.data.found, dim=-1)


def torso_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
) -> torch.Tensor:
    """Reward for getting torso to target height (standing).

    Args:
        env: The environment instance.
        target_height: Target torso height in meters.
        asset_cfg: Scene entity configuration for the torso.

    Returns:
        Reward tensor with exponential reward based on height proximity.
    """
    asset = env.scene[asset_cfg.name]
    torso_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], :]
    current_height = torso_pos_w[:, 2]

    # Exponential reward that peaks at target height
    height_error = torch.abs(current_height - target_height)
    reward = torch.exp(-height_error / 0.5)  # Broader peak to attract from ground

    return reward


def torso_upright_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
) -> torch.Tensor:
    """Reward for keeping torso upright (z-axis pointing up).

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the torso.

    Returns:
        Reward tensor based on torso uprightness.
    """
    asset = env.scene[asset_cfg.name]
    torso_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]

    # Get the z-axis of the torso in world frame
    # Quaternion rotation: z_world = quat_rotate(quat, [0, 0, 1])
    z_axis = torch.zeros(env.num_envs, 3, device=env.device)
    z_axis[:, 2] = 1.0

    # Rotate z-axis by torso quaternion
    torso_z = quat_rotate_vector(torso_quat_w, z_axis)

    # Reward is dot product with world up vector (how aligned is torso z with world z)
    upright_reward = torso_z[:, 2]  # This is the z-component, ranges from -1 to 1

    # Map from [-1, 1] to [0, 1] with exponential emphasis on being upright
    upright_reward = torch.exp(
        2.0 * (upright_reward - 1.0)
    )  # Peaks at 1 when perfectly upright

    return upright_reward


def recovery_time_bonus(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.65,
    upright_threshold: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
) -> torch.Tensor:
    """Bonus reward for achieving recovery quickly.

    Gives a time-decaying bonus when robot reaches standing position.

    Args:
        env: The environment instance.
        height_threshold: Minimum height to consider "standing".
        upright_threshold: Minimum uprightness (0-1) to consider "standing".
        asset_cfg: Scene entity configuration for the torso.

    Returns:
        Bonus reward tensor.
    """
    asset = env.scene[asset_cfg.name]
    torso_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], :]
    torso_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]

    current_height = torso_pos_w[:, 2]

    # Check uprightness
    z_axis = torch.zeros(env.num_envs, 3, device=env.device)
    z_axis[:, 2] = 1.0
    torso_z = quat_rotate_vector(torso_quat_w, z_axis)
    uprightness = torso_z[:, 2]

    # Check if standing
    is_standing = (current_height > height_threshold) & (
        uprightness > upright_threshold
    )

    # Time-based bonus (higher reward for faster recovery)
    # Assuming 50 steps per second, 10 seconds max episode
    max_steps = 500
    time_factor = 1.0 - (env.episode_length_buf.float() / max_steps)
    time_factor = torch.clamp(time_factor, 0.0, 1.0)

    bonus = is_standing.float() * time_factor * 10.0  # Large bonus for quick recovery

    return bonus


def feet_contact_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_contact",
) -> torch.Tensor:
    """Reward for having feet in contact with ground when upright.

    Args:
        env: The environment instance.
        sensor_name: Name of the feet contact sensor.

    Returns:
        Reward tensor.
    """
    contact_sensor = env.scene.sensors[sensor_name]
    feet_contact = contact_sensor.data.found.any(dim=-1).float()

    return feet_contact


def base_stability_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for stable base (low linear and angular velocity).

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot.

    Returns:
        Reward tensor.
    """
    asset = env.scene[asset_cfg.name]

    # Get base velocities
    lin_vel = asset.data.root_link_lin_vel_b
    ang_vel = asset.data.root_link_ang_vel_b

    # Penalize high velocities
    lin_vel_penalty = torch.sum(lin_vel**2, dim=-1)
    ang_vel_penalty = torch.sum(ang_vel**2, dim=-1)

    # Exponential reward for low velocities
    stability = torch.exp(-0.5 * (lin_vel_penalty + ang_vel_penalty))

    return stability


def energy_efficiency_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for high energy consumption (torque * velocity).

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot.

    Returns:
        Penalty tensor (negative values).
    """
    asset = env.scene[asset_cfg.name]

    # Get applied torques and joint velocities
    torques = asset.data.applied_torque
    joint_vel = asset.data.joint_vel

    # Power = torque * velocity
    power = torch.abs(torques * joint_vel)
    total_power = torch.sum(power, dim=-1)

    # Normalize by number of joints and return as penalty
    penalty = total_power / asset.num_joints

    return -penalty


def contact_progression_reward(
    env: ManagerBasedRlEnv,
    hand_sensor_name: str = "hand_contact",
    feet_sensor_name: str = "feet_contact",
) -> torch.Tensor:
    """Reward for logical contact progression during recovery.

    Encourages using hands/arms to push up before getting feet planted.

    Args:
        env: The environment instance.
        hand_sensor_name: Name of the hand contact sensor.
        feet_sensor_name: Name of the feet contact sensor.

    Returns:
        Reward tensor.
    """
    # Check if sensors exist
    if hand_sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    hand_contact = env.scene.sensors[hand_sensor_name].data.found.any(dim=-1).float()
    feet_contact = env.scene.sensors[feet_sensor_name].data.found.any(dim=-1).float()

    # Reward hand contact early in episode, feet contact later
    episode_progress = env.episode_length_buf.float() / 500.0  # Normalize to [0, 1]

    # Early: reward hand contact, Late: reward feet contact
    early_phase = (1.0 - episode_progress) * hand_contact
    late_phase = episode_progress * feet_contact

    return early_phase + late_phase


def quat_rotate_vector(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion.

    Args:
        quat: Quaternion tensor (N, 4) in (w, x, y, z) format.
        vec: Vector tensor (N, 3).

    Returns:
        Rotated vector tensor (N, 3).
    """
    # Extract quaternion components
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Extract vector components
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]

    # Quaternion rotation formula
    # v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)

    # First cross product: cross(q.xyz, v)
    cx1 = y * vz - z * vy
    cy1 = z * vx - x * vz
    cz1 = x * vy - y * vx

    # Add q.w * v
    cx1 += w * vx
    cy1 += w * vy
    cz1 += w * vz

    # Second cross product: cross(q.xyz, result)
    cx2 = y * cz1 - z * cy1
    cy2 = z * cx1 - x * cz1
    cz2 = x * cy1 - y * cx1

    # Final result: v + 2 * cross_result
    result = torch.stack(
        [
            vx + 2.0 * cx2,
            vy + 2.0 * cy2,
            vz + 2.0 * cz2,
        ],
        dim=-1,
    )

    return result


def standing_success_reward(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.65,
    upright_threshold: float = 0.9,
    stability_threshold: float = 0.5,
    min_duration: int = 50,  # Unused but kept for API compatibility
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
) -> torch.Tensor:
    """Shaped reward for standing success (height * upright * stability).

    This replaces the binary all-or-nothing reward with a dense shaped reward
    that guides the agent towards the success state.
    """
    asset = env.scene[asset_cfg.name]
    torso_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], :]
    torso_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]

    current_height = torso_pos_w[:, 2]

    # Check uprightness
    z_axis = torch.zeros(env.num_envs, 3, device=env.device)
    z_axis[:, 2] = 1.0
    torso_z = quat_rotate_vector(torso_quat_w, z_axis)
    uprightness = torso_z[:, 2]

    # Check stability
    lin_vel = asset.data.root_link_lin_vel_b
    ang_vel = asset.data.root_link_ang_vel_b
    vel_magnitude = torch.sqrt(
        torch.sum(lin_vel**2, dim=-1) + torch.sum(ang_vel**2, dim=-1)
    )

    # Shaped components
    # Map height [0.1, height_threshold] -> [0, 1]
    height_score = torch.clamp(
        (current_height - 0.1) / (height_threshold - 0.1), 0.0, 1.0
    )

    # Map uprightness [0.0, upright_threshold] -> [0, 1]
    # Use loosely defined range [0, 1] for uprightness score
    upright_score = torch.clamp(uprightness, 0.0, 1.0)

    # Map stability [2*threshold, 0] -> [0, 1]
    stability_score = torch.clamp(
        1.0 - (vel_magnitude / (stability_threshold * 2.0)), 0.0, 1.0
    )

    # Multiply to encourage satisfying ALL conditions
    # Result is in [0, 1] range. Config weight (25.0) scales this up.
    reward = height_score * upright_score * stability_score

    return reward
