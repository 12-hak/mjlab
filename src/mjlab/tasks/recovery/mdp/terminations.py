"""Recovery-specific termination functions."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def standing_success_termination(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.65,
    upright_threshold: float = 0.9,
    stability_threshold: float = 0.1,
    min_duration: int = 100,  # ~2 seconds at 50Hz
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
) -> torch.Tensor:
    """Terminate episode when robot successfully stands and maintains stability.

    Args:
        env: The environment instance.
        height_threshold: Minimum height to consider "standing".
        upright_threshold: Minimum uprightness to consider "standing".
        stability_threshold: Maximum velocity magnitude to consider "stable".
        min_duration: Minimum steps to maintain standing before termination.
        asset_cfg: Scene entity configuration for the torso.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    asset = env.scene[asset_cfg.name]
    torso_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], :]
    torso_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]

    current_height = torso_pos_w[:, 2]

    # Check uprightness
    z_axis = torch.zeros(env.num_envs, 3, device=env.device)
    z_axis[:, 2] = 1.0

    # Quaternion rotation
    w, x, y, z = (
        torso_quat_w[:, 0],
        torso_quat_w[:, 1],
        torso_quat_w[:, 2],
        torso_quat_w[:, 3],
    )
    vx, vy, vz = z_axis[:, 0], z_axis[:, 1], z_axis[:, 2]

    cx1 = y * vz - z * vy
    cy1 = z * vx - x * vz
    cz1 = x * vy - y * vx
    cx1 += w * vx
    cy1 += w * vy
    cz1 += w * vz
    cx2 = y * cz1 - z * cy1
    cy2 = z * cx1 - x * cz1
    cz2 = x * cy1 - y * cx1

    torso_z_component = vz + 2.0 * cz2
    uprightness = torso_z_component

    # Check stability
    lin_vel = asset.data.root_link_lin_vel_b
    ang_vel = asset.data.root_link_ang_vel_b
    vel_magnitude = torch.sqrt(
        torch.sum(lin_vel**2, dim=-1) + torch.sum(ang_vel**2, dim=-1)
    )

    # All conditions must be met
    is_standing = (
        (current_height > height_threshold)
        & (uprightness > upright_threshold)
        & (vel_magnitude < stability_threshold)
    )

    # For now, don't terminate early - let the episode run to learn stability
    # In future, could track consecutive standing steps and terminate after min_duration
    return torch.zeros_like(is_standing, dtype=torch.bool)




