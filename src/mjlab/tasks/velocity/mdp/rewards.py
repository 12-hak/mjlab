from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
    resolve_matching_names,
    resolve_matching_names_values,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity.

    The commanded z velocity is assumed to be zero.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + z_error
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_l2_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty for tracking error of base linear velocity (L2).

    This provides a consistent gradient for the robot to speed up even when
    the tracking error is large, avoiding the saturation of exponential rewards.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_lin_vel_b
    # (v_cmd - v_actual)^2
    return torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)


def track_lin_vel_y_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded lateral velocity specifically."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_lin_vel_b
    error = torch.square(command[:, 1] - actual[:, 1])
    return torch.exp(-error / std**2)


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize vertical base velocity (jumping/bouncing)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def lin_vel_y_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize lateral base velocity (drifting)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 1])


def lin_vel_x_highspeed(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float = 2.0,
    std: float = 0.2,
) -> torch.Tensor:
    """Track forward velocity error specifically above a speed threshold."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_lin_vel_b
    mask = (command[:, 0] > threshold).float()
    error = torch.square(command[:, 0] - actual[:, 0])
    return torch.exp(-error / std**2) * mask


def lin_vel_y_highspeed(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float = 2.0,
) -> torch.Tensor:
    """Track absolute lateral velocity (drift) above a speed threshold."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_lin_vel_b
    mask = (command[:, 0] > threshold).float()
    return torch.abs(actual[:, 1]) * mask


def ang_vel_z_highspeed(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float = 2.0,
) -> torch.Tensor:
    """Track absolute yaw rate (instability) above a speed threshold."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_ang_vel_b
    mask = (command[:, 0] > threshold).float()
    return torch.abs(actual[:, 2]) * mask


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward heading error for heading-controlled envs, angular velocity for others.

    The commanded xy angular velocities are assumed to be zero.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + xy_error
    return torch.exp(-ang_vel_error / std**2)


def base_height_l2(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from a target base height."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)


def knee_straightness_l2(
    env: ManagerBasedRlEnv,
    target_angle: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize bent knees to encourage straighter legs."""
    asset: Entity = env.scene[asset_cfg.name]
    # Assuming knee joints are named with 'knee'
    joint_indices, _ = asset.find_joints(".*knee.*")
    knee_pos = asset.data.joint_pos[:, joint_indices]
    return torch.sum(torch.square(knee_pos - target_angle), dim=1)


def flat_orientation(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward flat base orientation (robot being upright).

    If asset_cfg has body_ids specified, computes the projected gravity
    for that specific body. Otherwise, uses the root link projected gravity.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # If body_ids are specified, compute projected gravity for that body.
    if asset_cfg.body_ids:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
        body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
        gravity_w = asset.data.gravity_vec_w  # [3]
        projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    else:
        # Use root link projected gravity.
        xy_squared = torch.sum(
            torch.square(asset.data.projected_gravity_b[:, :2]), dim=1
        )
    return torch.exp(-xy_squared / std**2)


def speed_dependent_lean(
    env: ManagerBasedRlEnv,
    max_lean: float,
    speed_threshold: float,
    max_speed: float,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for leaning forward at high speeds."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("twist")
    speed = command[:, 0]

    # Calculate target lean based on speed
    # 0 lean at speed_threshold, max_lean at max_speed
    target_lean = (
        torch.clamp(
            (speed - speed_threshold) / (max_speed - speed_threshold),
            0.0,
            1.0,
        )
        * max_lean
    )

    # Target projected gravity: [sin(lean), 0, -cos(lean)]
    target_proj_gravity = torch.stack(
        [
            torch.sin(target_lean),
            torch.zeros_like(target_lean),
            -torch.cos(target_lean),
        ],
        dim=-1,
    )

    # Get orientations of the target bodies
    # If no bodies specified, default to root
    if asset_cfg.body_ids is not None:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
        gravity_w = asset.data.gravity_vec_w
        if gravity_w.dim() == 2:
            gravity_w = gravity_w.unsqueeze(1).repeat(1, body_quat_w.shape[1], 1)
        else:
            gravity_w = gravity_w.expand(body_quat_w.shape[0], body_quat_w.shape[1], 3)
        actual_proj_gravity = quat_apply_inverse(body_quat_w, gravity_w)

        # Error for each body
        target_proj_gravity = target_proj_gravity.unsqueeze(1)
        # Error only in X (forward lean) and Z (vertical alignment)
        # We ignore Y (lateral/roll) to allow the robot to bank into turns.
        actual_proj_gravity = actual_proj_gravity[:, :, [0, 2]]
        target_proj_gravity = target_proj_gravity[:, :, [0, 2]]
        error = torch.sum(
            torch.square(actual_proj_gravity - target_proj_gravity), dim=-1
        )
        return torch.exp(-torch.mean(error, dim=1) / std**2)
    else:
        actual_proj_gravity = asset.data.projected_gravity_b
        # Error only in X and Z
        actual_proj_gravity = actual_proj_gravity[:, [0, 2]]
        target_proj_gravity = target_proj_gravity[:, [0, 2]]
        error = torch.sum(
            torch.square(actual_proj_gravity - target_proj_gravity), dim=-1
        )
        return torch.exp(-error / std**2)


def knee_distance_penalty(
    env: ManagerBasedRlEnv,
    min_distance: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize knees being too close to each other.

    Args:
        min_distance: The minimum allowed distance between knees.
        asset_cfg: The asset configuration with body_names for the knees.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Resolve body IDs if not already provided
    if asset_cfg.body_ids is None:
        body_ids, _ = resolve_matching_names(asset_cfg.body_names, asset.body_names)
    else:
        body_ids = asset_cfg.body_ids

    if len(body_ids) != 2:
        return torch.zeros(asset.data.root_link_pos_w.shape[0], device=asset.device)

    # Get relative position between knees in world frame
    knee_poses = asset.data.body_link_pose_w[:, body_ids, :3]
    diff_w = knee_poses[:, 0, :] - knee_poses[:, 1, :]

    # Transform to base frame to isolate lateral (Y) distance
    diff_b = quat_apply_inverse(asset.data.root_link_quat_w, diff_w)
    lateral_distance = torch.abs(diff_b[:, 1])

    # Penalize if lateral distance is below min_distance
    return torch.square(torch.clamp(min_distance - lateral_distance, min=0.0))


def bilateral_joint_l2(
    env: ManagerBasedRlEnv,
    left_joint_names: str | list[str],
    right_joint_names: str | list[str],
    multiplier: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize asymmetry between left and right joints.

    Args:
        multiplier: Use 1.0 for symmetric (Roll/Yaw) and -1.0 for antisymmetric (Pitch).
    """
    asset: Entity = env.scene[asset_cfg.name]
    left_ids, _ = resolve_matching_names(left_joint_names, asset.joint_names)
    right_ids, _ = resolve_matching_names(right_joint_names, asset.joint_names)

    if len(left_ids) != len(right_ids):
        return torch.zeros(asset.data.root_link_pos_w.shape[0], device=asset.device)

    left_pos = asset.data.joint_pos[:, left_ids]
    right_pos = asset.data.joint_pos[:, right_ids]

    # Penalize (left - multiplier * right)^2
    # For pitch joints in running, multiplier=-1 makes them opposites.
    return torch.sum(torch.square(left_pos - multiplier * right_pos), dim=-1)


def bilateral_body_height_l2(
    env: ManagerBasedRlEnv,
    left_body_names: str | list[str],
    right_body_names: str | list[str],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize difference in height (Z) between left and right body links."""
    asset: Entity = env.scene[asset_cfg.name]
    left_ids, _ = resolve_matching_names(left_body_names, asset.body_names)
    right_ids, _ = resolve_matching_names(right_body_names, asset.body_names)

    if len(left_ids) != len(right_ids):
        return torch.zeros(asset.data.root_link_pos_w.shape[0], device=asset.device)

    # Use world height of links
    left_z = asset.data.body_link_pose_w[:, left_ids, 2]
    right_z = asset.data.body_link_pose_w[:, right_ids, 2]

    return torch.sum(torch.square(left_z - right_z), dim=-1)


def feet_roll_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize feet rolling (sideways tilt) when in contact.

    Unlike feet_flat_orientation, this allows pitching (toes down) for running.
    """
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]

    # Get projected gravity for the specified foot bodies
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    gravity_w = asset.data.gravity_vec_w

    if gravity_w.dim() == 2:
        gravity_w = gravity_w.unsqueeze(1).repeat(1, body_quat_w.shape[1], 1)
    else:
        gravity_w = gravity_w.expand(body_quat_w.shape[0], body_quat_w.shape[1], 3)

    projected_gravity = quat_apply_inverse(body_quat_w, gravity_w)

    # Only penalize Y component (Roll in body frame for gravity projection)
    y_squared = torch.square(projected_gravity[:, :, 1])

    contact_mask = (sensor.data.found > 0).float().squeeze(-1)

    # Return penalty for roll when in contact
    return torch.mean(y_squared * contact_mask, dim=-1)


def feet_yaw_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize feet yaw misalignment with the base forward direction.

    This ensures the feet face mostly forward and don't drift 'pigeon-toed'.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Get foot quaternions in world frame [B, N, 4]
    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]

    # Get base orientation in world frame [B, 4]
    base_quat_w = asset.data.root_link_quat_w

    # Create foot's local forward vector in world frame
    # Foot local X is forward
    local_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand_as(
        asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    )
    foot_forward_w = quat_apply(foot_quat_w, local_x)

    # Transform that world-frame forward vector into the base frame
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, foot_quat_w.shape[1], -1)
    foot_forward_b = quat_apply_inverse(base_quat_w_expanded, foot_forward_w)

    # In base frame, foot_forward_b should be [1, 0, 0].
    # The Y component [..., 1] is the yaw/roll deviation.
    # Since feet_roll_penalty handles gravity tilt, this penalizes yaw drift.
    return torch.mean(torch.square(foot_forward_b[:, :, 1]), dim=-1)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalize self-collisions and contacts.

    Returns the total count of detections across all slots of the specified contact sensor.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    # Sum over all slots to return a [B] tensor
    return torch.sum(sensor.data.found, dim=-1)


def body_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive body angular velocities."""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
    ang_vel = ang_vel.squeeze(1)
    ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
    return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize whole-body angular momentum to encourage natural arm swing."""
    angmom_sensor: BuiltinSensor = env.scene[sensor_name]
    angmom = angmom_sensor.data
    angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
    angmom_magnitude = torch.sqrt(angmom_magnitude_sq + 1e-6)
    env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
    return angmom_magnitude_sq


def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_name: str | None = None,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward feet air time."""
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    in_air = current_air_time > 0
    num_in_air = torch.sum(in_air.float())
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
        num_in_air, min=1
    )
    env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    command_name: str | None = None,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from target clearance height, weighted by foot velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
    delta = torch.abs(foot_z - target_height)  # [B, N]
    cost = torch.sum(delta * vel_norm, dim=1)  # [B]
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


class feet_swing_height:
    """Penalize deviation from target swing height, evaluated at landing."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = cfg.params["sensor_name"]
        self.site_names = cfg.params["asset_cfg"].site_names
        self.peak_heights = torch.zeros(
            (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
        )
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        target_height: float,
        command_name: str,
        command_threshold: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene[sensor_name]
        command = env.command_manager.get_command(command_name)
        assert command is not None
        foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
        in_air = contact_sensor.data.found == 0
        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )
        first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > command_threshold).float()
        error = self.peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
        num_landings = torch.sum(first_contact.float())
        peak_heights_at_landing = self.peak_heights * first_contact.float()
        mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
            num_landings, min=1
        )
        env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )
        return cost


def feet_slip(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact)."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
    vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
    num_in_contact = torch.sum(in_contact)
    mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
        num_in_contact, min=1
    )
    env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
    return cost


def soft_landing(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.force is not None
    forces = sensor_data.force  # [B, N, 3]
    force_magnitude = torch.sqrt(
        torch.sum(torch.square(forces), dim=-1) + 1e-6
    )  # [B, N]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]
    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


class variable_posture:
    """Penalize deviation from default pose, with tighter constraints when standing."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, std_standing = resolve_matching_names_values(
            data=cfg.params["std_standing"],
            list_of_strings=joint_names,
        )
        self.std_standing = torch.tensor(
            std_standing, device=env.device, dtype=torch.float32
        )

        _, _, std_walking = resolve_matching_names_values(
            data=cfg.params["std_walking"],
            list_of_strings=joint_names,
        )
        self.std_walking = torch.tensor(
            std_walking, device=env.device, dtype=torch.float32
        )

        _, _, std_running = resolve_matching_names_values(
            data=cfg.params["std_running"],
            list_of_strings=joint_names,
        )
        self.std_running = torch.tensor(
            std_running, device=env.device, dtype=torch.float32
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running  # Unused.

        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = (
            (total_speed >= walking_threshold) & (total_speed < running_threshold)
        ).float()
        running_mask = (total_speed >= running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def toe_based_run(
    env: ManagerBasedRlEnv,
    toe_sensor_name: str,
    heel_sensor_name: str,
    command_name: str,
    speed_threshold: float = 1.5,
) -> torch.Tensor:
    """Reward for landing on toes specifically at high speeds."""
    toe_sensor: ContactSensor = env.scene[toe_sensor_name]
    heel_sensor: ContactSensor = env.scene[heel_sensor_name]

    # Reduce to [B] by checking if any matched geom is in contact
    toe_contact = (torch.sum(toe_sensor.data.found, dim=1) > 0).float()
    heel_contact = (torch.sum(heel_sensor.data.found, dim=1) > 0).float()

    # Ideal: any toe contact and NO heel contact
    toe_only = toe_contact * (1.0 - heel_contact)
    heel_penalty = heel_contact

    # Scale by command speed
    command = env.command_manager.get_command(command_name)
    speed = torch.norm(command[:, :2], dim=1)
    scale = (speed > speed_threshold).float()

    # Combine: Reward toe-only contact, penalize any heel contact at high speed
    reward = 5.0 * toe_only - 3.0 * heel_penalty
    return reward * scale


def waist_stability(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize waist roll and pitch to force legs to command the turn.

    We want the torso to stay mostly aligned with the pelvis.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # waist_roll_joint and waist_pitch_joint are usually indices 1 and 2 in the waist group
    # but we'll just penalize all joints in the provided asset_cfg (excluding yaw if possible)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos), dim=-1)


def banking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    lean_scale: float = -0.15,
) -> torch.Tensor:
    """Reward leaning into turns (banking).

    Encourages projected gravity Y to match a factor of the commanded yaw rate.
    Uses Pelvis orientation.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    yaw_rate = command[:, 2]

    # Get projected gravity for base (pelvis)
    quat_w = asset.data.root_link_quat_w
    gravity_w = asset.data.gravity_vec_w
    projected_gravity = quat_apply_inverse(quat_w, gravity_w)

    # Lateral lean (Roll) is projected_gravity[:, 1]
    target_lean = yaw_rate * lean_scale
    lean_error = torch.square(projected_gravity[:, 1] - target_lean)

    return torch.exp(-lean_error / 0.1)  # Relaxed slightly (was 0.05)


def stride_length_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    speed_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward longer strides by rewarding air time proportional to velocity.

    This discourages 'pattering' with small, fast steps at high speed.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    lin_vel_x = command[:, 0]

    # Only active when moving forward above threshold
    active = (lin_vel_x > speed_threshold).float()

    # Sum of current air times across both feet
    air_time = torch.sum(torch.clamp(sensor.data.current_air_time, max=1.0), dim=1)

    # Reward = air_time * velocity
    # This forces the robot to maximize time in air for the given speed
    return air_time * lin_vel_x * active


def smooth_leg_motion(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize high joint acceleration specifically in the legs."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_acc), dim=1)


def joint_pos_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint position deviation from default pose using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(current_joint_pos - desired_joint_pos), dim=1)


def joint_velocity_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint velocities (L2 squared)."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel), dim=1)


def standing_stability_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward keeping the base stable (low velocity) when commanded to stand."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm

    # Active only when commanded to stand (below threshold)
    is_standing = (total_command < command_threshold).float()

    # Calculate velocity magnitude squared
    lin_vel_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1)
    ang_vel_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)

    # Exponential reward: 1.0 when still, decays as velocity increases
    return torch.exp(-(lin_vel_sq + ang_vel_sq) / 0.04) * is_standing
