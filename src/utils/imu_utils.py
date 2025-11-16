"""IMU data processing and rotation matrix computation."""

import numpy as np
from typing import Tuple, Dict, Optional


class IMUProcessor:
    """Process IMU data and compute rotation matrices."""

    def __init__(
        self,
        roll_std: float = 0.01,
        pitch_std: float = 0.01,
        yaw_std: float = 0.05,
        altitude_std: float = 2.0
    ):
        """
        Initialize IMU processor.

        Args:
            roll_std: Roll angle standard deviation (radians)
            pitch_std: Pitch angle standard deviation (radians)
            yaw_std: Yaw angle standard deviation (radians)
            altitude_std: Altitude standard deviation (meters)
        """
        self.roll_std = roll_std
        self.pitch_std = pitch_std
        self.yaw_std = yaw_std
        self.altitude_std = altitude_std

    @staticmethod
    def euler_to_rotation_matrix(
        roll: float,
        pitch: float,
        yaw: float,
        order: str = 'ZYX'
    ) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix.

        Args:
            roll: Roll angle (φ) in radians
            pitch: Pitch angle (θ) in radians
            yaw: Yaw angle (ψ) in radians
            order: Rotation order, default 'ZYX' (yaw-pitch-roll)

        Returns:
            3x3 rotation matrix
        """
        # Individual rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combine according to order
        if order == 'ZYX':
            R = Rz @ Ry @ Rx
        elif order == 'XYZ':
            R = Rx @ Ry @ Rz
        elif order == 'ZXY':
            R = Rz @ Rx @ Ry
        else:
            raise ValueError(f"Unsupported rotation order: {order}")

        return R

    @staticmethod
    def rotation_matrix_to_euler(R: np.ndarray, order: str = 'ZYX') -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles.

        Args:
            R: 3x3 rotation matrix
            order: Rotation order, default 'ZYX'

        Returns:
            (roll, pitch, yaw) in radians
        """
        if order == 'ZYX':
            # Check for gimbal lock
            if np.abs(R[2, 0]) >= 1:
                yaw = 0
                if R[2, 0] < 0:
                    pitch = np.pi / 2
                    roll = np.arctan2(R[0, 1], R[0, 2])
                else:
                    pitch = -np.pi / 2
                    roll = np.arctan2(-R[0, 1], -R[0, 2])
            else:
                pitch = np.arcsin(-R[2, 0])
                roll = np.arctan2(R[2, 1], R[2, 2])
                yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            raise ValueError(f"Unsupported rotation order: {order}")

        return roll, pitch, yaw

    def compute_rotation_jacobian(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        delta: float = 1e-6
    ) -> np.ndarray:
        """
        Compute Jacobian of rotation matrix w.r.t. Euler angles.

        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            delta: Finite difference step size

        Returns:
            9x3 Jacobian matrix (vectorized R w.r.t. [roll, pitch, yaw])
        """
        R0 = self.euler_to_rotation_matrix(roll, pitch, yaw)

        # Compute finite differences
        dR_droll = (self.euler_to_rotation_matrix(roll + delta, pitch, yaw) - R0) / delta
        dR_dpitch = (self.euler_to_rotation_matrix(roll, pitch + delta, yaw) - R0) / delta
        dR_dyaw = (self.euler_to_rotation_matrix(roll, pitch, yaw + delta) - R0) / delta

        # Stack into Jacobian (vectorized)
        J = np.column_stack([
            dR_droll.flatten(),
            dR_dpitch.flatten(),
            dR_dyaw.flatten()
        ])

        return J

    def compute_homography_uncertainty(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        altitude: float,
        K: np.ndarray,
        motion_scale: float = 1.0
    ) -> np.ndarray:
        """
        Compute homography covariance matrix from IMU uncertainty.

        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            altitude: Flight altitude in meters
            K: 3x3 camera intrinsic matrix
            motion_scale: Scale factor for motion-induced noise (>1 for high dynamics)

        Returns:
            9x9 covariance matrix for vectorized homography matrix
        """
        # Scale uncertainties by motion
        sigma_roll = self.roll_std * motion_scale
        sigma_pitch = self.pitch_std * motion_scale
        sigma_yaw = self.yaw_std * motion_scale
        sigma_alt = self.altitude_std * motion_scale

        # Uncertainty covariance for [roll, pitch, yaw, altitude]
        Sigma_params = np.diag([sigma_roll**2, sigma_pitch**2, sigma_yaw**2, sigma_alt**2])

        # Compute Jacobian of homography w.r.t. parameters
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)
        n = np.array([0, 0, 1]).reshape(3, 1)
        t = np.array([0, 0, -altitude]).reshape(3, 1)
        K_inv = np.linalg.inv(K)

        # Compute H and its Jacobian numerically
        delta = 1e-6

        def compute_H(params):
            r, p, y, h = params
            R_temp = self.euler_to_rotation_matrix(r, p, y)
            t_temp = np.array([0, 0, -h]).reshape(3, 1)
            H_temp = K @ (R_temp - (t_temp @ n.T) / h) @ K_inv
            return H_temp.flatten()

        H0 = compute_H([roll, pitch, yaw, altitude])

        # Compute Jacobian using finite differences
        J_H = np.zeros((9, 4))
        params0 = np.array([roll, pitch, yaw, altitude])

        for i in range(4):
            params_plus = params0.copy()
            params_plus[i] += delta
            H_plus = compute_H(params_plus)
            J_H[:, i] = (H_plus - H0) / delta

        # Propagate uncertainty: Sigma_H = J * Sigma_params * J^T
        Sigma_H = J_H @ Sigma_params @ J_H.T

        return Sigma_H

    def estimate_dynamic_uncertainty(
        self,
        angular_velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate motion scale factor based on IMU dynamics.

        Args:
            angular_velocity: 3D angular velocity vector (rad/s)
            acceleration: 3D acceleration vector (m/s^2)

        Returns:
            Motion scale factor (>=1.0)
        """
        scale = 1.0

        if angular_velocity is not None:
            # Higher angular velocity increases attitude uncertainty
            ang_vel_magnitude = np.linalg.norm(angular_velocity)
            scale *= (1.0 + 0.1 * ang_vel_magnitude)  # Example: 0.1 rad/s -> 10% increase

        if acceleration is not None:
            # Higher acceleration increases uncertainty
            acc_magnitude = np.linalg.norm(acceleration)
            scale *= (1.0 + 0.05 * acc_magnitude)  # Example: 1 m/s^2 -> 5% increase

        return scale

    def validate_imu_data(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        altitude: float
    ) -> bool:
        """
        Validate IMU data for reasonable ranges.

        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            altitude: Altitude in meters

        Returns:
            True if data is valid
        """
        # Check attitude angles (should be within reasonable ranges)
        if np.abs(roll) > np.pi / 2:  # ±90 degrees
            return False
        if np.abs(pitch) > np.pi / 2:  # ±90 degrees
            return False

        # Yaw can be any angle (periodic)

        # Check altitude (should be positive and reasonable)
        if altitude <= 0 or altitude > 10000:  # 0-10km
            return False

        return True


def compute_ground_normal(terrain_slope_x: float = 0.0, terrain_slope_y: float = 0.0) -> np.ndarray:
    """
    Compute ground plane normal vector.

    Args:
        terrain_slope_x: Terrain slope in x direction (radians)
        terrain_slope_y: Terrain slope in y direction (radians)

    Returns:
        3x1 unit normal vector
    """
    # For flat ground, normal is [0, 0, 1]
    # For sloped terrain, adjust based on slopes
    n = np.array([
        -np.tan(terrain_slope_x),
        -np.tan(terrain_slope_y),
        1.0
    ])

    # Normalize
    n = n / np.linalg.norm(n)

    return n.reshape(3, 1)
