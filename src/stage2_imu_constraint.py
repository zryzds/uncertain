"""
Stage 2: IMU-Guided Geometric Constraint Optimization

Uses UAV IMU data to compute homography prior and refines matches
using Bayesian optimization and optical flow.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from scipy.optimize import least_squares

from .utils.camera_utils import (
    compute_homography_from_pose,
    check_homography_validity,
    apply_homography,
    compute_reprojection_error
)
from .utils.imu_utils import IMUProcessor


class IMUConstraintOptimizer:
    """Optimize feature matches using IMU-guided geometric constraints."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        imu_processor: Optional[IMUProcessor] = None,
        optical_flow_window: int = 21,
        optical_flow_levels: int = 3,
        optical_flow_iters: int = 30,
        max_bayesian_iters: int = 50,
        convergence_threshold: float = 1e-5
    ):
        """
        Initialize IMU constraint optimizer.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            imu_processor: IMU data processor (creates default if None)
            optical_flow_window: Lucas-Kanade window size
            optical_flow_levels: Number of pyramid levels
            optical_flow_iters: Maximum optical flow iterations
            max_bayesian_iters: Maximum Bayesian optimization iterations
            convergence_threshold: Convergence threshold for optimization
        """
        self.K = camera_matrix
        self.imu_processor = imu_processor or IMUProcessor()

        # Optical flow parameters
        self.lk_params = dict(
            winSize=(optical_flow_window, optical_flow_window),
            maxLevel=optical_flow_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                     optical_flow_iters, 0.01)
        )

        # Optimization parameters
        self.max_bayesian_iters = max_bayesian_iters
        self.convergence_threshold = convergence_threshold

    def compute_homography_prior(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        altitude: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute homography matrix prior from IMU data.

        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            altitude: Flight altitude in meters

        Returns:
            (H_prior, Sigma_H): Homography matrix and its covariance
        """
        # Validate IMU data
        if not self.imu_processor.validate_imu_data(roll, pitch, yaw, altitude):
            raise ValueError("Invalid IMU data")

        # Compute rotation matrix
        R = self.imu_processor.euler_to_rotation_matrix(roll, pitch, yaw)

        # Ground plane normal (assuming flat terrain)
        n = np.array([[0], [0], [1]])

        # Translation vector (mainly altitude)
        t = np.array([[0], [0], [-altitude]])

        # Compute homography: H = K(R - t*n^T/d)K^(-1)
        H_prior = compute_homography_from_pose(R, t, n, altitude, self.K)

        # Compute uncertainty
        Sigma_H = self.imu_processor.compute_homography_uncertainty(
            roll, pitch, yaw, altitude, self.K
        )

        return H_prior, Sigma_H

    def bayesian_optimization(
        self,
        points_sat: np.ndarray,
        points_uav: np.ndarray,
        H_prior: np.ndarray,
        Sigma_H: np.ndarray,
        visual_noise: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bayesian optimization to fuse visual matching and IMU prior.

        Maximizes: P(H | matches, IMU) ∝ P(matches | H) * P(H | IMU)

        Args:
            points_sat: Nx2 satellite image points
            points_uav: Nx2 UAV image points (initial matches)
            H_prior: 3x3 prior homography from IMU
            Sigma_H: 9x9 covariance matrix for H_prior
            visual_noise: Visual matching noise standard deviation (pixels)

        Returns:
            (H_optimized, weights): Optimized homography and point weights
        """
        n_points = len(points_sat)

        # Initialize with prior
        H_current = H_prior.copy()

        # Compute IMU confidence (inverse of uncertainty)
        imu_confidence = 1.0 / (1.0 + np.trace(Sigma_H.reshape(9, 9)))

        # Regularization weight
        lambda_imu = imu_confidence

        # Iterative optimization (Expectation-Maximization style)
        for iteration in range(self.max_bayesian_iters):
            # E-step: Compute weights for each match point
            errors = compute_reprojection_error(points_sat, points_uav, H_current)

            # Visual likelihood (Gaussian)
            visual_likelihood = np.exp(-errors**2 / (2 * visual_noise**2))

            # Normalize to get weights
            weights = visual_likelihood / (visual_likelihood + 1e-10)

            # M-step: Update homography with weighted least squares
            def residual_function(h_vec):
                H_test = h_vec.reshape(3, 3)

                # Visual matching term
                projected = apply_homography(H_test, points_sat)
                visual_residuals = (projected - points_uav).flatten() * np.repeat(weights, 2)

                # IMU prior term (Mahalanobis distance)
                h_diff = (h_vec - H_prior.flatten())
                Sigma_H_inv = np.linalg.pinv(Sigma_H.reshape(9, 9))
                imu_residuals = lambda_imu * Sigma_H_inv @ h_diff

                # Combine residuals
                return np.concatenate([visual_residuals, imu_residuals])

            # Optimize
            result = least_squares(
                residual_function,
                H_current.flatten(),
                method='lm',
                ftol=self.convergence_threshold,
                max_nfev=20
            )

            H_new = result.x.reshape(3, 3)

            # Normalize homography
            H_new = H_new / H_new[2, 2]

            # Check convergence
            change = np.linalg.norm(H_new - H_current)
            H_current = H_new

            if change < self.convergence_threshold:
                break

        # Validate final homography
        if not check_homography_validity(H_current):
            print("Warning: Optimized homography may be invalid, using prior")
            H_current = H_prior

        return H_current, weights

    def optical_flow_refinement(
        self,
        img_sat: np.ndarray,
        img_uav: np.ndarray,
        points_sat: np.ndarray,
        points_uav_init: np.ndarray,
        backward_check: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine match points using Lucas-Kanade optical flow.

        Args:
            img_sat: Satellite image (grayscale)
            img_uav: UAV image (grayscale)
            points_sat: Nx2 satellite image points
            points_uav_init: Nx2 initial UAV image points
            backward_check: Whether to perform backward consistency check

        Returns:
            (points_uav_refined, status): Refined points and validity status
        """
        # Ensure images are grayscale uint8
        if img_sat.dtype != np.uint8:
            img_sat = (img_sat * 255).astype(np.uint8) if img_sat.max() <= 1.0 else img_sat.astype(np.uint8)
        if img_uav.dtype != np.uint8:
            img_uav = (img_uav * 255).astype(np.uint8) if img_uav.max() <= 1.0 else img_uav.astype(np.uint8)

        # Convert points to float32 for OpenCV
        points_sat_cv = points_sat.astype(np.float32).reshape(-1, 1, 2)
        points_uav_cv = points_uav_init.astype(np.float32).reshape(-1, 1, 2)

        # Forward optical flow: sat -> uav
        points_uav_refined, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            img_sat, img_uav, points_sat_cv, points_uav_cv, **self.lk_params
        )

        status = status_fwd.flatten().astype(bool)

        # Backward consistency check
        if backward_check and np.any(status):
            points_sat_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
                img_uav, img_sat,
                points_uav_refined[status],
                points_sat_cv[status],
                **self.lk_params
            )

            # Compute backward error
            backward_error = np.linalg.norm(
                points_sat_back.reshape(-1, 2) - points_sat[status],
                axis=1
            )

            # Update status (consistent if backward error < 1 pixel)
            consistent = backward_error < 1.0
            status_indices = np.where(status)[0]
            status[status_indices[~consistent]] = False

        points_uav_refined = points_uav_refined.reshape(-1, 2)

        return points_uav_refined, status

    def refine_matches(
        self,
        img_sat: np.ndarray,
        img_uav: np.ndarray,
        points_sat: np.ndarray,
        points_uav: np.ndarray,
        roll: float,
        pitch: float,
        yaw: float,
        altitude: float,
        alpha: float = 0.6
    ) -> Dict:
        """
        Complete Stage 2 refinement pipeline.

        Args:
            img_sat: Satellite image (grayscale)
            img_uav: UAV image (grayscale)
            points_sat: Nx2 satellite image points
            points_uav: Nx2 UAV image points (initial matches)
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            altitude: Flight altitude in meters
            alpha: Fusion weight for IMU prediction (higher = trust IMU more)

        Returns:
            Dictionary containing:
                - points_sat: Input satellite points
                - points_uav_refined: Refined UAV points
                - H_optimized: Optimized homography matrix
                - Sigma_H: Homography covariance
                - weights: Point confidence weights
                - status: Boolean array of valid points
        """
        # Step 1: Compute homography prior from IMU
        H_prior, Sigma_H = self.compute_homography_prior(roll, pitch, yaw, altitude)

        # Step 2: Bayesian optimization
        H_optimized, weights = self.bayesian_optimization(
            points_sat, points_uav, H_prior, Sigma_H
        )

        # Step 3: Predict UAV points using optimized homography
        points_uav_predicted = apply_homography(H_optimized, points_sat)

        # Step 4: Fuse with initial matching
        # Combined initial estimate: alpha * predicted + (1-alpha) * initial
        imu_confidence = 1.0 / (1.0 + np.trace(Sigma_H.reshape(9, 9)))
        alpha_adaptive = alpha * imu_confidence

        points_uav_combined = (
            alpha_adaptive * points_uav_predicted +
            (1 - alpha_adaptive) * points_uav
        )

        # Step 5: Optical flow refinement
        points_uav_refined, status = self.optical_flow_refinement(
            img_sat, img_uav, points_sat, points_uav_combined
        )

        # Step 6: Soft constraint projection back to homography manifold
        # Project refined points toward homography prediction
        beta = imu_confidence  # Projection strength
        points_uav_final = (
            beta * points_uav_predicted +
            (1 - beta) * points_uav_refined
        )

        # Only keep valid points
        points_uav_final[~status] = points_uav_refined[~status]

        # Compute final quality metrics
        final_errors = compute_reprojection_error(
            points_sat[status],
            points_uav_final[status],
            H_optimized
        )

        return {
            "points_sat": points_sat,
            "points_uav_refined": points_uav_final,
            "H_optimized": H_optimized,
            "Sigma_H": Sigma_H,
            "weights": weights,
            "status": status,
            "mean_error": np.mean(final_errors) if len(final_errors) > 0 else float('inf'),
            "median_error": np.median(final_errors) if len(final_errors) > 0 else float('inf'),
            "n_valid": np.sum(status)
        }


def robust_homography_estimation(
    points_sat: np.ndarray,
    points_uav: np.ndarray,
    method: int = cv2.RANSAC,
    reproj_threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust homography estimation using RANSAC.

    Args:
        points_sat: Nx2 satellite points
        points_uav: Nx2 UAV points
        method: OpenCV method (RANSAC, LMEDS, etc.)
        reproj_threshold: RANSAC reprojection threshold (pixels)

    Returns:
        (H, inlier_mask): Homography matrix and inlier mask
    """
    if len(points_sat) < 4:
        raise ValueError("Need at least 4 points for homography estimation")

    H, mask = cv2.findHomography(
        points_sat.astype(np.float32),
        points_uav.astype(np.float32),
        method,
        reproj_threshold
    )

    if H is None:
        # Fallback to direct linear transform
        H, _ = cv2.findHomography(
            points_sat.astype(np.float32),
            points_uav.astype(np.float32),
            0
        )
        mask = np.ones(len(points_sat), dtype=np.uint8)

    return H, mask.flatten().astype(bool)
