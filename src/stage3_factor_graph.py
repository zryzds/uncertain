"""
Stage 3: Factor Graph Multi-Modal Fusion

Implements factor graph optimization to fuse visual matching, IMU constraints,
GPS data, and smoothness constraints for globally optimal matching.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import least_squares
from scipy.spatial import KDTree

from .utils.camera_utils import apply_homography, compute_reprojection_error


class Factor:
    """Base class for factor graph factors."""

    def __init__(self, variables: List[int], noise: float):
        """
        Initialize factor.

        Args:
            variables: List of variable indices this factor connects to
            noise: Noise standard deviation for this factor
        """
        self.variables = variables
        self.noise = noise
        self.weight = 1.0 / (noise ** 2)

    def error(self, values: np.ndarray) -> np.ndarray:
        """
        Compute error for this factor.

        Args:
            values: Current values of all variables

        Returns:
            Error vector
        """
        raise NotImplementedError

    def jacobian(self, values: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for this factor.

        Args:
            values: Current values of all variables

        Returns:
            Jacobian matrix
        """
        raise NotImplementedError


class UnaryFactor(Factor):
    """Unary factor (prior) that pulls a variable toward a target value."""

    def __init__(self, variable_idx: int, target: np.ndarray, noise: float):
        """
        Initialize unary factor.

        Args:
            variable_idx: Index of variable
            target: Target value (2D point)
            noise: Noise standard deviation
        """
        super().__init__([variable_idx], noise)
        self.target = target

    def error(self, values: np.ndarray) -> np.ndarray:
        """Compute error: current - target."""
        var_idx = self.variables[0]
        current = values[var_idx * 2:(var_idx + 1) * 2]
        return (current - self.target) * np.sqrt(self.weight)

    def jacobian(self, values: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Compute Jacobian (identity matrix for unary factor)."""
        var_idx = self.variables[0]
        J = np.eye(2) * np.sqrt(self.weight)
        indices = [var_idx * 2, var_idx * 2 + 1]
        return J, indices


class BetweenFactor(Factor):
    """Between factor that constrains relative displacement between two variables."""

    def __init__(
        self,
        variable_idx1: int,
        variable_idx2: int,
        relative_displacement: np.ndarray,
        noise: float
    ):
        """
        Initialize between factor.

        Args:
            variable_idx1: First variable index
            variable_idx2: Second variable index
            relative_displacement: Expected relative displacement
            noise: Noise standard deviation
        """
        super().__init__([variable_idx1, variable_idx2], noise)
        self.relative_displacement = relative_displacement

    def error(self, values: np.ndarray) -> np.ndarray:
        """Compute error: (p2 - p1) - expected_displacement."""
        idx1, idx2 = self.variables
        p1 = values[idx1 * 2:(idx1 + 1) * 2]
        p2 = values[idx2 * 2:(idx2 + 1) * 2]
        actual_displacement = p2 - p1
        return (actual_displacement - self.relative_displacement) * np.sqrt(self.weight)

    def jacobian(self, values: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Compute Jacobian."""
        idx1, idx2 = self.variables
        J = np.zeros((2, 4))
        J[:, 0:2] = -np.eye(2) * np.sqrt(self.weight)  # -I for first variable
        J[:, 2:4] = np.eye(2) * np.sqrt(self.weight)   # +I for second variable

        indices = [idx1 * 2, idx1 * 2 + 1, idx2 * 2, idx2 * 2 + 1]
        return J, indices


class FactorGraphOptimizer:
    """Factor graph optimizer using Levenberg-Marquardt algorithm."""

    def __init__(
        self,
        visual_noise: float = 1.5,
        imu_noise: float = 1.0,
        gps_noise: float = 5.0,
        smooth_noise: float = 0.5,
        neighbor_radius: float = 40.0,
        k_neighbors: int = 5,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-5,
        use_robust_kernel: bool = True,
        robust_kernel_type: str = 'huber',
        robust_threshold: float = 1.5
    ):
        """
        Initialize factor graph optimizer.

        Args:
            visual_noise: Visual matching noise (pixels)
            imu_noise: IMU constraint noise (pixels)
            gps_noise: GPS constraint noise (pixels)
            smooth_noise: Smoothness constraint noise (pixels)
            neighbor_radius: Neighborhood radius for smoothness (pixels)
            k_neighbors: Number of neighbors for smoothness constraints
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold
            use_robust_kernel: Whether to use robust kernel
            robust_kernel_type: Type of robust kernel ('huber' or 'cauchy')
            robust_threshold: Threshold for robust kernel
        """
        self.visual_noise = visual_noise
        self.imu_noise = imu_noise
        self.gps_noise = gps_noise
        self.smooth_noise = smooth_noise
        self.neighbor_radius = neighbor_radius
        self.k_neighbors = k_neighbors
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_robust_kernel = use_robust_kernel
        self.robust_kernel_type = robust_kernel_type
        self.robust_threshold = robust_threshold

        self.factors: List[Factor] = []

    def add_visual_factors(
        self,
        points_visual: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ):
        """
        Add visual matching factors (unary priors).

        Args:
            points_visual: Nx2 array of visual matching results
            confidence: Optional Nx1 confidence scores (used to scale noise)
        """
        n_points = len(points_visual)

        for i in range(n_points):
            # Adjust noise based on confidence
            if confidence is not None:
                noise = self.visual_noise / (confidence[i] + 0.1)
            else:
                noise = self.visual_noise

            factor = UnaryFactor(i, points_visual[i], noise)
            self.factors.append(factor)

    def add_imu_factors(
        self,
        points_sat: np.ndarray,
        H_imu: np.ndarray,
        Sigma_H: np.ndarray
    ):
        """
        Add IMU constraint factors.

        Args:
            points_sat: Nx2 satellite image points
            H_imu: 3x3 homography from IMU
            Sigma_H: 9x9 covariance matrix for homography
        """
        n_points = len(points_sat)

        # Compute IMU predictions
        points_imu_predicted = apply_homography(H_imu, points_sat)

        # Compute per-point uncertainty from Sigma_H
        for i in range(n_points):
            # Simplified: use trace of Sigma_H as overall uncertainty
            uncertainty = np.sqrt(np.trace(Sigma_H.reshape(9, 9))) / 3.0
            noise = max(self.imu_noise, uncertainty)

            factor = UnaryFactor(i, points_imu_predicted[i], noise)
            self.factors.append(factor)

    def add_gps_factors(
        self,
        points_sat: np.ndarray,
        gps_lat: float,
        gps_lon: float,
        gps_alt: float,
        satellite_georef: Dict
    ):
        """
        Add GPS absolute position factors.

        Args:
            points_sat: Nx2 satellite image points
            gps_lat: GPS latitude
            gps_lon: GPS longitude
            gps_alt: GPS altitude
            satellite_georef: Satellite image georeferencing info
                (should contain transform to convert GPS to pixel coords)
        """
        # This is a placeholder - actual implementation needs georeferencing
        # For now, we'll skip GPS factors in the example
        pass

    def add_smoothness_factors(
        self,
        points_sat: np.ndarray,
        points_uav_init: np.ndarray
    ):
        """
        Add smoothness factors (local consistency constraints).

        Args:
            points_sat: Nx2 satellite image points (for building neighborhood)
            points_uav_init: Nx2 initial UAV points
        """
        n_points = len(points_sat)

        # Build KD-tree for efficient neighborhood search
        tree = KDTree(points_sat)

        # For each point, find neighbors and add between factors
        for i in range(n_points):
            # Query k nearest neighbors (excluding self)
            distances, indices = tree.query(
                points_sat[i],
                k=min(self.k_neighbors + 1, n_points)
            )

            # Skip first index (self)
            for j, dist in zip(indices[1:], distances[1:]):
                if dist > self.neighbor_radius:
                    continue

                # Expected relative displacement in UAV image
                # Should match satellite image displacement
                expected_displacement = points_uav_init[j] - points_uav_init[i]

                factor = BetweenFactor(i, j, expected_displacement, self.smooth_noise)
                self.factors.append(factor)

    def robust_kernel(self, residuals: np.ndarray) -> np.ndarray:
        """
        Apply robust kernel to residuals.

        Args:
            residuals: Raw residuals

        Returns:
            Robustified residuals
        """
        if not self.use_robust_kernel:
            return residuals

        # Compute robust threshold (adaptive based on MAD)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        threshold = self.robust_threshold * mad

        if self.robust_kernel_type == 'huber':
            # Huber kernel
            mask = np.abs(residuals) <= threshold
            residuals_robust = residuals.copy()
            residuals_robust[~mask] = (
                np.sign(residuals[~mask]) *
                (threshold + threshold * np.log(np.abs(residuals[~mask]) / threshold))
            )
            return residuals_robust

        elif self.robust_kernel_type == 'cauchy':
            # Cauchy kernel
            c = threshold
            return c * np.log(1 + (residuals / c) ** 2)

        else:
            return residuals

    def optimize(
        self,
        initial_values: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Optimize factor graph using Levenberg-Marquardt.

        Args:
            initial_values: Nx2 initial variable values
            verbose: Whether to print optimization progress

        Returns:
            (optimized_values, info): Optimized values and info dict
        """
        n_vars = len(initial_values)
        x0 = initial_values.flatten()

        # Build residual function
        def residual_function(x):
            residuals = []

            for factor in self.factors:
                error = factor.error(x)
                residuals.extend(error)

            residuals = np.array(residuals)

            # Apply robust kernel
            residuals = self.robust_kernel(residuals)

            return residuals

        # Optimize using Levenberg-Marquardt
        result = least_squares(
            residual_function,
            x0,
            method='lm',
            ftol=self.convergence_threshold,
            max_nfev=self.max_iterations,
            verbose=2 if verbose else 0
        )

        optimized_values = result.x.reshape(-1, 2)

        # Compute posterior covariance (approximation)
        # Sigma_post = (J^T * J)^(-1)
        J = result.jac
        try:
            JtJ = J.T @ J
            Sigma_post = np.linalg.inv(JtJ)

            # Extract diagonal (variances for each variable)
            variances = np.diag(Sigma_post).reshape(-1, 2)
            uncertainties = np.sqrt(np.sum(variances, axis=1))
        except:
            # Singular matrix, use default uncertainty
            uncertainties = np.ones(n_vars)

        info = {
            "success": result.success,
            "cost": result.cost,
            "n_iterations": result.nfev,
            "optimality": result.optimality,
            "uncertainties": uncertainties,
            "n_factors": len(self.factors)
        }

        return optimized_values, info

    def detect_outliers(
        self,
        optimized_values: np.ndarray,
        chi2_threshold: float = 5.991  # 95% confidence for 2 DOF
    ) -> np.ndarray:
        """
        Detect outliers using chi-squared test.

        Args:
            optimized_values: Nx2 optimized variable values
            chi2_threshold: Chi-squared threshold

        Returns:
            Boolean array indicating inliers (True) and outliers (False)
        """
        n_vars = len(optimized_values)
        x = optimized_values.flatten()

        # Compute Mahalanobis distance for each variable
        mahalanobis_distances = np.zeros(n_vars)

        for factor in self.factors:
            if len(factor.variables) == 1:  # Unary factor
                var_idx = factor.variables[0]
                error = factor.error(x)
                mahalanobis_distances[var_idx] += np.sum(error ** 2)

        # Chi-squared test
        inliers = mahalanobis_distances < chi2_threshold

        return inliers

    def compute_quality_scores(
        self,
        optimized_values: np.ndarray,
        uncertainties: np.ndarray
    ) -> np.ndarray:
        """
        Compute quality score for each match point.

        Args:
            optimized_values: Nx2 optimized points
            uncertainties: N array of uncertainties

        Returns:
            N array of quality scores (0-1)
        """
        n_vars = len(optimized_values)
        x = optimized_values.flatten()

        # Component 1: Low uncertainty
        C_uncertainty = 1.0 / (1.0 + uncertainties)

        # Component 2: Visual matching quality
        C_visual = np.zeros(n_vars)
        for factor in self.factors:
            if len(factor.variables) == 1 and factor.noise < self.visual_noise * 2:
                var_idx = factor.variables[0]
                error = np.linalg.norm(factor.error(x))
                C_visual[var_idx] = np.exp(-error)

        # Component 3: Smoothness consistency
        C_smooth = np.ones(n_vars)
        # (simplified - could compute based on between factors)

        # Combine (weighted average)
        weights = [0.4, 0.4, 0.2]
        quality = (
            weights[0] * C_uncertainty +
            weights[1] * C_visual +
            weights[2] * C_smooth
        )

        # Normalize to [0, 1]
        quality = np.clip(quality, 0, 1)

        return quality


def build_and_optimize_factor_graph(
    points_sat: np.ndarray,
    points_uav_init: np.ndarray,
    H_imu: np.ndarray,
    Sigma_H: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    config: Optional[Dict] = None
) -> Dict:
    """
    Build and optimize factor graph (convenience function).

    Args:
        points_sat: Nx2 satellite image points
        points_uav_init: Nx2 initial UAV image points
        H_imu: 3x3 homography from IMU
        Sigma_H: 9x9 homography covariance
        confidence: Optional N array of confidence scores
        config: Optional configuration dictionary

    Returns:
        Dictionary containing optimization results
    """
    # Default configuration
    default_config = {
        "visual_noise": 1.5,
        "imu_noise": 1.0,
        "smooth_noise": 0.5,
        "neighbor_radius": 40.0,
        "k_neighbors": 5,
        "max_iterations": 100,
    }

    if config is not None:
        default_config.update(config)

    # Create optimizer
    optimizer = FactorGraphOptimizer(**default_config)

    # Add factors
    optimizer.add_visual_factors(points_uav_init, confidence)
    optimizer.add_imu_factors(points_sat, H_imu, Sigma_H)
    optimizer.add_smoothness_factors(points_sat, points_uav_init)

    # Optimize
    optimized_values, info = optimizer.optimize(points_uav_init, verbose=False)

    # Detect outliers
    inliers = optimizer.detect_outliers(optimized_values)

    # Compute quality scores
    quality = optimizer.compute_quality_scores(optimized_values, info["uncertainties"])

    return {
        "points_sat": points_sat,
        "points_uav_optimized": optimized_values,
        "uncertainties": info["uncertainties"],
        "quality": quality,
        "inliers": inliers,
        "optimization_info": info,
        "n_valid": np.sum(inliers)
    }
