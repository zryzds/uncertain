"""
Stage 5: Subpixel Template Matching

Refines match points to subpixel accuracy using normalized cross-correlation
and quadratic/bicubic peak localization.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from scipy.ndimage import map_coordinates
from scipy.interpolate import RectBivariateSpline


class SubpixelMatcher:
    """Subpixel template matching using NCC and peak localization."""

    def __init__(
        self,
        template_size_min: int = 24,
        template_size_max: int = 40,
        adaptive_template: bool = True,
        search_radius_base: int = 5,
        search_radius_scale: float = 3.0,
        ncc_threshold: float = 0.7,
        peak_uniqueness_threshold: float = 1.2,
        subpixel_method: str = 'quadratic',
        offset_limit: float = 0.5,
        max_iterations: int = 3,
        convergence_threshold: float = 0.01
    ):
        """
        Initialize subpixel matcher.

        Args:
            template_size_min: Minimum template size (pixels)
            template_size_max: Maximum template size (pixels)
            adaptive_template: Whether to adapt template size based on texture
            search_radius_base: Base search radius (pixels)
            search_radius_scale: Scale factor for uncertainty-based radius
            ncc_threshold: Minimum NCC value for valid match
            peak_uniqueness_threshold: Min ratio of max/second_max peak
            subpixel_method: 'quadratic' or 'bicubic'
            offset_limit: Maximum allowed subpixel offset from integer peak
            max_iterations: Maximum refinement iterations
            convergence_threshold: Convergence threshold (pixels)
        """
        self.template_size_min = template_size_min
        self.template_size_max = template_size_max
        self.adaptive_template = adaptive_template
        self.search_radius_base = search_radius_base
        self.search_radius_scale = search_radius_scale
        self.ncc_threshold = ncc_threshold
        self.peak_uniqueness_threshold = peak_uniqueness_threshold
        self.subpixel_method = subpixel_method
        self.offset_limit = offset_limit
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def compute_texture_complexity(
        self,
        img: np.ndarray,
        center: np.ndarray,
        window_size: int = 32
    ) -> float:
        """
        Compute local texture complexity using gradient variance.

        Args:
            img: Input image
            center: Center point (x, y)
            window_size: Window size for analysis

        Returns:
            Texture complexity score
        """
        x, y = int(center[0]), int(center[1])
        half_win = window_size // 2

        # Extract window (with boundary check)
        y1 = max(0, y - half_win)
        y2 = min(img.shape[0], y + half_win)
        x1 = max(0, x - half_win)
        x2 = min(img.shape[1], x + half_win)

        window = img[y1:y2, x1:x2]

        if window.size == 0:
            return 0.0

        # Compute gradients
        grad_x = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Return variance of gradient magnitude
        return np.var(grad_magnitude)

    def adaptive_template_size(
        self,
        img: np.ndarray,
        center: np.ndarray,
        texture_threshold: float = 100.0
    ) -> int:
        """
        Determine adaptive template size based on local texture.

        Args:
            img: Input image
            center: Center point
            texture_threshold: Threshold for high/low texture

        Returns:
            Template size
        """
        if not self.adaptive_template:
            return (self.template_size_min + self.template_size_max) // 2

        complexity = self.compute_texture_complexity(img, center)

        if complexity > texture_threshold:
            # High texture - use smaller template
            return self.template_size_min
        else:
            # Low texture - use larger template
            return self.template_size_max

    def extract_template(
        self,
        img: np.ndarray,
        center: np.ndarray,
        template_size: int
    ) -> Optional[np.ndarray]:
        """
        Extract template centered at given point.

        Args:
            img: Input image
            center: Center point (x, y)
            template_size: Template size (must be odd)

        Returns:
            Template image or None if extraction fails
        """
        # Ensure template size is odd
        if template_size % 2 == 0:
            template_size += 1

        x, y = int(center[0]), int(center[1])
        half_size = template_size // 2

        # Boundary check
        if (x - half_size < 0 or x + half_size >= img.shape[1] or
            y - half_size < 0 or y + half_size >= img.shape[0]):
            return None

        template = img[y - half_size:y + half_size + 1,
                      x - half_size:x + half_size + 1]

        return template

    def normalized_cross_correlation(
        self,
        template: np.ndarray,
        search_region: np.ndarray
    ) -> np.ndarray:
        """
        Compute normalized cross-correlation using FFT.

        Args:
            template: Template image
            search_region: Search region image

        Returns:
            NCC response map
        """
        # Ensure float type
        template = template.astype(np.float64)
        search_region = search_region.astype(np.float64)

        # Normalize template (zero mean, unit variance)
        template_norm = (template - np.mean(template))
        template_std = np.std(template_norm)
        if template_std > 1e-6:
            template_norm = template_norm / template_std
        else:
            return np.zeros_like(search_region)

        # Use OpenCV's matchTemplate for efficiency
        result = cv2.matchTemplate(
            search_region.astype(np.float32),
            template_norm.astype(np.float32),
            cv2.TM_CCOEFF_NORMED
        )

        return result

    def find_peak(
        self,
        response_map: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Find peak in response map.

        Args:
            response_map: NCC response map

        Returns:
            (peak_location, peak_value, second_peak_value)
        """
        # Find maximum
        max_val = np.max(response_map)
        max_loc = np.unravel_index(np.argmax(response_map), response_map.shape)
        peak_location = np.array([max_loc[1], max_loc[0]], dtype=np.float64)  # (x, y)

        # Find second maximum (for uniqueness check)
        response_copy = response_map.copy()
        # Suppress region around maximum
        y, x = max_loc
        suppress_radius = 5
        y1 = max(0, y - suppress_radius)
        y2 = min(response_map.shape[0], y + suppress_radius)
        x1 = max(0, x - suppress_radius)
        x2 = min(response_map.shape[1], x + suppress_radius)
        response_copy[y1:y2, x1:x2] = -np.inf

        second_max_val = np.max(response_copy)

        return peak_location, max_val, second_max_val

    def subpixel_quadratic_fit(
        self,
        response_map: np.ndarray,
        peak_location: np.ndarray
    ) -> np.ndarray:
        """
        Subpixel peak localization using quadratic surface fitting.

        Args:
            response_map: NCC response map
            peak_location: Integer peak location (x, y)

        Returns:
            Subpixel refined location (x, y)
        """
        x0, y0 = int(peak_location[0]), int(peak_location[1])

        # Boundary check
        if (x0 < 1 or x0 >= response_map.shape[1] - 1 or
            y0 < 1 or y0 >= response_map.shape[0] - 1):
            return peak_location

        # Extract 3x3 neighborhood
        R = response_map[y0-1:y0+2, x0-1:x0+2]

        # Compute gradients (first derivatives)
        g_x = (R[1, 2] - R[1, 0]) / 2.0
        g_y = (R[2, 1] - R[0, 1]) / 2.0

        # Compute Hessian (second derivatives)
        h_xx = R[1, 0] - 2 * R[1, 1] + R[1, 2]
        h_yy = R[0, 1] - 2 * R[1, 1] + R[2, 1]
        h_xy = (R[2, 2] - R[2, 0] - R[0, 2] + R[0, 0]) / 4.0

        # Hessian matrix
        H = np.array([[h_xx, h_xy],
                     [h_xy, h_yy]])

        # Check if Hessian is valid (negative definite for maximum)
        det_H = np.linalg.det(H)
        if det_H >= 0 or h_xx >= 0:  # Not a maximum
            return peak_location

        # Solve: H * [dx, dy]^T = -[g_x, g_y]^T
        try:
            offset = -np.linalg.solve(H, np.array([g_x, g_y]))
        except np.linalg.LinAlgError:
            return peak_location

        # Limit offset to prevent jumping to adjacent pixel
        offset = np.clip(offset, -self.offset_limit, self.offset_limit)

        # Subpixel location
        subpixel_location = peak_location + offset

        return subpixel_location

    def subpixel_bicubic_interpolation(
        self,
        response_map: np.ndarray,
        peak_location: np.ndarray
    ) -> np.ndarray:
        """
        Subpixel peak localization using bicubic interpolation.

        Args:
            response_map: NCC response map
            peak_location: Integer peak location (x, y)

        Returns:
            Subpixel refined location (x, y)
        """
        x0, y0 = int(peak_location[0]), int(peak_location[1])

        # Boundary check (need 7x7 neighborhood)
        margin = 3
        if (x0 < margin or x0 >= response_map.shape[1] - margin or
            y0 < margin or y0 >= response_map.shape[0] - margin):
            return peak_location

        # Extract larger neighborhood
        neighborhood = response_map[y0-margin:y0+margin+1, x0-margin:x0+margin+1]

        # Create bicubic spline interpolator
        x_indices = np.arange(x0 - margin, x0 + margin + 1)
        y_indices = np.arange(y0 - margin, y0 + margin + 1)

        spline = RectBivariateSpline(y_indices, x_indices, neighborhood, kx=3, ky=3)

        # Search on dense grid
        search_range = 1.0
        step = 0.1
        x_fine = np.arange(x0 - search_range, x0 + search_range + step, step)
        y_fine = np.arange(y0 - search_range, y0 + search_range + step, step)

        # Evaluate spline on fine grid
        response_fine = spline(y_fine, x_fine)

        # Find maximum on fine grid
        max_idx = np.unravel_index(np.argmax(response_fine), response_fine.shape)
        subpixel_location = np.array([x_fine[max_idx[1]], y_fine[max_idx[0]]])

        return subpixel_location

    def refine_single_point(
        self,
        img_ref: np.ndarray,
        img_query: np.ndarray,
        point_ref: np.ndarray,
        point_query_init: np.ndarray,
        uncertainty: float = 1.0
    ) -> Dict:
        """
        Refine a single match point to subpixel accuracy.

        Args:
            img_ref: Reference image (satellite)
            img_query: Query image (UAV)
            point_ref: Reference point (x, y)
            point_query_init: Initial query point (x, y)
            uncertainty: Uncertainty estimate for determining search radius

        Returns:
            Dictionary containing refined point and quality metrics
        """
        # Determine template size
        template_size = self.adaptive_template_size(img_ref, point_ref)

        # Determine search radius
        search_radius = max(
            self.search_radius_base,
            int(self.search_radius_scale * uncertainty)
        )

        # Extract template from reference image
        template = self.extract_template(img_ref, point_ref, template_size)
        if template is None:
            return {
                "point_refined": point_query_init,
                "confidence": 0.0,
                "ncc_max": 0.0,
                "status": False,
                "iterations": 0
            }

        # Iterative refinement
        point_current = point_query_init.copy()
        converged = False

        for iteration in range(self.max_iterations):
            # Define search region
            search_size = template_size + 2 * search_radius
            search_center = point_current

            # Extract search region
            x_c, y_c = int(search_center[0]), int(search_center[1])
            half_search = search_size // 2

            # Boundary check
            if (x_c - half_search < 0 or x_c + half_search >= img_query.shape[1] or
                y_c - half_search < 0 or y_c + half_search >= img_query.shape[0]):
                break

            search_region = img_query[
                y_c - half_search:y_c + half_search + 1,
                x_c - half_search:x_c + half_search + 1
            ]

            # Compute NCC
            response_map = self.normalized_cross_correlation(template, search_region)

            if response_map.size == 0:
                break

            # Find peak
            peak_loc_local, ncc_max, ncc_second = self.find_peak(response_map)

            # Check NCC threshold
            if ncc_max < self.ncc_threshold:
                break

            # Check peak uniqueness
            if ncc_second > 0:
                uniqueness = ncc_max / (ncc_second + 1e-10)
                if uniqueness < self.peak_uniqueness_threshold:
                    pass  # Continue but mark as low confidence

            # Subpixel refinement
            if self.subpixel_method == 'quadratic':
                peak_loc_subpixel = self.subpixel_quadratic_fit(response_map, peak_loc_local)
            elif self.subpixel_method == 'bicubic':
                peak_loc_subpixel = self.subpixel_bicubic_interpolation(response_map, peak_loc_local)
            else:
                peak_loc_subpixel = peak_loc_local

            # Convert to global coordinates
            # peak_loc_local is relative to search_region top-left corner
            offset_from_center = peak_loc_subpixel - np.array([half_search, half_search])
            point_new = search_center + offset_from_center

            # Check convergence
            displacement = np.linalg.norm(point_new - point_current)
            point_current = point_new

            if displacement < self.convergence_threshold:
                converged = True
                break

        # Compute confidence score
        confidence = self._compute_confidence(
            ncc_max if 'ncc_max' in locals() else 0.0,
            ncc_second if 'ncc_second' in locals() else 0.0,
            uncertainty
        )

        return {
            "point_refined": point_current,
            "confidence": confidence,
            "ncc_max": ncc_max if 'ncc_max' in locals() else 0.0,
            "status": converged and (ncc_max if 'ncc_max' in locals() else 0.0) >= self.ncc_threshold,
            "iterations": iteration + 1 if 'iteration' in locals() else 0
        }

    def _compute_confidence(
        self,
        ncc_max: float,
        ncc_second: float,
        uncertainty: float
    ) -> float:
        """
        Compute overall confidence score.

        Args:
            ncc_max: Maximum NCC value
            ncc_second: Second maximum NCC value
            uncertainty: Prior uncertainty

        Returns:
            Confidence score (0-1)
        """
        # Component 1: NCC response strength
        C1 = ncc_max

        # Component 2: Peak uniqueness
        C2 = (ncc_max - ncc_second) / (ncc_max + 1e-10)

        # Component 3: Low uncertainty
        C3 = 1.0 / (1.0 + uncertainty)

        # Weighted combination
        weights = [0.5, 0.3, 0.2]
        confidence = weights[0] * C1 + weights[1] * C2 + weights[2] * C3

        return np.clip(confidence, 0.0, 1.0)

    def refine_matches(
        self,
        img_sat: np.ndarray,
        img_uav: np.ndarray,
        points_sat: np.ndarray,
        points_uav_init: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Refine multiple match points to subpixel accuracy.

        Args:
            img_sat: Satellite image
            img_uav: UAV image
            points_sat: Nx2 satellite points
            points_uav_init: Nx2 initial UAV points
            uncertainties: Optional N array of uncertainty estimates

        Returns:
            Dictionary containing refined matches and quality metrics
        """
        n_points = len(points_sat)

        if uncertainties is None:
            uncertainties = np.ones(n_points)

        # Ensure images are grayscale
        if len(img_sat.shape) == 3:
            img_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2GRAY)
        if len(img_uav.shape) == 3:
            img_uav = cv2.cvtColor(img_uav, cv2.COLOR_BGR2GRAY)

        # Refine each point
        points_uav_refined = np.zeros_like(points_uav_init)
        confidence = np.zeros(n_points)
        ncc_values = np.zeros(n_points)
        status = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            result = self.refine_single_point(
                img_sat, img_uav,
                points_sat[i],
                points_uav_init[i],
                uncertainties[i]
            )

            points_uav_refined[i] = result["point_refined"]
            confidence[i] = result["confidence"]
            ncc_values[i] = result["ncc_max"]
            status[i] = result["status"]

        # Filter low-quality matches
        high_quality_mask = confidence > 0.6

        return {
            "points_sat": points_sat,
            "points_uav_refined": points_uav_refined,
            "confidence": confidence,
            "ncc_values": ncc_values,
            "status": status,
            "high_quality_mask": high_quality_mask,
            "n_valid": np.sum(status),
            "n_high_quality": np.sum(high_quality_mask),
            "mean_confidence": np.mean(confidence[status]) if np.any(status) else 0.0
        }


def cross_validation_check(
    img_sat: np.ndarray,
    img_uav: np.ndarray,
    points_sat: np.ndarray,
    points_uav: np.ndarray,
    matcher: SubpixelMatcher,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Perform backward matching for cross-validation.

    Args:
        img_sat: Satellite image
        img_uav: UAV image
        points_sat: Nx2 satellite points
        points_uav: Nx2 UAV points
        matcher: SubpixelMatcher instance
        threshold: Consistency threshold (pixels)

    Returns:
        Boolean array indicating consistent matches
    """
    n_points = len(points_sat)
    consistent = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        # Backward match: UAV -> Satellite
        result = matcher.refine_single_point(
            img_uav, img_sat,
            points_uav[i],
            points_sat[i],
            uncertainty=1.0
        )

        # Check consistency
        if result["status"]:
            backward_error = np.linalg.norm(result["point_refined"] - points_sat[i])
            consistent[i] = backward_error < threshold

    return consistent
