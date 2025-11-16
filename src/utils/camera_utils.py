"""Camera model and related utilities."""

import numpy as np
from typing import Tuple, Optional


class CameraModel:
    """Camera model for handling intrinsic parameters and projections."""

    def __init__(self, intrinsic_matrix: np.ndarray):
        """
        Initialize camera model.

        Args:
            intrinsic_matrix: 3x3 camera intrinsic matrix K
        """
        self.K = np.array(intrinsic_matrix, dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

        # Extract focal lengths and principal point
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

    @classmethod
    def from_params(cls, fx: float, fy: float, cx: float, cy: float):
        """
        Create camera model from individual parameters.

        Args:
            fx: Focal length in x direction (pixels)
            fy: Focal length in y direction (pixels)
            cx: Principal point x coordinate
            cy: Principal point y coordinate
        """
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        return cls(K)

    def pixel_to_normalized(self, points: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to normalized camera coordinates.

        Args:
            points: Nx2 array of pixel coordinates

        Returns:
            Nx2 array of normalized coordinates
        """
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        normalized = (self.K_inv @ points_homogeneous.T).T
        return normalized[:, :2] / normalized[:, 2:3]

    def normalized_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """
        Convert normalized camera coordinates to pixel coordinates.

        Args:
            points: Nx2 array of normalized coordinates

        Returns:
            Nx2 array of pixel coordinates
        """
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        pixels = (self.K @ points_homogeneous.T).T
        return pixels[:, :2] / pixels[:, 2:3]

    def compute_fov(self, image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Compute field of view angles.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            (horizontal_fov, vertical_fov) in radians
        """
        fov_h = 2 * np.arctan(image_width / (2 * self.fx))
        fov_v = 2 * np.arctan(image_height / (2 * self.fy))
        return fov_h, fov_v


def compute_homography_from_pose(
    R: np.ndarray,
    t: np.ndarray,
    n: np.ndarray,
    d: float,
    K: np.ndarray
) -> np.ndarray:
    """
    Compute homography matrix from camera pose.

    For planar scenes, H = K(R - t*n^T/d)K^(-1)

    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        n: 3x1 ground plane normal vector (typically [0, 0, 1])
        d: Distance to ground plane (altitude)
        K: 3x3 camera intrinsic matrix

    Returns:
        3x3 homography matrix
    """
    K_inv = np.linalg.inv(K)

    # Reshape vectors for matrix operations
    t = t.reshape(3, 1)
    n = n.reshape(1, 3)

    # H = K(R - t*n^T/d)K^(-1)
    H = K @ (R - (t @ n) / d) @ K_inv

    # Normalize so that H[2,2] = 1
    H = H / H[2, 2]

    return H


def check_homography_validity(H: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if homography matrix is valid.

    Args:
        H: 3x3 homography matrix
        threshold: Maximum allowed condition number

    Returns:
        True if homography is valid
    """
    # Check determinant (should be positive for orientation-preserving)
    det = np.linalg.det(H)
    if det <= 0:
        return False

    # Check condition number
    cond = np.linalg.cond(H)
    if cond > threshold:
        return False

    return True


def decompose_homography(
    H: np.ndarray,
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Decompose homography into rotation, translation, normal, and distance.

    Args:
        H: 3x3 homography matrix
        K: 3x3 camera intrinsic matrix

    Returns:
        (R, t, n, d) - rotation matrix, translation, normal, distance
    """
    K_inv = np.linalg.inv(K)

    # Compute normalized homography
    H_normalized = K_inv @ H @ K

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H_normalized)

    # Extract rotation (first solution)
    R = U @ Vt

    # Ensure proper rotation matrix (det = +1)
    if np.linalg.det(R) < 0:
        R = -R

    # Extract other parameters (simplified, one of multiple solutions)
    d = S[1]  # Middle singular value
    n = np.array([0, 0, 1])  # Assume ground plane
    t = np.array([0, 0, -d])  # Vertical translation

    return R, t, n, d


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply homography transformation to points.

    Args:
        H: 3x3 homography matrix
        points: Nx2 array of points

    Returns:
        Nx2 array of transformed points
    """
    points_homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = (H @ points_homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def compute_reprojection_error(
    points1: np.ndarray,
    points2: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    """
    Compute reprojection error for point correspondences.

    Args:
        points1: Nx2 array of source points
        points2: Nx2 array of target points
        H: 3x3 homography from points1 to points2

    Returns:
        N array of reprojection errors (Euclidean distances)
    """
    projected = apply_homography(H, points1)
    errors = np.linalg.norm(projected - points2, axis=1)
    return errors
