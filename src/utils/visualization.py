"""Visualization utilities for matching results."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import cv2


def visualize_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    title: str = "Feature Matches",
    max_display: int = 100,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize feature matches between two images.

    Args:
        img1: First image (satellite)
        img2: Second image (UAV)
        points1: Nx2 array of points in img1
        points2: Nx2 array of points in img2
        confidence: Optional N array of confidence scores (0-1)
        title: Plot title
        max_display: Maximum number of matches to display
        save_path: Optional path to save the figure
    """
    # Limit number of points to display
    n_points = min(len(points1), max_display)
    indices = np.random.choice(len(points1), n_points, replace=False) if len(points1) > max_display else np.arange(len(points1))

    points1_display = points1[indices]
    points2_display = points2[indices]
    conf_display = confidence[indices] if confidence is not None else None

    # Create side-by-side image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2

    # Convert to RGB if grayscale
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    # Create combined image
    combined = np.zeros((h, w, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1+w2] = img2

    # Plot
    plt.figure(figsize=(16, 8))
    plt.imshow(combined)

    # Draw matches
    for i in range(len(points1_display)):
        x1, y1 = points1_display[i]
        x2, y2 = points2_display[i]

        # Determine color based on confidence
        if conf_display is not None:
            c = conf_display[i]
            color = plt.cm.RdYlGn(c)  # Red (low) to Green (high)
        else:
            color = 'cyan'

        # Draw line
        plt.plot([x1, x2 + w1], [y1, y2], '-', color=color, linewidth=0.5, alpha=0.6)

        # Draw points
        plt.plot(x1, y1, 'o', color='red', markersize=3)
        plt.plot(x2 + w1, y2, 'o', color='blue', markersize=3)

    plt.title(f"{title} (showing {n_points} matches)")
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_factor_graph(
    points: np.ndarray,
    edges: List[Tuple[int, int]],
    title: str = "Factor Graph Structure",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize factor graph structure.

    Args:
        points: Nx2 array of point positions
        edges: List of (i, j) tuples representing edges
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 10))

    # Draw edges
    for i, j in edges:
        plt.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            'b-', alpha=0.3, linewidth=0.5
        )

    # Draw nodes
    plt.scatter(points[:, 0], points[:, 1], c='red', s=20, zorder=10)

    plt.title(f"{title}\n{len(points)} nodes, {len(edges)} edges")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_error_statistics(
    errors: np.ndarray,
    title: str = "Matching Error Statistics",
    save_path: Optional[str] = None
) -> None:
    """
    Plot error distribution and statistics.

    Args:
        errors: N array of errors
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
    axes[0].axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.3f}')
    axes[0].set_xlabel("Error (pixels)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CDF
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1].plot(sorted_errors, cdf, linewidth=2)
    axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    axes[1].axhline(0.95, color='green', linestyle='--', alpha=0.5, label='95th percentile')
    axes[1].set_xlabel("Error (pixels)")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_title("Cumulative Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Statistics:\n"
    stats_text += f"Mean: {np.mean(errors):.3f} px\n"
    stats_text += f"Median: {np.median(errors):.3f} px\n"
    stats_text += f"Std: {np.std(errors):.3f} px\n"
    stats_text += f"Min: {np.min(errors):.3f} px\n"
    stats_text += f"Max: {np.max(errors):.3f} px\n"
    stats_text += f"95th %ile: {np.percentile(errors, 95):.3f} px"

    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_convergence(
    errors_history: List[float],
    title: str = "Optimization Convergence",
    save_path: Optional[str] = None
) -> None:
    """
    Plot optimization convergence curve.

    Args:
        errors_history: List of error values at each iteration
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(errors_history, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel("Iteration")
    plt.ylabel("Total Error")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def create_quality_report(
    points_sat: np.ndarray,
    points_uav: np.ndarray,
    confidence: np.ndarray,
    uncertainty: np.ndarray,
    stage_name: str = "Stage",
    save_path: Optional[str] = None
) -> Dict:
    """
    Create a comprehensive quality report.

    Args:
        points_sat: Nx2 satellite image points
        points_uav: Nx2 UAV image points
        confidence: N array of confidence scores
        uncertainty: N array of uncertainty estimates
        stage_name: Name of the processing stage
        save_path: Optional path to save the report

    Returns:
        Dictionary containing quality metrics
    """
    report = {
        "stage": stage_name,
        "n_matches": len(points_sat),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_median": float(np.median(confidence)),
        "confidence_std": float(np.std(confidence)),
        "uncertainty_mean": float(np.mean(uncertainty)),
        "uncertainty_median": float(np.median(uncertainty)),
        "high_quality_ratio": float(np.sum(confidence > 0.8) / len(confidence)),
        "medium_quality_ratio": float(np.sum((confidence > 0.6) & (confidence <= 0.8)) / len(confidence)),
        "low_quality_ratio": float(np.sum(confidence <= 0.6) / len(confidence)),
    }

    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

    return report
