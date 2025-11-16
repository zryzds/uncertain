"""
Example script demonstrating the complete UAV-satellite matching pipeline.

This example shows how to:
1. Load images and initial matches
2. Configure the pipeline
3. Run the complete matching process
4. Save and visualize results
"""

import numpy as np
import cv2
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import MatchingPipeline, load_config
from src.utils.visualization import visualize_matches, plot_error_statistics, create_quality_report


def create_sample_data():
    """
    Create sample data for demonstration.
    In practice, you would load actual satellite/UAV images and initial matches.
    """
    # Create synthetic images (replace with actual image loading)
    img_sat = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    img_uav = np.random.randint(0, 256, (4000, 4000), dtype=np.uint8)

    # Add some texture
    img_sat = cv2.GaussianBlur(img_sat, (5, 5), 1.0)
    img_uav = cv2.GaussianBlur(img_uav, (5, 5), 1.0)

    # Sample initial matches (replace with actual feature matching)
    n_points = 200
    points_sat = np.random.rand(n_points, 2) * 800 + 100

    # Simulate transformation with some noise
    scale = 4.0  # UAV has higher resolution
    points_uav = points_sat * scale + np.random.randn(n_points, 2) * 3.0

    # IMU data (replace with actual IMU readings)
    imu_data = {
        "roll": 0.01,   # ~0.57 degrees
        "pitch": 0.02,  # ~1.15 degrees
        "yaw": 1.57,    # ~90 degrees
        "altitude": 100.0  # meters
    }

    return img_sat, img_uav, points_sat, points_uav, imu_data


def main():
    """Main execution function."""
    print("UAV-Satellite High-Precision Matching Pipeline")
    print("=" * 70)

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'matching_config.yaml')

    if os.path.exists(config_path):
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
    else:
        print("Configuration file not found, using defaults")
        config = create_default_config()

    # Load or create sample data
    print("\nLoading data...")
    img_sat, img_uav, points_sat, points_uav, imu_data = create_sample_data()

    print(f"  Satellite image: {img_sat.shape}")
    print(f"  UAV image: {img_uav.shape}")
    print(f"  Initial matches: {len(points_sat)}")
    print(f"  IMU data: roll={np.degrees(imu_data['roll']):.2f}°, "
          f"pitch={np.degrees(imu_data['pitch']):.2f}°, "
          f"yaw={np.degrees(imu_data['yaw']):.2f}°, "
          f"altitude={imu_data['altitude']:.1f}m")

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = MatchingPipeline(config)

    # Run matching
    print("\nRunning matching pipeline...\n")
    results = pipeline.run(
        img_sat=img_sat,
        img_uav=img_uav,
        points_sat=points_sat,
        points_uav=points_uav,
        imu_data=imu_data,
        verbose=True
    )

    # Save results
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("\nSaving results...")

    # Save in different formats
    pipeline.save_results(results, os.path.join(output_dir, "matches.npz"), format="npz")
    pipeline.save_results(results, os.path.join(output_dir, "matches.csv"), format="csv")
    pipeline.save_results(results, os.path.join(output_dir, "matches.json"), format="json")

    print(f"  Saved to: {output_dir}/matches.[npz|csv|json]")

    # Visualize results (if matplotlib available)
    try:
        print("\nGenerating visualizations...")

        # Visualize final matches
        visualize_matches(
            img_sat, img_uav,
            results["final"]["points_sat"],
            results["final"]["points_uav"],
            confidence=results["final"]["confidence"],
            title="Final High-Precision Matches",
            save_path=os.path.join(output_dir, "matches_visualization.png")
        )

        print(f"  Visualization saved to: {output_dir}/matches_visualization.png")

        # Create quality report
        report = create_quality_report(
            results["final"]["points_sat"],
            results["final"]["points_uav"],
            results["final"]["confidence"],
            results["final"]["uncertainty"] if results["final"]["uncertainty"] is not None else np.ones(len(results["final"]["points_sat"])),
            stage_name="Final Pipeline Output",
            save_path=os.path.join(output_dir, "quality_report.json")
        )

        print(f"  Quality report saved to: {output_dir}/quality_report.json")
        print("\nQuality Report Summary:")
        print(f"  Total matches: {report['n_matches']}")
        print(f"  High quality ratio: {report['high_quality_ratio']:.2%}")
        print(f"  Mean confidence: {report['confidence_mean']:.3f}")
        print(f"  Mean uncertainty: {report['uncertainty_mean']:.3f} pixels")

    except ImportError:
        print("\nMatplotlib not available, skipping visualization")

    print("\n" + "=" * 70)
    print("Pipeline execution completed successfully!")
    print("=" * 70)


def create_default_config():
    """Create default configuration."""
    return {
        "camera": {
            "intrinsic_matrix": [
                [3000.0, 0, 2000.0],
                [0, 3000.0, 1500.0],
                [0, 0, 1.0]
            ]
        },
        "imu": {
            "roll_std": 0.01,
            "pitch_std": 0.01,
            "yaw_std": 0.05,
            "altitude_std": 2.0
        },
        "stage2": {
            "optical_flow": {
                "window_size": 21,
                "pyramid_levels": 3,
                "max_iterations": 30
            },
            "bayesian_optimization": {
                "max_iterations": 50
            }
        },
        "stage3": {
            "factor_graph": {
                "visual_noise": 1.5,
                "smooth_noise": 0.5,
                "neighbor_radius": 40,
                "k_neighbors": 5,
                "max_iterations": 100
            }
        },
        "stage5": {
            "template": {
                "size_min": 24,
                "size_max": 40,
                "adaptive": True
            },
            "search": {
                "radius_base": 5
            },
            "ncc_threshold": 0.7,
            "subpixel": {
                "method": "quadratic"
            },
            "iteration": {
                "max_iterations": 3
            }
        },
        "output": {
            "cross_validation": False
        }
    }


if __name__ == "__main__":
    main()
