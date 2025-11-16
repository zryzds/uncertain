"""
Complete Integration Pipeline

Combines all stages (Stage 2, 3, and 5) into a unified high-precision
UAV-satellite image matching pipeline.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import time

from .stage2_imu_constraint import IMUConstraintOptimizer
from .stage3_factor_graph import build_and_optimize_factor_graph
from .stage5_subpixel_matching import SubpixelMatcher, cross_validation_check
from .utils.camera_utils import CameraModel
from .utils.imu_utils import IMUProcessor


class MatchingPipeline:
    """Complete high-precision UAV-satellite image matching pipeline."""

    def __init__(self, config: Dict):
        """
        Initialize matching pipeline.

        Args:
            config: Configuration dictionary containing all parameters
        """
        self.config = config

        # Initialize camera model
        camera_config = config.get("camera", {})
        self.camera = CameraModel(np.array(camera_config.get("intrinsic_matrix")))

        # Initialize IMU processor
        imu_config = config.get("imu", {})
        self.imu_processor = IMUProcessor(
            roll_std=imu_config.get("roll_std", 0.01),
            pitch_std=imu_config.get("pitch_std", 0.01),
            yaw_std=imu_config.get("yaw_std", 0.05),
            altitude_std=imu_config.get("altitude_std", 2.0)
        )

        # Initialize Stage 2 optimizer
        stage2_config = config.get("stage2", {})
        self.stage2_optimizer = IMUConstraintOptimizer(
            camera_matrix=self.camera.K,
            imu_processor=self.imu_processor,
            optical_flow_window=stage2_config.get("optical_flow", {}).get("window_size", 21),
            optical_flow_levels=stage2_config.get("optical_flow", {}).get("pyramid_levels", 3),
            optical_flow_iters=stage2_config.get("optical_flow", {}).get("max_iterations", 30),
            max_bayesian_iters=stage2_config.get("bayesian_optimization", {}).get("max_iterations", 50)
        )

        # Initialize Stage 5 matcher
        stage5_config = config.get("stage5", {})
        self.stage5_matcher = SubpixelMatcher(
            template_size_min=stage5_config.get("template", {}).get("size_min", 24),
            template_size_max=stage5_config.get("template", {}).get("size_max", 40),
            adaptive_template=stage5_config.get("template", {}).get("adaptive", True),
            search_radius_base=stage5_config.get("search", {}).get("radius_base", 5),
            ncc_threshold=stage5_config.get("ncc_threshold", 0.7),
            subpixel_method=stage5_config.get("subpixel", {}).get("method", "quadratic"),
            max_iterations=stage5_config.get("iteration", {}).get("max_iterations", 3)
        )

        # Store stage 3 config
        self.stage3_config = config.get("stage3", {})

    def run(
        self,
        img_sat: np.ndarray,
        img_uav: np.ndarray,
        points_sat: np.ndarray,
        points_uav: np.ndarray,
        imu_data: Dict,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete matching pipeline.

        Args:
            img_sat: Satellite image (grayscale or RGB)
            img_uav: UAV image (grayscale or RGB)
            points_sat: Nx2 initial satellite points
            points_uav: Nx2 initial UAV points
            imu_data: Dictionary containing IMU data:
                - roll: Roll angle (radians)
                - pitch: Pitch angle (radians)
                - yaw: Yaw angle (radians)
                - altitude: Flight altitude (meters)
            verbose: Whether to print progress

        Returns:
            Dictionary containing final results and intermediate outputs
        """
        results = {}
        timing = {}

        # Convert to grayscale if needed
        if len(img_sat.shape) == 3:
            img_sat_gray = cv2.cvtColor(img_sat, cv2.COLOR_BGR2GRAY)
        else:
            img_sat_gray = img_sat

        if len(img_uav.shape) == 3:
            img_uav_gray = cv2.cvtColor(img_uav, cv2.COLOR_BGR2GRAY)
        else:
            img_uav_gray = img_uav

        # ===== Stage 2: IMU-Guided Geometric Constraint Optimization =====
        if verbose:
            print("=" * 60)
            print("Stage 2: IMU-Guided Geometric Constraint Optimization")
            print("=" * 60)

        start_time = time.time()

        stage2_results = self.stage2_optimizer.refine_matches(
            img_sat_gray,
            img_uav_gray,
            points_sat,
            points_uav,
            roll=imu_data["roll"],
            pitch=imu_data["pitch"],
            yaw=imu_data["yaw"],
            altitude=imu_data["altitude"]
        )

        timing["stage2"] = time.time() - start_time

        if verbose:
            print(f"  Valid points: {stage2_results['n_valid']} / {len(points_sat)}")
            print(f"  Mean error: {stage2_results['mean_error']:.3f} pixels")
            print(f"  Median error: {stage2_results['median_error']:.3f} pixels")
            print(f"  Time: {timing['stage2']:.2f}s")

        results["stage2"] = stage2_results

        # Filter to valid points
        valid_mask = stage2_results["status"]
        points_sat_valid = stage2_results["points_sat"][valid_mask]
        points_uav_stage2 = stage2_results["points_uav_refined"][valid_mask]
        weights_stage2 = stage2_results["weights"][valid_mask]

        if len(points_sat_valid) < 10:
            print("Warning: Too few valid points after Stage 2, aborting")
            return results

        # ===== Stage 3: Factor Graph Multi-Modal Fusion =====
        if verbose:
            print("\n" + "=" * 60)
            print("Stage 3: Factor Graph Multi-Modal Fusion")
            print("=" * 60)

        start_time = time.time()

        stage3_results = build_and_optimize_factor_graph(
            points_sat_valid,
            points_uav_stage2,
            stage2_results["H_optimized"],
            stage2_results["Sigma_H"],
            confidence=weights_stage2,
            config=self.stage3_config.get("factor_graph", {})
        )

        timing["stage3"] = time.time() - start_time

        if verbose:
            print(f"  Optimized points: {len(stage3_results['points_uav_optimized'])}")
            print(f"  Inliers: {stage3_results['n_valid']} / {len(stage3_results['points_uav_optimized'])}")
            print(f"  Optimization iterations: {stage3_results['optimization_info']['n_iterations']}")
            print(f"  Final cost: {stage3_results['optimization_info']['cost']:.6f}")
            print(f"  Time: {timing['stage3']:.2f}s")

        results["stage3"] = stage3_results

        # Filter to inliers
        inlier_mask = stage3_results["inliers"]
        points_sat_inliers = stage3_results["points_sat"][inlier_mask]
        points_uav_stage3 = stage3_results["points_uav_optimized"][inlier_mask]
        uncertainties_stage3 = stage3_results["uncertainties"][inlier_mask]

        if len(points_sat_inliers) < 10:
            print("Warning: Too few inliers after Stage 3, aborting")
            return results

        # ===== Stage 5: Subpixel Template Matching =====
        if verbose:
            print("\n" + "=" * 60)
            print("Stage 5: Subpixel Template Matching")
            print("=" * 60)

        start_time = time.time()

        stage5_results = self.stage5_matcher.refine_matches(
            img_sat_gray,
            img_uav_gray,
            points_sat_inliers,
            points_uav_stage3,
            uncertainties=uncertainties_stage3
        )

        timing["stage5"] = time.time() - start_time

        if verbose:
            print(f"  Refined points: {stage5_results['n_valid']} / {len(points_sat_inliers)}")
            print(f"  High quality: {stage5_results['n_high_quality']}")
            print(f"  Mean confidence: {stage5_results['mean_confidence']:.3f}")
            print(f"  Mean NCC: {np.mean(stage5_results['ncc_values'][stage5_results['status']]):.3f}")
            print(f"  Time: {timing['stage5']:.2f}s")

        results["stage5"] = stage5_results

        # ===== Cross-Validation (Optional) =====
        if self.config.get("output", {}).get("cross_validation", False):
            if verbose:
                print("\n" + "=" * 60)
                print("Cross-Validation Check")
                print("=" * 60)

            start_time = time.time()

            consistent = cross_validation_check(
                img_sat_gray,
                img_uav_gray,
                stage5_results["points_sat"],
                stage5_results["points_uav_refined"],
                self.stage5_matcher
            )

            timing["cross_validation"] = time.time() - start_time

            if verbose:
                print(f"  Consistent matches: {np.sum(consistent)} / {len(consistent)}")
                print(f"  Time: {timing['cross_validation']:.2f}s")

            results["cross_validation"] = consistent
        else:
            consistent = np.ones(len(stage5_results["points_sat"]), dtype=bool)

        # ===== Final Output =====
        # Combine all quality criteria
        final_mask = (
            stage5_results["status"] &
            stage5_results["high_quality_mask"] &
            consistent
        )

        final_points_sat = stage5_results["points_sat"][final_mask]
        final_points_uav = stage5_results["points_uav_refined"][final_mask]
        final_confidence = stage5_results["confidence"][final_mask]
        final_uncertainty = uncertainties_stage3[final_mask] if len(uncertainties_stage3) == len(final_mask) else None

        results["final"] = {
            "points_sat": final_points_sat,
            "points_uav": final_points_uav,
            "confidence": final_confidence,
            "uncertainty": final_uncertainty,
            "n_matches": len(final_points_sat)
        }

        results["timing"] = timing
        results["timing"]["total"] = sum(timing.values())

        if verbose:
            print("\n" + "=" * 60)
            print("Final Results")
            print("=" * 60)
            print(f"  Total matches: {results['final']['n_matches']}")
            print(f"  Mean confidence: {np.mean(final_confidence):.3f}")
            print(f"  Total time: {results['timing']['total']:.2f}s")
            print("=" * 60)

        return results

    def save_results(self, results: Dict, output_path: str, format: str = "npz"):
        """
        Save results to file.

        Args:
            results: Results dictionary from run()
            output_path: Output file path
            format: Output format ('npz', 'csv', or 'json')
        """
        if format == "npz":
            np.savez(
                output_path,
                points_sat=results["final"]["points_sat"],
                points_uav=results["final"]["points_uav"],
                confidence=results["final"]["confidence"],
                uncertainty=results["final"]["uncertainty"]
            )

        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "id", "x_sat", "y_sat", "x_uav", "y_uav",
                    "confidence", "uncertainty", "status"
                ])

                for i in range(len(results["final"]["points_sat"])):
                    writer.writerow([
                        i,
                        results["final"]["points_sat"][i, 0],
                        results["final"]["points_sat"][i, 1],
                        results["final"]["points_uav"][i, 0],
                        results["final"]["points_uav"][i, 1],
                        results["final"]["confidence"][i],
                        results["final"]["uncertainty"][i] if results["final"]["uncertainty"] is not None else 0.0,
                        "success"
                    ])

        elif format == "json":
            import json
            output_dict = {
                "n_matches": int(results["final"]["n_matches"]),
                "matches": []
            }

            for i in range(len(results["final"]["points_sat"])):
                match = {
                    "id": i,
                    "sat": results["final"]["points_sat"][i].tolist(),
                    "uav": results["final"]["points_uav"][i].tolist(),
                    "confidence": float(results["final"]["confidence"][i]),
                    "uncertainty": float(results["final"]["uncertainty"][i]) if results["final"]["uncertainty"] is not None else None
                }
                output_dict["matches"].append(match)

            with open(output_path, 'w') as f:
                json.dump(output_dict, f, indent=2)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
