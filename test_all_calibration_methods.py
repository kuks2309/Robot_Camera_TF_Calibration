#!/usr/bin/env python3
"""
Test all hand-eye calibration methods to find the best one.
Uses fixed Euler order (xyz).
"""

import cv2
import numpy as np
import os
import json
import yaml
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple


def load_images_and_yaml(images_dir: str) -> List:
    """Load images and corresponding YAML files directly from images directory."""
    import yaml as yaml_module

    tcp_poses = []
    yaml_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.yaml')])

    for yaml_file in yaml_files:
        yaml_path = os.path.join(images_dir, yaml_file)
        with open(yaml_path, 'r') as f:
            data = yaml_module.safe_load(f)

        if data and 'tcp_pose' in data:
            pose_data = {
                'pose_id': data['id'],
                'timestamp': data.get('timestamp', ''),
                'tcp_pose': data['tcp_pose'],
                'image_file': os.path.join(images_dir, data['image_file']),
                'image_relative_path': data['image_file']
            }
            tcp_poses.append(pose_data)

    return tcp_poses


class ChessboardDetector:
    """Detects standard chessboard pattern and estimates pose relative to camera."""

    def __init__(self, pattern_size: Tuple[int, int] = (10, 7), square_size: float = 0.022):
        self.pattern_size = pattern_size
        self.square_size = square_size

        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def detect_and_estimate_pose(self, image: np.ndarray, camera_matrix: np.ndarray,
                                 dist_coeffs: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

        if not ret:
            return None

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        success, rvec, tvec = cv2.solvePnP(self.objp, corners_refined, camera_matrix, dist_coeffs)

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        return {
            'corners': corners_refined,
            'rvec': rvec,
            'tvec': tvec,
            'rotation_matrix': rotation_matrix,
            'num_corners': len(corners_refined)
        }


def test_calibration_method(method_name: str, method_code: int,
                           tcp_poses: List, camera_matrix: np.ndarray,
                           dist_coeffs: np.ndarray, detector,
                           euler_order: str = 'xyz') -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Test calibration with a specific method."""
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for pose_data in tcp_poses:
        img_path = pose_data['image_file']
        image = cv2.imread(img_path)
        if image is None:
            continue

        detection = detector.detect_and_estimate_pose(image, camera_matrix, dist_coeffs)
        if detection is None:
            continue

        tcp = pose_data['tcp_pose']
        x_m = tcp['x_mm'] / 1000.0
        y_m = tcp['y_mm'] / 1000.0
        z_m = tcp['z_mm'] / 1000.0

        try:
            R_tcp = R.from_euler(euler_order, [tcp['rx_deg'], tcp['ry_deg'], tcp['rz_deg']],
                                degrees=True).as_matrix()
            t_tcp = np.array([[x_m], [y_m], [z_m]], dtype=np.float64)

            R_gripper2base.append(R_tcp)
            t_gripper2base.append(t_tcp)

            R_board = detection['rotation_matrix'].astype(np.float64)
            t_board = detection['tvec'].reshape(3, 1).astype(np.float64)

            R_target2cam.append(R_board)
            t_target2cam.append(t_board)
        except Exception as e:
            continue

    if len(R_gripper2base) < 3:
        return float('inf'), float('inf'), None, None

    # Compute calibration
    try:
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=method_code
        )
    except Exception as e:
        print(f"  Calibration failed: {e}")
        return float('inf'), float('inf'), None, None

    # Compute validation error
    if len(R_gripper2base) < 2:
        return float('inf'), float('inf'), R_cam2gripper, t_cam2gripper

    errors = []

    # Use first pose as reference
    R_gripper_ref = R_gripper2base[0]
    t_gripper_ref = t_gripper2base[0]
    R_target_ref = R_target2cam[0]
    t_target_ref = t_target2cam[0]

    R_gripper_ref_inv = R_gripper_ref.T
    t_gripper_ref_inv = -R_gripper_ref_inv @ t_gripper_ref
    R_target_ref_inv = R_target_ref.T
    t_target_ref_inv = -R_target_ref_inv @ t_target_ref

    for i in range(1, len(R_gripper2base)):
        # A_i = T_gripper[i] * inv(T_gripper[0])
        R_A = R_gripper2base[i] @ R_gripper_ref_inv
        t_A = R_gripper2base[i] @ t_gripper_ref_inv + t_gripper2base[i]

        # B_i = T_target[i] * inv(T_target[0])
        R_B = R_target2cam[i] @ R_target_ref_inv
        t_B = R_target2cam[i] @ t_target_ref_inv + t_target2cam[i]

        # Left: A_i * X
        R_left = R_A @ R_cam2gripper
        t_left = R_A @ t_cam2gripper + t_A

        # Right: X * B_i
        R_right = R_cam2gripper @ R_B
        t_right = R_cam2gripper @ t_B + t_cam2gripper

        # Rotation error
        R_error = R_left.T @ R_right
        angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        angle_error_deg = np.degrees(angle_error)

        # Translation error
        t_error = np.linalg.norm(t_left - t_right)

        errors.append((angle_error_deg, t_error))

    avg_rot_error = np.mean([e[0] for e in errors])
    avg_trans_error = np.mean([e[1] for e in errors])

    return avg_rot_error, avg_trans_error, R_cam2gripper, t_cam2gripper


def main():
    print("\n" + "="*80)
    print("Testing All Hand-Eye Calibration Methods")
    print("="*80)

    images_dir = "calibration_data/images"

    if not os.path.exists(images_dir):
        print(f"\n❌ Images directory not found: {images_dir}")
        return 1

    # Load data
    print(f"\nLoading data from: {images_dir}")
    tcp_poses = load_images_and_yaml(images_dir)

    if not tcp_poses:
        print("\n❌ No valid YAML files found")
        return 1

    print(f"✅ Loaded {len(tcp_poses)} poses from YAML files")

    # Load camera intrinsics from YAML file
    camera_intrinsics_file = 'camera_intrinsics.yaml'

    if os.path.exists(camera_intrinsics_file):
        print(f"\n📂 Loading camera intrinsics from: {camera_intrinsics_file}")
        with open(camera_intrinsics_file, 'r') as f:
            cam_config = yaml.safe_load(f)
        camera_intrinsics = cam_config['camera_intrinsics']
        print(f"   fx={camera_intrinsics['fx']:.2f}, fy={camera_intrinsics['fy']:.2f}")
        print(f"   cx={camera_intrinsics['ppx']:.2f}, cy={camera_intrinsics['ppy']:.2f}")
    else:
        print(f"\n⚠️  Camera intrinsics file not found: {camera_intrinsics_file}")
        print("   Using default values")
        camera_intrinsics = {
            'width': 1920, 'height': 1080,
            'fx': 1380.0, 'fy': 1380.0,
            'ppx': 960.0, 'ppy': 540.0,
            'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
        }

    # Load config for chessboard
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    squares = tuple(config['chessboard']['squares'])
    pattern_size = (squares[0] - 1, squares[1] - 1)
    square_size_m = config['chessboard']['square_size_mm'] / 1000.0

    detector = ChessboardDetector(pattern_size=pattern_size, square_size=square_size_m)
    print(f"Using Chessboard detector ({squares[0]}x{squares[1]} squares)")

    camera_matrix = np.array([
        [camera_intrinsics['fx'], 0, camera_intrinsics['ppx']],
        [0, camera_intrinsics['fy'], camera_intrinsics['ppy']],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array(camera_intrinsics['coeffs'], dtype=np.float32)

    # Test all calibration methods
    methods = {
        'Tsai': cv2.CALIB_HAND_EYE_TSAI,
        'Park': cv2.CALIB_HAND_EYE_PARK,
        'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
        'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
        'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS
    }

    euler_order = 'xyz'
    print(f"\nUsing Euler order: {euler_order}")
    print(f"\n🔍 Testing {len(methods)} calibration methods...")
    print("="*80)

    results = []

    for method_name, method_code in methods.items():
        print(f"\nTesting: {method_name}")

        rot_error, trans_error, R_cam2gripper, t_cam2gripper = test_calibration_method(
            method_name, method_code, tcp_poses, camera_matrix, dist_coeffs, detector, euler_order
        )

        if rot_error != float('inf'):
            print(f"  ✓ Rotation error: {rot_error:.4f}°")
            print(f"  ✓ Translation error: {trans_error*1000:.4f} mm")

            results.append({
                'method': method_name,
                'rotation_error_deg': float(rot_error),
                'translation_error_mm': float(trans_error * 1000),
                'R_cam2gripper': R_cam2gripper.tolist() if R_cam2gripper is not None else None,
                't_cam2gripper': t_cam2gripper.tolist() if t_cam2gripper is not None else None
            })
        else:
            print(f"  ✗ Failed")

    # Sort by rotation error
    results.sort(key=lambda x: x['rotation_error_deg'])

    # Display results
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY (sorted by rotation error)")
    print("="*80)
    print(f"\n{'Rank':<6}{'Method':<15}{'Rotation Error':<20}{'Translation Error':<20}")
    print("-" * 80)

    for i, result in enumerate(results, 1):
        print(f"{i:<6}{result['method']:<15}{result['rotation_error_deg']:>8.4f}°{'':<11}"
              f"{result['translation_error_mm']:>8.4f} mm")

    # Save results
    results_file = os.path.join(images_dir, "calibration_methods_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Highlight best result
    if results:
        best = results[0]
        print("\n" + "="*80)
        print("🏆 BEST RESULT")
        print("="*80)
        print(f"Method: {best['method']}")
        print(f"Rotation error: {best['rotation_error_deg']:.4f}°")
        print(f"Translation error: {best['translation_error_mm']:.4f} mm")

        if best['rotation_error_deg'] < 5.0 and best['translation_error_mm'] < 10.0:
            print("\n✅ EXCELLENT - This calibration should work well!")
        elif best['rotation_error_deg'] < 10.0 and best['translation_error_mm'] < 50.0:
            print("\n✓ GOOD - This calibration is acceptable for most applications")
        else:
            print("\n⚠️  POOR - Consider reviewing your setup")
            print("\nPossible issues:")
            print("  1. Robot TCP coordinates may not match camera position")
            print("  2. Chessboard size incorrect (check config.yaml)")
            print("  3. Camera intrinsics inaccurate (run camera calibration)")
            print("  4. Pose variety insufficient (need diverse angles/distances)")

    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
