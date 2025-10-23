#!/usr/bin/env python3
"""
Calibration Computation Script

This script processes collected calibration data to compute the camera-to-TCP
transformation matrix using hand-eye calibration.

Input:
    calibration_data/session_YYYYMMDD_HHMMSS/
    ├── tcp_poses.json
    ├── images/pose_*.jpg
    └── metadata.json

Output:
    (added to same directory)
    ├── calibration_result.json
    ├── camera_to_tcp_transform.npy
    └── detected_poses/   # Images with board detection visualization
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime
from typing import Optional, List, Tuple, Dict
import json
from scipy.spatial.transform import Rotation as R


class ChessboardDetector:
    """Detects standard chessboard pattern and estimates pose relative to camera."""

    def __init__(self,
                 pattern_size: Tuple[int, int] = (9, 6),
                 square_size: float = 0.025):  # 25mm squares
        """
        Initialize Chessboard detector.

        Args:
            pattern_size: (width, height) number of INTERNAL corners (not squares!)
            square_size: Size of chessboard squares in meters
        """
        self.pattern_size = pattern_size
        self.square_size = square_size

        # Create 3D object points for the chessboard
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def detect_and_estimate_pose(self,
                                 image: np.ndarray,
                                 camera_matrix: np.ndarray,
                                 dist_coeffs: np.ndarray) -> Optional[Dict]:
        """Detect chessboard and estimate its pose."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

        if not ret:
            return None

        # Refine corner locations to sub-pixel accuracy
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

        # Estimate pose
        success, rvec, tvec = cv2.solvePnP(
            self.objp,
            corners_refined,
            camera_matrix,
            dist_coeffs
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Calculate camera pose in board frame (inverse transformation)
        camera_rotation_in_board = rotation_matrix.T
        camera_position_in_board = -camera_rotation_in_board @ tvec

        return {
            'corners': corners_refined,
            'rvec': rvec,
            'tvec': tvec,
            'rotation_matrix': rotation_matrix,
            'board_position_in_camera': tvec,
            'camera_position_in_board': camera_position_in_board,
            'camera_rotation_in_board': camera_rotation_in_board,
            'num_corners': len(corners_refined)
        }

    def draw_detection(self,
                      image: np.ndarray,
                      detection_result: Dict,
                      camera_matrix: np.ndarray,
                      dist_coeffs: np.ndarray) -> np.ndarray:
        """Draw detected board and coordinate frame on image."""
        img_copy = image.copy()

        # Draw detected corners
        cv2.drawChessboardCorners(img_copy, self.pattern_size,
                                 detection_result['corners'], True)

        # Draw coordinate frame
        cv2.drawFrameAxes(
            img_copy,
            camera_matrix,
            dist_coeffs,
            detection_result['rvec'],
            detection_result['tvec'],
            0.1  # Axis length in meters
        )

        # Add text info
        num_corners = detection_result['num_corners']
        cv2.putText(img_copy, f"Corners: {num_corners}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img_copy


class ChArUcoBoardDetector:
    """Detects ChArUco board and estimates pose relative to camera."""

    def __init__(self,
                 grid_size: Tuple[int, int] = (8, 6),
                 square_size: float = 0.0298,  # 30mm squares
                 marker_size: float = 0.02091,  # 21mm markers
                 aruco_dict: int = cv2.aruco.DICT_4X4_50):
        """Initialize ChArUco board detector."""
        self.grid_size = grid_size
        self.square_size = square_size
        self.marker_size = marker_size

        # Create ArUco dictionary and board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.board = cv2.aruco.CharucoBoard(
            grid_size,
            square_size,
            marker_size,
            self.aruco_dict
        )

        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)

    def detect_and_estimate_pose(self,
                                 image: np.ndarray,
                                 camera_matrix: np.ndarray,
                                 dist_coeffs: np.ndarray) -> Optional[Dict]:
        """Detect ChArUco board and estimate its pose."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect ChArUco board
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            self.charuco_detector.detectBoard(gray)

        if charuco_corners is None or len(charuco_corners) < 4:
            return None

        # Estimate pose
        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            self.board,
            camera_matrix,
            dist_coeffs,
            None,
            None
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Calculate camera pose in board frame (inverse transformation)
        camera_rotation_in_board = rotation_matrix.T
        camera_position_in_board = -camera_rotation_in_board @ tvec

        return {
            'charuco_corners': charuco_corners,
            'charuco_ids': charuco_ids,
            'marker_corners': marker_corners,
            'marker_ids': marker_ids,
            'rvec': rvec,
            'tvec': tvec,
            'rotation_matrix': rotation_matrix,
            'board_position_in_camera': tvec,
            'camera_position_in_board': camera_position_in_board,
            'camera_rotation_in_board': camera_rotation_in_board,
            'num_corners': len(charuco_corners)
        }

    def draw_detection(self,
                      image: np.ndarray,
                      detection_result: Dict,
                      camera_matrix: np.ndarray,
                      dist_coeffs: np.ndarray) -> np.ndarray:
        """Draw detected board and coordinate frame on image."""
        img_copy = image.copy()

        # Draw detected ChArUco corners
        if detection_result['charuco_corners'] is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                img_copy,
                detection_result['charuco_corners'],
                detection_result['charuco_ids']
            )

        # Draw ArUco markers
        if detection_result['marker_corners'] is not None:
            cv2.aruco.drawDetectedMarkers(
                img_copy,
                detection_result['marker_corners'],
                detection_result['marker_ids']
            )

        # Draw coordinate frame
        cv2.drawFrameAxes(
            img_copy,
            camera_matrix,
            dist_coeffs,
            detection_result['rvec'],
            detection_result['tvec'],
            0.1  # Axis length in meters
        )

        # Add text info
        num_corners = detection_result['num_corners']
        cv2.putText(img_copy, f"Corners: {num_corners}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img_copy


class HandEyeCalibration:
    """Performs hand-eye calibration using collected data."""

    def __init__(self, calibration_type: int = cv2.CALIB_HAND_EYE_TSAI):
        """Initialize hand-eye calibration solver."""
        self.calibration_type = calibration_type
        self.method_names = {
            cv2.CALIB_HAND_EYE_TSAI: "Tsai",
            cv2.CALIB_HAND_EYE_PARK: "Park",
            cv2.CALIB_HAND_EYE_HORAUD: "Horaud",
            cv2.CALIB_HAND_EYE_ANDREFF: "Andreff",
            cv2.CALIB_HAND_EYE_DANIILIDIS: "Daniilidis"
        }

    def compute_calibration(self,
                           R_gripper2base: List[np.ndarray],
                           t_gripper2base: List[np.ndarray],
                           R_target2cam: List[np.ndarray],
                           t_target2cam: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute hand-eye calibration (eye-in-hand)."""
        print(f"\n🔧 Computing hand-eye calibration using {self.method_names[self.calibration_type]} method...")
        print(f"   Number of poses: {len(R_gripper2base)}")

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=self.calibration_type
        )

        return R_cam2gripper, t_cam2gripper

    @staticmethod
    def compute_reprojection_error(R_cam2gripper: np.ndarray,
                                   t_cam2gripper: np.ndarray,
                                   R_gripper2base: List[np.ndarray],
                                   t_gripper2base: List[np.ndarray],
                                   R_target2cam: List[np.ndarray],
                                   t_target2cam: List[np.ndarray]) -> float:
        """Compute average reprojection error for calibration validation."""
        errors = []

        for i in range(len(R_gripper2base)):
            # Left side: T_gripper2base * T_cam2gripper
            R_left = R_gripper2base[i] @ R_cam2gripper
            t_left = R_gripper2base[i] @ t_cam2gripper + t_gripper2base[i]

            # Right side: T_cam2gripper * T_target2cam
            R_right = R_cam2gripper @ R_target2cam[i]
            t_right = R_cam2gripper @ t_target2cam[i] + t_cam2gripper

            # Compute rotation error (angle between two rotations)
            R_error = R_left.T @ R_right
            angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
            angle_error_deg = np.degrees(angle_error)

            # Compute translation error
            t_error = np.linalg.norm(t_left - t_right)

            errors.append((angle_error_deg, t_error))

        avg_rot_error = np.mean([e[0] for e in errors])
        avg_trans_error = np.mean([e[1] for e in errors])

        print(f"\n📊 Calibration Validation:")
        print(f"   Average rotation error: {avg_rot_error:.4f} degrees")
        print(f"   Average translation error: {avg_trans_error*1000:.4f} mm")

        return avg_rot_error


def load_session_data(session_dir: str) -> Tuple[List, Dict, Dict]:
    """
    Load collected data from session directory.

    Returns:
        Tuple of (tcp_poses, metadata, camera_intrinsics)
    """
    # Load TCP poses
    poses_file = os.path.join(session_dir, "tcp_poses.json")
    with open(poses_file, 'r') as f:
        tcp_poses = json.load(f)

    # Load metadata
    metadata_file = os.path.join(session_dir, "metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    camera_intrinsics = metadata['camera_intrinsics']

    return tcp_poses, metadata, camera_intrinsics


def process_calibration(session_dir: str, board_type: str):
    """
    Process calibration data and compute transformation matrix.

    Args:
        session_dir: Path to session directory
        board_type: "charuco" or "chessboard"
    """
    print("\n" + "="*80)
    print("Processing Calibration Data")
    print("="*80)
    print(f"Session: {session_dir}")
    print(f"Board type: {board_type}")

    # Load session data
    print("\n📂 Loading session data...")
    tcp_poses, metadata, camera_intrinsics = load_session_data(session_dir)
    print(f"✅ Loaded {len(tcp_poses)} poses")

    # Create detector based on board type
    if board_type.lower() == "chessboard":
        detector = ChessboardDetector(
            pattern_size=(9, 6),
            square_size=0.025
        )
        board_name = "Chessboard"
    else:  # charuco
        detector = ChArUcoBoardDetector(
            grid_size=(8, 6),
            square_size=0.0298,
            marker_size=0.02091
        )
        board_name = "ChArUco"

    print(f"🎯 Using {board_name} detector")

    # Build camera matrix and distortion coefficients
    camera_matrix = np.array([
        [camera_intrinsics['fx'], 0, camera_intrinsics['ppx']],
        [0, camera_intrinsics['fy'], camera_intrinsics['ppy']],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array(camera_intrinsics['coeffs'], dtype=np.float32)

    # Process each image
    print(f"\n🔍 Detecting {board_name} in images...")
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    detected_poses_dir = os.path.join(session_dir, "detected_poses")
    os.makedirs(detected_poses_dir, exist_ok=True)

    successful_detections = 0

    for pose_data in tcp_poses:
        pose_id = pose_data['pose_id']
        img_path = os.path.join(session_dir, pose_data['image_relative_path'])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️  Pose {pose_id}: Failed to load image {img_path}")
            continue

        # Detect board
        detection = detector.detect_and_estimate_pose(image, camera_matrix, dist_coeffs)

        if detection is None:
            print(f"❌ Pose {pose_id}: Board not detected")
            continue

        # Save visualization
        img_with_detection = detector.draw_detection(image, detection, camera_matrix, dist_coeffs)
        detection_img_path = os.path.join(detected_poses_dir, f"detected_pose_{pose_id:03d}.jpg")
        cv2.imwrite(detection_img_path, img_with_detection)

        # Extract TCP pose
        tcp = pose_data['tcp_pose']
        x_m = tcp['x_mm'] / 1000.0
        y_m = tcp['y_mm'] / 1000.0
        z_m = tcp['z_mm'] / 1000.0

        R_tcp = R.from_euler('xyz', [tcp['rx_deg'], tcp['ry_deg'], tcp['rz_deg']],
                            degrees=True).as_matrix()
        t_tcp = np.array([[x_m], [y_m], [z_m]], dtype=np.float64)

        R_gripper2base.append(R_tcp)
        t_gripper2base.append(t_tcp)

        # Extract board pose
        R_board = detection['rotation_matrix'].astype(np.float64)
        t_board = detection['tvec'].reshape(3, 1).astype(np.float64)

        R_target2cam.append(R_board)
        t_target2cam.append(t_board)

        successful_detections += 1
        print(f"✅ Pose {pose_id}: Board detected ({detection['num_corners']} features)")

    print(f"\n📊 Detection Summary:")
    print(f"   Total poses: {len(tcp_poses)}")
    print(f"   Successful detections: {successful_detections}")
    print(f"   Failed detections: {len(tcp_poses) - successful_detections}")

    if successful_detections < 3:
        print(f"\n❌ Insufficient successful detections (need at least 3, have {successful_detections})")
        print("   Please collect more data or check calibration board visibility")
        return None

    # Compute hand-eye calibration
    calibrator = HandEyeCalibration()
    R_cam2gripper, t_cam2gripper = calibrator.compute_calibration(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
    )

    # Compute validation error
    avg_error = calibrator.compute_reprojection_error(
        R_cam2gripper, t_cam2gripper,
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
    )

    # Build calibration result
    calibration_result = {
        'timestamp': datetime.now().isoformat(),
        'session_dir': session_dir,
        'board_type': board_type,
        'num_poses_collected': len(tcp_poses),
        'num_poses_used': successful_detections,
        'calibration_method': calibrator.method_names[calibrator.calibration_type],
        'R_cam2gripper': R_cam2gripper.tolist(),
        't_cam2gripper': t_cam2gripper.tolist(),
        'avg_rotation_error_deg': float(avg_error),
        'camera_to_tcp_transformation': {
            'rotation_matrix': R_cam2gripper.tolist(),
            'translation_vector_m': t_cam2gripper.flatten().tolist(),
            'euler_angles_deg': R.from_matrix(R_cam2gripper).as_euler('xyz', degrees=True).tolist()
        }
    }

    print("\n✅ Calibration computed successfully")
    print(f"\n📊 Camera-to-TCP Transformation:")
    print(f"   Translation (m): {t_cam2gripper.flatten()}")
    print(f"   Euler angles (deg): {calibration_result['camera_to_tcp_transformation']['euler_angles_deg']}")
    print(f"   Average error: {avg_error:.4f} degrees")

    # Save results
    print(f"\n💾 Saving calibration results...")

    # Save JSON result
    result_file = os.path.join(session_dir, "calibration_result.json")
    with open(result_file, 'w') as f:
        json.dump(calibration_result, f, indent=2)
    print(f"   Saved: {result_file}")

    # Save transformation matrix in numpy format
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3:4] = t_cam2gripper

    transform_file = os.path.join(session_dir, "camera_to_tcp_transform.npy")
    np.save(transform_file, T_cam2gripper)
    print(f"   Saved: {transform_file}")

    print(f"\n📁 All results saved to: {session_dir}")

    return calibration_result


def main():
    """Main calibration computation workflow."""
    print("\n" + "="*80)
    print("Robot-Camera Hand-Eye Calibration - Computation")
    print("="*80)

    # Find calibration data directory
    calib_dir = "calibration_data"
    if not os.path.exists(calib_dir):
        print(f"\n❌ Calibration data directory not found: {calib_dir}")
        print("   Please run collect_calibration_data.py first")
        return 1

    # List available sessions
    sessions = [d for d in os.listdir(calib_dir) if d.startswith("session_")]
    if not sessions:
        print(f"\n❌ No calibration sessions found in {calib_dir}")
        print("   Please run collect_calibration_data.py first")
        return 1

    sessions.sort()

    print(f"\nAvailable calibration sessions:")
    for i, session in enumerate(sessions, 1):
        session_path = os.path.join(calib_dir, session)
        # Check if already processed
        if os.path.exists(os.path.join(session_path, "calibration_result.json")):
            status = " [PROCESSED]"
        else:
            status = ""
        print(f"  {i}. {session}{status}")

    # Select session
    if len(sessions) == 1:
        selected_idx = 0
        print(f"\n✓ Auto-selected: {sessions[0]}")
    else:
        while True:
            try:
                choice = input(f"\nSelect session (1-{len(sessions)}) [default: {len(sessions)}]: ").strip()
                if choice == "":
                    selected_idx = len(sessions) - 1
                    break
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    selected_idx = idx
                    break
                else:
                    print(f"Invalid choice. Enter 1-{len(sessions)}")
            except ValueError:
                print("Invalid input. Enter a number.")

    session_dir = os.path.join(calib_dir, sessions[selected_idx])
    print(f"✓ Selected: {sessions[selected_idx]}")

    # Select board type
    print("\nSelect calibration board type:")
    print("  1. ChArUco (8x6 grid) - Recommended")
    print("  2. Chessboard (9x6 corners)")

    while True:
        choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip()
        if choice == "" or choice == "1":
            board_type = "charuco"
            break
        elif choice == "2":
            board_type = "chessboard"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    print(f"✓ Selected: {board_type}")

    input("\nPress ENTER to start calibration computation...")

    # Process calibration
    try:
        result = process_calibration(session_dir, board_type)

        if result is None:
            return 1

        print("\n" + "="*80)
        print("✅ Hand-Eye Calibration Complete!")
        print("="*80)
        print(f"\nCalibration files saved to:")
        print(f"  {session_dir}/")
        print("\nFiles:")
        print(f"  - calibration_result.json      # Full calibration data")
        print(f"  - camera_to_tcp_transform.npy  # 4x4 transformation matrix")
        print(f"  - detected_poses/              # Visualization images")
        print("\nYou can now use this calibration for:")
        print("  - Converting camera coordinates to robot base coordinates")
        print("  - Visual servoing applications")
        print("  - Object pose estimation in robot frame")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\n❌ Calibration computation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Computation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
