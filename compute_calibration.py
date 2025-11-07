#!/usr/bin/env python3
"""
Hand-Eye Calibration with XYZ Extrinsic Euler angles (confirmed from manual page 344)

Manual states: "Rx → Ry → Rz 순서로" (Rx → Ry → Rz order)
This is XYZ extrinsic rotation order.
"""

import cv2
import numpy as np
import yaml
import glob
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_camera_intrinsics(yaml_path):
    """Load camera intrinsic parameters"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    cam = data['camera_intrinsics']
    camera_matrix = np.array([
        [cam['fx'], 0, cam['ppx']],
        [0, cam['fy'], cam['ppy']],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array(cam['coeffs'], dtype=np.float64)

    return camera_matrix, dist_coeffs


def load_chessboard_config(yaml_path):
    """Load chessboard configuration"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    config = data['chessboard']
    squares = config['squares']
    square_size_mm = config['square_size_mm']
    pattern_size = (squares[0] - 1, squares[1] - 1)

    return pattern_size, square_size_mm


def create_chessboard_object_points(pattern_size, square_size_mm):
    """Create 3D object points for chessboard"""
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm
    return objp


def detect_chessboard_pose(image_path, camera_matrix, dist_coeffs, pattern_size, objp):
    """Detect chessboard and get camera-to-board transformation"""
    img = cv2.imread(image_path)
    if img is None:
        return False, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if not ret:
        return False, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    success, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix,
                                        dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    return success, rvec, tvec


def euler_to_rotation_matrix_intrinsic_xyz(rx_deg, ry_deg, rz_deg):
    """
    Convert Euler angles to rotation matrix using INTRINSIC xyz rotation
    As confirmed in manual page 344: Rx → Ry → Rz 순서 (Tool 좌표계 기준)

    Tool 좌표계는 움직이는 좌표계이므로 intrinsic rotation입니다.

    Args:
        rx_deg, ry_deg, rz_deg: Euler angles in degrees

    Returns:
        3x3 rotation matrix
    """
    r = Rotation.from_euler('xyz', [rx_deg, ry_deg, rz_deg], degrees=True)
    return r.as_matrix()


def load_calibration_data(images_dir, camera_matrix, dist_coeffs, pattern_size, objp):
    """Load all calibration data with xyz intrinsic rotation"""
    data_list = []
    yaml_files = sorted(glob.glob(str(images_dir / "*.yaml")))

    print(f"\nLoading calibration data...")
    for yaml_file in yaml_files:
        yaml_path = Path(yaml_file)

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        if 'tcp_pose' not in yaml_data:
            continue

        tcp = yaml_data['tcp_pose']
        image_file = yaml_data.get('image_file', yaml_path.stem + '.jpg')
        image_path = images_dir / image_file

        success, rvec, tvec = detect_chessboard_pose(
            str(image_path), camera_matrix, dist_coeffs, pattern_size, objp
        )

        if not success:
            print(f"  Skipped {image_file}: chessboard not detected")
            continue

        # Convert TCP Euler angles to rotation matrix (INTRINSIC xyz)
        R_tcp = euler_to_rotation_matrix_intrinsic_xyz(
            tcp['rx_deg'], tcp['ry_deg'], tcp['rz_deg']
        )
        t_tcp = np.array([[tcp['x_mm']], [tcp['y_mm']], [tcp['z_mm']]], dtype=np.float64)

        # Convert board pose to rotation matrix
        R_board, _ = cv2.Rodrigues(rvec)
        t_board = tvec.astype(np.float64)

        data_list.append({
            'id': yaml_data.get('id'),
            'R_gripper2base': R_tcp,
            't_gripper2base': t_tcp,
            'R_target2cam': R_board,
            't_target2cam': t_board
        })
        print(f"  Loaded {image_file}")

    return data_list


def perform_hand_eye_calibration(data_list):
    """Perform hand-eye calibration with all methods"""

    methods = {
        'Tsai': cv2.CALIB_HAND_EYE_TSAI,
        'Park': cv2.CALIB_HAND_EYE_PARK,
        'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
        'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
        'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS
    }

    R_gripper2base = [d['R_gripper2base'] for d in data_list]
    t_gripper2base = [d['t_gripper2base'] for d in data_list]
    R_target2cam = [d['R_target2cam'] for d in data_list]
    t_target2cam = [d['t_target2cam'] for d in data_list]

    results = {}

    print(f"\nPerforming hand-eye calibration with {len(data_list)} poses...")
    for method_name, method_code in methods.items():
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=method_code
            )

            # Compute error
            error = compute_reprojection_error(
                R_cam2gripper, t_cam2gripper,
                data_list
            )

            results[method_name] = {
                'R_cam2gripper': R_cam2gripper,
                't_cam2gripper': t_cam2gripper,
                'error': error
            }

        except Exception as e:
            print(f"  {method_name} failed: {e}")
            results[method_name] = None

    return results


def compute_reprojection_error(R_cam2gripper, t_cam2gripper, data_list):
    """
    Compute reprojection error

    For eye-in-hand: T_gripper2base * T_cam2gripper * T_target2cam = T_target2base (constant)
    """
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3:4] = t_cam2gripper

    # Compute board position in base frame for each pose
    board_in_base = []

    for data in data_list:
        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = data['R_gripper2base']
        T_gripper2base[:3, 3:4] = data['t_gripper2base']

        T_target2cam = np.eye(4)
        T_target2cam[:3, :3] = data['R_target2cam']
        T_target2cam[:3, 3:4] = data['t_target2cam']

        # Board position in base frame
        T_target2base = T_gripper2base @ T_cam2gripper @ T_target2cam
        board_in_base.append(T_target2base[:3, 3])

    board_in_base = np.array(board_in_base)

    # Board should be at same position for all poses
    mean_board = np.mean(board_in_base, axis=0)
    deviations = board_in_base - mean_board
    errors = np.linalg.norm(deviations, axis=1)

    return {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'rms_error': np.sqrt(np.mean(errors**2)),
        'board_positions': board_in_base,
        'mean_position': mean_board
    }


def rotation_matrix_to_euler_xyz(R):
    """Convert rotation matrix to xyz intrinsic Euler angles"""
    r = Rotation.from_matrix(R)
    return r.as_euler('xyz', degrees=True)


def main():
    base_dir = Path(__file__).parent
    intrinsics_path = base_dir / "camera_intrinsics.yaml"
    config_path = base_dir / "config.yaml"
    images_dir = base_dir / "calibration_data" / "images"
    output_dir = images_dir

    print("="*60)
    print("Hand-Eye Calibration (xyz Intrinsic - Manual Page 344)")
    print("="*60)
    print("\nManual confirms: Rx → Ry → Rz rotation order (Tool 좌표계 기준)")
    print("This corresponds to xyz intrinsic rotation")

    # Load parameters
    print("\nLoading parameters...")
    camera_matrix, dist_coeffs = load_camera_intrinsics(intrinsics_path)
    pattern_size, square_size_mm = load_chessboard_config(config_path)
    objp = create_chessboard_object_points(pattern_size, square_size_mm)

    # Load data with XYZ extrinsic rotation
    data_list = load_calibration_data(
        images_dir, camera_matrix, dist_coeffs, pattern_size, objp
    )
    print(f"\nTotal valid poses loaded: {len(data_list)}")

    if len(data_list) < 3:
        print("\nERROR: Not enough poses for calibration (minimum 3 required)")
        return

    # Try all calibration methods
    results = perform_hand_eye_calibration(data_list)

    # Find best result
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}\n")

    best_method = None
    best_error = float('inf')
    best_result = None

    for method_name, result in results.items():
        if result is None:
            print(f"{method_name:12s}: Failed")
        else:
            error = result['error']
            print(f"{method_name:12s}: RMS={error['rms_error']:7.2f} mm, "
                  f"Mean={error['mean_error']:7.2f} mm, Max={error['max_error']:7.2f} mm")

            if error['rms_error'] < best_error:
                best_error = error['rms_error']
                best_result = result
                best_method = method_name

    # Display best result
    if best_result is None:
        print("\nNo successful calibration found!")
        return

    print(f"\n{'='*60}")
    print(f"BEST RESULT: {best_method}")
    print(f"{'='*60}")
    print(f"RMS Error: {best_error:.2f} mm")

    R_cam2gripper = best_result['R_cam2gripper']
    t_cam2gripper = best_result['t_cam2gripper']

    euler = rotation_matrix_to_euler_xyz(R_cam2gripper)

    print(f"\nCamera to TCP Transformation:")
    print(f"\nTranslation (mm):")
    print(f"  X: {t_cam2gripper[0, 0]:8.2f}")
    print(f"  Y: {t_cam2gripper[1, 0]:8.2f}")
    print(f"  Z: {t_cam2gripper[2, 0]:8.2f}")

    print(f"\nRotation (xyz Intrinsic - Rx → Ry → Rz):")
    print(f"  RX: {euler[0]:8.2f} deg")
    print(f"  RY: {euler[1]:8.2f} deg")
    print(f"  RZ: {euler[2]:8.2f} deg")

    print(f"\nRotation Matrix:")
    print(R_cam2gripper)

    # Additional statistics
    error = best_result['error']
    print(f"\n{'='*60}")
    print("Error Statistics")
    print(f"{'='*60}")
    print(f"Mean Error:     {error['mean_error']:8.2f} mm")
    print(f"Std Deviation:  {error['std_error']:8.2f} mm")
    print(f"Max Error:      {error['max_error']:8.2f} mm")
    print(f"RMS Error:      {error['rms_error']:8.2f} mm")

    print(f"\nBoard position in base frame (should be constant):")
    print(f"  Mean: X={error['mean_position'][0]:8.2f}, "
          f"Y={error['mean_position'][1]:8.2f}, Z={error['mean_position'][2]:8.2f} mm")

    # Save result
    output_path = output_dir / "calibration_result.yaml"
    result_data = {
        'camera_to_tcp': {
            'translation_mm': {
                'x': float(t_cam2gripper[0, 0]),
                'y': float(t_cam2gripper[1, 0]),
                'z': float(t_cam2gripper[2, 0])
            },
            'rotation_deg': {
                'rx': float(euler[0]),
                'ry': float(euler[1]),
                'rz': float(euler[2])
            },
            'rotation_matrix': R_cam2gripper.tolist(),
            'euler_order': 'xyz',
            'euler_type': 'intrinsic',
            'method': best_method,
            'manual_reference': 'Page 344: Rx → Ry → Rz rotation order'
        },
        'calibration_quality': {
            'rms_error_mm': float(best_error),
            'mean_error_mm': float(error['mean_error']),
            'std_error_mm': float(error['std_error']),
            'max_error_mm': float(error['max_error']),
            'num_poses': len(data_list)
        },
        'board_position_base_frame': {
            'x_mm': float(error['mean_position'][0]),
            'y_mm': float(error['mean_position'][1]),
            'z_mm': float(error['mean_position'][2])
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(result_data, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
