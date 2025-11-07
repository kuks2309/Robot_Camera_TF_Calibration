#!/usr/bin/env python3
"""
Simple calibration data collection - Just capture images + TCP poses
No chessboard detection during live view
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import yaml
import time
from datetime import datetime
from modbus_robot_interface import ModbusRobotInterface


def calculate_tcp_difference(tcp1, tcp2):
    """Calculate position and rotation difference between two TCP poses"""
    if tcp1 is None or tcp2 is None:
        return None, None

    x1, y1, z1, rx1, ry1, rz1 = tcp1
    x2, y2, z2, rx2, ry2, rz2 = tcp2

    # Position difference (Euclidean distance in mm)
    position_diff = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    # Rotation difference (max absolute difference in degrees)
    rotation_diff = max(abs(rx2-rx1), abs(ry2-ry1), abs(rz2-rz1))

    return position_diff, rotation_diff


def main():
    print("Starting calibration data collection...")

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    robot_ip = config['robot']['ip']
    robot_port = config['robot']['port']
    images_dir = config['storage']['images_dir']
    data_file = config['storage']['data_file']

    # Load thresholds
    position_threshold = config['thresholds']['position_mm']
    rotation_threshold = config['thresholds']['rotation_deg']

    print(f"Config loaded: Robot={robot_ip}:{robot_port}")
    print(f"Position threshold: {position_threshold} mm")
    print(f"Rotation threshold: {rotation_threshold} deg")

    # Create pipeline
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    print("Starting camera...")
    pipeline.start(rs_config)

    # Warm up camera
    for _ in range(30):
        pipeline.wait_for_frames()

    print("Camera ready!")

    # Connect to robot
    print("\nConnecting to robot...")
    robot = ModbusRobotInterface(robot_ip, robot_port)
    robot_connected = False

    if robot.connect():
        print("✅ Robot connected")
        robot_connected = True
        tcp = robot.read_camera_pose()
        if tcp:
            x, y, z, rx, ry, rz = tcp
            print(f"Current TCP: x={x:.1f}, y={y:.1f}, z={z:.1f}, rx={rx:.1f}, ry={ry:.1f}, rz={rz:.1f}")
    else:
        print("⚠️  Robot connection failed - will save images without TCP data")
        robot = None

    # Create images directory
    os.makedirs(images_dir, exist_ok=True)
    print(f"\nImages directory: {images_dir}")
    print("\nControls:")
    print("  SPACE - Capture image")
    print("  Q or ESC - Quit")
    print("\n" + "="*60 + "\n")

    # Load existing data if file exists
    collected_data = []
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                collected_data = yaml.safe_load(f) or []
            print(f"📂 Loaded {len(collected_data)} existing captures from {data_file}")
        except Exception as e:
            print(f"⚠️  Could not load existing data: {e}")
            collected_data = []

    capture_count = len(collected_data)

    while True:
        # Get frame with error handling
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue
        except RuntimeError as e:
            print(f"\n⚠️  Camera frame timeout: {e}")
            print("Retrying...")
            continue

        # Convert to numpy
        image = np.asanyarray(color_frame.get_data())

        # Resize to 1/2 for display
        height, width = image.shape[:2]
        display_image = cv2.resize(image, (width // 2, height // 2))
        h, w = display_image.shape[:2]

        # Display robot status
        if robot_connected:
            cv2.putText(display_image, "Robot: Connected", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(display_image, "Robot: Disconnected", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Display capture count
        cv2.putText(display_image, f"Captured: {capture_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # Display instructions
        cv2.putText(display_image, "SPACE: Capture  |  Q: Quit", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Show image
        cv2.imshow('Calibration Data Collection', display_image)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):  # ESC or Q
            break

        elif key == ord(' '):  # SPACE - capture
            print(f"\n[Capture {capture_count + 1}]")

            # Step 1: Read TCP pose BEFORE image capture
            tcp_before = None
            if robot_connected:
                tcp_before = robot.read_camera_pose()
                if tcp_before:
                    x, y, z, rx, ry, rz = tcp_before
                    print(f"TCP Before: x={x:.1f}, y={y:.1f}, z={z:.1f}, rx={rx:.2f}, ry={ry:.2f}, rz={rz:.2f}")
                else:
                    print("⚠️  Failed to read TCP before capture")

            # Step 2: Capture image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # Small delay to ensure image is captured
            time.sleep(0.05)

            # Step 3: Read TCP pose AFTER image capture
            tcp_after = None
            if robot_connected:
                tcp_after = robot.read_camera_pose()
                if tcp_after:
                    x, y, z, rx, ry, rz = tcp_after
                    print(f"TCP After:  x={x:.1f}, y={y:.1f}, z={z:.1f}, rx={rx:.2f}, ry={ry:.2f}, rz={rz:.2f}")
                else:
                    print("⚠️  Failed to read TCP after capture")

            # Step 4: Check TCP stability
            if tcp_before is not None and tcp_after is not None:
                position_diff, rotation_diff = calculate_tcp_difference(tcp_before, tcp_after)

                print(f"TCP Stability Check:")
                print(f"  Position difference: {position_diff:.3f} mm (threshold: {position_threshold} mm)")
                print(f"  Rotation difference: {rotation_diff:.3f} deg (threshold: {rotation_threshold} deg)")

                if position_diff > position_threshold or rotation_diff > rotation_threshold:
                    print("❌ TCP UNSTABLE - Robot moved during capture!")
                    print("   Capture rejected. Please wait for robot to stabilize and try again.")
                    continue  # Skip saving this capture

                print("✅ TCP stable - proceeding with save")
                # Use average of before and after TCP as the final value
                tcp = tuple((b + a) / 2.0 for b, a in zip(tcp_before, tcp_after))
            elif tcp_before is not None:
                tcp = tcp_before  # Fallback to before if after failed
            else:
                tcp = None

            # Step 5: Save image and data (only if TCP check passed)
            base_filename = f"image_{capture_count + 1:03d}"
            img_filename = os.path.join(images_dir, f"{base_filename}.jpg")
            yaml_filename = os.path.join(images_dir, f"{base_filename}.yaml")

            cv2.imwrite(img_filename, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

            capture_count += 1

            # Create data entry for individual YAML file
            pose_data = {
                'id': capture_count,
                'timestamp': timestamp,
                'image_file': f"{base_filename}.jpg",
            }

            if tcp is not None:
                x, y, z, rx, ry, rz = tcp
                pose_data['tcp_pose'] = {
                    'x_mm': float(x),
                    'y_mm': float(y),
                    'z_mm': float(z),
                    'rx_deg': float(rx),
                    'ry_deg': float(ry),
                    'rz_deg': float(rz)
                }

                # Add stability info
                if tcp_before is not None and tcp_after is not None:
                    pose_data['stability'] = {
                        'position_diff_mm': float(position_diff),
                        'rotation_diff_deg': float(rotation_diff),
                        'stable': True
                    }

            # Save individual YAML file for this image
            with open(yaml_filename, 'w') as f:
                yaml.dump(pose_data, f, default_flow_style=False, sort_keys=False)

            # Also append to collected data list
            collected_data.append(pose_data)

            # Save combined data file immediately after each capture (incremental save)
            with open(data_file, 'w') as f:
                yaml.dump(collected_data, f, default_flow_style=False, sort_keys=False)

            print(f"✅ Image saved: {img_filename}")
            print(f"✅ Pose saved: {yaml_filename}")
            print(f"✅ Combined data saved: {data_file}")
            print(f"Total captured: {capture_count}")

    # Cleanup
    cv2.destroyAllWindows()
    pipeline.stop()
    if robot is not None:
        robot.disconnect()

    # Final summary (data already saved incrementally)
    if len(collected_data) > 0:
        print(f"\n✅ Total {len(collected_data)} captures saved to {data_file}")
    else:
        print("\n⚠️  No data collected")

    print("Done!")


if __name__ == "__main__":
    main()
