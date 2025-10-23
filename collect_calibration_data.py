#!/usr/bin/env python3
"""
Calibration Data Collection Script

This script collects calibration data by capturing images and robot TCP poses.
Use the teaching pendant to move the robot to different positions, then press
SPACE to capture each pose.

Workflow:
    1. Move robot using teaching pendant
    2. View live camera feed
    3. Press SPACE when calibration board is visible
    4. Robot TCP and image are saved
    5. Repeat for 10-15 different poses
    6. Press Q to quit

Output:
    calibration_data/session_YYYYMMDD_HHMMSS/
    ├── tcp_poses.json           # All robot TCP poses
    ├── images/
    │   ├── pose_001.jpg
    │   ├── pose_002.jpg
    │   └── ...
    └── metadata.json            # Session information
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import sys
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, List
import json

# Import modbus interface for TCP reading
from modbus_robot_interface import ModbusRobotInterface


class RealSenseCamera:
    """Manages Intel RealSense camera operations."""

    def __init__(self,
                 width: int = 1280,
                 height: int = 720,
                 fps: int = 15):
        """
        Initialize RealSense camera.

        Args:
            width: Image width
            height: Image height
            fps: Frame rate
        """
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure color stream
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = None
        self.intrinsics = None

    def start(self) -> bool:
        """Start camera pipeline."""
        try:
            self.profile = self.pipeline.start(self.config)

            # Get camera intrinsics
            color_stream = self.profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            print(f"✅ RealSense camera started: {self.width}x{self.height} @ {self.fps} FPS")
            print(f"   Intrinsics: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, "
                  f"cx={self.intrinsics.ppx:.2f}, cy={self.intrinsics.ppy:.2f}")

            # Warm up camera
            for _ in range(30):
                self.pipeline.wait_for_frames()

            return True

        except Exception as e:
            print(f"❌ Failed to start RealSense camera: {e}")
            return False

    def stop(self):
        """Stop camera pipeline."""
        self.pipeline.stop()
        print("🔌 RealSense camera stopped")

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Capture a frame from camera.

        Returns:
            Tuple of (color_image, intrinsics_dict) or (None, None)
        """
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                return None, None

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Build intrinsics dict
            intrinsics_dict = {
                'width': self.intrinsics.width,
                'height': self.intrinsics.height,
                'fx': float(self.intrinsics.fx),
                'fy': float(self.intrinsics.fy),
                'ppx': float(self.intrinsics.ppx),
                'ppy': float(self.intrinsics.ppy),
                'coeffs': [float(c) for c in self.intrinsics.coeffs]
            }

            return color_image, intrinsics_dict

        except Exception as e:
            print(f"❌ Failed to get frame: {e}")
            return None, None


class CalibrationDataCollector:
    """Manages calibration data collection session."""

    def __init__(self,
                 robot_ip: str = "192.168.0.29",
                 robot_port: int = 1502,
                 save_dir: str = "calibration_data",
                 use_robot: bool = True):
        """
        Initialize data collection session.

        Args:
            robot_ip: Robot Modbus IP address
            robot_port: Robot Modbus port
            save_dir: Directory to save calibration data
            use_robot: If True, read TCP from robot. If False, enter manually.
        """
        self.use_robot = use_robot
        if use_robot:
            self.robot = ModbusRobotInterface(robot_ip, robot_port)
        else:
            self.robot = None

        self.camera = RealSenseCamera()

        self.save_dir = save_dir
        self.session_dir = None
        self.images_dir = None

        # Data storage
        self.collected_poses = []
        self.camera_intrinsics = None

    def start(self) -> bool:
        """Start data collection session."""
        print("\n" + "="*80)
        print("Hand-Eye Calibration Data Collection")
        print("="*80)

        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.save_dir, f"session_{timestamp}")
        self.images_dir = os.path.join(self.session_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        print(f"📁 Session directory: {self.session_dir}")

        # Connect to robot (if enabled)
        if self.use_robot:
            print("\n🔌 Connecting to robot...")
            if not self.robot.connect():
                print("❌ Failed to connect to robot")
                print("⚠️  Continuing in manual TCP entry mode...")
                self.use_robot = False
        else:
            print("\n⚙️  Manual TCP entry mode enabled")

        # Start camera
        print("\n📷 Starting camera...")
        if not self.camera.start():
            print("❌ Failed to start camera")
            if self.use_robot:
                self.robot.disconnect()
            return False

        print("\n✅ System ready for data collection")
        return True

    def stop(self):
        """Stop data collection session."""
        self.camera.stop()
        if self.use_robot and self.robot:
            self.robot.disconnect()
        print("\n🔌 Session ended")

    def get_tcp_pose(self) -> Optional[Dict]:
        """
        Get robot TCP pose either from robot or manual input.

        Returns:
            Dictionary with TCP pose or None
        """
        if self.use_robot:
            # Read from robot via Modbus
            tcp_pose = self.robot.read_camera_pose()
            if tcp_pose is None:
                print("❌ Failed to read robot TCP")
                return None

            x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = tcp_pose
            return {
                'x_mm': float(x_mm),
                'y_mm': float(y_mm),
                'z_mm': float(z_mm),
                'rx_deg': float(rx_deg),
                'ry_deg': float(ry_deg),
                'rz_deg': float(rz_deg),
                'source': 'robot_modbus'
            }
        else:
            # Manual entry from teaching pendant
            print("\n" + "-"*60)
            print("Enter TCP pose from teaching pendant:")
            print("-"*60)
            try:
                x_mm = float(input("X (mm): "))
                y_mm = float(input("Y (mm): "))
                z_mm = float(input("Z (mm): "))
                rx_deg = float(input("Rx (deg): "))
                ry_deg = float(input("Ry (deg): "))
                rz_deg = float(input("Rz (deg): "))

                return {
                    'x_mm': x_mm,
                    'y_mm': y_mm,
                    'z_mm': z_mm,
                    'rx_deg': rx_deg,
                    'ry_deg': ry_deg,
                    'rz_deg': rz_deg,
                    'source': 'manual_entry'
                }
            except (ValueError, EOFError, KeyboardInterrupt):
                print("\n❌ Invalid input or cancelled")
                return None

    def capture_pose(self, pose_id: int) -> bool:
        """
        Capture a single calibration pose.

        Args:
            pose_id: Unique identifier for this pose

        Returns:
            True if capture successful
        """
        print(f"\n📸 Capturing pose #{pose_id}...")

        # Get TCP pose
        tcp_pose = self.get_tcp_pose()
        if tcp_pose is None:
            return False

        print(f"   Robot TCP: x={tcp_pose['x_mm']:.2f}mm, y={tcp_pose['y_mm']:.2f}mm, "
              f"z={tcp_pose['z_mm']:.2f}mm")
        print(f"              rx={tcp_pose['rx_deg']:.2f}°, ry={tcp_pose['ry_deg']:.2f}°, "
              f"rz={tcp_pose['rz_deg']:.2f}°")

        # Capture camera frame
        color_image, intrinsics = self.camera.get_frame()
        if color_image is None:
            print("❌ Failed to capture camera frame")
            return False

        # Store camera intrinsics (only once)
        if self.camera_intrinsics is None:
            self.camera_intrinsics = intrinsics

        # Save image
        img_filename = os.path.join(self.images_dir, f"pose_{pose_id:03d}.jpg")
        cv2.imwrite(img_filename, color_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # Store pose data
        pose_data = {
            'pose_id': pose_id,
            'timestamp': datetime.now().isoformat(),
            'tcp_pose': tcp_pose,
            'image_file': img_filename,
            'image_relative_path': f"images/pose_{pose_id:03d}.jpg"
        }

        self.collected_poses.append(pose_data)

        print(f"✅ Pose #{pose_id} captured successfully")
        print(f"   Image saved: {img_filename}")

        return True

    def run_interactive_collection(self):
        """Run interactive data collection loop."""
        print("\n" + "="*80)
        print("Interactive Data Collection")
        print("="*80)
        print("\nInstructions:")
        print("  1. Move robot using TEACHING PENDANT to a new pose")
        print("  2. Ensure calibration board is visible in camera view")
        print("  3. Press SPACE to capture current pose")
        print("  4. Repeat for at least 10-15 different poses")
        print("  5. Press Q to finish data collection")
        print("\nTips:")
        print("  - Vary robot position (closer/farther from board)")
        print("  - Vary robot orientation (different angles)")
        print("  - Keep board fully visible in all images")
        print("  - Avoid motion blur (move slowly)")
        print("="*80)

        pose_id = 1
        window_name = "Data Collection - Live View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            # Get live camera frame
            color_image, _ = self.camera.get_frame()
            if color_image is None:
                continue

            # Create display image
            display_image = color_image.copy()

            # Add UI overlay
            status_text = "Ready to capture"
            status_color = (0, 255, 0)

            cv2.putText(display_image, status_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
            cv2.putText(display_image, f"Captured: {len(self.collected_poses)} poses", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Instructions
            cv2.putText(display_image, "SPACE: Capture Pose | Q: Quit",
                       (10, display_image.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw center crosshair
            h, w = display_image.shape[:2]
            cv2.drawMarker(display_image, (w//2, h//2), (0, 255, 255),
                          cv2.MARKER_CROSS, 30, 2)

            cv2.imshow(window_name, display_image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space - capture
                # Flash screen
                flash = np.ones_like(display_image) * 255
                cv2.imshow(window_name, flash)
                cv2.waitKey(100)

                success = self.capture_pose(pose_id)
                if success:
                    pose_id += 1

                # Brief pause
                cv2.waitKey(500)

            elif key == ord('q'):  # Q - quit
                break

        cv2.destroyAllWindows()

        print(f"\n✅ Data collection complete: {len(self.collected_poses)} poses captured")

    def save_results(self):
        """Save collection results to files."""
        if len(self.collected_poses) == 0:
            print("⚠️  No poses captured, nothing to save")
            return

        # Save TCP poses
        poses_file = os.path.join(self.session_dir, "tcp_poses.json")
        with open(poses_file, 'w') as f:
            json.dump(self.collected_poses, f, indent=2)
        print(f"\n💾 TCP poses saved: {poses_file}")

        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'num_poses': len(self.collected_poses),
            'camera_intrinsics': self.camera_intrinsics,
            'tcp_source': self.collected_poses[0]['tcp_pose']['source'] if self.collected_poses else 'unknown',
            'image_resolution': f"{self.camera.width}x{self.camera.height}",
            'session_directory': self.session_dir
        }

        metadata_file = os.path.join(self.session_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Metadata saved: {metadata_file}")

        print(f"\n📁 All data saved to: {self.session_dir}")
        print(f"\n📊 Collection Summary:")
        print(f"   Total poses: {len(self.collected_poses)}")
        print(f"   Images directory: {self.images_dir}/")
        print(f"   TCP data: {poses_file}")


def main():
    """Main data collection workflow."""
    print("\n" + "="*80)
    print("Robot-Camera Hand-Eye Calibration - Data Collection")
    print("="*80)
    print("\nThis script collects calibration data:")
    print("  - Camera images")
    print("  - Robot TCP poses")
    print("\nPrerequisites:")
    print("  ✓ Camera mounted to robot end-effector")
    print("  ✓ Calibration board placed in workspace")
    print("  ✓ Intel RealSense camera connected")
    print("  ✓ Robot teaching pendant ready")
    print("="*80)

    # Ask about TCP reading mode
    print("\nHow do you want to provide TCP poses?")
    print("  1. Automatic - Read from robot via Modbus (192.168.0.29:1502)")
    print("  2. Manual - Enter TCP values from teaching pendant")

    while True:
        choice = input("\nEnter choice (1 or 2) [default: 2]: ").strip()
        if choice == "1":
            use_robot = True
            print("✓ Selected: Automatic TCP reading via Modbus")
            break
        elif choice == "" or choice == "2":
            use_robot = False
            print("✓ Selected: Manual TCP entry")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    input("\nPress ENTER to start data collection...")

    # Create data collector
    collector = CalibrationDataCollector(
        robot_ip="192.168.0.29",
        robot_port=1502,
        save_dir="calibration_data",
        use_robot=use_robot
    )

    # Start session
    if not collector.start():
        print("\n❌ Failed to start data collection session")
        return 1

    try:
        # Run interactive data collection
        collector.run_interactive_collection()

        # Save results
        collector.save_results()

        print("\n" + "="*80)
        print("✅ Data Collection Complete!")
        print("="*80)
        print("\nNext step:")
        print("  Run: python3 compute_calibration.py")
        print("  This will process the collected data and compute calibration.")
        print("="*80)

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Data collection interrupted by user")
        # Still try to save partial data
        if len(collector.collected_poses) > 0:
            print("\n💾 Saving partial data...")
            collector.save_results()
        return 1

    finally:
        collector.stop()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
