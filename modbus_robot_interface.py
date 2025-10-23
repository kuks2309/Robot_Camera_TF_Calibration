#!/usr/bin/env python3
# modbus_robot_interface.py

from pymodbus.client import ModbusTcpClient
from typing import Optional, Tuple, List
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R


class ModbusRobotInterface:
    """Handles all Modbus communication with the robot controller."""
    
    # Register addresses
    """
    address 301 ~ 306: pose_back
    address 307 ~ 312: pose_main 
    address 351: (write 1: grab gun(from empty port), write 2: empty port, write 3: grab gun(from car), write 4: car charging port)
    address 352: (write 1: pose_back, write 2: pose_main)    
    """
    
    REGISTER_CMD = 351
    REGISTER_RESP = 352
    REGISTER_POSE_MAIN = 301  # 301~306
    REGISTER_POSE_BACK = 307  # 307~312
    REGISTER_BASE_CAM = 158   # 158~169 (12 registers for camera pose)
    
    def __init__(self, ip: str = "192.168.0.29", port: int = 1502, timeout: float = 0.1):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.client: Optional[ModbusTcpClient] = None
        self._connected = False
    
    def connect(self) -> bool:
        """Establish connection to robot Modbus server."""
        try:
            self.client = ModbusTcpClient(self.ip, port=self.port, timeout=self.timeout)
            self._connected = self.client.connect()
            if self._connected:
                print(f"✅ Connected to robot at {self.ip}:{self.port}")
            else:
                print(f"❌ Failed to connect to robot at {self.ip}:{self.port}")
            return self._connected
        except Exception as e:
            print(f"❌ Connection error: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Close Modbus connection."""
        if self.client:
            self.client.close()
            self._connected = False
            print("🔌 Modbus connection closed")
    
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._connected and self.client is not None
    
    # ==================== Command Operations ====================
    
    def read_command(self) -> Optional[int]:
        """Read command from register 351."""
        if not self.is_connected():
            return None
        
        try:
            rr = self.client.read_holding_registers(address=self.REGISTER_CMD, count=1)
            if rr.isError():
                print("⚠️  Failed to read CMD(351)")
                return None
            val = rr.registers[0]
            print(f"📖 Read CMD(351) = {val}")
            return val
        except Exception as e:
            print(f"❌ read_command error: {e}")
            return None
    
    def write_response(self, value: int) -> bool:
        """Write response to register 352."""
        if not self.is_connected():
            return False
        
        try:
            self.client.write_registers(self.REGISTER_RESP, [value])
            print(f"✍️  Wrote RESP(352) = {value}")
            return True
        except Exception as e:
            print(f"❌ write_response error: {e}")
            return False
    
    def reset_command(self) -> bool:
        """Reset command register 351 to 0."""
        if not self.is_connected():
            return False
        
        try:
            # self.client.write_registers(self.REGISTER_CMD, [0])
            print("🔄 Reset CMD(351) → 0")
            return True
        except Exception as e:
            print(f"❌ reset_command error: {e}")
            return False
    
    # ==================== Pose Operations ====================
    
    def write_pose(self, pose_main: List[int], pose_back: List[int]) -> bool:
        """Write pose data to registers 301-312."""
        if not self.is_connected():
            return False
        
        try:
            print("POSE MAIN", pose_main)
            self.client.write_registers(self.REGISTER_POSE_MAIN, pose_main)
            self.client.write_registers(self.REGISTER_POSE_BACK, pose_back)
            print(f"📤 Pose written 301~306: {pose_main}")
            print(f"📤 Pose written 307~312: {pose_back}")
            return True
        except Exception as e:
            print(f"❌ write_pose error: {e}")
            return False
    
    def read_camera_pose(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Read camera pose from registers 158-169.
        
        Returns:
            Tuple of (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) or None
        """
        if not self.is_connected():
            return None
        
        try:
            rr = self.client.read_holding_registers(address=self.REGISTER_BASE_CAM, count=12)
            if rr.isError():
                print("⚠️  Failed to read camera pose")
                return None
            
            
            pose = self._modbus_to_pose(rr)
            print(f"📖 Camera pose: x={pose[0]}, y={pose[1]}, z={pose[2]}, rx={pose[3]}, ry={pose[4]}, rz={pose[5]}")
            return pose
        
        except Exception as e:
            print(f"❌ read_camera_pose error: {e}")
            return None
    
    def get_robot_transform_matrices(self) -> Optional[dict]:
        """Get camera transformation matrices for vision processing.
        
        Returns:
            Dictionary with transformation matrices or None
        """
        pose = self.read_camera_pose()
        if pose is None:
            return None
        
        x_bc, y_bc, z_bc, rx_bc, ry_bc, rz_bc = pose
        
        # Translation vector (meters)
        tvec = np.array([[x_bc/1000.0, y_bc/1000.0, z_bc/1000.0]], dtype=np.float64)
        
        # Rotation matrix from Euler angles
        rotation_matrix = R.from_euler("xyz", [rx_bc, ry_bc, rz_bc], degrees=True).as_matrix()
        
        # Camera to end-effector transform
        cam_to_ee = np.array([[-1, 0, 0],
                              [0, -1, 0],
                              [0, 0, 1]])

        rotation_matrix = rotation_matrix @ cam_to_ee.T
        robot_rotation_in_camera = rotation_matrix.T
        robot_position_in_camera = -robot_rotation_in_camera @ tvec.T
        
        return {
            'camera_position': tvec,
            'camera_rotation': rotation_matrix,
            'board_rotation_in_camera': robot_rotation_in_camera,
            'board_position_in_camera': robot_position_in_camera,
        }
    
    # ==================== Utility Methods ====================
    
    @staticmethod
    def pose_to_modbus_data(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg):
        """
        Encode pose (x,y,z [mm], rx,ry,rz [deg]) → [uint16 * 6]
        - Position: millimeters, normalized to int16
        - Rotation: degrees, normalized to (-180, 180]
        - Each value stored as int16, then mapped to uint16 register
        """

        def to_uint16(val: int):
            # int16 → uint16 매핑
            return int(val + 65536) if val < 0 else int(val)

        def wrap_deg(deg: float):
            return ((deg + 180) % 360) - 180  # normalize to (-180, 180]

        # 1. Position 그대로 사용 (mm 단위 입력)
        x_val = int(round(x_mm*1000))
        y_val = int(round(y_mm*1000))
        z_val = int(round(z_mm*1000))

        # 2. Rotation: deg normalize
        rx_val = int(round(wrap_deg(rx_deg)))
        ry_val = int(round(wrap_deg(ry_deg)))
        rz_val = int(round(wrap_deg(rz_deg)))
            
        # 3. Clamp to int16 범위, uint16 변환
        vals = [x_val, y_val, z_val, rx_val, ry_val, rz_val]
        regs = [
            to_uint16(v if -32768 <= v <= 32767 else max(min(v, 32767), -32768))
            for v in vals
        ]

        return regs
    
    @staticmethod
    def _modbus_to_pose(modbus_tool_cam_tcp):
        # float32 (mm, deg) from robot
        mod_list = modbus_tool_cam_tcp.registers 
        modbus_x = [mod_list[0], mod_list[1]] 
        byte_data_x = modbus_x[1].to_bytes(2, 'big') + modbus_x[0].to_bytes(2, 'big') 
        x = struct.unpack('>f', byte_data_x)[0] 
        modbus_y = [mod_list[2], mod_list[3]] 
        byte_data_y = modbus_y[1].to_bytes(2, 'big') + modbus_y[0].to_bytes(2, 'big') 
        y = struct.unpack('>f', byte_data_y)[0] 
        modbus_z = [mod_list[4], mod_list[5]] 
        byte_data_z = modbus_z[1].to_bytes(2, 'big') + modbus_z[0].to_bytes(2, 'big') 
        z = struct.unpack('>f', byte_data_z)[0] 
        modbus_rx = [mod_list[6], mod_list[7]] 
        byte_data_rx = modbus_rx[1].to_bytes(2, 'big') + modbus_rx[0].to_bytes(2, 'big') 
        rx = struct.unpack('>f', byte_data_rx)[0] 
        modbus_ry = [mod_list[8], mod_list[9]] 
        byte_data_ry = modbus_ry[1].to_bytes(2, 'big') + modbus_ry[0].to_bytes(2, 'big') 
        ry = struct.unpack('>f', byte_data_ry)[0] 
        modbus_rz = [mod_list[10], mod_list[11]] 
        byte_data_rz = modbus_rz[1].to_bytes(2, 'big') + modbus_rz[0].to_bytes(2, 'big') 
        rz = struct.unpack('>f', byte_data_rz)[0]
        return (x, y, z, rx, ry, rz)
    
    def __enter__(self):
        """Context manager support."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.disconnect()
        