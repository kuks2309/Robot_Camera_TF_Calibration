# Robot-Camera Hand-Eye Calibration

이 패키지는 로봇 End-Effector에 장착된 카메라의 **eye-in-hand** 캘리브레이션을 수행하여 로봇의 Tool Center Point (TCP)와 카메라 프레임 간의 변환 행렬을 계산합니다.

## 특징

- **2단계 워크플로우**: 데이터 수집과 캘리브레이션 계산 분리
- **티칭 패드 사용**: 로봇을 수동으로 움직이며 데이터 수집
- **두 가지 보드 지원**: ChArUco 또는 일반 Chessboard
- **자동/수동 TCP 입력**: Modbus 자동 읽기 또는 수동 입력 선택
- **RealSense 통합**: Intel RealSense 카메라 지원
- **검증 기능**: Reprojection error 자동 계산

## 요구사항

### 하드웨어
- Modbus TCP 인터페이스 지원 로봇 (옵션: 192.168.0.29:1502)
- Intel RealSense 카메라 (D400 시리즈 권장)
- 로봇 End-Effector에 견고하게 고정된 카메라
- 캘리브레이션 보드:
  - **ChArUco**: 8x6 그리드, 30mm 사각형, 21mm 마커 (권장)
  - **Chessboard**: 9x6 내부 코너 (10x7 사각형), 25mm 사각형

### 소프트웨어
```bash
pip install -r requirements.txt
```

## 설치

```bash
cd Robot_Camera_TF_Calibration
pip install -r requirements.txt
```

## 사용법

### 단계 0: 캘리브레이션 보드 생성

ChArUco 보드 또는 Chessboard를 생성하여 출력합니다:

```bash
# ChArUco 보드 생성 (권장)
python3 generate_calibration_board.py --type charuco

# Chessboard 생성
python3 generate_calibration_board.py --type chessboard
```

출력된 보드를 평평한 표면(폼보드, 아크릴 등)에 부착합니다.

### 단계 1: 데이터 수집

```bash
python3 collect_calibration_data.py
```

#### 데이터 수집 절차:

1. **TCP 입력 방식 선택**:
   - **자동 (Modbus)**: 로봇에서 자동으로 TCP 읽기
   - **수동**: 티칭 패드에서 TCP 값을 수동으로 입력

2. **카메라 라이브 뷰** 확인
   - 화면에 카메라 영상이 실시간으로 표시됩니다

3. **로봇 이동 및 캡처**:
   - 티칭 패드로 로봇을 새로운 포즈로 이동
   - 캘리브레이션 보드가 화면에 보이는지 확인
   - **SPACE 키** 누름 → 이미지 + TCP 저장
   - 10-15개의 서로 다른 포즈에서 반복

4. **Q 키**를 눌러 종료

#### 데이터 수집 팁:
- 최소 10-15개 포즈 수집 (더 많을수록 좋음)
- 로봇 위치 다양화 (보드와의 거리 변경)
- 로봇 방향 다양화 (다양한 각도에서 촬영)
- 보드가 모든 이미지에서 완전히 보이도록 유지
- 모션 블러 방지 (천천히 이동)

#### 출력 데이터:
```
calibration_data/session_YYYYMMDD_HHMMSS/
├── tcp_poses.json          # 모든 TCP 포즈 데이터
├── metadata.json           # 세션 메타데이터
└── images/
    ├── pose_001.jpg
    ├── pose_002.jpg
    └── ...
```

### 단계 2: 캘리브레이션 계산

```bash
python3 compute_calibration.py
```

#### 계산 절차:

1. **세션 선택**: 처리할 데이터 수집 세션 선택
2. **보드 타입 선택**: ChArUco 또는 Chessboard
3. **자동 처리**:
   - 각 이미지에서 보드 검출
   - Hand-Eye Calibration 수행
   - 변환 행렬 계산 및 저장

#### 출력 결과:
```
calibration_data/session_YYYYMMDD_HHMMSS/
├── calibration_result.json        # 캘리브레이션 결과
├── camera_to_tcp_transform.npy    # 4x4 변환 행렬 (numpy)
└── detected_poses/                 # 보드 검출 시각화 이미지
    ├── detected_pose_001.jpg
    ├── detected_pose_002.jpg
    └── ...
```

## 캘리브레이션 결과 형식

### JSON 출력 (`calibration_result.json`)

```json
{
  "timestamp": "2025-01-23T10:30:45",
  "num_poses_collected": 15,
  "num_poses_used": 14,
  "calibration_method": "Tsai",
  "avg_rotation_error_deg": 0.234,
  "camera_to_tcp_transformation": {
    "rotation_matrix": [[...], [...], [...]],
    "translation_vector_m": [0.05, -0.02, 0.1],
    "euler_angles_deg": [180.5, 2.3, -1.8]
  }
}
```

### Numpy 행렬 (`camera_to_tcp_transform.npy`)

4x4 동차 변환 행렬:
```
T_cam_to_tcp = [R | t]
               [0 | 1]
```

여기서:
- `R`: 3x3 회전 행렬
- `t`: 3x1 이동 벡터 (미터)

## 캘리브레이션 결과 사용

### 변환 행렬 로드

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# 4x4 변환 행렬 로드
T_cam_to_tcp = np.load('camera_to_tcp_transform.npy')

# 회전 및 이동 추출
R_cam_to_tcp = T_cam_to_tcp[:3, :3]
t_cam_to_tcp = T_cam_to_tcp[:3, 3]

print(f"Translation (m): {t_cam_to_tcp}")
print(f"Rotation (deg): {R.from_matrix(R_cam_to_tcp).as_euler('xyz', degrees=True)}")
```

### 카메라 좌표를 로봇 베이스 좌표로 변환

```python
def camera_to_robot_base(point_in_camera, T_cam_to_tcp, T_tcp_to_base):
    """
    카메라 프레임의 점을 로봇 베이스 프레임으로 변환

    Args:
        point_in_camera: 카메라 프레임의 3D 점 [x, y, z]
        T_cam_to_tcp: 4x4 카메라-TCP 변환 행렬
        T_tcp_to_base: 4x4 TCP-베이스 변환 행렬 (로봇에서 읽음)

    Returns:
        로봇 베이스 프레임의 3D 점
    """
    # 동차 좌표로 변환
    point_homo = np.array([*point_in_camera, 1.0])

    # 변환: camera -> TCP -> base
    point_in_tcp = T_cam_to_tcp @ point_homo
    point_in_base = T_tcp_to_base @ point_in_tcp

    return point_in_base[:3]
```

### 실제 사용 예제

```python
# 캘리브레이션 로드
import numpy as np
from scipy.spatial.transform import Rotation as R
from modbus_robot_interface import ModbusRobotInterface

T_cam_to_tcp = np.load('calibration_data/session_XXX/camera_to_tcp_transform.npy')

# 로봇 TCP 포즈 읽기 (Modbus)
robot = ModbusRobotInterface("192.168.0.29", 1502)
robot.connect()
tcp_pose = robot.read_camera_pose()  # (x_mm, y_mm, z_mm, rx, ry, rz) 반환

# TCP-to-base 변환 행렬 생성
T_tcp_to_base = np.eye(4)
T_tcp_to_base[:3, :3] = R.from_euler('xyz', tcp_pose[3:], degrees=True).as_matrix()
T_tcp_to_base[:3, 3] = np.array(tcp_pose[:3]) / 1000.0  # mm를 m로 변환

# 카메라 프레임에서 객체 검출 (비전 시스템 사용)
object_position_in_camera = np.array([0.5, 0.0, 1.0, 1.0])  # 동차 좌표

# 로봇 베이스 프레임으로 변환
object_in_tcp = T_cam_to_tcp @ object_position_in_camera
object_in_base = T_tcp_to_base @ object_in_tcp

print(f"Object position in robot base: {object_in_base[:3]}")

# 이제 로봇에 이 위치로 이동 명령을 내릴 수 있습니다!
```

## 문제 해결

### 보드가 검출되지 않음

**ChArUco:**
- 조명 개선
- 마커 크기 확인 (21mm)
- 로봇 위치/각도 조정
- ArUco 딕셔너리 확인 (DICT_4X4_50)

**Chessboard:**
- 전체 보드가 보이는지 확인
- 보드가 평평한지 확인 (휘어지지 않음)
- 대비 향상을 위한 조명 개선
- 보드가 너무 작으면 로봇을 가까이 이동

### 캘리브레이션 오차가 큼

`avg_rotation_error_deg` > 1.0도일 경우:
- 더 많은 포즈 수집 (20개 이상 권장)
- 포즈 다양성 개선 (더 많은 방향)
- 카메라 장착이 실제로 견고한지 확인
- 보드 측정값이 정확한지 확인
- 로봇 TCP 읽기가 정확한지 확인

### 연결 오류

**Robot Modbus:**
- IP 주소 확인: 192.168.0.29
- 포트 확인: 1502
- `ping 192.168.0.29`로 테스트

**RealSense 카메라:**
- USB 연결 확인
- `realsense-viewer`로 테스트
- 필요시 RealSense SDK 업데이트

## 워크플로우 요약

```
┌──────────────────────────────────────────┐
│ 1. 캘리브레이션 보드 생성 및 출력      │
│    python3 generate_calibration_board.py │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│ 2. 데이터 수집                           │
│    python3 collect_calibration_data.py   │
│                                           │
│    - 티칭 패드로 로봇 이동              │
│    - SPACE: 이미지 + TCP 저장           │
│    - 10-15 포즈 반복                     │
│    - Q: 종료                             │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│ 3. 캘리브레이션 계산                    │
│    python3 compute_calibration.py        │
│                                           │
│    - 보드 검출                           │
│    - Hand-Eye Calibration 수행          │
│    - 변환 행렬 저장                      │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│ 4. 결과 사용                             │
│    - 변환 행렬 로드                      │
│    - 카메라 좌표 -> 로봇 좌표 변환     │
│    - Visual Servoing 응용               │
└──────────────────────────────────────────┘
```

## 파일 구조

```
Robot_Camera_TF_Calibration/
├── collect_calibration_data.py      # 1단계: 데이터 수집
├── compute_calibration.py            # 2단계: 캘리브레이션 계산
├── generate_calibration_board.py    # 보드 생성 유틸리티
├── modbus_robot_interface.py        # Robot TCP 통신
├── requirements.txt                  # Python 의존성
├── README.md                         # 이 파일
└── calibration_data/                 # 출력 디렉토리 (자동 생성)
    └── session_YYYYMMDD_HHMMSS/
        ├── tcp_poses.json
        ├── metadata.json
        ├── images/
        ├── calibration_result.json
        ├── camera_to_tcp_transform.npy
        └── detected_poses/
```

## 보드 사양

### ChArUco 보드 파라미터

```python
ChArUcoBoardDetector(
    grid_size=(8, 6),           # 8x6 사각형 그리드
    square_size=0.0298,         # 30mm 사각형
    marker_size=0.02091,        # 21mm ArUco 마커
    aruco_dict=cv2.aruco.DICT_4X4_50
)
```

### Chessboard 파라미터

```python
ChessboardDetector(
    pattern_size=(9, 6),        # 9x6 내부 코너 (10x7 사각형)
    square_size=0.025           # 25mm 사각형
)
```

## 기술 세부사항

### Hand-Eye Calibration

스크립트는 **AX = XB** 방정식을 풉니다:
- **A**: 로봇 TCP 변환 (gripper-to-base)
- **B**: 보드 변환 (target-to-camera)
- **X**: 카메라-TCP 변환 (우리가 구하는 값)

### 캘리브레이션 방법

지원되는 OpenCV 방법:
- **Tsai** (기본) - 빠르고 강건함
- **Park** - 높은 정확도
- **Horaud** - 대안 공식
- **Andreff** - 나사 운동 기반
- **Daniilidis** - 이중 쿼터니언

### 좌표 프레임

```
Robot Base Frame
    ↓
Robot TCP Frame (Modbus로 읽음)
    ↓
Camera Frame (X = 계산된 변환)
    ↓
Calibration Board Frame (B = ChArUco/Chessboard로 검출)
```

## 참고 자료

- [OpenCV Hand-Eye Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b)
- [ChArUco Board Detection](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)
- [RealSense SDK](https://github.com/IntelRealSense/librealsense)

## 라이선스

MIT License

## 지원

문제나 질문이 있으면 프로젝트 관리자에게 문의하세요.
