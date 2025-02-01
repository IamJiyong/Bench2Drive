import cv2
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R
from vis_utils import overlay_trajectory

def compute_extrinsic_matrix(position, rotation):
    """
    차량 좌표계를 카메라 좌표계로 변환하는 Extrinsic Matrix를 계산하는 함수
    """
    roll, pitch, yaw = np.radians(rotation)
    rot_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    cam_to_vehicle = np.array([
        [1,  0,  0],  
        [0,  0, -1],  
        [0,  1,  0]   
    ])

    final_rot_matrix = rot_matrix @ cam_to_vehicle

    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = final_rot_matrix
    extrinsic_matrix[:3, 3] = np.array([position[0], position[2], position[1]])

    return extrinsic_matrix

def extract_waypoints(json_path):
    """
    JSON 파일에서 'plan' 필드에 해당하는 waypoints를 읽어오는 함수
    """
    try:
        with open(json_path, "r") as file:
            json_data = json.load(file)
        
        if "plan" not in json_data:
            raise KeyError("❌ Error: 'plan' 키가 JSON 데이터에 없습니다.")

        waypoints_2d = np.array(json_data["plan"], dtype=np.float32)
        waypoints_3d = np.hstack((waypoints_2d, np.zeros((waypoints_2d.shape[0], 1), dtype=np.float32)))  

        return waypoints_3d
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return None

def process_trajectory_sequence(camera_type, image_folder, json_folder, intrinsic_matrix, extrinsic_matrix, output_dir):
    """
    이미지와 JSON 데이터를 로드하여 trajectory를 시각화한 후 저장하는 함수.

    Args:
        camera_type (str): "rgb_front" 또는 "bev"
        image_folder (str): 이미지 파일이 있는 폴더 경로.
        json_folder (str): JSON 파일이 있는 폴더 경로.
        intrinsic_matrix (numpy.ndarray): 카메라 내부 행렬.
        extrinsic_matrix (numpy.ndarray): 카메라 외부 행렬.
        output_dir (str): 결과 이미지를 저장할 폴더 경로.
    """
    # 저장할 폴더 생성 (존재하지 않으면 생성)
    output_cam_dir = os.path.join(output_dir, camera_type)
    os.makedirs(output_cam_dir, exist_ok=True)

    frame_ids = sorted([f.split(".")[0] for f in os.listdir(image_folder) if f.endswith(".png")])

    for frame_id in frame_ids:
        image_path = os.path.join(image_folder, f"{frame_id}.png")
        json_path = os.path.join(json_folder, f"{frame_id}.json")

        if not os.path.exists(json_path):
            print(f"⚠️ Warning: JSON file {json_path} not found. Skipping...")
            continue

        trajectory = extract_waypoints(json_path)
        if trajectory is None:
            print(f"⚠️ Warning: Could not extract waypoints from {json_path}. Skipping...")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image from {image_path}. Skipping...")
            continue

        # Trajectory Overlay 적용
        output_image = overlay_trajectory(camera_type, image, trajectory, intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix)

        # 이미지 저장
        output_path = os.path.join(output_cam_dir, f"{frame_id}.png")
        cv2.imwrite(output_path, output_image)
        print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    # 설정 가능한 카메라 타입: "rgb_front" 또는 "bev"
    camera_type = "rgb_front"  # "rgb_front" 또는 "bev" 선택 가능

    # 기본 경로 설정
    base_path = "/home/ysh/jiyong/b2d_carla/Bench2Drive/eval_v1/RouteScenario_2082_rep0_Town12_OppositeVehicleRunningRedLight_1_22_01_31_01_13_50/"
    json_folder = os.path.join(base_path, "meta")

    # 결과 저장 폴더
    output_dir = "/home/ysh/jiyong/b2d_carla/Bench2Drive/eval_results_calibration_debug"

    # 카메라 타입에 따른 설정 적용
    if camera_type == "rgb_front":
        image_folder = os.path.join(base_path, "rgb_front")
        intrinsic_matrix = np.array([
            [1142.5184, 0, 800],
            [0, 1142.5184, 450],
            [0, 0, 1]
        ])
        extrinsic_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, -1.0, 3.2], 
            [0.0, 1.0, 0.0, 0.8], 
            [0.0, 0.0, 0.0, 1.0]
        ])
        camera_position = (0.0, 0.0, 1.6)  # 차량 앞쪽 카메라 위치
        camera_rotation = (0, 0, 0)  # 정면 방향

    elif camera_type == "bev":
        image_folder = os.path.join(base_path, "bev")
        intrinsic_matrix = np.array([
            [548.993771650447, 0.0, 256.0],  
            [0.0, 548.993771650447, 256.0],  
            [0.0, 0.0, 1.0]
        ])
        camera_position = (0.0, 0.0, 50.0)  # 차량 위 50m 높이
        camera_rotation = (0, 90, 0)  # 아래를 바라보는 -90도 Pitch 회전

    else:
        raise ValueError(f"❌ Unsupported camera type: {camera_type}")

    # BEV Extrinsic Matrix 설정 (필요하면 적용)
    if camera_type == "bev":
        extrinsic_matrix = np.array([
            [0.0, -1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, -1.0, 50.0], 
            [0.0, 0.0, 0.0, 1.0]
        ])

    print(extrinsic_matrix)

    # 실행 - trajectory 결과를 특정 폴더에 저장
    process_trajectory_sequence(camera_type, image_folder, json_folder, intrinsic_matrix, extrinsic_matrix, output_dir)
