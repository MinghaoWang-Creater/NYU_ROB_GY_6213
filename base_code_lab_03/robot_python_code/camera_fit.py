# External Libraries
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
import cv2
from scipy.optimize import least_squares
import json
from scipy.spatial.transform import Rotation as R

tvec_scaling_factor = 0.095 / 0.068


def load_data():
    data_path = Path("points_data.json")
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def parse_data(data):
    measured_points = []
    camera_readings = []
    for entry in data:
        measured_points.append(entry[0])
        camera_readings.append(entry[1:])
    return measured_points, camera_readings


def transformation_matrix(tvec, rvec):
    # Standard OpenCV Rodrigues conversion
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=float))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


data = load_data()
measured_points, camera_readings = parse_data(data)

measured_points_repeated = []
camera_readings_repeated = []

# Rotation matrix to rotate tag by 180 degrees around Y axis
R_cam_tag_offset = cv2.Rodrigues(np.array([0.0, np.pi, 0.0], dtype=float))[0]
print("R_robot_tag offset:\n", R_cam_tag_offset)

for i in range(len(measured_points)):
    for reading in camera_readings[i]:
        measured = measured_points[i]

        # Convert measured points from cm/deg to meters/rad
        measured_points_repeated.append(
            [
                measured[0] * 0.01,
                measured[1] * 0.01,
                0.0,
                0.0,
                0.0,
                measured[2] * math.pi / 180.0,
            ]
        )

        # Extract and scale tvec immediately
        tvec_orig = np.array(reading[:3], dtype=float) * tvec_scaling_factor

        # Apply 180-degree rotation to rvec
        rvec_orig = np.array(reading[3:], dtype=float)
        # R_original, _ = cv2.Rodrigues(rvec_orig)
        # R_new = R_original# @ R_cam_tag_offset
        # rvec_new, _ = cv2.Rodrigues(R_new)

        camera_readings_repeated.append(
            np.concatenate([tvec_orig, rvec_orig.flatten()])
        )
measured_points_repeated = np.array(measured_points_repeated)
camera_readings_repeated = np.array(camera_readings_repeated)
np.set_printoptions(precision=4, suppress=True)

# Static offsets
t_robot_tag = np.array([0.05, -0.05, 0.12], dtype=float)
T_robot_tag = np.eye(4)
T_robot_tag[:3, 3] = t_robot_tag
T_robot_tag[:3, :3] = R_cam_tag_offset
print("-------------------------------------------------------")

for i in range(len(camera_readings_repeated)):
    tvec = camera_readings_repeated[i][:3]
    rvec = camera_readings_repeated[i][3:]
    T_cam_tag = transformation_matrix(tvec, rvec)

    tvec_m = measured_points_repeated[i][:3]
    rvec_m = measured_points_repeated[i][3:]
    T_world_robot = transformation_matrix(tvec_m, rvec_m)
    print(
        f"Measured Point {i}: {tvec_m} with Yaw {rvec_m[2] * 180.0 / math.pi:.2f} deg"
    )

    # Compute T_cam_robot
    # T_cam_robot = T_cam_tag @ np.linalg.inv(T_robot_tag)
    T_world_tag = T_world_robot @ T_robot_tag
    T_tag_cam = np.linalg.inv(T_cam_tag)
    t_vec_tag_cam = T_tag_cam[:3, 3]
    r_vec_tag_cam, _ = cv2.Rodrigues(T_tag_cam[:3, :3])
    euler_tag_cam = R.from_matrix(T_tag_cam[:3, :3]).as_euler('zyx', degrees=True)
    print("t_vec_tag_cam (m):", t_vec_tag_cam)
    print("r_vec_tag_cam (rvec):", r_vec_tag_cam.flatten())
    print("euler_tag_cam (deg):", euler_tag_cam)
    T_world_cam = T_world_tag @ np.linalg.inv(T_cam_tag)

    print(f"Camera Reading {i}:")
    t_vec_world_cam = T_world_cam[:3, 3]
    r_vec_world_cam, _ = cv2.Rodrigues(T_world_cam[:3, :3])
    euler_world_cam = R.from_matrix(T_world_cam[:3, :3]).as_euler('zyx', degrees=True)
    print("Estimated Camera Position (m):", t_vec_world_cam)
    print("Estimated Camera Orientation (rvec):", r_vec_world_cam.flatten())
    print("Estimated Camera Orientation (euler deg):", euler_world_cam)
    # print("T_world_robot:\n", T_world_robot)
    # print("T_world_cam:\n", T_world_cam)
    print("-" * 30)
