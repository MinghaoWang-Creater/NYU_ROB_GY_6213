# External Libraries
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
import cv2
from scipy.optimize import least_squares
import json
from scipy.spatial.transform import Rotation as R


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
        measure = np.array(entry[0], dtype=np.float64) * 100 / 300  # Convert ft to m
        measure[2] = entry[0][2] / 180.0 * np.pi  # Convert degrees to radians
        measured_points[-1] = np.array(
            [measure[0], measure[1], 0, 0, 0, measure[2]], dtype=np.float64
        )
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

# fit the linear relationship between the measured points and camera readings
for i in range(len(measured_points)):
    for cam_read in camera_readings[i]:
        measured_points_repeated.append(measured_points[i])
        camera_readings_repeated.append(cam_read)
measured_points_repeated = np.array(measured_points_repeated, dtype=np.float64)
camera_readings_repeated = np.array(camera_readings_repeated, dtype=np.float64)


# Fit a linear model to the data using scipy
# Now mapping camera reading -> measured point (Ground Truth)
# params define T_cam_to_world
def residuals(params, measured_points, camera_readings):
    tvec = params[:3]
    rvec = params[3:]
    T_cam_to_world = transformation_matrix(tvec, rvec)
    
    residuals_list = []
    
    # Pre-compute rotation matrix for speed
    R_cam_to_world = T_cam_to_world[:3, :3]
    
    for i in range(len(measured_points)):
        # Camera Reading (Source)
        # Position
        p_cam = np.array([camera_readings[i][0], camera_readings[i][1], camera_readings[i][2], 1.0])
        # Rotation Vector -> Matrix
        cam_rvec = camera_readings[i][3:6]
        R_cam, _ = cv2.Rodrigues(cam_rvec)
        
        # Transform Position to World
        p_est_homog = T_cam_to_world @ p_cam
        p_est = p_est_homog[:3]
        
        # Transform Orientation to World: R_world_est = R_cam_to_world * R_cam
        R_est = R_cam_to_world @ R_cam
        # Convert back to Rodrigues for residual calculation
        rvec_est, _ = cv2.Rodrigues(R_est)
        rvec_est = rvec_est.flatten()
        
        # Measured Point (Target / Ground Truth)
        p_gt = measured_points[i][:3]  # x, y, z (usually z=0 in measurement setup)
        # measured_points layout described in previous block: [x, y, 0, 0, 0, theta]
        # The rotation part in measured_points is just [0, 0, theta]. 
        # Ideally we convert that Euler/Vector to Rotation Matrix or compare vectors directly if aligned.
        # Here we compare the resulting rotation vectors.
        # Note: measured_points[i][3:] is [0, 0, theta]. Let's assume this is a rotation vector format.
        rvec_gt = measured_points[i][3:6]
        
        # Calculate residuals
        res_t = p_est - p_gt
        res_r = rvec_est - rvec_gt
        
        residuals_list.extend(res_t)
        residuals_list.extend(res_r)
        
    return residuals_list

# Initial guess for optimization
initial_guess = np.zeros(6) 
# Perform optimization
result = least_squares(
    residuals, initial_guess, args=(measured_points_repeated, camera_readings_repeated)
)

# Extract optimized parameters
optimized_tvec = result.x[:3]
optimized_rvec = result.x[3:]
print("Optimized Camera->World Translation (tvec):", optimized_tvec)
print("Optimized Camera->World Rotation (rvec):", optimized_rvec)

# Check the fit
T_opt = transformation_matrix(optimized_tvec, optimized_rvec)
R_opt = T_opt[:3, :3]

est_world_points = []
est_world_rvecs = []

for i in range(len(camera_readings_repeated)):
    # Transform position
    p_cam = np.array([
        camera_readings_repeated[i][0], 
        camera_readings_repeated[i][1], 
        camera_readings_repeated[i][2], 
        1.0
    ])
    p_est = (T_opt @ p_cam)[:3]
    est_world_points.append(p_est)
    
    # Transform rotation
    cam_rvec = camera_readings_repeated[i][3:6]
    R_cam, _ = cv2.Rodrigues(cam_rvec)
    R_est = R_opt @ R_cam
    r_est_vec, _ = cv2.Rodrigues(R_est)
    est_world_rvecs.append(r_est_vec.flatten())

est_world_points = np.array(est_world_points)
est_world_rvecs = np.array(est_world_rvecs)

gt_points = measured_points_repeated[:, :3]
gt_rvecs = measured_points_repeated[:, 3:6]

# Calculate errors (Estimated - Ground Truth)
pos_errors = est_world_points - gt_points
rot_errors = est_world_rvecs - gt_rvecs

# Compute Std Dev for EKF noise covariance (R matrix)
# We assume the error is zero-mean for std calculation, or just use RMSE if bias exists.
# Standard Deviation of the error represents the sensor noise mapped to state space.

std_x = np.std(pos_errors[:, 0])
std_y = np.std(pos_errors[:, 1])
std_z = np.std(pos_errors[:, 2])

# For rotation (specifically Yaw/Theta about Z)
std_yaw = np.std(rot_errors[:, 2])

print("-" * 30)
print("Observation Noise Statistics (for EKF):")
print(f"Std Dev X: {std_x:.6f} m")
print(f"Std Dev Y: {std_y:.6f} m")
print(f"Std Dev Yaw: {std_yaw:.6f} rad")
print("-" * 30)

# MSE for general reference
mse_translation = np.mean(np.sum(pos_errors**2, axis=1))
mse_rotation = np.mean(np.sum(rot_errors**2, axis=1))

print("Mean Squared Error (Translation):", mse_translation)
print("Mean Squared Error (Rotation):", mse_rotation)