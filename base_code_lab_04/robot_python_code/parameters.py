# External libraries
import math
import numpy as np

# UDP parameters
localIP = "192.168.0.200" # Put your laptop computer's IP here 199
arduinoIP = "192.168.0.199" # Put your arduino's IP here 200
localPort = 4010
arduinoPort = 4010
bufferSize = 1024

# Camera parameters
camera_id = 1
marker_length = 0.095 #
aruco_s = marker_length / 2
aruco_obj_points = np.array([
    [-aruco_s,  aruco_s, 0],  # Top-left
    [ aruco_s,  aruco_s, 0],  # Top-right
    [ aruco_s, -aruco_s, 0],  # Bottom-right
    [-aruco_s, -aruco_s, 0]   # Bottom-left
], dtype=np.float32)
camera_matrix = np.array([[1043.8076991308567, 0.0, 567.6461917597342], 
                          [0.0, 1045.5137931512506, 339.3373122636459], 
                          [0.0, 0.0, 1.0]], dtype=np.float32)
dist_coeffs = np.array([-0.3611480623680854, -0.005147783780657834, 0.0006081964431444159, 0.0022315832746950598, 0.2948753728273205], dtype=np.float32)


# Robot parameters
num_robot_sensors = 2 # encoder, steering
num_robot_control_signals = 2 # speed, steering

# Logging parameters
max_num_lines_before_write = 1
filename_start = './data/robot_data'
data_name_list = ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal', 'state_mean', 'state_covariance']

# Experiment trial parameters
trial_time = 2000 # milliseconds
extra_trial_log_time = 2000 # milliseconds

# KF parameters
I3 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
covariance_plot_scale = 100

# PF parameters, modify the map and num particles as you see fit.
num_particles = 100
wall_corner_list = np.array([
    [-1, -2, -1, 2], # back wall
    [4, -2, 4, 2], # front wall
    [-1, -2, 4, -2], # right wall
    [-1, 2, 4, 2], # left wall
    [3, 0, 3, 0] # obstacle wall
    ]) * 0.33 # ft to m
wall_corner_list[-1][2] += 0.067
wall_corner_list[-1][3] -= 0.171
lidar_pos = np.array([0.177, -0.183/2, 0.])