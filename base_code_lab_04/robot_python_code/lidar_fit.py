import parameters
import robot_python_code
import numpy as np
from particle_filter import Map, State


# Open a file and return data in a form ready to plot
def get_file_data_for_pf(filename):
    data_loader = robot_python_code.DataLoader(filename)
    data_dict = data_loader.load()

    # The dictionary should have keys ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal']
    time_list = data_dict["time"]
    control_signal_list = data_dict["control_signal"]
    robot_sensor_signal_list = data_dict["robot_sensor_signal"]

    # Pack up what is needed for KF
    t0 = time_list[0]
    pf_data = []
    for i in range(len(time_list)):
        row = [time_list[i] - t0, control_signal_list[i], robot_sensor_signal_list[i]]
        pf_data.append(row)

    return pf_data


def convert_lidar_pos_to_robot(): ...


def load_file(filename):
    data = get_file_data_for_pf(filename)
    xy = filename.split("/")[-1].split(".")[0]
    x, y = xy.split("_")
    x = float(x) * 0.33  # ft to m
    y = float(y) * 0.33  # ft to m
    lidar_reading = []
    for row in data:
        sensor_signal: robot_python_code.RobotSensorSignal = row[2]
        lidar_reading.extend(
            [
                (
                    sensor_signal.convert_hardware_angle(angle),
                    sensor_signal.convert_hardware_distance(distance),
                )
                for angle, distance in zip(
                    sensor_signal.angles, sensor_signal.distances
                )
            ]
        )

    return x, y, lidar_reading


# find all files in the data folder and load them
import os


def load_all_files_in_data_folder(data_folder: str):
    files = os.listdir(data_folder)
    data = {}
    for file in files:
        if file.endswith(".pkl"):
            filename = os.path.join(data_folder, file)
            x, y, lidar_reading = load_file(filename)
            data[(x, y)] = lidar_reading
    return data


all_data = load_all_files_in_data_folder("static_data")
print(len(all_data))
print(all_data.keys())

map = Map(parameters.wall_corner_list)
# get the std deviation of the lidar readings at each (x, y) position
lidar_error = {pos: [] for pos in all_data.keys()}
for key, readings in all_data.items():
    print(f"Processing data for position {key} with {len(readings)} lidar readings...")
    robot_gt = np.array(key)  # robot ground truth position
    state = State(robot_gt[0], robot_gt[1], 0)  # we only care about x, y for the lidar
    # compute the error
    for angle, distance in readings:
        distance_gt, wall = map.simulate_lidar(state, angle)
        if np.isinf(distance_gt):
            continue  # skip if the ground truth distance is infinite (no wall in that direction)
        if np.abs(distance - distance_gt) <= 0.15:
            # print(f"At position {key}, angle {angle:.2f} rad, wall {wall} measured distance: {distance:.2f} m, ground truth distance: {distance_gt:.2f} m")
            lidar_error[key].append(distance - distance_gt)

# fit the bias and std deviation of the lidar error
flatten_error = [item for sublist in list(lidar_error.values()) for item in sublist]
# print("Lidar error: ", lidar_error.values())
bias = np.mean(flatten_error)
std_dev = np.std(flatten_error)

print("Lidar bias: ", bias)
print("Lidar std deviation: ", std_dev)
