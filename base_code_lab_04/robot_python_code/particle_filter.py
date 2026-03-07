# External libraries
import copy
import matplotlib.pyplot as plt
import math
import numpy as np
import random

# Local libraries
import parameters
import data_handling
from motion_models import MyMotionModel

from scipy.spatial.transform import Rotation as R

def estimate_pose_from_camera_measurement(camera_measurement):
    T_optimal = np.array(
        [
            [ 0.99796608,  0.01102755, -0.06278616,  0.0094643 ],
            [-0.01221634,  0.99975271, -0.01858174, -0.04666413],
            [ 0.06256572,  0.01931097,  0.99785401,  0.0046057 ],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ]
    )
    cam_meas_homog = np.array(
        [camera_measurement[0], camera_measurement[1], camera_measurement[2], 1],
        dtype=np.float64,
    )
    rot = T_optimal[:3, :3] @ R.from_rotvec(camera_measurement[3:]).as_matrix()
    theta = R.from_matrix(rot).as_euler("zyx", degrees=False)[0]
    world_meas_homog = T_optimal @ cam_meas_homog
    return np.array([world_meas_homog[0], world_meas_homog[1], theta], dtype=np.float64)


# Helper function to make sure all angles are between -pi and pi
def angle_wrap(angle):
    while angle > math.pi:
        angle -= 2*math.pi
    while angle < -math.pi:
        angle += 2*math.pi
    return angle

# Helper class to store and manipulate your states.
class State:

    # Constructor
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    # Get the euclidean distance between 2 states
    def distance_to(self, other_state):
        return math.sqrt(math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2))
        
    # Get the distance squared between two states
    def distance_to_squared(self, other_state):
        return math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2)

    # return a deep copy of the state.
    def deepcopy(self):
        return copy.deepcopy(self)
        
    # Print the state
    def print(self):
        print("State: ",self.x, self.y, self.theta)


# Class to store walls as objects (specifically when represented as line segments in a 2D map.)
class Wall:

    # Constructor
    def __init__(self, wall_corners):
        self.corner1 = State(wall_corners[0], wall_corners[1], 0)
        self.corner2 = State(wall_corners[2], wall_corners[3], 0)
        self.corner1_mm = State(wall_corners[0] * 1000, wall_corners[1] * 1000, 0)
        self.corner2_mm = State(wall_corners[2] * 1000, wall_corners[3] * 1000, 0)
        
        self.m = (wall_corners[3] - wall_corners[1])/(0.0001 + wall_corners[2] -  wall_corners[0])
        self.b = wall_corners[3] - self.m * wall_corners[2]
        self.b_mm =  wall_corners[3] * 1000 - self.m * wall_corners[2] * 1000
        self.length = self.corner1.distance_to(self.corner2)
        self.length_mm_squared = self.corner1_mm.distance_to_squared(self.corner2_mm)
        
        if self.m > 1000:
            self.vertical = True
        else:
            self.vertical = False
        if abs(self.m) < 0.1:
            self.horizontal = True
        else:
            self.horizontal = False


# A class to store 2D maps
class Map:
    def __init__(self, wall_corner_list):
        self.wall_list = []
        for wall_corners in wall_corner_list:
            self.wall_list.append(Wall(wall_corners))
        min_x = 999999
        max_x = -99999
        min_y = 999999
        max_y = -99999
        for wall in self.wall_list:
            min_x = min(min_x, min(wall.corner1.x, wall.corner2.x))
            max_x = max(max_x, max(wall.corner1.x, wall.corner2.x))
            min_y = min(min_y, min(wall.corner1.y, wall.corner2.y))
            max_y = max(max_y, max(wall.corner1.y, wall.corner2.y))
        border = 0.5
        self.plot_range = [min_x - border, max_x + border, min_y - border, max_y + border]
        
        self.particle_range = [min_x , max_x , min_y, max_y]

    # Function to calculate the distance between any state and its closest wall, accounting for directon of the state.
    def closest_distance_to_walls(self, state):
        closest_distance = np.inf
        dist = [self.get_distance_to_wall(state, wall) for wall in self.wall_list]
        closest_distance = min(dist)
        
        return closest_distance, np.argmin(dist)
        
    # Function to get distance to a wall from a state, in the direction of the state's theta angle.
    # Or return the distance currently believed to be the closest if its closer.
    def get_distance_to_wall(self, state, wall):
        # 1. Ray: Origin (x, y) and Direction (dx, dy)
        x, y, theta = state.x, state.y, state.theta
        dx = math.cos(theta)
        dy = math.sin(theta)

        # 2. Get Wall Boundaries (handles corners in any order)
        x_min, x_max = min(wall.corner1.x, wall.corner2.x), max(wall.corner1.x, wall.corner2.x)
        y_min, y_max = min(wall.corner1.y, wall.corner2.y), max(wall.corner1.y, wall.corner2.y)

        # 3. If robot is already inside the brick, distance is 0
        # Add a tiny epsilon (1e-9) to handle floating point edges
        if (x_min - 1e-9 <= x <= x_max + 1e-9) and (y_min - 1e-9 <= y <= y_max + 1e-9):
            return 0.0

        hits = []

        # 4. Check Vertical Boundaries (X = constant)
        if abs(dx) > 1e-9:
            for wx in [x_min, x_max]:
                t = (wx - x) / dx
                if t > 0:
                    hit_y = y + t * dy
                    if y_min <= hit_y <= y_max:
                        hits.append(t)

        # 5. Check Horizontal Boundaries (Y = constant)
        if abs(dy) > 1e-9:
            for wy in [y_min, y_max]:
                t = (wy - y) / dy
                if t > 0:
                    hit_x = x + t * dx
                    if x_min <= hit_x <= x_max:
                        hits.append(t)
        # print(f"State at ({x:.2f}, {y:.2f}), angle {theta:.2f} rad, wall {self.wall_list.index(wall)}, distance to wall: {min(hits) if hits else float('inf'):.2f} m")
        # 3. Return the closest hit
        return min(hits) if hits else float('inf')

    def simulate_lidar(self, state, angle):
        '''
        simulate a lidar reading from a state and angle
        '''
        lidar_state = State(state.x + math.cos(state.theta) * parameters.lidar_pos[0] - math.sin(state.theta) * parameters.lidar_pos[1],
                            state.y + math.sin(state.theta) * parameters.lidar_pos[0] + math.cos(state.theta) * parameters.lidar_pos[1],
                            state.theta + angle)
        return self.closest_distance_to_walls(lidar_state)

# Class to hold a particle
class Particle:
    
    def __init__(self):
        self.state = State(0, 0, 0)
        self.weight = 1
        self.model = MyMotionModel([0, 0, 0], 0)
        self.model.step_with_noise = True
        self.model.return_noise_scale = False
        
    # Function to create a new random particle state within a range
    def randomize_uniformly(self, xy_range):
        x = np.random.uniform(xy_range[0], xy_range[1])
        y = np.random.uniform(xy_range[2], xy_range[3])
        theta = np.random.uniform(-math.pi, math.pi)
        self.state = State(x, y, theta)
        self.weight = 1

    # Function to create a new random particle state with a normal distribution
    def randomize_around_initial_state(self, initial_state, state_stdev):
        x = np.random.normal(initial_state.x, state_stdev.x)
        y = np.random.normal(initial_state.y, state_stdev.y)
        theta = angle_wrap(np.random.normal(initial_state.theta, state_stdev.theta))
        self.state = State(x, y, theta)
        self.weight = 1
        
    # Function to take a particle and "randomly" propagate it forward according to a motion model.
    def propagate_state(self, last_state, delta_encoder_counts, steering, delta_t):
        self.model.state = [last_state.x, last_state.y, last_state.theta]
        new_state_arr = self.model.step_update(delta_encoder_counts, steering, delta_t)
        self.state = State(new_state_arr[0], new_state_arr[1], angle_wrap(new_state_arr[2]))
        
    # Function to determine a particles weight based how well the lidar measurement matches up with the map.
    def calculate_weight(self, lidar_signal, map):
        self.weight = 1
        for i in range(len(lidar_signal.angles)):
            measured_distance = lidar_signal.convert_hardware_distance(lidar_signal.distances[i]) - parameters.lidar_bias
            angle = lidar_signal.convert_hardware_angle(lidar_signal.angles[i])
            expected_distance, _ = map.simulate_lidar(self.state, angle)
            # Skip ray if map returns no hit or residual is too large (outlier)
            if expected_distance == math.inf:
                continue
            if abs(expected_distance - measured_distance) > parameters.lidar_outlier_threshold:
                continue
            self.weight *= self.gaussian(expected_distance, measured_distance)
        
    # Return the normal distribution function output.
    def gaussian(self, expected_distance, distance):
        return math.exp(-math.pow(expected_distance - distance, 2)/ 2 / parameters.distance_variance)

    # Deep copy the particle
    def deepcopy(self):
        return copy.deepcopy(self)
        
    # Print the particle
    def print(self):
        print("Particle: ", self.state.x, self.state.y, self.state.theta, " w: ", self.weight)


# This class holds the collection of particles.
class ParticleSet:
    
    # Constructor, which calls the known start or unknown start initialization.
    def __init__(self, num_particles, xy_range, initial_state, state_stdev, known_start_state):
        self.num_particles = num_particles
        self.particle_list = []
        if known_start_state:
            self.generate_initial_state_particles(initial_state, state_stdev)
        else:
            self.generate_uniform_random_particles(xy_range)
        self.mean_state = State(0, 0, 0)
        self.update_mean_state()
        
    # Function to reset particles and random locations in the workspace.
    def generate_uniform_random_particles(self, xy_range):
        for i in range(self.num_particles):
            random_particle = Particle()
            random_particle.randomize_uniformly(xy_range)
            self.particle_list.append(random_particle)

    # Function to reset particles, normally distributed around the initial state. 
    def generate_initial_state_particles(self, initial_state, state_stdev):
        for i in range(self.num_particles):
            random_particle = Particle()
            random_particle.randomize_around_initial_state(initial_state, state_stdev)
            self.particle_list.append(random_particle)

    # Function to resample the particles set, i.e. make a new one with more copies of particles with higher weights.  
    def resample(self, max_weight):
        # Multinomial resampling: randomly draw N particles with replacement,
        # with probability proportional to each particle's weight.
        weights = np.array([p.weight for p in self.particle_list], dtype=np.float64)
        total_weight = weights.sum()
        if total_weight == 0:
            print("Warning: All particle weights are zero. Resampling uniformly.")
            # All weights zero — fall back to uniform weights
            weights[:] = 1.0
            total_weight = float(self.num_particles)
        weights /= total_weight  # normalize to a probability distribution
        indices = np.random.choice(self.num_particles, size=self.num_particles, replace=True, p=weights)
        self.particle_list = [self.particle_list[i].deepcopy() for i in indices]
            
    # Calculate the mean state. 
    def update_mean_state(self):
        ## Be careful how you calculate the mean theta
        n = len(self.particle_list)
        if n == 0:
            return
        self.mean_state.x = sum(p.state.x for p in self.particle_list) / n
        self.mean_state.y = sum(p.state.y for p in self.particle_list) / n
        # Circular mean for theta
        sin_sum = sum(math.sin(p.state.theta) for p in self.particle_list)
        cos_sum = sum(math.cos(p.state.theta) for p in self.particle_list)
        self.mean_state.theta = math.atan2(sin_sum, cos_sum)
        
    # Print the particle set. Useful for debugging.
    def print_particles(self):
        for particle in self.particle_list:
            particle.print()
        print()

# Class to hold the particle filter and its functions.
class ParticleFilter:
    
    # Constructor
    def __init__(self, num_particles, map, initial_state, state_stdev, known_start_state, encoder_counts_0):
        self.map = map
        self.particle_set = ParticleSet(num_particles, map.particle_range, initial_state, state_stdev, known_start_state)
        self.state_estimate = self.particle_set.mean_state
        self.state_estimate_list = []
        self.last_time = 0
        self.last_encoder_counts = encoder_counts_0

    # Update the states given new measurements
    def update(self, odometery_signal, measurement_signal, delta_t):
        self.prediction(odometery_signal, delta_t)
        if len(measurement_signal.angles)>0:
            self.correction(measurement_signal)
        self.particle_set.update_mean_state()
        self.state_estimate_list.append(self.state_estimate.deepcopy())

    # Predict the current state from the last state.
    def prediction(self, odometry_signal, delta_t):
        encoder_counts = odometry_signal[0]
        steering = odometry_signal[1]
        delta_encoder_counts = encoder_counts - self.last_encoder_counts
        for particle in self.particle_set.particle_list:
            last_state = particle.state.deepcopy()
            particle.propagate_state(last_state, delta_encoder_counts, steering, delta_t)
        self.last_encoder_counts = encoder_counts
        
    # Corrrect the predicted states.
    def correction(self, measurement_signal):
        for particle in self.particle_set.particle_list:
            particle.calculate_weight(measurement_signal, self.map)
        max_weight = max(p.weight for p in self.particle_set.particle_list)
        self.particle_set.resample(max_weight)
        
    # Output to terminal the mean state.
    def print_state_estimate(self):
        print("Mean state: ", self.particle_set.mean_state.x, self.particle_set.mean_state.y, self.particle_set.mean_state.theta)
    

# Class to help with plotting PF data.
class ParticleFilterPlot:

    # Constructor
    def __init__(self, map):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig
        self.map = map

    # Clear and update the plot with new PF data
    def update(self, state_mean, particle_set, lidar_signal, hold_show_plot):
        plt.clf()
        
        # Plot walls
        for wall in self.map.wall_list:
            plt.plot([wall.corner1.x, wall.corner2.x],[wall.corner1.y, wall.corner2.y],'k')

        # Plot lidar
        for i in range(len(lidar_signal.angles)):
            distance = lidar_signal.convert_hardware_distance(lidar_signal.distances[i])
            angle = lidar_signal.convert_hardware_angle(lidar_signal.angles[i]) + state_mean.theta
            x_ray = [state_mean.x, state_mean.x + distance * math.cos(angle)]
            y_ray = [state_mean.y, state_mean.y + distance * math.sin(angle)]
            plt.plot(x_ray, y_ray, 'r')


        # Plot state estimate
        plt.plot(state_mean.x, state_mean.y,'ro')
        plt.plot([state_mean.x, state_mean.x+ self.dir_length*math.cos(state_mean.theta) ], [state_mean.y, state_mean.y+ self.dir_length*math.sin(state_mean.theta) ],'r')
        x_particles, y_particles = self.to_plot_data(particle_set)
        plt.plot(x_particles, y_particles, 'g.')
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.axis(self.map.plot_range)
        plt.grid()
        if hold_show_plot:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.1)

    # Helper function to make the particles easy to plot.
    def to_plot_data(self, particle_set):
        x_list = []
        y_list = []
        for p in particle_set.particle_list:
            x_list.append(p.state.x)
            y_list.append(p.state.y)
        return x_list, y_list
        

# Function used to test your PF offline with logged data.
def offline_pf():
    
    # Make a map of walls
    map = Map(parameters.wall_corner_list)

    # Get data to filter
    filename = './data/robot_data_0_0_25_02_26_21_41_33.pkl'
    pf_data = data_handling.get_file_data_for_pf(filename)

    # Instantiate PF with no initial guess
    particle_filter = ParticleFilter(parameters.num_particles, map, 
                                     initial_state = State(0.5, 2.0, 1.57), 
                                     state_stdev = State(0.1,0.1,0.1), 
                                     known_start_state=True, 
                                     encoder_counts_0=pf_data[0][2].encoder_counts)

    # Create plotting tool for particles
    particle_filter_plot = ParticleFilterPlot(map)

    # Loop over pf data
    for t in range(1, len(pf_data)):
        row = pf_data[t]
        delta_t = pf_data[t][0] - pf_data[t-1][0] # time step size
        u_t = np.array([row[2].encoder_counts, row[2].steering]) # robot_sensor_signal
        z_t = row[2] # lidar_sensor_signal

        # Run the PF for a time step
        particle_filter.update(u_t, z_t, delta_t)
        particle_filter_plot.update(particle_filter.particle_set.mean_state, particle_filter.particle_set, z_t, False)

    particle_filter_plot.update(particle_filter.particle_set.mean_state, particle_filter.particle_set, z_t, False)

        


####### MAIN #######
if __name__ == '__main__':
    offline_pf()
