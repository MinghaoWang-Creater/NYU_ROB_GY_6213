# External libraries
import copy
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.patches import Ellipse
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
            [0.99796608, 0.01102755, -0.06278616, 0.0094643],
            [-0.01221634, 0.99975271, -0.01858174, 0],#-0.04666413],
            [0.06256572, 0.01931097, 0.99785401, 0.0046057],
            [0.0, 0.0, 0.0, 1.0],
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
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
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
        return math.sqrt(
            math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2)
        )

    # Get the distance squared between two states
    def distance_to_squared(self, other_state):
        return math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2)

    # return a deep copy of the state.
    def deepcopy(self):
        return copy.deepcopy(self)

    # Print the state
    def print(self):
        print("State: ", self.x, self.y, self.theta)


# Class to store walls as objects (specifically when represented as line segments in a 2D map.)
class Wall:
    # Constructor
    def __init__(self, wall_corners):
        self.corner1 = State(wall_corners[0], wall_corners[1], 0)
        self.corner2 = State(wall_corners[2], wall_corners[3], 0)
        self.corner1_mm = State(wall_corners[0] * 1000, wall_corners[1] * 1000, 0)
        self.corner2_mm = State(wall_corners[2] * 1000, wall_corners[3] * 1000, 0)

        self.m = (wall_corners[3] - wall_corners[1]) / (
            0.0001 + wall_corners[2] - wall_corners[0]
        )
        self.b = wall_corners[3] - self.m * wall_corners[2]
        self.b_mm = wall_corners[3] * 1000 - self.m * wall_corners[2] * 1000
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
        self.plot_range = [
            min_x - border,
            max_x + border,
            min_y - border,
            max_y + border,
        ]

        self.particle_range = [min_x, max_x, min_y, max_y]

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
        x_min, x_max = (
            min(wall.corner1.x, wall.corner2.x),
            max(wall.corner1.x, wall.corner2.x),
        )
        y_min, y_max = (
            min(wall.corner1.y, wall.corner2.y),
            max(wall.corner1.y, wall.corner2.y),
        )

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
        return min(hits) if hits else float("inf")

    def simulate_lidar(self, state, angle):
        """
        simulate a lidar reading from a state and angle
        """
        lidar_state = State(
            state.x
            + math.cos(state.theta) * parameters.lidar_pos[0]
            - math.sin(state.theta) * parameters.lidar_pos[1],
            state.y
            + math.sin(state.theta) * parameters.lidar_pos[0]
            + math.cos(state.theta) * parameters.lidar_pos[1],
            state.theta + angle,
        )
        return self.closest_distance_to_walls(lidar_state)


def robot_pose_to_lidar_pose(robot_pose):
    """
    Convert a robot pose (x, y, theta) to the corresponding lidar pose (x, y, theta)
    """
    x = (
        robot_pose[0]
        + math.cos(robot_pose[2]) * parameters.lidar_pos[0]
        - math.sin(robot_pose[2]) * parameters.lidar_pos[1]
    )
    y = (
        robot_pose[1]
        + math.sin(robot_pose[2]) * parameters.lidar_pos[0]
        + math.cos(robot_pose[2]) * parameters.lidar_pos[1]
    )
    theta = robot_pose[2]
    return np.array([x, y, theta], dtype=np.float64)


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
        self.state = State(
            new_state_arr[0], new_state_arr[1], angle_wrap(new_state_arr[2])
        )

    # Function to determine a particles weight based how well the lidar measurement matches up with the map.
    def calculate_weight(self, lidar_signal, map):
        self.weight = 0
        self.angle_count = 0
        log_weight = 0.0
        # Check if particle is outside the map bounds
        x_min, x_max = map.particle_range[0], map.particle_range[1]
        y_min, y_max = map.particle_range[2], map.particle_range[3]
        if not (x_min <= self.state.x <= x_max and y_min <= self.state.y <= y_max):
            self.weight = 0
            return
        for i in range(len(lidar_signal.angles)):
            measured_distance = (
                lidar_signal.convert_hardware_distance(lidar_signal.distances[i])
                - parameters.lidar_bias
            )
            angle = lidar_signal.convert_hardware_angle(lidar_signal.angles[i])
            expected_distance, _ = map.simulate_lidar(self.state, angle)
            # Skip ray if map returns no hit or residual is too large (outlier)
            # if expected_distance == math.inf:
            #     continue
            log_weight += self.gaussian(expected_distance, measured_distance) + 1e-300
            self.angle_count += 1
        if self.angle_count > len(lidar_signal.angles) / 2:
            self.weight = math.exp(log_weight)

    # Return the normal distribution function output.
    def gaussian(self, expected_distance, distance):
        return (
            -math.pow(expected_distance - distance, 2)
            / 2
            / parameters.distance_variance
        )

    # Deep copy the particle
    def deepcopy(self):
        return copy.deepcopy(self)

    # Print the particle
    def print(self):
        print(
            "Particle: ",
            self.state.x,
            self.state.y,
            self.state.theta,
            " w: ",
            self.weight,
        )


# This class holds the collection of particles.
class ParticleSet:
    # Constructor, which calls the known start or unknown start initialization.
    def __init__(
        self, num_particles, xy_range, initial_state, state_stdev, known_start_state
    ):
        self.num_particles = num_particles
        self.particle_list = []
        self.xy_range = xy_range
        if known_start_state:
            self.generate_initial_state_particles(initial_state, state_stdev)
        else:
            self.generate_uniform_random_particles(xy_range)
        self.mean_state = initial_state
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
            self.particle_list.clear()
            self.generate_uniform_random_particles(self.xy_range)
            return
            # All weights zero — fall back to uniform weights
            weights[:] = 1.0
            total_weight = float(self.num_particles)
        weights /= total_weight  # normalize to a probability distribution
        indices = np.random.choice(
            self.num_particles, size=self.num_particles, replace=True, p=weights
        )
        # self.particle_list = [self.particle_list[i].deepcopy()
        #                                 for i in indices]
        #         # Add small random noise to resampled particles to avoid degeneracy
        self.particle_list = [self.particle_list[i].deepcopy() for i in indices]
        # for p in self.particle_list:
        #     p.state.x += np.random.normal(0, parameters.resample_noise_x)
        #     p.state.y += np.random.normal(0, parameters.resample_noise_y)
        #     p.state.theta = angle_wrap(
        #         p.state.theta + np.random.normal(0, parameters.resample_noise_theta)
        #     )

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
    def __init__(
        self,
        num_particles,
        map,
        initial_state,
        state_stdev,
        known_start_state,
        encoder_counts_0,
    ):
        self.map = map
        self.particle_set = ParticleSet(
            num_particles,
            map.particle_range,
            initial_state,
            state_stdev,
            known_start_state,
        )
        self.state_estimate = self.particle_set.mean_state
        self.state_estimate_list = []
        self.last_time = 0
        self.last_encoder_counts = encoder_counts_0

    # Update the states given new measurements
    def update(self, odometery_signal, measurement_signal, delta_t):
        self.prediction(odometery_signal, delta_t)
        if len(measurement_signal.angles) > 0:
            self.correction(measurement_signal)
        else:
            print("No valid lidar measurements received, skipping correction step.")
        self.particle_set.update_mean_state()
        self.state_estimate_list.append(self.state_estimate.deepcopy())

    # Predict the current state from the last state.
    def prediction(self, odometry_signal, delta_t):
        encoder_counts = odometry_signal[0]
        steering = odometry_signal[1]
        delta_encoder_counts = encoder_counts - self.last_encoder_counts
        for particle in self.particle_set.particle_list:
            last_state = particle.state.deepcopy()
            particle.propagate_state(
                last_state, delta_encoder_counts, steering, delta_t
            )
        self.last_encoder_counts = encoder_counts

    # Corrrect the predicted states.
    def correction(self, measurement_signal):
        for particle in self.particle_set.particle_list:
            particle.calculate_weight(measurement_signal, self.map)
        max_weight = max(p.weight for p in self.particle_set.particle_list)
        sorted_particle = sorted(self.particle_set.particle_list, key=lambda p: p.weight, reverse=True)
        print("sorted", [p.weight for p in sorted_particle[:10]])  # Print top 5 particles by weight
        for p in sorted_particle[:10]:  # Print top 5 particles by weight
            print(f"Particle weight: {p.weight:.3e}")
            print(f"Particle angle count: {p.angle_count}")
            print(f"Particle state: x={p.state.x:.3f}, y={p.state.y:.3f}, theta={p.state.theta:.3f}")
        self.particle_set.resample(max_weight)
        print("normalized weights after resampling:", [p.weight for p in self.particle_set.particle_list[:5]])  # Print weights of top 5 particles after resampling

    # Output to terminal the mean state.
    def print_state_estimate(self):
        print(
            "Mean state: ",
            self.particle_set.mean_state.x,
            self.particle_set.mean_state.y,
            self.particle_set.mean_state.theta,
        )


# Draw each wall as a rectangle defined by two diagonal corners
def draw_walls_as_bricks(
    wall_list, thickness=0.04, color="#8B4513", first_label="Wall"
):
    ax = plt.gca()
    for i, wall in enumerate(wall_list):
        x1, y1 = wall.corner1.x, wall.corner1.y
        x2, y2 = wall.corner2.x, wall.corner2.y

        # Calculate width and height from the two diagonal corners
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Lower-left corner (minimum x and y)
        lower_left_x = min(x1, x2)
        lower_left_y = min(y1, y2)

        label = first_label if i == 0 else "_nolegend_"

        # For point obstacles (width and height both ~0), draw a small square
        if width < 1e-6 and height < 1e-6:
            thickness_sq = 0.1  # Small square size for point obstacles
            rect = matplotlib.patches.Rectangle(
                (x1 - thickness_sq / 2, y1 - thickness_sq / 2),
                thickness_sq,
                thickness_sq,
                linewidth=1,
                edgecolor="k",
                facecolor=color,
                label=label,
            )
        # For line segments (either width or height is ~0), draw with thickness
        elif width < 1e-6:
            rect = matplotlib.patches.Rectangle(
                (lower_left_x - thickness / 2, lower_left_y),
                thickness,
                height,
                linewidth=1,
                edgecolor="k",
                facecolor=color,
                label=label,
            )
        elif height < 1e-6:
            rect = matplotlib.patches.Rectangle(
                (lower_left_x, lower_left_y - thickness / 2),
                width,
                thickness,
                linewidth=1,
                edgecolor="k",
                facecolor=color,
                label=label,
            )
        # For actual rectangular regions, draw the rectangle from diagonal corners
        else:
            rect = matplotlib.patches.Rectangle(
                (lower_left_x, lower_left_y),
                width,
                height,
                linewidth=1,
                edgecolor="k",
                facecolor=color,
                label=label,
            )

        ax.add_patch(rect)


# Class to help with plotting PF data.
class ParticleFilterPlot:
    # Constructor
    def __init__(self, map):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig
        self.map: Map = map

    # Clear and update the plot with new PF data
    def update(
        self, state_mean, particle_set, lidar_signal, hold_show_plot, camera_pose=None
    ):
        plt.clf()

        # Plot walls as bricks
        draw_walls_as_bricks(self.map.wall_list)

        # Plot lidar measurements
        # for i in range(len(lidar_signal.angles)):
        #     distance = lidar_signal.convert_hardware_distance(lidar_signal.distances[i])
        #     angle = lidar_signal.convert_hardware_angle(lidar_signal.angles[i]) + state_mean.theta
        #     lidar_pose = robot_pose_to_lidar_pose((state_mean.x, state_mean.y, state_mean.theta))
        #     x_ray = [lidar_pose[0], lidar_pose[0] + distance * math.cos(angle)]
        #     y_ray = [lidar_pose[1], lidar_pose[1] + distance * math.sin(angle)]
        #     plt.plot(x_ray, y_ray, 'r', alpha=0.5,
        #              label='Lidar Measurement' if i == 0 else '_nolegend_')

        # Plot camera ground truth
        if camera_pose is not None:
            plt.plot(
                camera_pose[0],
                camera_pose[1],
                "g^",
                markersize=8,
                label="Ground Truth (Camera)",
            )
            plt.plot(
                [
                    camera_pose[0],
                    camera_pose[0] + self.dir_length * math.cos(camera_pose[2]),
                ],
                [
                    camera_pose[1],
                    camera_pose[1] + self.dir_length * math.sin(camera_pose[2]),
                ],
                "g-",
                label="_nolegend_",
            )
            for i in range(len(lidar_signal.angles)):
                distance = lidar_signal.convert_hardware_distance(
                    lidar_signal.distances[i]
                )
                angle = (
                    lidar_signal.convert_hardware_angle(lidar_signal.angles[i])
                    + camera_pose[2]
                )
                lidar_pose = robot_pose_to_lidar_pose(camera_pose)
                x_ray = [lidar_pose[0], lidar_pose[0] + distance * math.cos(angle)]
                y_ray = [lidar_pose[1], lidar_pose[1] + distance * math.sin(angle)]
                plt.plot(
                    x_ray,
                    y_ray,
                    "r",
                    alpha=0.5,
                    label="Lidar Measurement (GT)" if i == 0 else "_nolegend_",
                )

        # Plot state estimate
        plt.plot(state_mean.x, state_mean.y, "bo", label="PF Estimation")
        plt.plot(
            [state_mean.x, state_mean.x + self.dir_length * math.cos(state_mean.theta)],
            [state_mean.y, state_mean.y + self.dir_length * math.sin(state_mean.theta)],
            "b-",
            label="Heading",
        )
        x_particles, y_particles = self.to_plot_data(particle_set)
        plt.plot(x_particles, y_particles, "b.", alpha=0.5, label="Particles")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis(self.map.plot_range)
        plt.legend(loc="upper left", fontsize="8", framealpha=0.3)
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
        print(f"Particle set has {len(particle_set.particle_list)} particles.")
        for p in particle_set.particle_list:
            # print(
            #     f"Particle state: x={p.state.x:.3f}, y={p.state.y:.3f}, theta={p.state.theta:.3f}, weight={p.weight:.3e}"
            # )
            x_list.append(p.state.x)
            y_list.append(p.state.y)
        return x_list, y_list


# Plot a single PF trajectory with camera ground truth and covariance ellipses.
def plot_traj_single(state_list, camera_list, covariance_list, map, label, color="r"):
    state = np.array([[s.x, s.y, s.theta] for s in state_list])
    camera = np.array(camera_list) if camera_list is not None else None

    # Mark start and end
    plt.plot(
        state[0, 0], state[0, 1], f"{color}*", markersize=15, label=f"{label} Start"
    )
    plt.plot(
        state[-1, 0], state[-1, 1], f"{color}+", markersize=15, label=f"{label} End"
    )
    plt.plot(
        state[:, 0],
        state[:, 1],
        f"{color}o-",
        markersize=3,
        label=f"{label} Estimation",
    )

    if camera is not None:
        plt.plot(
            camera[:, 0],
            camera[:, 1],
            "ko-",
            markersize=3,
            label="Ground Truth (Camera)",
            alpha=0.4,
        )

    # Covariance ellipses every 20 steps
    for i in range(0, len(state_list), 20):
        cov = covariance_list[i]
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(np.maximum(lambda_, 0)) * parameters.cov_plot_scale
        xy = (state[i, 0], state[i, 1])
        angle = np.rad2deg(np.arctan2(*v[:, 0][::-1]))
        ell = plt.matplotlib.patches.Ellipse(
            xy,
            width=lambda_[0],
            height=lambda_[1],
            angle=angle,
            alpha=0.3,
            facecolor=color,
        )
        plt.gca().add_artist(ell)

    # Draw walls as bricks
    draw_walls_as_bricks(map.wall_list)


# Plot one or multiple PF trajectories.
def plot_traj(state_raw, camera_raw, covariance_raw, map):
    multiple_traj = isinstance(state_raw[0], (list, tuple))
    plt.cla()
    if multiple_traj:
        colors = ["r", "g", "b", "c", "m", "y"]
        for i in range(len(state_raw)):
            plot_traj_single(
                state_raw[i],
                camera_raw[i] if i == len(state_raw) - 1 else None,
                covariance_raw[i],
                map,
                label=f"PF {i + 1}",
                color=colors[i % len(colors)],
            )
        plt.legend(loc="upper left", fontsize="8", ncol=4, framealpha=0.3)
    else:
        plot_traj_single(
            state_raw, camera_raw, covariance_raw, map, label="PF Estimate"
        )
        plt.legend()

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid()
    plt.axis("equal")
    plt.show()


# Run the PF on one dataset and return the trajectory, camera ground truth, and covariances.
def run_test(
    initial_state, state_stdev, pf_data, map, known_start_state=True, realtime=False
):
    print(
        f"Initial PF state: x={initial_state.x:.3f}, y={initial_state.y:.3f}, theta={initial_state.theta:.3f}"
    )

    particle_filter = ParticleFilter(
        parameters.num_particles,
        map,
        initial_state=initial_state,
        state_stdev=state_stdev,
        known_start_state=known_start_state,
        encoder_counts_0=pf_data[0][2].encoder_counts,
    )
    particle_filter_plot = ParticleFilterPlot(map)

    state_traj = [particle_filter.particle_set.mean_state.deepcopy()]
    camera_traj = [estimate_pose_from_camera_measurement(pf_data[0][5])]
    covariance_traj = []  # placeholder for t=0

    for t in range(1, len(pf_data)):
        row = pf_data[t]
        delta_t = pf_data[t][0] - pf_data[t - 1][0]
        u_t = np.array([row[2].encoder_counts, row[2].steering])
        z_t = row[2]  # RobotSensorSignal with lidar

        particle_filter.update(u_t, z_t, delta_t)

        mean = particle_filter.particle_set.mean_state
        state_traj.append(mean.deepcopy())
        camera_traj.append(estimate_pose_from_camera_measurement(row[5]))

        # 2x2 position covariance from particle spread
        xs = np.array([p.state.x for p in particle_filter.particle_set.particle_list])
        ys = np.array([p.state.y for p in particle_filter.particle_set.particle_list])
        cov = np.cov(np.stack([xs, ys]))
        print(f"Covariance at t={t}:")
        print(cov)
        covariance_traj.append(cov)

        print(f"  t={t:4d}  x={mean.x:.3f}  y={mean.y:.3f}  theta={mean.theta:.3f}")
        if realtime:
            particle_filter_plot.update(
                mean,
                particle_filter.particle_set,
                z_t,
                False,
                camera_pose=camera_traj[-1],
            )
            input("Press Enter to continue to next step...")

    return state_traj, camera_traj, covariance_traj


# Run the PF offline on a data file and plot the result.
def offline_pf(multi_random_mode=False, num_random_tests=5):
    map = Map(parameters.wall_corner_list)

    filename = "./data/offline/simple_0_0.pkl"
    pf_data = data_handling.get_file_data_for_pf(filename)

    # Use first camera measurement as initial state
    cam_0 = estimate_pose_from_camera_measurement(pf_data[0][5])
    base_initial_state = State(cam_0[0], cam_0[1], cam_0[2])
    state_stdev = State(0.5, 0.5, 0.5)
    
    if multi_random_mode:
        # Test multiple random initial states
        all_state_trajs = []
        all_camera_trajs = []
        all_covariance_trajs = []
        
        for i in range(num_random_tests):
            print(f"\n========== Running test {i+1}/{num_random_tests} ==========")
            # Generate random initial state
            initial_state = State(
                np.random.normal(base_initial_state.x, state_stdev.x),
                np.random.normal(base_initial_state.y, state_stdev.y),
                angle_wrap(np.random.normal(base_initial_state.theta, state_stdev.theta))
            )
            initial_state.x = np.clip(initial_state.x, map.particle_range[0], map.particle_range[1])
            initial_state.y = np.clip(initial_state.y, map.particle_range[2], map.particle_range[3])
            
            state_traj, camera_traj, covariance_traj = run_test(
                initial_state, state_stdev, pf_data, map, known_start_state=True, realtime=False
            )
            
            all_state_trajs.append(state_traj)
            all_camera_trajs.append(camera_traj)
            all_covariance_trajs.append(covariance_traj)
        
        # Plot multiple trajectories
        plot_traj(all_state_trajs, all_camera_trajs, all_covariance_trajs, map)
    else:
        # Single test mode (original behavior)
        initial_state = State(
            np.random.normal(base_initial_state.x, state_stdev.x),
            np.random.normal(base_initial_state.y, state_stdev.y),
            angle_wrap(np.random.normal(base_initial_state.theta, state_stdev.theta))
        )
        initial_state.x = np.clip(initial_state.x, map.particle_range[0], map.particle_range[1])
        initial_state.y = np.clip(initial_state.y, map.particle_range[2], map.particle_range[3])

        state_traj, camera_traj, covariance_traj = run_test(
            initial_state, state_stdev, pf_data, map, known_start_state=True, realtime=False
        )
        plot_traj(state_traj, camera_traj, covariance_traj, map)


# Plot only the logged EKF state mean and camera ground truth from a saved file (no re-running PF).
def plot_traj_only():
    map = Map(parameters.wall_corner_list)

    filename = "./data/robot_data_0_0_25_02_26_21_41_33.pkl"
    pf_data = data_handling.get_file_data_for_pf(filename)

    state_list = []
    camera_list = []
    covariance_list = []

    for t in range(1, len(pf_data)):
        row = pf_data[t]
        ekf_mean = row[3]  # logged EKF state mean [x, y, theta]
        ekf_cov = row[4]  # logged EKF covariance (3x3)
        cam = estimate_pose_from_camera_measurement(row[5])
        state_list.append(State(ekf_mean[0], ekf_mean[1], ekf_mean[2]))
        camera_list.append(cam)
        covariance_list.append(np.array(ekf_cov)[:2, :2])

    plot_traj(state_list, camera_list, covariance_list, map)


####### MAIN #######
if __name__ == "__main__":
    # Set multi_random_mode=True to test multiple random initial states
    offline_pf(multi_random_mode=True, num_random_tests=5)
    # offline_pf(multi_random_mode=False)  # Single test mode
    # plot_traj_only()
