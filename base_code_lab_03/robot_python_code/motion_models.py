# External Libraries
import math
import random
import numpy as np

# Motion Model constants


# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_distance_travelled_s(encoder_counts):
    # Add student code here
    a = -5.38659836735178e-08
    b = 0.0002806804940722607
    c = 0.1051219368000114

    var_s = a * encoder_counts**2 + b * encoder_counts + c

    return var_s * 0.01**2  # convert to meters^2


# Function to calculate distance from encoder counts
def distance_travelled_s(encoder_counts):
    # Add student code here
    s = 0.028308198909415782 * encoder_counts

    return s * 0.01  # convert to meters


# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_rotational_velocity_w(steering_angle_command, speed_command):
    # Add student code here
    c = 0.9126683501683492
    var_w = c

    return var_w


def rotational_velocity_w(steering_angle_command, speed_command):
    # Add student code here
    def model_rot(xdata, a, b, c, d, e, f):
        return (
            a * xdata[0]
            # + b * xdata[1]
            # + c * xdata[0] ** 2
            # + d * xdata[1] ** 2
            # + e * xdata[0] * xdata[1]
            + c * xdata[0] ** 3 * xdata[1]
            # + d * xdata[0] ** 2 * xdata[1]
            + e * xdata[0] * xdata[1]
            + f
        )

    p = [
        1.57500000e00,
        1.00000000e00,
        -8.88888889e-05,
        1.00000000e00,
        2.77777778e-04,
        -1.28205128e-01,
    ]
    # p = [-0.59469697, 0.10597015, -0.00808458, -0.00160586, 0.03744949]
    w = model_rot((steering_angle_command, speed_command), *p)

    return w


# This class is an example structure for implementing your motion model.
class MyMotionModel:
    # Constructor, change as you see fit.
    def __init__(self, initial_state, last_encoder_count):
        self.state = initial_state
        self.last_encoder_count = last_encoder_count
        self.step_with_noise = True
        self.gen_noise = False
        self.return_noise_scale = False

    # This is the key step of your motion model, which implements x_t = f(x_{t-1}, u_t)
    def step_update(self, encoder_counts, steering_angle_command, delta_t):
        # Add student code here
        distance = distance_travelled_s(encoder_counts - self.last_encoder_count)
        if self.step_with_noise:
            self.gen_noise = True
        if self.gen_noise or self.return_noise_scale:
            dist_std_err = math.sqrt(
                variance_distance_travelled_s(encoder_counts - self.last_encoder_count)
            )
        if self.gen_noise:
            dist_noise = random.normalvariate(0, dist_std_err)
        if self.step_with_noise:
            distance += dist_noise
        self.distance_step = distance  # Store the distance step for use in the Jacobian
        self.last_encoder_count = encoder_counts
        # our tested cmd to actual speed ratio is speed / cmd = 0.42
        est_vel = distance / delta_t / 0.42 * 100
        w = rotational_velocity_w(steering_angle_command, est_vel)
        if self.gen_noise or self.return_noise_scale:
            w_std_err = (
                math.sqrt(
                    max(
                        0,
                        variance_rotational_velocity_w(steering_angle_command, est_vel),
                    )
                ),
            )

        if self.gen_noise:
            w_noise = np.sign(w) * random.normalvariate(0, w_std_err)
        if self.step_with_noise:
            w += w_noise
        self.w_Rad = w / 180 * math.pi
        self.state[2] = self.state[2] + self.w_Rad * delta_t
        self.state[0] = self.state[0] + distance * math.cos(self.state[2])
        self.state[1] = self.state[1] + distance * math.sin(self.state[2])

        if self.return_noise_scale:
            x_noise_scale = np.array(
                [
                    dist_std_err * math.cos(self.state[2]),
                    dist_std_err * math.sin(self.state[2]),
                    w_std_err / 180 * math.pi * delta_t,
                ]
            )
            return self.state, x_noise_scale
        else:
            return self.state

    # This is a great tool to take in data from a trial and iterate over the data to create
    # a robot trajectory in the global frame, using your motion model.
    def traj_propagation(self, time_list, encoder_count_list, steering_angle_list):
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]
        self.last_encoder_count = encoder_count_list[0]
        for i in range(1, len(encoder_count_list)):
            delta_t = time_list[i] - time_list[i - 1]
            new_state = self.step_update(
                encoder_count_list[i], steering_angle_list[i], delta_t
            )
            x_list.append(new_state[0])
            y_list.append(new_state[1])
            theta_list.append(new_state[2])

        return x_list, y_list, theta_list

    # Coming soon
    def generate_simulated_traj(self, duration, steering_angle):
        delta_t = 0.1
        t_list = []
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]
        t = 0
        encoder_counts = 0
        while t < duration:
            # 1000 -> 30cm
            # speed 5cm /s -> 166.67 encoder counts / s
            encoder_counts += int(166.67 * delta_t)
            self.step_update(encoder_counts, steering_angle, delta_t)
            x_list.append(self.state[0])
            y_list.append(self.state[1])
            theta_list.append(self.state[2])
            t += delta_t
            t_list.append(t)
        return x_list, y_list, theta_list
