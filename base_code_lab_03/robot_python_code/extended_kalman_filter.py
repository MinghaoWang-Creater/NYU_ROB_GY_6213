# External libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Local libraries
import parameters
import data_handling

from motion_models import MyMotionModel

from scipy.spatial.transform import Rotation as R


def estimate_pose_from_camera_measurement(camera_measurement):
    T_optimal = np.array(
        [
            [0.99660771, 0.07871153, 0.02403284, 0.03141616],
            [-0.07888354, 0.99686397, 0.00629397, 0.00124132],
            [-0.02346206, -0.00816841, 0.99969136, 0.07092353],
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


# Main class
class ExtendedKalmanFilter:
    def __init__(self, x_0, Sigma_0, encoder_counts_0, use_corrector=True):
        self.state_mean = np.array(x_0)
        self.state_covariance = Sigma_0
        self.predicted_state_mean = np.array([0, 0, 0], dtype=np.float64)
        self.predicted_state_covariance = parameters.I3 * 1.0
        self.last_encoder_counts = encoder_counts_0
        self.model = MyMotionModel(x_0, encoder_counts_0)
        self.model.step_with_noise = False  # EKF prediction step should not have noise, since we account for noise in the R matrix
        self.model.return_noise_scale = True
        self.use_corrector = use_corrector

    # Call the prediction and correction steps
    def update(self, u_t, z_t, delta_t):
        """
        u_t is not real u but encoder_counts reading and steering command, since we need encoder counts to calculate distance travelled in the motion model
        z_t is camera measurement of x, y, theta
        """
        self.prediction_step(u_t, delta_t)
        self.correction_step(z_t)
        return

    # Set the EKF's predicted state mean and covariance matrix
    def prediction_step(self, u_t, delta_t):
        self.model.state = self.state_mean  # sync states
        self.model.last_encoder_count = (
            self.last_encoder_counts
        )  # sync encoder counts for distance calculation
        x_tp_mean, s = self.model.step_update(u_t[0], u_t[1], delta_t)
        # print("noise scale:", s)  # Debug print for noise scale
        G_x = self.get_G_x(self.state_mean, s, delta_t)
        R_t = self.get_R(s)
        G_u = self.get_G_u(self.state_mean, delta_t)
        self.predicted_state_mean = x_tp_mean
        self.predicted_state_covariance = (
            G_x @ self.state_covariance @ G_x.T + G_u @ R_t @ G_u.T
        )
        self.last_encoder_counts = u_t[0]
        return

    # Set the EKF's corrected state mean and covariance matrix
    def correction_step(self, z_t):
        H_t = self.get_H()
        Q_t = self.get_Q()
        S_t = H_t @ self.predicted_state_covariance @ H_t.T + Q_t
        K_t = self.predicted_state_covariance @ H_t.T @ np.linalg.inv(S_t)
        diff = z_t - self.get_h_function(self.predicted_state_mean)

        if self.use_corrector:
            self.state_mean = self.predicted_state_mean + K_t @ diff
            self.state_covariance = (
                parameters.I3 - K_t @ H_t
            ) @ self.predicted_state_covariance
        else:
            self.state_mean = np.array(
                self.predicted_state_mean
            )  # No correction, only prediction
            self.state_covariance = np.array(
                self.predicted_state_covariance
            )  # No correction, only prediction
        # print(self.state_covariance)
        return

    # The nonlinear transition equation that provides new states from past states
    def g_function(self, x_tm1, u_t, delta_t):
        raise NotImplementedError(
            "use self.model.step_update instead of g_function for prediction step since we need noise scale from motion model for EKF"
        )

    # The nonlinear measurement function
    def get_h_function(self, x_t):
        return x_t

    # This function returns a matrix with the partial derivatives dg/dx
    def get_G_x(self, x_tm1, s, delta_t):
        G = np.array(
            [
                [1, 0, -self.model.distance_step * math.sin(self.model.state[2])],
                [0, 1, self.model.distance_step * math.cos(self.model.state[2])],
                [0, 0, self.model.w_Rad * delta_t],
            ]
        )
        return G

    # This function returns a matrix with the partial derivatives dg/du
    def get_G_u(self, x_tm1, delta_t):
        return np.array(
            [
                [math.cos(self.model.state[2]), 0],
                [math.sin(self.model.state[2]), 0],
                [0, delta_t],
            ]
        )

    # This function returns a matrix with the partial derivatives dh_t/dx_t
    def get_H(self):
        return parameters.I3

    # This function returns the R_t matrix which contains transition function covariance terms.
    def get_R(self, s):
        # Scale process noise with distance travelled or a baseline
        return np.diag(s)

    # This function returns the Q_t matrix which contains measurement covariance terms.
    def get_Q(self):
        return np.diag(
            [
                0.0048478566654710465 * 2,
                0.007572909242886893 * 2,
                0.009776259224975919,
            ]
        )


class KalmanFilterPlot:
    def __init__(self):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig

    def update(self, state_mean, z_t, state_covaraiance, realtime=True):
        plt.clf()

        # Plot covariance ellipse
        lambda_, v = np.linalg.eig(state_covaraiance)
        lambda_ = np.sqrt(lambda_) * 10
        xy = (state_mean[0], state_mean[1])
        angle = np.rad2deg(np.arctan2(*v[:, 0][::-1]))
        ell = Ellipse(
            xy,
            alpha=0.5,
            facecolor="red",
            width=lambda_[0],
            height=lambda_[1],
            angle=angle,
        )
        ax = self.fig.gca()
        ax.add_artist(ell)

        # Plot state estimate
        plt.plot(state_mean[0], state_mean[1], "ro")
        plt.plot(
            [state_mean[0], state_mean[0] + self.dir_length * math.cos(state_mean[2])],
            [state_mean[1], state_mean[1] + self.dir_length * math.sin(state_mean[2])],
            "r",
        )
        plt.plot(z_t[0], z_t[1], "bo")
        plt.plot(
            [z_t[0], z_t[0] + self.dir_length * math.cos(z_t[2])],
            [z_t[1], z_t[1] + self.dir_length * math.sin(z_t[2])],
            "b",
        )
        plt.xlabel("X(m)")
        plt.ylabel("Y(m)")
        plt.axis([-0.25, 2, -2, 2])
        plt.grid()
        if realtime:
            plt.draw()
            plt.pause(0.1)


def plot_traj_single(state, camera, covariance, label, color="r"):
    # mark the start by *, and by +
    plt.plot(
        state[0][0], state[0][1], f"{color}*", markersize=20, label=f"{label} Start"
    )
    plt.plot(
        state[-1][0], state[-1][1], f"{color}+", markersize=20, label=f"{label} End"
    )
    plt.plot(state[:, 0], state[:, 1], f"{color}o-", label=f"{label} State")
    if camera is not None:
        plt.plot(
            camera[:, 0], camera[:, 1], "ko-", label="Camera Measurement", alpha=0.3
        )
    # plot covariance ellipses at every 10th point
    for i in range(0, len(state), 20):
        lambda_, v = np.linalg.eig(covariance[i])
        lambda_ = np.sqrt(lambda_) * 5
        print(lambda_)
        xy = (state[i][0], state[i][1])
        angle = np.rad2deg(np.arctan2(*v[:, 0][::-1]))
        ell = Ellipse(
            xy,
            alpha=0.3,
            facecolor=color,
            width=lambda_[0],
            height=lambda_[1],
            angle=angle,
        )
        ax = plt.gca()
        ax.add_artist(ell)


def plot_traj(state_raw, camera_raw, covariance_raw):
    multiple_traj = isinstance(
        state_raw[0], list
    )  # Check if we have multiple trajectories
    plt.cla()
    if multiple_traj:
        color = ["r", "g", "b", "c", "m", "y"]  # Colors for different trajectories
        for i in range(len(state_raw)):
            state = np.array(state_raw[i])
            camera = np.array(camera_raw[i])
            plot_traj_single(
                state,
                camera if i == len(state_raw) - 1 else None,
                covariance_raw[i],
                label=f"EKF {i + 1}",
                color=color[i % len(color)],
            )
        plt.legend(loc="upper left", fontsize="8", ncol=4, framealpha=0.3)
    else:
        print("Single trajectory provided, plotting without labels.")
        state = np.array(state_raw)
        camera = np.array(camera_raw)
        plot_traj_single(state, camera, covariance_raw, label="EKF Estimate")
        plt.legend()

    # plt.axis("equal")
    plt.xlabel("X(m)")
    plt.ylabel("Y(m)")
    # plt.axis([-0.25, 2, -2, 2])
    plt.grid()
    plt.show()


def run_test(x_0, ekf_data, use_corrector):
    print("Initial EKF state guess:", x_0)  # Debug print for initial state guess
    Sigma_0 = parameters.I3
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    extended_kalman_filter = ExtendedKalmanFilter(
        x_0, Sigma_0, encoder_counts_0, use_corrector
    )
    # Create plotting tool for ekf
    kalman_filter_plot = KalmanFilterPlot()
    state_traj = []
    state_traj_camera = []
    covariance = []
    state_traj.append(extended_kalman_filter.state_mean)
    state_traj_camera.append(estimate_pose_from_camera_measurement(ekf_data[0][3]))
    # Loop over sim data
    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        delta_t = ekf_data[t][0] - ekf_data[t - 1][0]  # time step size
        u_t = np.array([row[2].encoder_counts, row[2].steering])  # robot_sensor_signal
        # z_t = np.array([row[3][0], row[3][1], row[3][5]])  # camera_sensor_signal
        z_t = estimate_pose_from_camera_measurement(row[3])
        print(
            f"Time step {t}, delta_t: {delta_t:.3f} sec, x: {extended_kalman_filter.state_mean}, z_t: {z_t}, u_t: {u_t}"
        )  # Debug print
        # Run the EKF for a time step
        extended_kalman_filter.update(u_t, z_t, delta_t)
        kalman_filter_plot.update(
            extended_kalman_filter.state_mean,
            z_t,
            extended_kalman_filter.state_covariance[0:2, 0:2],
            realtime=False,
        )
        covariance.append(extended_kalman_filter.state_covariance[0:2, 0:2])
        state_traj.append(extended_kalman_filter.state_mean)
        state_traj_camera.append(z_t)
    return state_traj, state_traj_camera, covariance


# Code to run your EKF offline with a data file.
def offline_efk():

    # Get data to filter
    filename = "./data/simple_c_c.pkl"
    ekf_data = data_handling.get_file_data_for_kf(filename)
    use_corrector = True#"0_0" not in filename
    use_random_initial_state = False
    # (
    #     "r_r" in filename
    # )  # use random initial state for the trial with large estimation error to show EKF convergence
    N_random_test = 10
    state_traj = []
    state_traj_camera = []
    covariance = []
    if use_random_initial_state:
        print(
            "Using random initial state for EKF to show convergence from a bad initial guess."
        )
        random_x_0 = np.random.uniform(-1.0, 1.0, size=N_random_test)
        random_y_0 = np.random.uniform(-1.0, 1.0, size=N_random_test)
        random_theta_0 = np.random.uniform(-math.pi, math.pi, size=N_random_test)
        # Instantiate PF with no initial guess
        for i in range(N_random_test):
            x_0 = [
                ekf_data[0][3][0] + random_x_0[i],
                ekf_data[0][3][1] + random_y_0[i],
                ekf_data[0][3][5] + random_theta_0[i],
            ]
            state_traj_, state_traj_camera_, covariance_ = run_test(
                x_0, ekf_data, use_corrector
            )
            state_traj.append(state_traj_)
            state_traj_camera.append(state_traj_camera_)
            covariance.append(covariance_)
    else:
        x_0 = [
            ekf_data[0][3][0],
            ekf_data[0][3][1],
            ekf_data[0][3][5],
        ]  # use first camera measurement as initial state guess
        state_traj, state_traj_camera, covariance = run_test(
            x_0, ekf_data, use_corrector
        )
    # if "r_r" in filename:
    #     x_0 = np.zeros(3)  # [x, y, theta]
    plot_traj(state_traj, state_traj_camera, covariance)

def plot_traj_only():
    # Get data to filter
    filename = "./data/robot_data_40_-15_25_02_26_19_55_04.pkl"
    # filename = "./data/simple_c_c.pkl"
    ekf_data = data_handling.get_file_data_for_kf(filename)
    state_mean = []
    ground_truth = []
    covariance = []
    kalman_plot =  KalmanFilterPlot()
    for t in range(1, len(ekf_data)):
        state_mean.append(np.array(ekf_data[t][4]))
        z_t = estimate_pose_from_camera_measurement(np.array(ekf_data[t][3]))
        ground_truth.append(z_t)
        covariance.append(np.array(ekf_data[t][5])[:2,:2])
        # kalman_plot.update(state_mean[-1], z_t, covariance[-1], realtime=True)

    plot_traj(state_mean, ground_truth, covariance)

####### MAIN #######
if __name__ == "__main__":
    # offline_efk()
    plot_traj_only()
