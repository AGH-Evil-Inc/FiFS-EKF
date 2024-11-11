import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pandas as pd

gravity = 9.81

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_to_euler(q):
    w, x, y, z = q # inny sposób zapisu kwaternionu w użytej bibliotece
    rotation = R.from_quat((x, y, z, w))
    return rotation.as_euler('xyz', degrees=False) # Radiany

class OrientationEKF:
    def __init__(self, delta_t, x_init):
        self.dt = delta_t #

        # State vector (quaternion and bias): [q_w, q_x, q_y, q_z, b_x, b_y, b_z]
        self.x = x_init
        self.P = np.eye(7) * 0.1  # Covariance matrix

        # Process noise covariance
        #self.Q = np.zeros(7)
        self.Q = np.eye(7) * 0.01
        self.Q[4:, 4:] *= 0.001  # Lower process noise for gyroscope bias

        # Measurement noise covariance (tune based on your sensor's accuracy)
        self.R = np.eye(3) * 0.1

    def predict(self, angular_velocities):
        # bias żyroskopu w osiach x, y, z
        biases = self.x[4:]

        # # Odejmujemy bias od nowych pomiarów żyroskopu
        # w_x, w_y, w_z = angular_velocities - biases
        #
        # # Tworzymy macierz przejścia A
        # A = np.array([[0, -w_x, -w_y, -w_z, 0, 0, 0],
        #               [w_x, 0, w_z, -w_y, 0, 0, 0],
        #               [w_y, -w_z, 0, w_x, 0, 0, 0],
        #               [w_z, w_y, -w_x, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0]])
        #
        # # Obliczamy nowy stan (a w zasadzie jego predykcję)
        # self.x += (0.5 * self.dt * A) @ self.x

        # Odejmujemy bias od nowych pomiarów żyroskopu
        w_x, w_y, w_z = 0.5 * self.dt * (angular_velocities - biases)

        # Tworzymy macierz przejścia A (Jakobian)
        A = np.array([[1, -w_x, -w_y, -w_z, 0, 0, 0],
                      [w_x, 1, w_z, -w_y, 0, 0, 0],
                      [w_y, -w_z, 1, w_x, 0, 0, 0],
                      [w_z, w_y, -w_x, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1]])

        # Obliczamy nowy stan (a w zasadzie jego predykcję)
        self.x = A @ self.x

        # Przewidujemy macierz kowariancji
        self.P = A @ self.P @ A.T + self.Q

    def update(self, accelerations):
        # Normalize the accelerometer reading to represent gravity direction
        #accelerations = accelerations / np.linalg.norm(accelerations)

        # Ensure quaternion is normalized to avoid zero-norm issues
        #self.x[:4] = normalize_quaternion(self.x[:4])

        # # Convert quaternion to rotation matrix to predict gravity direction in sensor frame
        # q = self.x[:4]
        # w, x, y, z = q # inny sposób zapisu w użytej bibliotece
        # R_q = R.from_quat((x, y, z, w)).as_matrix()  # Convert normalized quaternion to rotation matrix
        # accel_pred = R_q.T @ np.array([0, 0, -1])

        # Compute the measurement residual
        #e = accelerations - accel_pred
        #print(e)

        # # Measurement Jacobian
        # H = np.zeros((3, 7))
        # H[:3, :4] = 2 * np.array([
        #     [-q[2], q[1], -q[0], q[3]],
        #     [q[3], q[0], -q[1], -q[2]],
        #     [q[0], q[3], q[2], q[1]]
        # ])

        e = self.calculate_measurement_residual(accelerations)
        #e = np.array([0.01, 0.01, 0.01])
        H = self.compute_H_matrix()

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S) # Wzmocnienie Kalmana
        print(K)

        # Update the state vector and covariance
        self.x += K @ e
        self.P = (np.eye(7) - K @ H) @ self.P

        # Normalize quaternion again after update
        #self.x[:4] = normalize_quaternion(self.x[:4])

    def get_orientation(self):
        return self.x[:4]

    def compute_H_matrix(self, delta=1e-5):
        """
        Computes the measurement Jacobian H matrix for the accelerometer.

        Parameters:
        - quaternion: The current orientation quaternion [q_w, q_x, q_y, q_z]
        - delta: Small change to approximate partial derivatives

        Returns:
        - H: The 3x7 measurement Jacobian matrix
        """
        quaternion = self.x[:4]

        H = np.zeros((3, 7))
        g_world = np.array([0, 0, gravity])  # Gravity in the world frame

        # Rotate gravity vector to sensor frame using the current quaternion
        def rotate_gravity(quat):
            w, x, y, z = quat
            rotation = R.from_quat((x, y, z, w))
            return rotation.apply(g_world)

        # Calculate the accelerometer reading in the sensor frame at the current quaternion
        accel_base = rotate_gravity(quaternion)

        # For each quaternion component, calculate the partial derivative
        for i in range(4):
            # Create a slightly perturbed quaternion
            perturbed_quat = np.array(quaternion)
            perturbed_quat[i] += delta

            # Calculate the perturbed accelerometer reading
            perturbed_accel = rotate_gravity(perturbed_quat)

            # Approximate partial derivative with respect to quaternion component
            H[:, i] = (perturbed_accel - accel_base) / delta

        # The last three columns are zeros for gyroscope biases
        return H

    def calculate_measurement_residual(self, accelerations):
        """
        Computes the measurement residual y = z - h(x).

        Parameters:
        - accelerometer_measurement: Actual accelerometer reading [AccX, AccY, AccZ]
        - quaternion: Current orientation quaternion [q_w, q_x, q_y, q_z]

        Returns:
        - residual: The measurement residual y
        """
        quaternion = self.x[:4]

        # Actual measurement
        z_actual = np.array(accelerations)

        # Gravity vector in world frame
        g_world = np.array([0, 0, gravity])

        # Rotate gravity vector to sensor frame using the current quaternion
        w, x, y, z = quaternion
        rotation = R.from_quat((x, y, z, w))
        h_x = rotation.apply(g_world)  # Predicted measurement based on current orientation

        # Calculate the measurement residual
        residual = z_actual - h_x

        return residual


if __name__ == '__main__':
    data = pd.read_csv("data/train.csv")
    print(data.head())

    # Dane z akcelerometru w g -> m/s^2
    data['AccX'] *= gravity
    data['AccY'] *= gravity
    data['AccZ'] *= gravity

    # Dane z żyroskopu w mdps (mili degrees per second) -> rad/s
    data['GyroX'] *= 0.001 * (np.pi / 180)
    data['GyroY'] *= 0.001 * (np.pi / 180)
    data['GyroZ'] *= 0.001 * (np.pi / 180)

    g_bias_x = np.mean(data['GyroX'][:1000])
    g_bias_y = np.mean(data['GyroY'][:1000])
    g_bias_z = np.mean(data['GyroZ'][:1000])

    dt = 0.005
    ekf = OrientationEKF(dt, np.array([1, 0, 0, 0, g_bias_x, g_bias_y, g_bias_z], dtype=float))

    pred = []
    for _, row in data.iterrows():
        gyroscope_measurement = np.array([row['GyroX'], row['GyroY'], row['GyroZ']])
        accelerometer_measurement = np.array([row['AccX'], row['AccY'], row['AccZ']])

        # Predykcja za pomocą żyroskopu
        ekf.predict(gyroscope_measurement)

        # Korekcja za pomocą akcelerometru (NIE DZIAŁA)
        ekf.update(accelerometer_measurement)

        orientation_euler = quaternion_to_euler(ekf.get_orientation())
        pred.append(orientation_euler)

    # Pracujemy na radianach, plotuję w kątach
    plt.plot(data['Time'], [p[0] * 180 / np.pi for p in pred], label="Roll prediction")
    plt.plot(data['Time'], data['roll'], label="Roll actual")
    plt.legend()
    plt.show()

    plt.plot(data['Time'], [p[1] * 180 / np.pi for p in pred], label="Pitch prediction")
    plt.plot(data['Time'], data['pitch'], label="Pitch actual")
    plt.legend()
    plt.show()

    plt.plot(data['Time'], [p[2] * 180 / np.pi for p in pred], label="Yaw prediction")
    plt.plot(data['Time'], data['yaw'], label="Yaw actual")
    plt.legend()
    plt.show()
