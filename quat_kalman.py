import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pandas as pd

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def quaternion_to_euler(q):
    w, x, y, z = q # inny sposób zapisu kwaternionu w użytej bibliotece
    rotation = R.from_quat((x, y, z, w))
    return rotation.as_euler('xyz', degrees=False) # Radiany

def integrate_gyroscope(q, omega, dt):
    omega_quat = np.array([0, omega[0], omega[1], omega[2]])
    dq = 0.5 * quaternion_multiply(q, omega_quat) * dt
    q_new = q + dq
    return normalize_quaternion(q_new)


class OrientationEKF:
    def __init__(self, dt):
        self.dt = dt

        # State vector (quaternion and bias): [q_w, q_x, q_y, q_z, b_x, b_y, b_z]
        self.x = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)  # Initial quaternion and zero bias
        self.P = np.eye(7) * 0.1  # Covariance matrix

        # Process noise covariance
        self.Q = np.eye(7) * 0.01
        self.Q[4:, 4:] *= 0.001  # Lower process noise for gyroscope bias

        # Measurement noise covariance (tune based on your sensor's accuracy)
        self.R = np.eye(3) * 0.1

    def predict(self, omega):
        # Extract current quaternion and biases
        q = self.x[:4]
        b = self.x[4:]

        # Bias-corrected angular velocity
        omega_corrected = omega - b

        # Predict new quaternion using gyroscope integration
        q_pred = integrate_gyroscope(q, omega_corrected, self.dt)
        self.x[:4] = q_pred

        # Compute the Jacobian of the transition model
        F = np.eye(7)
        F[:4, 4:] = -0.5 * self.dt * np.array([
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]]
        ])

        # Predict the covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, accel):
        # Normalize the accelerometer reading to represent gravity direction
        accel = accel / np.linalg.norm(accel)

        # Ensure quaternion is normalized to avoid zero-norm issues
        self.x[:4] = normalize_quaternion(self.x[:4])

        # Convert quaternion to rotation matrix to predict gravity direction in sensor frame
        q = self.x[:4]
        w, x, y, z = q # inny sposób zapisu w użytej bibliotece
        R_q = R.from_quat((x, y, z, w)).as_matrix()  # Convert normalized quaternion to rotation matrix
        accel_pred = R_q.T @ np.array([0, 0, -1])

        # Compute the measurement residual
        y = accel - accel_pred

        # Measurement Jacobian
        H = np.zeros((3, 7))
        H[:3, :4] = 2 * np.array([
            [-q[2], q[1], -q[0], q[3]],
            [q[3], q[0], -q[1], -q[2]],
            [q[0], q[3], q[2], q[1]]
        ])

        # Compute Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update the state vector and covariance
        self.x += K @ y
        self.P = (np.eye(7) - K @ H) @ self.P

        # Normalize quaternion again after update
        self.x[:4] = normalize_quaternion(self.x[:4])

    def get_orientation(self):
        return self.x[:4]





data = pd.read_csv("data/train.csv")
print(data.head())

# Dane z akcelerometru w g -> m/s^2
data['AccX'] *= 9.81
data['AccY'] *= 9.81
data['AccZ'] *= 9.81

# Dane z żyroskopu w mdps (mili degrees per second) -> rad/s
data['GyroX'] *= 0.001 * (np.pi / 180)
data['GyroY'] *= 0.001 * (np.pi / 180)
data['GyroZ'] *= 0.001 * (np.pi / 180)

dt = 0.005
ekf = OrientationEKF(dt)

pred = []
for i, row in data.iterrows():
    gyroscope_measurement = np.array([row['GyroX'], row['GyroY'], row['GyroZ']])
    accelerometer_measurement = np.array([row['AccX'], row['AccY'], row['AccZ']])

    # Predykcja za pomocą żyroskopu
    ekf.predict(gyroscope_measurement)

    # Korekcja za pomocą akcelerometru (NIE DZIAŁA)
    #ekf.update(accelerometer_measurement)

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
