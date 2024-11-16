import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pandas as pd
import math


gravity = 9.81

def normalize_vector(v):
    norm = math.sqrt(np.sum(v**2))
    return v/norm

def get_quaternion_rot_from_accelerometer(accelerations):
    acc_norm = normalize_vector(accelerations)
    g_ref = np.array([0, 0, 1])
    r = np.array([-acc_norm[1], acc_norm[0], 0])
    cos_theta = np.dot(g_ref, acc_norm)
    theta = math.acos(cos_theta)

    # r[2] == 0, sam akcelerometr nie daje nam informacji o yaw
    return np.array([math.cos(theta/2), r[0]*math.sin(theta/2), r[1]*math.sin(theta/2), r[2]*math.sin(theta/2)])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def quaternion_inverse(q):
    w, x, y, z = q
    norm_squared = w**2 + x**2 + y**2 + z**2
    return np.array([w/norm_squared, -x/norm_squared, -y/norm_squared, -z/norm_squared])

def quaternion_to_euler(q):
    w, x, y, z = q.reshape(4) # inny sposób zapisu kwaternionu w użytej bibliotece
    rotation = R.from_quat((x, y, z, w))
    return rotation.as_euler('xyz', degrees=False) # Radiany

class OrientationKF:
    def __init__(self, delta_t, x_init):
        self.dt = delta_t

        # Wektor stanu (kwaternion i bias): [q_w, q_x, q_y, q_z, b_x, b_y, b_z]
        self.x = np.array(x_init).reshape((7, 1))
        self.P = np.eye(7) * 0.1  # Macierz kowariancji

        # Macierz wyjścia C
        self.C = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]])

        # Kowariancja szumu procesu
        self.Q = np.eye(7) * 0.001

        # Kowariancja szumu pomiaru
        self.R = np.eye(4) * 1000

    def predict(self, angular_velocities):
        # bias żyroskopu w osiach x, y, z
        biases = self.x[4:].reshape(3)

        # Odejmujemy bias od nowych pomiarów żyroskopu
        w_x, w_y, w_z = 0.5 * self.dt * (angular_velocities - biases)

        # Tworzymy macierz przejścia A
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
        y = get_quaternion_rot_from_accelerometer(accelerations)
        y = quaternion_inverse(y) # To nie powinno być potrzebne ale bez tego pomiary z akcelerometru są na odwrót

        #e = różnica kąta (kwaternion) między y a C@x
        Cx = (self.C @ self.x).reshape(4)
        e = quaternion_multiply(y, quaternion_inverse(Cx)).reshape((4, 1))

        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S) # Wzmocnienie Kalmana

        # Aktualizacja wektora stanu i kowariancji
        self.x += K @ e
        self.P = (np.eye(7) - K @ self.C) @ self.P
        #self.x[:4] = normalize_vector(self.x[:4])

    def get_orientation(self):
        return self.x[:4]


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
    ekf = OrientationKF(dt, np.array([1, 0, 0, 0, g_bias_x, g_bias_y, g_bias_z], dtype=float))

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
    fig, axs = plt.subplots(3, 1, figsize=(5, 5))
    axs[0].plot(data['Time'], [p[0] * 180 / np.pi for p in pred], label="Roll prediction")
    axs[0].plot(data['Time'], data['roll'], label="Roll actual")
    axs[0].legend()

    axs[1].plot(data['Time'], [p[1] * 180 / np.pi for p in pred], label="Pitch prediction")
    axs[1].plot(data['Time'], data['pitch'], label="Pitch actual")
    axs[1].legend()

    axs[2].plot(data['Time'], [p[2] * 180 / np.pi for p in pred], label="Yaw prediction")
    axs[2].plot(data['Time'], data['yaw'], label="Yaw actual")
    axs[2].legend()
    fig.tight_layout()
    plt.show()
