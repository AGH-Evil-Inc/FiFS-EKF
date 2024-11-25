import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

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

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
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
    def __init__(self, delta_t, process_noise, measurment_noise, x_init):
        self.dt = delta_t
        # Wektor stanu (kwaternion i bias): [q_w, q_x, q_y, q_z, b_x, b_y, b_z]
        self.x = np.array(x_init).reshape((7, 1))
        self.P = np.eye(7) * 0.1  # Macierz kowariancji

        # Macierz wyjścia C
        self.C = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]])

        self.Q = np.zeros(shape=(7, 7))
        self.Q[:4, :4] = process_noise

        self.R = measurment_noise

        # Szum procesu
        # self.Q = np.eye(7) * 0.00000001
        # self.Q[4:, 4:] = 0

        # Szum pomiaru
        # self.R = np.eye(4) * 0.0001

    def predict(self, angular_velocities):
        # bias żyroskopu w osiach x, y, z
        biases = self.x[4:].reshape(3)

        # Odejmujemy bias od nowych pomiarów żyroskopu
        w_x, w_y, w_z = 0.5 * self.dt * (angular_velocities - biases)

        # Tworzymy macierz przejścia A
        A = np.array([[0, -w_x, -w_y, -w_z, 0, 0, 0],
                      [w_x, 0, w_z, -w_y, 0, 0, 0],
                      [w_y, -w_z, 0, w_x, 0, 0, 0],
                      [w_z, w_y, -w_x, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])

        # Obliczamy nowy stan (a w zasadzie jego predykcję)
        self.x += A @ self.x
        self.x[0:4] = normalize_vector(self.x[0:4])

        # Przewidujemy macierz kowariancji
        self.P = A @ self.P @ A.T + self.Q

    def update(self, accelerations):
        state_rot_mat = quaternion_to_rotation_matrix(self.x[:4]) # Macierz rotacji stanu
        acc_predict_vec = state_rot_mat.T @ np.array([0, 0, gravity]) # Przewidywany pomiar akcelerometru
        e_vec = accelerations - acc_predict_vec.reshape(3) # Różnica wektora zmierzonego i przewidzianego
        e = np.array([0, e_vec[0], e_vec[1], e_vec[2]]).reshape((4, 1)) # Na kwaternion

        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S) # Wzmocnienie Kalmana

        # Aktualizacja wektora stanu i kowariancji
        self.x = self.x + (K @ e)
        self.x[:4] = normalize_vector(self.x[:4])
        self.P = (np.eye(7) - K @ self.C) @ self.P
        #self.x[:4] = normalize_vector(self.x[:4]) # Bez normalizacji pomiar z akcelerometru nie działa poprawnie
        #print(math.sqrt(np.sum(self.x**2)))

    def get_orientation(self):
        return self.x[:4]


if __name__ == '__main__':
    data_train = pd.read_csv("data/train.csv")

    # Komentować zamiennie - tylko [train part] albo [train part]+[entire test]
    # data = data_train
    data = pd.read_csv("data/test.csv")

    calibration_samples = 1000

    # Dane z akcelerometru w g -> m/s^2
    data['AccX'] *= gravity
    data['AccY'] *= gravity
    data['AccZ'] *= gravity

    acc_bias_x = np.mean(data['AccX'][:calibration_samples])
    acc_bias_y = np.mean(data['AccY'][:calibration_samples])
    acc_bias_z = np.mean(data['AccZ'][:calibration_samples]) - gravity

    acc_var = np.var(data[['AccX', 'AccY', 'AccZ']][:calibration_samples], axis=0)

    # Dane z żyroskopu w mdps (mili degrees per second) -> rad/s
    data['GyroX'] *= 0.001 * (np.pi / 180)
    data['GyroY'] *= 0.001 * (np.pi / 180)
    data['GyroZ'] *= 0.001 * (np.pi / 180)

    g_bias_x = np.mean(data['GyroX'][:calibration_samples])
    g_bias_y = np.mean(data['GyroY'][:calibration_samples])
    g_bias_z = np.mean(data['GyroZ'][:calibration_samples])

    gyro_var = np.var(data[['GyroX', 'GyroY', 'GyroZ']][:calibration_samples], axis=0)

    s1 = 0.0000001
    s2 = 0.0001

    proc_noise = np.diag(np.append(gyro_var, [s1 ** 2])) * 4e-2  # *(dt**2/4)
    meas_noise = np.diag(np.append(acc_var, [s2 ** 2]))

    dt = 0.005
    ekf = OrientationKF(dt, proc_noise, meas_noise, np.array([1, 0, 0, 0, g_bias_x, g_bias_y, g_bias_z], dtype=float))

    pred = []
    for _, row in data.iterrows():
        gyroscope_measurement = np.array([row['GyroX'], row['GyroY'], row['GyroZ']])
        accelerometer_measurement = np.array([row['AccX'] - acc_bias_x, row['AccY'] - acc_bias_y, row['AccZ'] - acc_bias_z])

        # Predykcja za pomocą żyroskopu
        ekf.predict(gyroscope_measurement)

        # Korekcja za pomocą akcelerometru
        ekf.update(accelerometer_measurement)

        orientation_euler = quaternion_to_euler(ekf.get_orientation())
        pred.append(orientation_euler)

    # Pracujemy na radianach, plotuję w kątach
    pred_roll = np.unwrap([p[0] * 180 / np.pi for p in pred])
    pred_pitch = np.unwrap([p[1] * 180 / np.pi for p in pred])
    pred_yaw = np.unwrap([p[2] * 180 / np.pi for p in pred])

    l = len(pred_roll)
    result_df = pd.DataFrame(
        data={'Id': list(range(1, l + 1)), 'pitch': pred_pitch, 'roll': pred_roll, 'yaw': pred_yaw})
    print(result_df.head())

    result_df.to_csv("results/submission.csv", index=False)

    # Liczone tylko dla [train part]
    print(f"Roll MSE = {mean_squared_error(data_train['roll'], pred_roll[:7000])}")
    print(f"Pitch MSE = {mean_squared_error(data_train['pitch'], pred_pitch[:7000])}")
    print(f"Yaw MSE = {mean_squared_error(data_train['yaw'], pred_yaw[:7000])}")

    # Pierwsze 7000 - nałożone prawdziwe pozycja z [train part]
    fig, axs = plt.subplots(3, 1, figsize=(5, 5))
    axs[0].plot(data['Time'], pred_roll, label="Roll prediction")
    axs[0].plot(data_train['Time'], data_train['roll'], label="Roll actual")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(data['Time'], pred_pitch, label="Pitch prediction")
    axs[1].plot(data_train['Time'], data_train['pitch'], label="Pitch actual")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(data['Time'], pred_yaw, label="Yaw prediction")
    axs[2].plot(data_train['Time'], data_train['yaw'], label="Yaw actual")
    axs[2].legend()
    axs[2].grid()
    fig.tight_layout()
    plt.show()