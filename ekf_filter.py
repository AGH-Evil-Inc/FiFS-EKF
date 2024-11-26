import numpy as np
from scipy.spatial.transform import Rotation as R


gravity = 9.81


def normalize_vector(v):
    norm = np.sqrt(np.sum(v**2))
    return v/norm

def get_quaternion_rot_from_accelerometer(accelerations):
    acc_norm = normalize_vector(accelerations)
    g_ref = np.array([0, 0, 1])
    r = np.array([-acc_norm[1], acc_norm[0], 0])
    cos_theta = np.dot(g_ref, acc_norm)
    theta = np.arccos(cos_theta)

    # r[2] == 0, sam akcelerometr nie daje nam informacji o yaw
    return np.array([np.cos(theta/2), r[0]*np.sin(theta/2), r[1]*np.sin(theta/2), r[2]*np.sin(theta/2)])

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
    # forma Inhomogenous
    # TODO: może użyć Homogenous? (https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Rotation_matrices)
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
    # TODO: może użyć tego? - (stąd: https://cookierobotics.com/073/)
    # qw, qx, qy, qz = q.flat
    # roll  = np.arctan2(2 * (qw * qx + qy * qz),
    #                    1 - 2 * (qx * qx + qy * qy))
    # pitch = np.arcsin(2 * (qw * qy - qz * qx))
    # yaw   = np.arctan2(2 * (qw * qz + qx * qy),
    #                    1 - 2 * (qy * qy + qz * qz))
    # return roll, pitch, yaw

    w, x, y, z = q.reshape(4) # inny sposób zapisu kwaternionu w użytej bibliotece
    rotation = R.from_quat((x, y, z, w))
    return rotation.as_euler('xyz', degrees=False) # Radiany

class EKF:
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