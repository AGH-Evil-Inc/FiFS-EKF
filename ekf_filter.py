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

def get_Q(gyro_noises):
    gyro_noise_x, gyro_noise_y, gyro_noise_z = gyro_noises
    Q = np.array([[gyro_noise_x**2,               0,               0],
                  [              0, gyro_noise_y**2,               0],
                  [              0,               0, gyro_noise_z**2]])
    return Q

def get_Q_bias(gyro_bias_noises):
    gbn_x, gbn_y, gbn_z = gyro_bias_noises
    Q_bias = np.zeros((7, 7))
    Q_bias[4:7, 4:7] = np.array([[gbn_x**2,        0,        0],
                                 [       0, gbn_y**2,        0],
                                 [       0,        0, gbn_z**2]])
    return Q_bias

def get_R(accelerometer_noises):
    an_x, an_y, an_z = accelerometer_noises
    R = np.array([[an_x**2,       0,       0],
                  [      0, an_y**2,       0],
                  [      0,       0, an_z**2]])
    return R

def get_W(x, dt):
    qw, qx, qy, qz = x.flat[0:4]
    return dt/2 * np.array([
        [-qx, -qy, -qz],
        [ qw, -qz,  qy],
        [ qz,  qw, -qx],
        [-qy,  qx,  qw],
        [  0,   0,   0],
        [  0,   0,   0],
        [  0,   0,   0]
    ])

def get_F(x, angular_velocities, dt):
    qw, qx, qy, qz, bx, by, bz = x.flat
    wx, wy, wz = angular_velocities.flat
    return np.array([
        [             1, dt*(-wx + bx)/2, dt*(-wy + by)/2, dt*(-wz + bz)/2,  dt*qx/2,  dt*qy/2,  dt*qz/2],
        [dt*(wx - bx)/2,               1, dt*( wz - bz)/2, dt*(-wy + by)/2, -dt*qw/2,  dt*qz/2, -dt*qy/2],
        [dt*(wy - by)/2, dt*(-wz + bz)/2,               1, dt*( wx - bx)/2, -dt*qz/2, -dt*qw/2,  dt*qx/2],
        [dt*(wz - bz)/2, dt*( wy - by)/2, dt*(-wx + bx)/2,               1,  dt*qy/2, -dt*qx/2, -dt*qw/2],
        [             0,               0,               0,               0,        1,        0,        0],
        [             0,               0,               0,               0,        0,        1,        0],
        [             0,               0,               0,               0,        0,        0,        1]
    ])

class EKF:
    def __init__(self, q0=[1,0,0,0], b0=[0,0,0], delta_t=0.005, 
                 init_gyro_bias_err=0.1, gyro_noises=[0.015,0.015,0.015], gyro_bias_noises=[0.002,0.002,0.002],
                 accelerometer_noise=1):
        """
        Inicjalizacja EKF (Extended Kalman Filer)

        Parameters:
            q0 (List[qw, qx, qy, qz]): Początkowa orientacja [kwaternion znormalizowany]
            b0 (List[bx, by, bz]): Początkowe biasy GYRO [rad/sec]
            delta_t: TODO: usunąć
            init_gyro_bias_err (float): początkowa niepewność biasu GYRO w P (1 odch. stand.) [rad/sec]
            gyro_noises (List[float, float, float]): Szum GYRO (żyroskop) [rad/sec]
            gyro_bias_noises (List[float, float, float]): Szum biasu GYRO [rad/sec]
            accelerometer_noise (List[float, float, float]): Szum ACC (akcelerometr) [m/s^2]
        """
        # TODO: usunąć stałe dt, zrobić na parametr w predict() i update()
        self.dt = delta_t
        # Wektor stanu (kwaternion i bias): [7x1] [q_w, q_x, q_y, q_z, b_x, b_y, b_z]
        self.x = np.c_[q0 + b0]

        # Macierz kowariancji P [7x7]
        self.P = np.identity(7) # zainicjalizowana czymkolwiek
        # - Część kwaternionowa (orientacji/obrotu/pozycji)
        self.P[0:4, 0:4] = np.identity(4) * 0.01
        # - Część biasu żyroskopu
        self.P[4:7, 4:7] = np.identity(3) * (init_gyro_bias_err ** 2)

        # Macierz wyjścia C
        self.C = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]])
        
        # Macierz kowariancji szumu procesu (GYRO) [3x3]
        # Przemnażana w predykcji przez W
        # [rad/sec]
        self.Q = get_Q(gyro_noises)
        # Macierz kowariancji szumu biasu GYRO [7x7]
        # [rad/sec]
        self.Q_bias = get_Q_bias(gyro_bias_noises)

        #TODO: przerobić na poniższą (3x3) - gdy porównanie w update() będzie wektorem grawitacji
        # Macierz kowariancji szumu obserwacji (ACC) [3x3]
        # Przemnażana w korekcji przez V
        # [m/s^2]
        #self.R = get_R(accelerometer_noises)
        self.R = accelerometer_noise

        # Szum procesu
        # self.Q = np.eye(7) * 0.00000001
        # self.Q[4:, 4:] = 0

        # Szum pomiaru
        # self.R = np.eye(4) * 0.0001

    def predict(self, angular_velocities):
        """
        Predykcja EKF - użyty GYRO (żyroskop) jako proces

        Parameters:
            angular_velocities (List[wx, wy, wz]): Prędkości kątowe GYRO [rad/sec]
            dt (float): krok czasu [sec] TODO: dodać
        """
        # Prędkości kątowe na wektor pionowy
        # angular_velocities = np.c_[angular_velocities]

        # Macierz F - Jakobian z f()
        F = get_F(self.x, angular_velocities, self.dt)

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
        
        # Macierz W do przekształcenia Q procesu [prędkości x,y,z] -> [kwaternion]
        W = get_W(self.x, self.dt)

        # Obliczamy nowy stan (a w zasadzie jego predykcję)
        self.x += A @ self.x
        self.x[0:4] = normalize_vector(self.x[0:4])

        # Przewidujemy macierz kowariancji
        self.P = A @ self.P @ A.T + W @ self.Q @ W.T + self.Q_bias

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
    