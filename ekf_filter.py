import numpy as np
from scipy.spatial.transform import Rotation as R


GRAVITY = 9.81

def quaternion_identity():
    """
    Zwraca kwaternion jednostkowy (1, 0i, 0j ,0k) - część rzeczywista 1, czyli brak obrotu.
    Returns:
        identity_quaternion (np.array[4x1]): Kwaterinon jednostkowy (1, 0i, 0j, 0k), jako wektor pionowy
    """
    # c_[] - pionizacja wektora [4] na [4x1]
    return np.c_[[1, 0, 0, 0]]

def calculate_vector_norm(v):
    return np.sqrt(np.sum(v**2))

def normalize_vector(v):
    norm = calculate_vector_norm(v)
    return v / norm

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

def quaternion_to_rotation_matrix(q_unit):
    """
    Wyznacza macierz obrotu z kwaternionu obrotu

    Parameters:
        q_unit (np.array[4x1]): Kwaternion obrotu (qw, qx, qy, qz); wektor pionowy

    Returns:
        rotation_matrix (np.array[3x3]): Macierz obrotu
    """
    # forma Inhomogenous
    # w, x, y, z = q_unit
    # return np.array([
    #     [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    #     [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
    #     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
    # ])

    # forma Homogenous
    w, x, y, z = q_unit.flat
    w2 = w * w
    x2 = x * x
    y2 = y * y
    z2 = z * z
    # Row 1
    r11 = w2 + x2 - y2 - z2
    r12 = 2 * (x * y - w * z)
    r13 = 2 * (w * y + x * z)
    # Row 2
    r21 = 2 * (x * y + w * z)
    r22 = w2 - x2 + y2 - z2
    r23 = 2 * (y * z - w * x)
    # Row 3:
    r31 = 2 * (x * z - w * y)
    r32 = 2 * (y * z + w * x)
    r33 = w2 - x2 - y2 + z2
    # Combine:
    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
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


def quaternion_from_axis_angle(unit_axis, angle_rad):
    """
    Wyznacza kwaternion obrotu z reprezentacji axis-angle.

    Parameters:
        unit_axis (np.array[3x1]): Oś obrotu (znormalizowany wektor) w reprezentacji axis-angle
        angle_rad (float): Kąt obrotu w reprezentacji axis-angle [rad]

    Returns:
        quaternion (np.array[4x1]): Kwaternion obrotu (qw, qx i, qy j, qz k), wektor pionowy
    """
    # Ekstrakcja wyrazów osi obrotu (reprezentacja axis-angle)
    ux, uy, uz = unit_axis.flat

    # sin(angle/2) do obliczeń niżej
    sin_half = np.sin(angle_rad / 2)

    # Wyznaczanie wyrazów kwaternionu obrotu (qw, qx, qy, qz) z reprezentacji axis-angle
    qw = np.cos(angle_rad / 2)
    qx = ux * sin_half
    qy = uy * sin_half
    qz = uz * sin_half

    # c_[] - pionizacja wektora [4] na [4x1]
    return np.c_[[qw, qx, qy, qz]]


def quaternion_from_rotation_vector(rotation_vector, eps=0):
    """
    Wyznacza kwaternion obrotu z wektora obrotu [rad]

    Parameters:
        rotation_vector (np.array[3x1]): Wektor obrotu [rad]
        eps (float): Epsilon maszynowy, aby zapobiec dzieleniu przez 0

    Returns:
        quaternion (np.array[4x1]): Kwaternion obrotu (qw, qx i, qy j, qz k), wektor pionowy
    """
    # Ektrakcja kąta obrotu (reprezentacja axis-angle) z wektora-obrotu [rad]
    angle_rad = calculate_vector_norm(rotation_vector)

    # Ochrona przed dzieleniem przez 0; poniżej eps -> przyjmujemy zero
    if angle_rad > eps:
        # Wyznaczenie osi obrotu (do reprezentacji axis-angle)
        unit_axis = rotation_vector / angle_rad
        # axis-angle -> kwaternion obrotu
        q_unit = quaternion_from_axis_angle(unit_axis, angle_rad)
    else:
        # Dla kąta=0 -> brak obrotu (kwaternion jednostkowy)
        q_unit = quaternion_identity()

    # print(f"q_unit: {calculate_vector_norm(q_unit)}")
    return q_unit


def get_Q(gyro_noises):
    gyro_noise_x, gyro_noise_y, gyro_noise_z = gyro_noises
    Q = np.array([[gyro_noise_x,               0,               0],
                  [              0, gyro_noise_y,               0],
                  [              0,               0, gyro_noise_z]])
    return Q

def get_Q_bias(gyro_bias_noises):
    gbn_x, gbn_y, gbn_z = gyro_bias_noises
    Q_bias = np.zeros((7, 7))
    Q_bias[4:7, 4:7] = np.array([[gbn_x,        0,        0],
                                 [       0, gbn_y,        0],
                                 [       0,        0, gbn_z]])
    return Q_bias

def get_R(accelerometer_noises):
    an_x, an_y, an_z = accelerometer_noises
    R = np.array([[an_x,       0,       0],
                  [      0, an_y,       0],
                  [      0,       0, an_z]])
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

def get_V():
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

def f(x, angular_velocities, dt):
    """
    Funkcja predykcji stanu x (kwaternion+biasy) na podstawie pomiaru z żyroskopu 3-osiowego.

    Parameters:
        x (np.array[7x1]): Wektor stanu (qw, qx, qy, qz, bx, by, bz); wektor pionowy
            - wyrazy kwaternionu orientacji: qw, qx, qy, qz
            - biasy żyroskopu: bx, by, bz [rad/s]
        angular_velocities (np.array[3x1]): Prędkości kątowe (wx, wy, wz) z żyroskopu; wektor pionowy [rad/s]
        dt (float): Okres próbki [s]

    Returns:
        x (np.array[7x1]): Wektor stanu (qw, qx, qy, qz, bx, by, bz) po predykcji; wektor pionowy
            - wyrazy kwaternionu orientacji: qw, qx, qy, qz
            - biasy żyroskopu: bx, by, bz [rad/s]
    """
    # Ekstrakcja z wektora stanu [x]
    # kwaternion orientacji (pozycji obrotowej)
    q = x[0:4]
    # biasy żyroskopu
    gyro_biases = x[4:7]

    # Wektor obrotu (kompaktowe axis-angle)
    delta_angles = (angular_velocities - gyro_biases) * dt

    # Wektor obrotu -> kwaternion obrotu
    dq = quaternion_from_rotation_vector(delta_angles)
    # TODO: normalizacja dq

    # Predykcja nowej orientacji (orientacja += zmiana, ale mnożenie dla kwaternionu)
    q_new = quaternion_multiply(q, dq)
    q_new = normalize_vector(q_new)     # Normalizacja w celu uniknięcia błędów komputera
    
    # Złożenie powrotne wektora stanu po aktualizacji orientacji (pionowy)
    return np.r_[q_new, gyro_biases]


def get_F(x, angular_velocities, dt):
    """
    Zwraca Jakobian funkcji f()
    """
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


def h(x):
    # Ekstrakcja kwaternionu
    q = x[0:4]

    # Macierz obrotu - układ odniesienie obiektu (body)
    R_from_body = quaternion_to_rotation_matrix(q)

    # Konwersja wektora grawitacji układu-otoczenia do układu-obiektu
    # return R_from_body.T @ np.c_[[0, 0, -GRAVITY]]
    return R_from_body.T @ np.c_[[0, 0, 1]] # (1) for gravity


def get_H(x):
    """
    Zwraca Jakobian funkcji h()
    """
    qw, qx, qy, qz = x.flat[0:4]

    H = 2 * (-1) * np.array([    #(-1) for gravity flip
        [ qy, -qz,  qw, -qx, 0, 0, 0],
        [-qx, -qw, -qz, -qy, 0, 0, 0],
        [-qw,  qx,  qy, -qz, 0, 0, 0]
    ])

    # Wersja bez skróconych obliczeń (z wyprowadzenia: https://ahrs.readthedocs.io/en/latest/filters/ekf.html)
    # q_w, q_x, q_y, q_z = x.flat[0:4]
    # g_x, g_y, g_z = [0, 0, 1]

    # H_00 = g_x*q_w + g_y*q_z - g_z*q_y
    # H_01 = g_x*q_x + g_y*q_y + g_z*q_z
    # H_02 = -g_x*q_y + g_y*q_x - g_z*q_w
    # H_03 = -g_x*q_z + g_y*q_w + g_z*q_x

    # H_10 = -g_x*q_z + g_y*q_w + g_z*q_x
    # H_11 = g_x*q_y - g_y*q_x + g_z*q_w
    # H_12 = g_x*q_x + g_y*q_y + g_z*q_z
    # H_13 = -g_x*q_w - g_y*q_z + g_z*q_y

    # H_20 = g_x*q_y - g_y*q_x + g_z*q_w
    # H_21 = g_x*q_z - g_y*q_w - g_z*q_x
    # H_22 = g_x*q_w + g_y*q_z - g_z*q_y
    # H_23 = g_x*q_x + g_y*q_y + g_z*q_z

    # H = 2 * np.array([[H_00, H_01, H_02, H_03, 0, 0, 0],
    #                   [H_10, H_11, H_12, H_13, 0, 0, 0],
    #                   [H_20, H_21, H_22, H_23, 0, 0, 0]])
    return H

class EKF:
    def __init__(self, q0=[1,0,0,0], b0=[0,0,0], delta_t=0.005, 
                 init_gyro_quat_err=0.1, init_gyro_bias_err=0.1, gyro_noises=[0.015,0.015,0.015], gyro_bias_noises=[0.002,0.002,0.002],
                 accelerometer_noises=[0.002, 0.002, 0.002]):
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
        self.P[0:4, 0:4] = np.identity(4) * init_gyro_quat_err   # 0 na start, bo startujemy z pewnej pozycji (0,0,0)
        # - Część biasu żyroskopu
        self.P[4:7, 4:7] = np.identity(3) * (init_gyro_bias_err ** 2)
        
        # Macierz kowariancji szumu procesu (GYRO) [3x3]
        # Przemnażana w predykcji przez W
        # [rad/sec]
        self.Q = get_Q(gyro_noises)
        # Macierz kowariancji szumu biasu GYRO [7x7]
        # [rad/sec]
        self.Q_bias = get_Q_bias(gyro_bias_noises)

        # Macierz kowariancji szumu obserwacji (ACC) [3x3]
        # Przemnażana w korekcji przez V
        # [m/s^2]
        self.R = get_R(accelerometer_noises)

        print(self.P)
        print(self.Q)
        print(self.Q_bias)
        print(self.R)

    def predict(self, angular_velocities):
        """
        Predykcja EKF - użyty GYRO (żyroskop) jako proces

        Parameters:
            angular_velocities (List[wx, wy, wz]): Prędkości kątowe GYRO [rad/sec]
            dt (float): krok czasu [sec] TODO: dodać
        """
        # Prędkości kątowe na wektor pionowy [3x1]
        angular_velocities = np.c_[angular_velocities]

        # Macierz F - Jakobian z f() [7x7]
        F = get_F(self.x, angular_velocities, self.dt)
        
        # Macierz W do przekształcenia Q procesu [prędkości x,y,z] -> [kwaternion] [7x3]
        W = get_W(self.x, self.dt)

        # Obliczamy predykcję stanu [kwaternion+biasy, 7x1]
        self.x = f(self.x, angular_velocities, self.dt)

        # Przewidujemy macierz kowariancji [7x7]
        self.P = F @ self.P @ F.T + W @ self.Q @ W.T + self.Q_bias

    def update(self, accelerations):
        """
        Korekcja EKF - użyty ACC (akcelerometr) jako obserwacja

        Parameters:
            accelerations (List[ax, ay, az]): Przyspieszenia z akcelerometru [m/s^2]
            dt (float): krok czasu [sec] TODO: dodać
        """
        # Pozyskanie pomiaru z ACC [3x1]
        accelerations = np.c_[accelerations]
        z = normalize_vector(accelerations)

        # Predykcja pomiaru ACC na podstawie stanu [3x1]
        h_prediction = h(self.x)
        
        # Innowacja - różnica między predykcją a pomiarem ACC [3x1]
        # porównywanie w postaci wektora grawitacji
        y = z - h_prediction
        
        # Macierz H - Jakobian z h() [3x7]
        H = get_H(self.x)

        # Macierz kowariancji innowacji [3x3]
        S = H @ self.P @ H.T + self.R
        # Wzmocnienie Kalmana [7x3]
        K = (self.P @ H.T) @ np.linalg.inv(S)

        # Aktualizacja wektora stanu (kwaternionowo)
        # na razie nie działa :/
        # k_temp = (K @ y)
        # k_temp[:4] = normalize_vector(k_temp[:4])
        # self.x[:4] = quaternion_multiply(self.x[:4], k_temp[:4])
        # self.x[4:] = self.x[4:] + k_temp[4:]

        # Aktualizacja wektora stanu
        self.x = self.x + (K @ y)
        self.x[:4] = normalize_vector(self.x[0:4])  # Normalizacja kwaternionu, wynika z dodawania zamiast mnożenia

        # Aktualizacja macierzy kowariancji [7x7]
        self.P = (np.eye(7) - K @ H) @ self.P

        return z.reshape(3), h_prediction.reshape(3)


    def get_orientation(self):
        return self.x[:4].reshape(4)
    
    def get_orientation_std(self):
        return np.sqrt(np.diag(self.P)[0:4])
    
    def get_gyro_biases(self):
        return self.x[4:].reshape(3)
    
    def get_gyro_bias_std(self):
        return np.sqrt(np.diag(self.P)[4:7])
    