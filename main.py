import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

from ekf_filter import EKF
from ekf_filter import GRAVITY
from ekf_filter import quaternion_to_euler


if __name__ == '__main__':
    data_train = pd.read_csv("data/train.csv")

    # Komentować zamiennie - tylko [train part] albo [train part]+[entire test]
    # data = data_train
    data = pd.read_csv("data/test.csv")

    calibration_samples = 1000

    # Dane z akcelerometru w g -> m/s^2
    data['AccX'] *= GRAVITY
    data['AccY'] *= GRAVITY
    # data['AccX'], data['AccY'] = data['AccY'], data['AccX']
    data['AccZ'] *= GRAVITY

    acc_bias_x = np.mean(data['AccX'][:calibration_samples])
    acc_bias_y = np.mean(data['AccY'][:calibration_samples])
    # acc_bias_z = np.mean(data['AccZ'][:calibration_samples]) - 1
    acc_bias_z = np.mean(data['AccZ'][:calibration_samples]) - GRAVITY

    # ACC calibration matrix from [./get_calib_parameters.ipynb]
    acc_calibration_matrix = np.array([
        [0.9884646573407392,        -0.005196578337050819,  -0.003962203739630031],
        [-0.015449996787821385,     1.0139083437181962,     -0.041641932456625924],
        [0.01265817110981942,       0.04266824758539578,     1.0029777251748326  ],
        [-0.00028401797239294146,   0.0223789880167591,     -0.004273695060949001]
    ])

    acc_vars = np.var(data[['AccX', 'AccY', 'AccZ']][:calibration_samples], axis=0)

    # Dane z żyroskopu w mdps (mili degrees per second) -> rad/s
    data['GyroX'] *= 0.001 * (np.pi / 180)
    data['GyroY'] *= 0.001 * (np.pi / 180)
    data['GyroZ'] *= 0.001 * (np.pi / 180)

    g_bias_x = np.mean(data['GyroX'][:calibration_samples])
    g_bias_y = np.mean(data['GyroY'][:calibration_samples])
    g_bias_z = np.mean(data['GyroZ'][:calibration_samples])

    gyro_vars = np.var(data[['GyroX', 'GyroY', 'GyroZ']][:calibration_samples], axis=0)

    gyro_bias_noice_var = 0.00000001
    # s2 = 0.0001

    # gyro_noise = np.diag(np.append(gyro_vars, [s1 ** 2])) * 4e-2  # *(dt**2/4)
    # acc_noise = np.diag(np.append(acc_vars, [s2 ** 2]))

    dt = 0.005
    ekf = EKF(q0=[1, 0, 0, 0],
              b0=[g_bias_x, g_bias_y, g_bias_z],
            #   b0=[0,0,0],
              delta_t=dt,
              init_gyro_bias_err=0.0,
              gyro_noises=gyro_vars,
              gyro_bias_noises=[gyro_bias_noice_var,gyro_bias_noice_var,gyro_bias_noice_var],
              accelerometer_noises=acc_vars)

    pred = []

    z_vec = []
    h_vec = []
    for _, row in data.iterrows():
        gyroscope_measurement = np.array([row['GyroX'], row['GyroY'], row['GyroZ']])
        # gyroscope_measurement = np.array([row['GyroX'] - g_bias_x, row['GyroY'] - g_bias_y, row['GyroZ'] - g_bias_z])

        # # Get raw ACC sample
        # raw_acc_sample = np.array([row['AccX'], row['AccY'], row['AccZ'], 1])    # 1 at the end for correct matrix multiplication
        # # Calibrate ACC sample
        # accelerometer_measurement = raw_acc_sample @ acc_calibration_matrix
        accelerometer_measurement = np.array([row['AccX'] - acc_bias_x, row['AccY'] - acc_bias_y, row['AccZ'] - acc_bias_z])
        # accelerometer_measurement = np.array([row['AccX'], row['AccY'], row['AccZ']])

        # Predykcja za pomocą żyroskopu
        ekf.predict(gyroscope_measurement)

        # Korekcja za pomocą akcelerometru
        z, h = ekf.update(accelerometer_measurement)

        z_vec.append(z)
        h_vec.append(h)

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


    # MSE (Mean Square Error)
    # Liczone tylko dla [train part] (piersze 7000)
    mse_7_roll  = mean_squared_error(data_train['roll'], pred_roll[:7000])
    mse_7_pitch = mean_squared_error(data_train['pitch'], pred_pitch[:7000])
    mse_7_yaw   = mean_squared_error(data_train['yaw'], pred_yaw[:7000])
    mse_7_mean  = np.mean([mse_7_roll, mse_7_pitch, mse_7_yaw])
    # Liczone tylko dla pierwszych 5000
    mse_5_roll  = mean_squared_error(data_train['roll'][:5000], pred_roll[:5000])
    mse_5_pitch = mean_squared_error(data_train['pitch'][:5000], pred_pitch[:5000])
    mse_5_yaw   = mean_squared_error(data_train['yaw'][:5000], pred_yaw[:5000])
    mse_5_mean  = np.mean([mse_5_roll, mse_5_pitch, mse_5_yaw])
    # Porównanie MSE
    print("Sample count |  Roll MSE | Pitch MSE |  Yaw MSE  | Combined MSE")
    print(f"     7000    | {mse_7_roll:9.6f} | {mse_7_pitch:9.6f} | {mse_7_yaw:9.6f} | {mse_7_mean:.6f}")
    print(f"     5000    | {mse_5_roll:9.6f} | {mse_5_pitch:9.6f} | {mse_5_yaw:9.6f} | {mse_5_mean:.6f}")

    
    z_vec = np.array(z_vec)
    h_vec = np.array(h_vec)
    
    # Pierwsze 7000 - nałożone prawdziwe pozycja z [train part]
    fig, axs = plt.subplots(3, 2, figsize=(12, 5))
    axs[0,0].plot(data['Time'], pred_roll, label="Roll prediction")
    axs[0,0].plot(data_train['Time'], data_train['roll'], label="Roll actual")
    axs[0,0].legend()
    axs[0,0].grid()

    axs[1,0].plot(data['Time'], pred_pitch, label="Pitch prediction")
    axs[1,0].plot(data_train['Time'], data_train['pitch'], label="Pitch actual")
    axs[1,0].legend()
    axs[1,0].grid()

    axs[2,0].plot(data['Time'], pred_yaw, label="Yaw prediction")
    axs[2,0].plot(data_train['Time'], data_train['yaw'], label="Yaw actual")
    axs[2,0].legend()
    axs[2,0].grid()
    
    axs[0,1].plot(data['Time'], h_vec[:,0], marker='.', markersize=1, linewidth=0.5, label="Predykcja ACC  X", zorder=10)
    axs[0,1].plot(data['Time'], z_vec[:,0], marker='.', markersize=1, linewidth=0.5, label="Pomiar ACC  X")
    axs[0,1].legend()
    axs[0,1].grid()

    axs[1,1].plot(data['Time'], h_vec[:,1], marker='.', markersize=1, linewidth=0.5, label="Predykcja ACC  Y", zorder=10)
    axs[1,1].plot(data['Time'], z_vec[:,1], marker='.', markersize=1, linewidth=0.5, label="Pomiar ACC  Y")
    axs[1,1].legend()
    axs[1,1].grid()

    axs[2,1].plot(data['Time'], h_vec[:,2], marker='.', markersize=1, linewidth=0.5, label="Predykcja ACC  Z", zorder=10)
    axs[2,1].plot(data['Time'], z_vec[:,2], marker='.', markersize=1, linewidth=0.5, label="Pomiar ACC  Z")
    axs[2,1].legend()
    axs[2,1].grid()

    
    fig.tight_layout()
    plt.show()