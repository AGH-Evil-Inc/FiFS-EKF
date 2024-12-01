import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
gravity = 9.81

if __name__ == '__main__':
    data = pd.read_csv("data/train.csv")
    calibration_samples = 1000

    # Dane z akcelerometru w g -> m/s^2
    data['AccX'] *= gravity
    data['AccY'] *= gravity
    data['AccZ'] *= gravity

    acc_bias_x = np.mean(data['AccX'][:calibration_samples])
    acc_bias_y = np.mean(data['AccY'][:calibration_samples])
    acc_bias_z = np.mean(data['AccZ'][:calibration_samples]) - gravity

    
    data['AccX'] -= acc_bias_x
    data['AccY'] -= acc_bias_y
    data['AccZ'] -= acc_bias_z

    # Dane z żyroskopu w mdps (mili degrees per second) -> rad/s
    data['GyroX'] *= 0.001 * (np.pi / 180)
    data['GyroY'] *= 0.001 * (np.pi / 180)
    data['GyroZ'] *= 0.001 * (np.pi / 180)

    g_bias_x = np.mean(data['GyroX'][:calibration_samples])
    g_bias_y = np.mean(data['GyroY'][:calibration_samples])
    g_bias_z = np.mean(data['GyroZ'][:calibration_samples])

    plt.plot(data['Time'], data['AccX'], label="AccX")
    plt.plot(data['Time'], data['AccY'], label="AccY")
    plt.plot(data['Time'], data['AccZ'], label="AccZ")
    plt.legend()
    plt.show()
