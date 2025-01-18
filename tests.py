import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
gravity = 9.81

if __name__ == '__main__':
    data = pd.read_csv("data/test.csv")
    calibration_samples = 1000

    # Dane z akcelerometru w g -> m/s^2
    # data['AccX'] *= gravity
    # data['AccY'] *= gravity
    # data['AccZ'] *= gravity

    acc_bias_x = np.mean(data['AccX'][:calibration_samples])
    acc_bias_y = np.mean(data['AccY'][:calibration_samples])
    # acc_bias_z = np.mean(data['AccZ'][:calibration_samples]) - gravity
    acc_bias_z = np.mean(data['AccZ'][:calibration_samples]) - 1.0

    
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

    fig, ax = plt.subplots(1, 1)
    ax.grid()
    ax.plot(data['Id'], data['AccX'], label="AccX")
    ax.plot(data['Id'], data['AccY'], label="AccY")
    ax.plot(data['Id'], data['AccZ'], label="AccZ")
    ax.legend()
    plt.show()
