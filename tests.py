import pandas as pd
import matplotlib.pyplot as plt

gravity = 9.81

if __name__ == '__main__':
    data = pd.read_csv("data/train.csv")
    print(data.head())

    # Dane z akcelerometru w g -> m/s^2
    data['AccX'] *= gravity
    data['AccY'] *= gravity
    data['AccZ'] *= gravity

    plt.plot(data['Time'], data['AccX'], label="AccX")
    plt.plot(data['Time'], data['AccY'], label="AccY")
    plt.plot(data['Time'], data['AccZ'], label="AccZ")
    plt.legend()
    plt.show()
