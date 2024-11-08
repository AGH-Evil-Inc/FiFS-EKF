import pandas
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # data_board_acc = pandas.read_csv("data/plytka_ACC_Data_Log_2024_10_16_13_20_44.csv")
    # data_phone_acc = pandas.read_csv("data/telefon_ACC.csv")
    #
    # print(data_board_acc.columns)
    # print(data_phone_acc.columns)
    #
    # plt.plot(data_board_acc['time[us]'], data_board_acc['accX[mg]'])
    # #plt.plot(data_board['time[us]'], data_board['accY[mg]'])
    # #plt.plot(data_board['time[us]'], data_board['accZ[mg]'])
    #
    # plt.plot((data_phone_acc['Time (s)'] + 2.411) * 1000000, data_phone_acc['Acceleration x (m/s^2)'] * 1000 / 9.81)
    # plt.legend(["dane z płytki", "dane z telefonu"])
    # plt.show()

    #print(data_board['accX[mg]'])
    #print(data_phone['Acceleration x (m/s^2)'])

    data_board_gyro = pandas.read_csv("data/plytka_GYRO_Data_Log_2024_10_16_14_15_32.csv")
    data_phone_gyro = pandas.read_csv("data/telefon_GYRO.csv")

    print(data_board_gyro.columns)
    print(data_phone_gyro.columns)

    plt.plot(data_board_gyro['time[us]'], data_board_gyro['gyroX[mdps]'])
    plt.plot((data_phone_gyro['Time (s)'] + 8.5) * 1000000, data_phone_gyro['Gyroscope x (rad/s)']*57300)
    plt.title("Porównanie pomiaru osi X żyroskopu płytki i telefonu")
    plt.xlabel("Czas [us]")
    plt.ylabel("Prędkość kątowa [mdps] (mili stopnie na sekundę)")
    plt.legend(["dane z płytki", "dane z telefonu"])
    plt.show()


