import pandas as pd
import numpy as np
import functions as fn
import config as cfg
import exps.exp_22102015.config_exp as cfg_exp
import exps.fn_exp as fn_exp
from sklearn.ensemble import RandomForestRegressor
import glob
import csv
import matplotlib.pyplot as plt

__author__ = 'YBeer'


def parse_rssi(rssi):
    rssi[0] = rssi[0][2:]
    rssi[-1] = rssi[-1][:4]
    rssi = map(lambda x: float(x), rssi)
    return rssi

"""
Get files
"""
# insert the AP's MAC
fnames = glob.glob('calibration\\00140608*.csv')

# Placing and directing APs, direction is the AP main direction. phi = 0 -> y+, phi = 90 -> x+. like a compass
aps_raw = pd.DataFrame.from_csv('ap_pos.csv')
heading = np.array(aps_raw)[-1, :2]
aps_raw = np.array(aps_raw)[:-1, :]

# get APs dictionary
aps_dict = {}
for i in range(aps_raw.shape[0]):
    aps_dict[aps_raw[i, -1]] = i
# # manually add AP
# aps_dict['0014060849b0'] = 4

"""
Init variables for all the APs
"""
dataset = []
dataset_angle = []
test_data = []
rfc = []
for i in range(len(fnames)):
    dataset.append([])
    dataset_angle.append([])
    test_data.append([])
    rfc.append(RandomForestRegressor(n_estimators=50, max_depth=8, max_features=0.5))

"""
Parse training data
"""
for file_name in fnames:
    new_dataset = []
    new_angle = []
    with open(file_name, 'rb') as csvfile:
        cal_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        # read headline
        cal_title = cal_reader.next()
        cal_AP = cal_title[1]
        cal_ms = cal_title[0]
        print file_name, aps_dict[cal_AP]
        cal_index = aps_dict[cal_AP]
        # read 1st angle
        angle = float(cal_reader.next()[0])

        # read rest of file
        for row in cal_reader:
            if len(row) == 1:
                angle = float(row[0])
            else:
                row = parse_rssi(row[2:10])
                max_rssi = np.max(row)
                if cfg.min_rssi <= max_rssi <= cfg.max_rssi:
                    new_dataset.append(row + [angle])
    csvfile.close()

    dataset[cal_index] = np.array(new_dataset)
    dataset_angle[cal_index] = dataset[cal_index][:, -1]
    """
    Create test data, one for each AP
    """
    test_data[cal_index] = fn.arrange_data(dataset[cal_index][:, :-1])
    dataseta_for_sd = pd.DataFrame(dataset[cal_index][:, :-1], columns=['V0', 'V1', 'V2', 'V3', 'H0', 'H1',
                                                                        'H2', 'H3'])

    # filter noise below -10 dB
    test_data[cal_index] = fn.noise_filter(np.array(test_data[cal_index]))

    # read database from file
    test_data[cal_index] = pd.DataFrame(test_data[cal_index], columns=['V0', 'V1', 'V2', 'V3', 'H0', 'H1',
                                                                       'H2', 'H3', 'V_m_H'])

    dataset_angle[cal_index] = pd.DataFrame(dataset_angle[cal_index], columns=['Angle'])

    # filter angles out of range
    valid_angle = (cfg.min_angle <= dataset_angle[cal_index]['Angle']) & \
                  (cfg.max_angle >= dataset_angle[cal_index]['Angle'])

    test_data[cal_index] = test_data[cal_index].loc[valid_angle]
    dataset_angle[cal_index] = dataset_angle[cal_index].loc[valid_angle]

    dataset_angle[cal_index] = np.array(dataset_angle[cal_index]).ravel()

    """
    create trainset
    """
    ant_pat = fn_exp.find_pattern(test_data[cal_index], dataset_angle[cal_index])
    train = np.repeat(ant_pat, cfg_exp.cal_rep, axis=0)
    noise = np.random.normal(0, cfg_exp.cal_noise, train.shape)
    train += noise

    train_max = np.max(train[:, :-1], axis=1).reshape((train.shape[0], 1))

    train -= train_max

    # train angles
    ant_angles = [cfg.min_angle]
    while ant_angles[-1] < cfg.max_angle:
        ant_angles.append(ant_angles[-1] + 2.5)
    ant_angles = np.array(ant_angles)
    ant_angles = np.repeat(ant_angles, cfg_exp.cal_rep)

    # Fitting to RF
    rfc[cal_index].fit(train, ant_angles)

    # plot test prediction(angle) - for checking
    x = range(ant_angles.shape[0])

    # # creating predicted test set angles
    test_prediction = rfc[cal_index].predict(test_data[cal_index])
    RSME = np.sqrt(np.sum((dataset_angle[cal_index] - test_prediction) ** 2) / test_prediction.shape[0])
    print 'Calibrated AP: ', cal_AP, 'RSME is: ', RSME
    plt.plot(dataset_angle[cal_index], test_prediction, 'ro')
    plt.xlabel('Angle')
    plt.ylabel('Predition')
    plt.show()

    """
    Plot antennas' graph
    """
    # # Find parameters SD (from not arranged data)
    # sd_ant = fn_exp.cal_sd(dataseta_for_sd, dataset_angle[cal_index])
    #
    # ant_angles = [cfg.min_angle]
    # while ant_angles[-1] < cfg.max_angle:
    #     ant_angles.append(ant_angles[-1] + 2.5)
    # ant_angles = np.array(ant_angles)
    #
    # # plotting
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(ant_angles, ant_pat[:, 0], 'r', ant_angles, ant_pat[:, 0] + sd_ant[:, 0], 'r--',
    #          ant_angles, ant_pat[:, 0] - sd_ant[:, 0], 'r--',
    #          ant_angles, ant_pat[:, 1], 'g', ant_angles, ant_pat[:, 1] + sd_ant[:, 1], 'g--',
    #          ant_angles, ant_pat[:, 1] - sd_ant[:, 1], 'g--',
    #          ant_angles, ant_pat[:, 2], 'b', ant_angles, ant_pat[:, 2] + sd_ant[:, 2], 'b--',
    #          ant_angles, ant_pat[:, 2] - sd_ant[:, 2], 'b--',
    #          ant_angles, ant_pat[:, 3], 'k', ant_angles, ant_pat[:, 3] + sd_ant[:, 3], 'k--',
    #          ant_angles, ant_pat[:, 3] - sd_ant[:, 3], 'k--')
    # plt.title('V pol')
    # plt.xlim((cfg.min_angle, cfg.max_angle))
    # plt.ylim((-20, 10))
    #
    # plt.subplot(212)
    # plt.plot(ant_angles, ant_pat[:, 4], 'r', ant_angles, ant_pat[:, 4] + sd_ant[:, 4], 'r--',
    #          ant_angles, ant_pat[:, 4] - sd_ant[:, 4], 'r--',
    #          ant_angles, ant_pat[:, 5], 'g', ant_angles, ant_pat[:, 5] + sd_ant[:, 5], 'g--',
    #          ant_angles, ant_pat[:, 5] - sd_ant[:, 5], 'g--',
    #          ant_angles, ant_pat[:, 6], 'b', ant_angles, ant_pat[:, 6] + sd_ant[:, 6], 'b--',
    #          ant_angles, ant_pat[:, 6] - sd_ant[:, 6], 'b--',
    #          ant_angles, ant_pat[:, 7], 'k', ant_angles, ant_pat[:, 7] + sd_ant[:, 7], 'k--',
    #          ant_angles, ant_pat[:, 7] - sd_ant[:, 7], 'k--')
    # plt.title('H pol')
    # plt.xlim((cfg.min_angle, cfg.max_angle))
    # plt.ylim((-20, 10))
    # plt.show()
    #
    # ant_rssi = ant_pat.reshape((ant_pat.shape[0]*ant_pat.shape[1]))
    # ant_sd = sd_ant.reshape((sd_ant.shape[0]*sd_ant.shape[1]))
    #
    # plt.plot(ant_rssi, ant_sd, 'ro')
    # plt.show()

    # pd.DataFrame(dataset_angle[cal_index]).to_csv(file_name[:12] + 'noisycal.csv')
