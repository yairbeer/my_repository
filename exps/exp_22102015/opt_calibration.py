import pandas as pd
import numpy as np
import functions as fn
import config as cfg
import exps.exp_22102015.config_exp as cfg_exp
import exps.fn_exp as fn_exp
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import ParameterGrid
import glob
import csv

__author__ = 'YBeer'


def parse_rssi(rssi):
    rssi[0] = rssi[0][2:]
    rssi[-1] = rssi[-1][:4]
    rssi = map(lambda x: float(x), rssi)
    return rssi

"""
init parameter sweep
"""
param_grid = {'n_estimators': [50], 'max_depth': [8], 'max_features': [0.5]}
rsme_mim = 20
params_min = []
for params in ParameterGrid(param_grid):
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
        rfc.append(RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                         max_features=params['max_features']))

    """
    Parse training data
    """
    cur_rsme = []
    for file_name in fnames:
        new_dataset = []
        new_angle = []
        with open(file_name, 'rb') as csvfile:
            cal_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

            # read headline
            cal_title = cal_reader.next()
            cal_AP = cal_title[1]
            cal_ms = cal_title[0]
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
        cur_rsme.append(RSME)
    cur_rsme = np.mean(cur_rsme)
    if cur_rsme < rsme_mim:
        rsme_mim = cur_rsme
        params_min = params
    print 'Params: ', params, 'RSME is: ', cur_rsme
    print 'Best Params: ', params_min, 'minimum RSME is: ', rsme_mim
