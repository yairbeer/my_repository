import pandas as pd
import math
import track_exp1308 as track
import config_exp1308 as cfg_exp
import numpy as np
import functions as fn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

__author__ = 'YBeer'


def parse_rssi(rssi):
    rssi = rssi[1:-1].split(', ')
    rssi = map(lambda x: float(x), rssi)
    return rssi

predictions = np.zeros(())
ap_timed_pred = np.zeros((len(track.time_frames), len(track.valid_ants)))
ap_timed_sd = np.zeros((len(track.time_frames), len(track.valid_ants)))
for i in track.valid_ants:
    """
    Create training data
    """
    # read database from file
    dataset = pd.read_csv('dataset_ap' + str(i) + '.csv',
                          names=['V0', 'V1', 'V2', 'V3', 'H0', 'H1', 'H2', 'H3', 'V_m_H'])
    dataset_time = np.zeros((dataset.shape[0], 1))
    dataset = np.array(dataset)
    dataset = np.hstack((dataset_time, dataset))

    # filter noise below -10 dB
    dataset = fn.noise_filter(np.array(dataset))

    dataset_angle = pd.read_csv('dataset_angle_ap' + str(i) + '.csv', names=['Angle'])
    # filter angles out of range
    valid_angle = (cfg_exp.min_angle <= dataset_angle['Angle']) & (cfg_exp.max_angle >= dataset_angle['Angle'])

    dataset = fn.filter_rows(dataset, valid_angle)
    dataset_angle = dataset_angle.loc[valid_angle]

    dataset_angle = np.array(dataset_angle).ravel()

    # Fitting to RF
    clf = RandomForestClassifier()
    clf.fit(np.array(dataset[:, 1:]), dataset_angle)

    # creating predicted test set angles
    test_prediction = clf.predict(dataset[:, 1:]).tolist()

    # # plot test data
    # plt.plot(dataset_angle, test_prediction)
    # plt.show()

    # Fitting to RF
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1, min_samples_leaf=1,
                                 max_features=3, criterion="gini", min_samples_split=2)
    clf.fit(dataset[:, 1:], dataset_angle)

    """
    Get experiment data
    """
    ap = pd.read_csv('ap' + str(i) + '.csv', index_col=0, names=['MAC', 'Time', 'RSSIs', 'channel'])
    ap = ap.loc[ap.index == cfg_exp.mac]
    ap['Time'] /= 1000
    ap['Time'] = ap['Time'] - track.t_0
    ap = ap.loc[ap['Time'] >= 0]
    ap = ap.loc[ap['Time'] <= track.time_stop]

    ap_rssis = list(ap['RSSIs'])

    for j in range(len(ap_rssis)):
        ap_rssis[j] = parse_rssi(ap_rssis[j])

    del ap['RSSIs']

    ap_rssis = pd.DataFrame(ap_rssis, columns=['V0', 'V1', 'V2', 'V3', 'H0', 'H1', 'H2', 'H3'], index=ap.index)
    ap = pd.concat([ap, ap_rssis], axis=1)
    ap_rssis = np.array(ap_rssis)

    # Arranging model data
    ap_max = np.apply_along_axis(np.max, 1, ap_rssis)

    ap_rssis = np.hstack((np.array(np.transpose(np.matrix(ap['Time']))), ap_rssis))

    ap_arranged = fn.arrange_data(ap_rssis)
    ap_arranged = fn.noise_filter(ap_arranged)

    # conditions
    not_sat_power = ap_max <= cfg_exp.rssi_max
    not_low_power = ap_max >= cfg_exp.rssi_min
    not_erroneous = ap_arranged[:, 9] > -10

    # Filtered
    ap_arranged_filtered = fn.filter_rows(ap_arranged, not_sat_power & not_low_power & not_erroneous)

    # Predicting basic model result
    ap_pred = clf.predict(ap_arranged_filtered[:, 1:])
    ap_pred = ap_pred.reshape((ap_pred.shape[0], 1))
    ap_pred = np.hstack((ap_arranged_filtered[:, [0]], ap_pred))

    # # plot predictions
    # plt.plot(ap_pred[:, 0], ap_pred[:, 1], 'go')
    # plt.show()

    # Building time frames
    time_frames = track.time_frames

    # calculate real angles from AP
    ap_angles = np.zeros((len(time_frames), 3))
    ap_angles[:, 0] = time_frames

    # global
    # Get global angles
    deltax = track.track_position[:, 1] - track.aps[i, 0]
    deltay = track.track_position[:, 2] - track.aps[i, 1]

    atan_vectorized = np.vectorize(math.atan)
    g_angle = 180 / np.pi * atan_vectorized(deltax / deltay)

    ap_angles[:, 1] = g_angle
    ap_angles[:, 2] = g_angle - track.aps[i, 2]

    # not valid prediction are saved as 100
    ap_timed_pred[:, i], ap_timed_sd[:, i] = fn.timed_predictions(ap_pred, time_frames)

    # # prediction vs angle
    # plt.plot(track.time_frames, ap_timed_pred[:, 0], 'go', track.time_frames, ap_angles[:, 1], 'r')
    # plt.show()

    # # deltax VS angle
    # plt.plot(track.track_position[:, 0], track.track_position[:, 1] - track.aps[i, 0],
    #          ap_angles[:, 0], ap_angles[:, 1])
    # plt.show()
    #
    # # deltay VS angle
    # plt.plot(track.track_position[:, 0], track.track_position[:, 2] - track.aps[i, 1],
    #          ap_angles[:, 0], ap_angles[:, 1])
    # plt.show()

ap_direction = np.zeros((ap_timed_pred.shape[0], len(track.valid_ants)))

for i in range(len(track.valid_ants)):
    ap_direction[:, i] = track.aps[track.valid_ants[i], 2]
