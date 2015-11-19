import glob
import pandas as pd
import numpy as np
import exps.exp_22102015.track as track
import config
import exps.exp_22102015.config_exp as cfg_exp
import functions as fn
import exps.exp_22102015.calibration_noisy as cal
import scipy.optimize as opt
import exps.fn_exp as fn_exp
import matplotlib.pyplot as plt

__author__ = 'YBeer'

"""
This script is for minimizing the experiment setup errors and optimizing it's paramaters.
AP's x, y, heading, time
Kalman filter parameters

The AP parameters would be optimizied for AP alone
The Kalman for the mean of all of them
"""


def timed_predictions(preds, pred_times, time_frames):
    timed_pred = np.ones((len(time_frames), 1))
    timed_std = np.ones((len(time_frames), 1))
    for i_fun in range(len(time_frames)):
        cur_frame_pred = []
        cur_frame_weight = []
        for j_fun in range(preds.shape[0]):
            # Building subsets
            if time_frames[i_fun] < pred_times[j_fun] < (time_frames[i_fun] + cfg_exp.time_step):
                cur_frame_pred.append(float(preds[j_fun]))
                cur_frame_weight.append(1)
        # initializing with a value that isn't a legal outcome
        cur_frame_mean = float('nan')
        cur_frame_sd = float('nan')
        if cur_frame_pred:
            cur_frame_mean = np.average(a=cur_frame_pred, weights=cur_frame_weight)
            cur_frame_sd = np.std(cur_frame_pred)
        cur_frame_pred_filtered = []
        cur_frame_weight_filtered = []
        for j_fun in range(len(cur_frame_pred)):
            if abs(cur_frame_pred[j_fun] - cur_frame_mean) < 25:
                cur_frame_pred_filtered.append(cur_frame_pred[j_fun])
                cur_frame_weight_filtered.append(cur_frame_weight[j_fun])
        if cur_frame_pred_filtered:
            timed_pred[i_fun] = np.average(cur_frame_pred_filtered, weights=cur_frame_weight_filtered)
            timed_std[i_fun] = cur_frame_sd
        else:
            if cur_frame_pred:
                cur_frame_mean = np.average(cur_frame_pred, weights=cur_frame_weight)
            timed_pred[i_fun] = cur_frame_mean
            timed_std[i_fun] = cur_frame_sd
    return timed_pred, timed_std


def parse_rssi(rssi):
    rssi = rssi[1:-1].split(', ')
    rssi = map(lambda x: float(x), rssi)
    return rssi

"""
Get files
"""
# insert the AP's MAC
fnames = glob.glob('measurements\\00140608*.csv')

ap_pred = []
ap_pred_times = []
ap_pred_frames = []
ap_sd_frames = []

"""
opt parameters
"""

"""
start opt
"""
for i in range(len(track.track_list)):
    ap_pred.append([])
    ap_pred_times.append([])
    ap_pred_frames.append([])
    ap_sd_frames.append([])
    for j in range(track.aps_raw.shape[0]):
        ap_pred[i].append([])
        ap_pred_times[i].append([])

for i in range(len(track.track_time)):
    predictions = np.zeros(())

    """
    get doa, for now only for the first track
    """
    for fname in [fnames[0]]:
        cur_ap = fname[fname.find('00140608'): (fname.find('00140608') + 12)]
        cur_ap_i = cal.aps_dict[cur_ap]

        def minimize_time_delay(dt):
            """
            Get experiment data
            """
            ap = pd.read_csv(fname, index_col=0, names=['MAC', 'Time', 'RSSIs', 'channel'])
            ap = ap.loc[ap.index == cfg_exp.mac]
            ap['Time'] += dt
            # ap = ap.loc[ap['Time'] >= track.track_time[i][0]]
            # ap = ap.loc[ap['Time'] <= track.track_time[i][-1]]
            ap_rssis = list(ap['RSSIs'])

            for k in range(len(ap_rssis)):
                ap_rssis[k] = parse_rssi(ap_rssis[k])

            del ap['RSSIs']
            if ap_rssis:
                ap_rssis = pd.DataFrame(ap_rssis, columns=['V0', 'V1', 'V2', 'V3', 'H0', 'H1', 'H2', 'H3'],
                                        index=ap.index)
                ap = pd.concat([ap, ap_rssis], axis=1)

                ap_rssis = np.array(ap_rssis)
                ap_arranged = fn.arrange_data(ap_rssis)

                # Arranging model data
                ap_max = np.apply_along_axis(np.max, 1, ap_rssis)

                # conditions
                not_sat_power = ap_max <= config.max_rssi
                not_low_power = ap_max >= config.min_rssi
                not_erroneous = ap_arranged[:, 8] > -10

                ap_arranged = pd.DataFrame(ap_arranged, columns=['V0', 'V1', 'V2', 'V3', 'H0', 'H1', 'H2',
                                                                                   'H3', 'VmH'], index=ap.index)
                ap_arranged_filtered = ap_arranged.loc[not_sat_power & not_low_power & not_erroneous]
                ap_arranged_filtered = np.array(ap_arranged_filtered)

                ap_arranged_filtered = fn.noise_filter(ap_arranged_filtered)

                # Filtered
                ap_pred_times[i][j] = np.array(ap['Time'].loc[not_sat_power & not_low_power & not_erroneous])
                # Predicting basic model result
                if ap_arranged_filtered.shape[0]:
                    ap_pred[i][j] = cal.rfc[cur_ap_i].predict(ap_arranged_filtered)
                    ap_pred[i][j] = ap_pred[i][j].reshape((ap_pred[i][j].shape[0], 1))

                # # plot predictions
                # plt.plot(ap_pred_times[i][j], ap_pred[i][j], 'go')
                # plt.show()

            """
            amalgamating predictions for each time frame
            """

            # not valid prediction are saved as 100
            ap_timed_pred, ap_timed_sd = timed_predictions(ap_pred[i][j], ap_pred_times[i][j],
                                                           track.track_time_int[i])
            # remove NaNs
            ap_timed_pred = fn.remove_nan(ap_timed_pred)
            # plt.plot(track.track_time_int[i], track.doa_true[0][:, j], 'r',
            # track.track_time_int[i], ap_timed_pred, 'go')
            # plt.show()

            rsme = np.sqrt(np.sum((track.doa_true[0][:, j].reshape(ap_timed_pred.shape) - ap_timed_pred) ** 2) /
                           ap_timed_pred.shape[0])
            # print 'AP: ', j, 'RSME is: ', rsme
            return rsme

        # bruteforce coarse optimizer
        print 'AP ', fname, '0 dt is', minimize_time_delay(0)
        goal = 60
        dt_best = None
        for dt_cur in dt_interval:
            cur_goal = minimize_time_delay(dt_cur)
            if cur_goal < goal:
                goal = cur_goal
                dt_best = dt_cur
        print 'AP ', fname, 'coarse dt is', dt_best, 'and RSME is', goal
