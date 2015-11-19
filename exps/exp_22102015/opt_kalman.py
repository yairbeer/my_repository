import glob
import pandas as pd
import numpy as np
import track
import config
import exps.exp_22102015.config_exp as cfg_exp
import functions as fn
import calibration_noisy as cal
import exps.fn_exp as fn_exp
import matplotlib.pyplot as plt
import scipy.optimize as opt

__author__ = 'YBeer'


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
ap_timed_pred = []
ap_timed_sd = []
ap_timed_holt = []
ap_timed_kaplan = []

for i in range(len(track.track_list)):
    ap_pred.append([])
    ap_pred_times.append([])
    ap_pred_frames.append([])
    ap_sd_frames.append([])
    ap_timed_pred.append([])
    ap_timed_holt.append([])
    ap_timed_kaplan.append([])
    ap_timed_sd.append([])
    for j in range(track.aps_raw.shape[0]):
        ap_pred[i].append([])
        ap_pred_times[i].append([])

time_frames = []
for i in range(len(track.track_time)):
    predictions = np.zeros(())

    """
    get doa, for now only for the first track
    """
    for j, fname in enumerate(fnames):
        cur_ap = fname[fname.find('00140608'): (fname.find('00140608') + 12)]
        cur_ap_i = cal.aps_dict[cur_ap]
        """
        Get experiment data
        """
        ap = pd.read_csv(fname, index_col=0, names=['MAC', 'Time', 'RSSIs', 'channel'])
        ap = ap.loc[ap.index == cfg_exp.mac]
        ap['Time'] += cfg_exp.delay[cur_ap_i]
        # ap = ap.loc[ap['Time'] >= track.track_time[i][0]]
        # ap = ap.loc[ap['Time'] <= track.track_time[i][-1]]
        ap_rssis = list(ap['RSSIs'])

        for k in range(len(ap_rssis)):
            ap_rssis[k] = parse_rssi(ap_rssis[k])

        del ap['RSSIs']
        if ap_rssis:
            ap_rssis = pd.DataFrame(ap_rssis, columns=['V0', 'V1', 'V2', 'V3', 'H0', 'H1', 'H2', 'H3'], index=ap.index)
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
    ap_timed_pred[i], ap_timed_sd[i] = fn_exp.timed_predictions(ap_pred[i], ap_pred_times[i], track.track_time_int[i])

    # remove NaNs
    ap_timed_pred[i] = fn.remove_nan(ap_timed_pred[i])

    # Kalman's filtering algorithm
    kalman_x_p = np.zeros(ap_timed_pred[i].shape)
    kalman_x_p[0, :] = ap_timed_pred[i][0, :]
    kalman_p_m = np.zeros(ap_timed_pred[i].shape)
    kalman_p_p = np.zeros(ap_timed_pred[i].shape)
    kalman_p_p[0, :] = np.ones((1, 4)) * cfg_exp.kalman_q

    for j in range(1, ap_timed_pred[i].shape[0]):
        kalman_p_m[j, :] = kalman_p_p[j-1, :] + cfg_exp.kalman_q
        kalman_k_j = kalman_p_m[j, :] / (kalman_p_m[j, :] + cfg_exp.kalman_r)
        kalman_p_p[j, :] = (1 - kalman_k_j) * kalman_p_m[j, :]
        kalman_x_p[j, :] = kalman_x_p[j-1, :] + kalman_k_j * (ap_timed_pred[i][j, :] - kalman_x_p[j-1, :])

    ap_timed_kaplan[i] = kalman_x_p


def kalman_opt(kalman_params):

    # Kalman's filtering algorithm
    kalman_x_p = np.zeros(ap_timed_pred[0].shape)
    kalman_x_p[0, :] = ap_timed_pred[0][0, :]
    kalman_p_m = np.zeros(ap_timed_pred[0].shape)
    kalman_p_p = np.zeros(ap_timed_pred[0].shape)
    kalman_p_p[0, :] = np.ones((1, 4)) * kalman_params[1]

    for i in range(1, ap_timed_pred[0].shape[0]):
        kalman_p_m[i, :] = kalman_p_p[i-1, :] + kalman_params[1]
        kalman_k_i = kalman_p_m[i, :] / (kalman_p_m[i, :] + kalman_params[0])
        kalman_p_p[i, :] = (1 - kalman_k_i) * kalman_p_m[i, :]
        kalman_x_p[i, :] = kalman_x_p[i-1, :] + kalman_k_i * (ap_timed_pred[0][i, :] - kalman_x_p[i-1, :])

    RSME = np.ones((4))
    for i in range(kalman_p_m.shape[1]):
        RSME_tmp = np.sqrt(np.sum((track.doa_true[0][:, i] - kalman_x_p[:, i]) ** 2) /
                           kalman_x_p.shape[0])
        RSME[i] = RSME_tmp
    return np.mean(RSME)

print kalman_opt([8.0, 4.0])
print kalman_opt([6.0, 4.0])

opt_results = opt.minimize(kalman_opt, x0=[8.0, 4.0], method='Nelder-Mead', options={'disp': True})
print opt_results.x

print kalman_opt(opt_results.x)
