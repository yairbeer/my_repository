import glob
import pandas as pd
import numpy as np
import exps.exp_22102015.track as track
import config
import exps.exp_22102015.config_exp as cfg_exp
import functions as fn
import exps.exp_22102015.calibration_noisy as cal
import exps.fn_exp as fn_exp
import matplotlib.pyplot as plt

__author__ = 'YBeer'


class AccessPoint(object):
    def __init__(self, ap_stats, ap_heading_point, x_local, y_local):
        self.mac = ap_stats[3]
        self.x = ap_stats[0]
        self.y = ap_stats[1]

        # convert to local coordinates
        coor_to_m_cls = 100000
        self.x = (self.x - x_local) * coor_to_m_cls
        self.y = (self.y - y_local) * coor_to_m_cls
        heading_x = (ap_heading_point[0] - x_local) * coor_to_m_cls
        heading_y = (ap_heading_point[1] - y_local) * coor_to_m_cls

        # find manual heading
        dx = heading_x - self.x
        dy = heading_y - self.y
        ap_ga = global_angle_calc(dx, dy)
        self.heading = ap_ga


class TrackDoa(object):
    def __init__(self, ap_cls, my_track, my_track_time):
        dx = my_track[:, 0] - ap_cls.x
        dy = my_track[:, 1] - ap_cls.y
        global_angle = global_angle_calc(dx, dy)

        self.mac = ap_cls.mac
        self.doa = global_angle - ap_cls.heading
        self.doa = (self.doa + 180) % 360 - 180
        self.time = my_track_time


def parse_rssi(rssi):
    rssi = rssi[1:-1].split(', ')
    rssi = map(lambda x: float(x), rssi)
    return rssi


def calc_rsme_w_na(arr1, arr2):
    rsme = 0
    valid_samp = np.sum(1 - np.isnan(arr1) * 1)
    for j in range(arr1.shape[0]):
        if not (np.isnan(arr1[j]) or np.isnan(arr2[j])):
            rsme += (arr1[j] - arr2[j]) ** 2
    rsme = np.sqrt(rsme / valid_samp)
    return rsme


def global_angle_calc(deltax, deltay):
    global_angle = np.arctan(deltax / deltay) * 180 / np.pi
    # fix for negative dy
    global_angle += (deltay < 0) * 180
    # fix for negative dx positive dy
    global_angle += ((deltay > 0) * (deltax < 0)) * 360
    return global_angle

"""
read APs parameters
"""
# Placing and directing APs, direction is the AP main direction. phi = 0 -> y+, phi = 90 -> x+. like a compass
aps_raw = pd.DataFrame.from_csv('ap_pos.csv')
heading = np.array(aps_raw)[-1, :2]
aps_raw = np.array(aps_raw)[:-1, :]

x_min = np.min(aps_raw[:, 0])
y_min = np.min(aps_raw[:, 1])
coor_to_m = 100000

"""
init APs
"""
# Convert the track from GPS to local coordinates
for i in range(len(track.track)):
    track.track[i][:, 0] = (track.track[i][:, 0] - x_min) * coor_to_m
    track.track[i][:, 1] = (track.track[i][:, 1] - y_min) * coor_to_m

aps_doa = []
for i in range(len(track.track)):
    aps_doa.append([])
    for j in range(aps_raw.shape[0]):
        aps_doa[i].append(AccessPoint(aps_raw[j, :], heading, x_min, y_min))

"""
Get true DOA from each antenna
"""
# Converting to predicted global angle
doa_true = []
for i in range(len(track.track)):
    doa_true.append([])
    for j in range(aps_raw.shape[0]):
        doa_true[i].append(TrackDoa(aps_doa[i][j], track.track[i], track.track_time_int[i]))

"""
Get files
"""
# insert the AP's MAC
fnames = glob.glob('measurements\\00140608*.csv')

ap_pred = []
ap_pred_times = []
ap_pred_frames = []
ap_sd_frames = []
ap_timed_pred_raw = []
ap_timed_sd = []
ap_timed_holt = []
ap_timed_kaplan = []
ap_timeout = []

for i in range(len(track.track_list)):
    ap_pred.append([])
    ap_pred_times.append([])
    ap_pred_frames.append([])
    ap_sd_frames.append([])
    ap_timed_pred_raw.append([])
    ap_timed_holt.append([])
    ap_timed_kaplan.append([])
    ap_timed_sd.append([])
    ap_timeout.append([])
    for j in range(aps_raw.shape[0]):
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
    # Building time frames
    time_frames.append(np.array(range(int(track.track_time[i][0]),
                                      int(track.track_time[i][-1] + cfg_exp.time_step),
                                      cfg_exp.time_step)))

    # not valid prediction are saved as 100
    ap_timed_pred_raw[i], ap_timed_sd[i] = fn_exp.timed_predictions(ap_pred[i], ap_pred_times[i], track.track_time_int[i])

    timeout_matrix = np.zeros(ap_timed_pred_raw[i].shape).astype('int')
    timeout_matrix[0, :] = np.isnan(ap_timed_pred_raw[i][0, :]) * (cfg_exp.timeout + 1)
    for j in range(1, ap_timed_pred_raw[i].shape[0]):
        for k in range(ap_timed_pred_raw[i].shape[1]):
            if np.isnan(ap_timed_pred_raw[i][j, k]):
                timeout_matrix[j, k] = timeout_matrix[j-1, k] + 1
            else:
                timeout_matrix[j, k] = 0

    ap_timeout[i] = timeout_matrix

    # Kalman's filtering algorithm
    kalman_x_p = np.zeros(ap_timed_pred_raw[i].shape)
    kalman_x_m = np.zeros(ap_timed_pred_raw[i].shape)
    kalman_x_p[0, :] = ap_timed_pred_raw[i][0, :]
    kalman_p_m = np.zeros(ap_timed_pred_raw[i].shape)
    kalman_p_p = np.zeros(ap_timed_pred_raw[i].shape)
    kalman_p_p[0, :] = np.ones((1, 4)) * cfg_exp.kalman_q

    for j in range(1, ap_timed_pred_raw[i].shape[0]):
        for k in range(ap_timed_pred_raw[i].shape[1]):
            # if sample is in the relevant timeout
            if timeout_matrix[j, k] <= cfg_exp.timeout:
                # if it's the first sample in the subset of frames in timeout
                if timeout_matrix[j-1, k] > cfg_exp.timeout:
                    kalman_x_p[j, k] = ap_timed_pred_raw[i][j, k]
                else:
                    # if the current sample exists
                    if not np.isnan(ap_timed_pred_raw[i][j, k]):
                        kalman_p_m[j, k] = kalman_p_p[j-1, k] + cfg_exp.kalman_q
                        kalman_k_j = kalman_p_m[j, k] / (kalman_p_m[j, k] + cfg_exp.kalman_r)
                        kalman_p_p[j, k] = (1 - kalman_k_j) * kalman_p_m[j, k]
                        kalman_x_p[j, k] = kalman_x_p[j-1, k] + \
                                           kalman_k_j * (ap_timed_pred_raw[i][j, k] - kalman_x_p[j-1, k])
                    else:
                        kalman_x_p[j, k] = kalman_x_p[j-1, k]
            else:
                kalman_x_p[j, k] = np.nan

    ap_timed_kaplan[i] = kalman_x_p

    """
    Finding errors and plotting
    """

    # show DOAs on track
    plt.figure(1)
    plt.subplot(411)
    plt.plot(doa_true[i][0].time, doa_true[i][0].doa, 'r',
             track.track_time_int[i], ap_timed_pred_raw[i][:, 0], 'b',
             track.track_time_int[i], kalman_x_p[:, 0], 'k')
    plt.title(fnames[0][fnames[0].find('00140608'): (fnames[0].find('00140608') + 12)])

    plt.subplot(412)
    plt.plot(doa_true[i][1].time, doa_true[i][1].doa, 'r',
             track.track_time_int[i], ap_timed_pred_raw[i][:, 1], 'b',
             track.track_time_int[i], kalman_x_p[:, 1], 'k')
    plt.title(fnames[1][fnames[1].find('00140608'): (fnames[1].find('00140608') + 12)])

    plt.subplot(413)
    plt.plot(doa_true[i][2].time, doa_true[i][2].doa, 'r',
             track.track_time_int[i], ap_timed_pred_raw[i][:, 2], 'b',
             track.track_time_int[i], kalman_x_p[:, 2], 'k')
    plt.title(fnames[2][fnames[2].find('00140608'): (fnames[2].find('00140608') + 12)])

    plt.subplot(414)
    plt.plot(doa_true[i][3].time, doa_true[i][3].doa, 'r',
             track.track_time_int[i], ap_timed_pred_raw[i][:, 3], 'b',
             track.track_time_int[i], kalman_x_p[:, 3], 'k')
    plt.title(fnames[3][fnames[3].find('00140608'): (fnames[3].find('00140608') + 12)])
    plt.show()

RSME = np.ones((4))
# RSME over 1st track

# print 'raw DOA errors'
# for i in range(len(track.track_time_int)):
#     for j in range(ap_timed_kaplan[i].shape[1]):
#         RSME_tmp = calc_rsme_w_na(track.doa_true[i][:, j], ap_timed_pred_raw[i][:, j])
#         print 'track', i, 'AP: ', j, 'RSME is: ', RSME_tmp
#         RSME[j] = RSME_tmp
#     print 'track', i, np.mean(RSME)
#
# print 'DOA after kalman errors'
# for i in range(len(track.track_time_int)):
#     for j in range(ap_timed_kaplan[i].shape[1]):
#         RSME_tmp = calc_rsme_w_na(track.doa_true[i][:, j], ap_timed_kaplan[i][:, j])
#         print 'track', i, 'AP: ', j, 'RSME is: ', RSME_tmp
#         RSME[j] = RSME_tmp
#     print 'track', i, np.mean(RSME)

# pd.DataFrame(np.vstack((ap_timed_kaplan[0], ap_timed_kaplan[1]))).to_csv('doa_kalman_22102015.csv')
# pd.DataFrame(np.vstack((ap_timeout[0], ap_timeout[1]))).to_csv('timeout_kalman_22102015.csv')
