__author__ = 'YBeer'

import numpy as np
import config as cfg
import config_exp1308 as cfg_exp

# # find coordinates to meters factor
# print (61.8 / np.sqrt((0.103667 - 0.1032189) ** 2 + (0.812750 - 0.81235563) ** 2))

"""
DOA
"""


def arrange_data(rssi):
    # Arranging init
    rssi = np.array(rssi)
    n_row = rssi.shape[0]
    arranged_data = np.zeros((n_row, 9))
    arranged_data[:, 0] = rssi[:, 0]
    for i in range(n_row):
        # split to V and H pol
        v_pol = rssi[i, :4]
        h_pol = rssi[i, 4:8]

        # get max RSSI in each polarization
        rssi_v = np.max(v_pol)
        rssi_h = np.max(h_pol)

        # get V maximum RSSI - H maximum RSSI
        v_minus_h_RSSI = rssi_v - rssi_h

        # normalize antennas
        v_pol -= rssi_v
        h_pol -= rssi_h
        row = np.hstack((v_pol, h_pol, np.array(v_minus_h_RSSI)))
        # build arrange row
        arranged_data[i, :] = row

    return arranged_data


def noise_filter(arr):
    # Arranging init
    noise = (arr[:, :8] < -10) * 1
    not_noise = 1 - noise
    not_noise *= arr[:, :8]
    noise *= -10
    arr = np.hstack((not_noise + noise, arr[:, [8]]))
    return arr


def filter_rows(arr, log_vector):
    n_row = arr.shape[0]
    n_filtered = np.sum(log_vector * 1)
    filtered_arr = np.zeros((n_filtered, arr.shape[1]))

    j = 0
    for i in range(n_row):
        if log_vector[i]:
            filtered_arr[j, :] = arr[i, :]
            j += 1

    return filtered_arr


def divide_to_slots(arr, time_slots, func=np.mean):
    timed_arr = []
    for i in range(len(time_slots)):
        # split attributes into different time frames
        cur_att = []
        for j in range(arr.shape[0]):
            if time_slots[i] <= arr[j, 0] < time_slots[i] + cfg_exp.time_step:
                cur_att.append(arr[j, 1:])
        # if time frame has attribute, add row in to the time slots
        if cur_att:
            cur_att = np.array(cur_att)
            row = [time_slots[i]] + list(np.apply_along_axis(func, 0, cur_att))
            timed_arr.append(row)
    timed_arr = np.array(timed_arr)
    return timed_arr


def timed_predictions(preds, time_frames):
    timed_pred = []
    timed_std = []
    row = []
    for i in range(len(time_frames)):
        cur_frame_pred = []
        cur_frame_weight = []
        for j in range(preds.shape[0]):
            # Building subsets
            if time_frames[i] < preds[j, 0] < time_frames[i] + cfg_exp.time_step:
                cur_frame_pred.append(preds[j, 1])
                cur_frame_weight.append(1)
        # initializing with a value that isn't a legal outcome
        cur_frame_mean = float('nan')
        cur_frame_sd = float('nan')
        if cur_frame_pred:
            cur_frame_mean = np.average(a=cur_frame_pred, weights=cur_frame_weight)
            cur_frame_sd = np.std(cur_frame_pred)
        cur_frame_pred_filtered = []
        cur_frame_weight_filtered = []
        for j in range(len(cur_frame_pred)):
            if abs(cur_frame_pred[j] - cur_frame_mean) < 25:
                cur_frame_pred_filtered.append(cur_frame_pred[j])
                cur_frame_weight_filtered.append(cur_frame_weight[j])
        if cur_frame_pred_filtered:
            timed_pred.append(np.average(cur_frame_pred_filtered, weights=cur_frame_weight_filtered))
            timed_std.append(cur_frame_sd)
        else:
            if cur_frame_pred:
                cur_frame_mean = np.average(cur_frame_pred, weights=cur_frame_weight)
            timed_pred.append(cur_frame_mean)
            timed_std.append(cur_frame_sd)

    timed_pred = np.array(timed_pred).reshape((len(time_frames)))
    timed_std = np.array(timed_std).reshape((len(time_frames)))
    return timed_pred, timed_std

"""
Simulating Position
"""


# Find global angle
def find_global_angle(aps_vector_x, aps_vector_y, track_x, track_y):
    deltax = track_x-aps_vector_x
    deltay = track_y-aps_vector_y
    global_angle = np.arctan(deltax / deltay) * 180 / np.pi
    # fix for negative dy
    neg_dy_fix = (deltay < 0) * 180
    global_angle += neg_dy_fix
    # fix for negative dx positive dy
    neg_dx_pos_dy_fix = ((deltay > 0) * (deltax < 0)) * 360
    global_angle += neg_dx_pos_dy_fix
    return global_angle


# Adding error to the measurement
def add_error(local_angle_sim):
    in_angle_error = (local_angle_sim < 60) * (local_angle_sim > -60) * 1.0
    out_angle_error = 1 - in_angle_error

    in_angle_random = np.random.normal(0, cfg.std, local_angle_sim.shape[0] * local_angle_sim.shape[1])\
        .reshape(local_angle_sim.shape)
    in_angle_error *= in_angle_random

    out_angle_random = np.random.uniform(cfg.min_angle, cfg.max_angle, local_angle_sim.shape[0] *
                                         local_angle_sim.shape[1]).reshape(local_angle_sim.shape)
    out_angle_error *= out_angle_random

    local_angle_sim += in_angle_error + out_angle_error
    return local_angle_sim

"""
Positioning
"""


# Find crossing points
def crossings(slopes, y_intercept, cur_aps):
    x_tmp = (y_intercept[:, cur_aps[1]] - y_intercept[:, cur_aps[0]]) / (slopes[:, cur_aps[0]] - slopes[:, cur_aps[1]])

    y_tmp = x_tmp * slopes[:, cur_aps[0]] + y_intercept[:, cur_aps[0]]
    return x_tmp, y_tmp


# Calculate predicted distance
def crossings_dist(cur_aps, crossing, x_cross, y_cross):
    dist0 = np.sqrt((cur_aps[crossing[0], 0] - x_cross) ** 2 + (cur_aps[crossing[0], 1] - y_cross) ** 2)
    dist1 = np.sqrt((cur_aps[crossing[1], 0] - x_cross) ** 2 + (cur_aps[crossing[1], 1] - y_cross) ** 2)
    return dist0, dist1


def add_sd(angle0, angle1, dist0, dist1):
    # Calculate angle between the access points
    angle0 %= 180
    angle0 = angle0.reshape((angle0.shape[0], 1))
    angle1 %= 180
    angle1 = angle1.reshape((angle0.shape[0], 1))

    angle_difference = np.hstack((np.abs(angle0 - angle1), np.abs(180 - np.abs(angle0 - angle1))))
    angle_difference = np.apply_along_axis(np.min, 1, angle_difference)

    # Using that the SD is proportional to the distance from each AP.
    cur_sd = np.zeros((angle_difference.shape[0], len(range(0, 180, 2))))
    for i, x in enumerate(range(0, 180, 2)):
        cur_sd[:, i] = 1/(np.abs(np.cos(np.radians(x)) / dist0) +
                          np.abs(np.cos(np.radians((x + angle_difference))) / dist1))
    cur_sd_max = np.apply_along_axis(np.max, 1, cur_sd)
    cur_sd_min = np.apply_along_axis(np.min, 1, cur_sd)
    return cur_sd_max, cur_sd_min


def find_weights(sdmax):
    return 1 / sdmax


def crossings_same_slopes(slopes, couples, length):
    same_slopes = np.zeros((length, len(couples)))
    i = 0
    for crossing in couples:
        # Calculating cross-section
        same_slopes[:, i] = (slopes[:, crossing[0]] == slopes[:, crossing[1]]) * 1
        i += 1
    not_same_slopes = 1 - same_slopes

    return not_same_slopes


def crossings_sat(aps_sat, couples, length):
    cross_sat = np.zeros((length, len(couples)))
    i = 0
    for crossing in couples:
        # Calculating cross-section
        cross_sat[:, i] = aps_sat[:, crossing[0]] * 1 + aps_sat[:, crossing[1]] * 1 - \
                          ((aps_sat[:, crossing[0]] * 1) * (aps_sat[:, crossing[1]] * 1))
        i += 1
    not_cross_sat = 1 - cross_sat

    return not_cross_sat


def remove_nan(arr):
    for col in range(arr.shape[1]):
        if np.isnan(arr[0, col]):
            arr[0, col] = 0
    for col in range(arr.shape[1]):
        for row in range(1, arr.shape[0]):
            if np.isnan(arr[row, col]):
                arr[row, col] = arr[row-1, col]
    return arr


def add_nans(attribute, valid):
    for i in range(attribute.shape[0]):
        for j in range(attribute.shape[1]):
            if not(valid[i, j]):
                attribute[i, j] = 'NaN'
    return attribute


def estimate_xy(x_cross, y_cross, weights, valid=False):
    x = np.ones((x_cross.shape[0], 1))
    y = np.ones((x_cross.shape[0], 1))
    if x_cross.shape[0] > 1:
        for i in range(x_cross.shape[0]):
            if valid is None:
                cur_weights = []
                cur_x = []
                cur_y = []
                for j in range(x_cross.shape[0]):
                    if valid[i, j]:
                        cur_weights.append(weights[i, j])
                        cur_x.append(x_cross[i, j])
                        cur_y.append(y_cross[i, j])
                x[i] = np.average(cur_x, weights=cur_weights)
                y[i] = np.average(cur_y, weights=cur_weights)
            else:
                x[i] = np.average(x_cross, weights=weights)
                y[i] = np.average(y_cross, weights=weights)
    else:
        x = x_cross
        y = y_cross
    return np.hstack((x.reshape((x_cross.shape[0], 1)), y.reshape((x_cross.shape[0], 1))))


def remove_outside(position, bound):
    position_x_too_low = (position[:, 0] < bound[0][0]) * (bound[0][0] - position[:, 0])
    position[:, 0] += position_x_too_low

    position_x_too_high = (position[:, 0] > bound[0][1]) * (bound[0][1] - position[:, 0])
    position[:, 0] += position_x_too_high

    position_y_too_low = (position[:, 1] < bound[1][0]) * (bound[1][0] - position[:, 1])
    position[:, 1] += position_y_too_low

    position_y_too_high = (position[:, 1] > bound[1][1]) * (bound[1][1] - position[:, 1])
    position[:, 1] += position_y_too_high

    return position

"""
Ellipse
"""


# convert covariance matrix to ellipse coefficients
def cov_coefs_to_ellipse(xx, yy, xy):
    w = np.zeros((xx.shape[0], 2))
    angle = np.zeros((xx.shape[0], 1))
    for i in range(xx.shape[0]):
        if not (np.isnan(xx[i]) or np.isnan(yy[i]) or np.isnan(xy[i])):
            cov = np.array([[xx[i], xy[i]], [xy[i], yy[i]]])

            # constant for deciding the ellipse size, how much time is correct
            confidence_const = np.e

            w_cur, v_cur = np.linalg.eig(np.array(cov))
            angle[i] = 180 / np.pi * np.arctan(v_cur[1, 0] / v_cur[0, 0])

            w_cur = np.abs(w_cur)
            w[i, :] = 2 * np.sqrt(confidence_const * w_cur)
    return w, angle
