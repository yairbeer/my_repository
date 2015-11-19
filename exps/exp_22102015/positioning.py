import numpy as np
import exps.exp_22102015.config_exp as cfg_exp
import exps.fn_exp as fn_exp
import functions as fn
import matplotlib.pyplot as plt
import exps.exp_22102015.track as track
import exps.exp_22102015.doa as doa
import itertools
import pandas as pd

__author__ = 'YBeer'

"""
Using minimum difference between meta positions
"""
estimate_pos = []
estimate_pos_kalman = []
for i in range(len(track.track_list)):
    # single repeat
    ap_direction = np.repeat(track.aps[:, 2].reshape((1, track.aps.shape[0])), doa.ap_timed_kaplan[i].shape[0], axis=0)

    # Converting to predicted global angle
    global_angle = ap_direction + doa.ap_timed_kaplan[i]

    # Converting predicted angles into slopes
    slopes = 1 / np.tan(np.radians(global_angle))
    
    # Finding y intercept
    y_intercept = track.aps[:, 1] * np.ones(slopes.shape) - slopes * track.aps[:, 0]

    pos = np.ones((global_angle.shape[0], 2)) * np.nan
    for j in range(slopes.shape[0]):
        valid_aps = fn_exp.find_crossing(global_angle[j, :])
        if len(valid_aps) > 1:
            couples = list(itertools.combinations(valid_aps, 2))
            prelim_pos = []
            weights = []
            for crossing in couples:
                # Calculating cross-points
                prelim_pos.append(fn_exp.crossings(slopes[j, :], y_intercept[j, :], crossing))
                # Calculate distance between exp.aps and cross point
                dist0, dist1 = fn_exp.crossings_dist(track.aps, crossing, prelim_pos[-1])

                # Find angles from both exp.aps
                angle0 = global_angle[j, crossing[0]]
                angle1 = global_angle[j, crossing[1]]

                # Calculate total SD
                sdmax, sdmin = fn_exp.add_sd(angle0, angle1, dist0, dist1)

                weights.append(fn.find_weights(sdmax))

            # Calculate position_error(crossing) in order to find optimal weights
            # [x, y]
            prelim_pos = np.array(prelim_pos)
            weights = np.array(weights)
            prelim_pos_x = prelim_pos[:, 0]
            prelim_pos_y = prelim_pos[:, 1]
            pos[j] = fn_exp.estimate_xy(prelim_pos_x, prelim_pos_y, weights)

    print pos[:30, :]
    # # Change NaN to last known position
    # pos = fn.remove_nan(pos)

    # Remove points from outside
    # pos = fn.remove_outside(pos)
    estimate_pos.append(pos)

    # Kalman's filtering algorithm
    kalman_x_p = np.zeros(estimate_pos[i].shape)
    kalman_x_m = np.zeros(estimate_pos[i].shape)
    kalman_x_p[0, :] = estimate_pos[i][0, :]
    kalman_p_m = np.zeros(estimate_pos[i].shape)
    kalman_p_p = np.zeros(estimate_pos[i].shape)
    kalman_p_p[0, :] = np.ones((1, 2)) * cfg_exp.kalman_q

    for j in range(1, estimate_pos[i].shape[0]):
        for k in range(estimate_pos[i].shape[1]):
            # if sample is in the relevant timeout
            if doa.ap_timeout[i][j, k] <= cfg_exp.timeout:
                # if it's the first sample in the subset of frames in timeout
                if doa.ap_timeout[i][j-1, k] > cfg_exp.timeout:
                    kalman_x_p[j, k] = estimate_pos[i][j, k]
                else:
                    # if the current sample exists
                    if not np.isnan(estimate_pos[i][j, k]):
                        kalman_p_m[j, k] = kalman_p_p[j-1, k] + cfg_exp.kalman_pos_q
                        kalman_k_j = kalman_p_m[j, k] / (kalman_p_m[j, k] + cfg_exp.kalman_pos_r)
                        kalman_p_p[j, k] = (1 - kalman_k_j) * kalman_p_m[j, k]
                        kalman_x_p[j, k] = kalman_x_p[j-1, k] + \
                                           kalman_k_j * (estimate_pos[i][j, k] - kalman_x_p[j-1, k])
                    else:
                        kalman_x_p[j, k] = kalman_x_p[j-1, k]
            else:
                kalman_x_p[j, k] = np.nan

    estimate_pos_kalman.append(kalman_x_p)

# 1D plot
plt.figure(1)
plt.subplot(221)
plt.plot(track.track_time_int[0], track.track[0][:, 0], 'r',  track.track_time_int[0], estimate_pos[0][:, 0], 'b',
         track.track_time_int[0], estimate_pos_kalman[0][:, 0], 'k')
plt.title('track 0 x(t) axis tracking')
plt.ylim((-5, 130))

plt.subplot(222)
plt.plot(track.track_time_int[0], track.track[0][:, 1], 'r', track.track_time_int[0], estimate_pos[0][:, 1], 'b',
         track.track_time_int[0], estimate_pos_kalman[0][:, 1], 'k')
plt.title('track 0 y(t) axis tracking')
plt.ylim((-5, 100))

plt.subplot(223)
plt.plot(track.track_time_int[1], track.track[1][:, 0], 'r',  track.track_time_int[1], estimate_pos[1][:, 0], 'b',
         track.track_time_int[1], estimate_pos_kalman[1][:, 0], 'k')
plt.title('track 1 x(t) axis tracking')
plt.ylim((-5, 130))

plt.subplot(224)
plt.plot(track.track_time_int[1], track.track[1][:, 1], 'r', track.track_time_int[1], estimate_pos[1][:, 1], 'b',
         track.track_time_int[1], estimate_pos_kalman[1][:, 1], 'k')
plt.title('track 1 y(t) axis tracking')
plt.ylim((-5, 100))
plt.show()

# 2D plot
plt.figure(1)
plt.subplot(211)
plt.plot(track.track[0][:, 0], track.track[0][:, 1], 'r', estimate_pos[0][:, 0], estimate_pos[0][:, 1], 'b',
         estimate_pos_kalman[0][:, 0], estimate_pos_kalman[0][:, 1], 'k')
plt.title('track 0 pos tracking')
plt.subplot(212)
plt.plot(track.track[1][:, 0], track.track[1][:, 1], 'r', estimate_pos[1][:, 0], estimate_pos[1][:, 1], 'b',
         estimate_pos_kalman[1][:, 0], estimate_pos_kalman[1][:, 1], 'k')
plt.title('track 1 pos tracking')
plt.show()

pd.DataFrame(np.vstack((estimate_pos[0], estimate_pos[1]))).to_csv('estimated_pos_22102015.csv')
pd.DataFrame(np.vstack((track.track[0], track.track[1]))).to_csv('pos_22102015.csv')
