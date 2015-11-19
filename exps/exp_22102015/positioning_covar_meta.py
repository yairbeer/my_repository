import numpy as np
import exps.exp_22102015.config_exp as cfg_exp
import exps.fn_exp as fn_exp
import functions as fn
import matplotlib.pyplot as plt
import exps.exp_22102015.track as track
import exps.exp_22102015.doa as doa
import itertools
import matplotlib.pyplot as plt

__author__ = 'YBeer'

"""
calculating probability density for each meta position. Then finding the density max.
"""
estimate_pos = []
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
    covars = []
    for j in range(slopes.shape[0]):
        valid_aps = fn_exp.find_crossing(global_angle[j, :])
        if len(valid_aps) > 1:
            couples = list(itertools.combinations(valid_aps, 2))
            prelim_pos = []
            weights = []
            ellipse = []
            for crossing in couples:
                # Calculating cross-points
                prelim_pos.append(fn_exp.crossings(slopes[j, :], y_intercept[j, :], crossing))

                # Calculate distance between exp.aps and cross point
                dist0, dist1 = fn_exp.crossings_dist(track.aps, crossing, prelim_pos[-1])

                # Find angles from both exp.aps
                angle0 = global_angle[j, crossing[0]]
                angle1 = global_angle[j, crossing[1]]

                # Calculate SD covariance
                cur_eigen_val, cur_eigen_angles = fn_exp.sd_eigen(angle0, angle1, dist0, dist1)
                cur_covars = fn_exp.sd_covar(cur_eigen_val, cur_eigen_angles)
                covars.append(cur_covars)
                ellipse.append(fn_exp.create_ellipse(prelim_pos[-1], cur_eigen_val, cur_eigen_angles))
            pos[j] = fn_exp.estimate_xy_covar(prelim_pos, covars)
            # print prelim_pos, covars
            # if len(valid_aps) == 3:
            #     plt.plot(prelim_pos[0][0], prelim_pos[0][1], 'ro', ellipse[0][:, 0], ellipse[0][:, 1], 'r--',
            #              prelim_pos[1][0], prelim_pos[1][1], 'go', ellipse[1][:, 0], ellipse[1][:, 1], 'g--',
            #              prelim_pos[2][0], prelim_pos[2][1], 'bo', ellipse[2][:, 0], ellipse[2][:, 1], 'b--',
            #              pos[j][0], pos[j][1], 'ko',
            #              track.track[i][j, 0], track.track[i][j, 1], 'k^',)
            #     plt.title(str([cur_eigen_val, cur_eigen_angles[0], cur_covars]))
            #     plt.show()
    # Change NaN to last known position
    pos = fn.remove_nan(pos)

    # Remove points from outside
    # pos = fn.remove_outside(pos)
    estimate_pos.append(pos)

    # Holt's filtering algorithm
    holt = np.zeros(pos.shape)
    holt[0, :] = pos[0, :]
    holt_trend = np.zeros(pos.shape)
    
    for j in range(1, pos.shape[0]):
        holt[j, :] = (1 - cfg_exp.alpha) * (holt[j-1, :] + holt_trend[j-1, :]) + cfg_exp.alpha * pos[j, :]
        holt_trend[j, :] = cfg_exp.trend * (holt[j, :] - holt[j-1, :]) + (1 - cfg_exp.trend) * holt_trend[j-1, :]

# RSME over 1st track
RSME = np.sqrt(np.sum((track.track[0][:, 0] - estimate_pos[0][:, 0]) ** 2 +
                      (track.track[0][:, 1] - estimate_pos[0][:, 1]) ** 2) / estimate_pos[0].shape[0])
print RSME

# 1D plot
plt.figure(1)
plt.subplot(221)
plt.plot(track.track_time_int[0], track.track[0][:, 0], 'r',  track.track_time_int[0], estimate_pos[0][:, 0], 'b')
plt.title('track 0 x(t) axis tracking')
plt.ylim((-5, 130))

plt.subplot(222)
plt.plot(track.track_time_int[0], track.track[0][:, 1], 'r', track.track_time_int[0], estimate_pos[0][:, 1], 'b')
plt.title('track 0 y(t) axis tracking')
plt.ylim((-5, 100))

plt.subplot(223)
plt.plot(track.track_time_int[1], track.track[1][:, 0], 'r',  track.track_time_int[1], estimate_pos[1][:, 0], 'b')
plt.title('track 1 x(t) axis tracking')
plt.ylim((-5, 130))

plt.subplot(224)
plt.plot(track.track_time_int[1], track.track[1][:, 1], 'r', track.track_time_int[1], estimate_pos[1][:, 1], 'b')
plt.title('track 1 y(t) axis tracking')
plt.ylim((-5, 100))
plt.show()

# 2D plot
plt.figure(1)
plt.subplot(211)
plt.plot(track.track[0][:, 0], track.track[0][:, 1], 'r', estimate_pos[0][:, 0], estimate_pos[0][:, 1], 'b')
plt.title('track 0 pos tracking')
plt.subplot(212)
plt.plot(track.track[1][:, 0], track.track[1][:, 1], 'r', estimate_pos[1][:, 0], estimate_pos[1][:, 1], 'b')
plt.title('track 1 pos tracking')
plt.show()
