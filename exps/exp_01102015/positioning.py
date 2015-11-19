import numpy as np
import exps.exp_01102015.config_exp as cfg_exp
import exps.fn_exp as fn_exp
import functions as fn
import matplotlib.pyplot as plt
import track
import doa
import itertools

__author__ = 'YBeer'

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

# # 1D plot
# plt.figure(1)
# plt.subplot(321)
# plt.plot(doa.time_frames[0], track.track[0][:, 0], 'r',  doa.time_frames[0], estimate_pos[0][:, 0], 'b')
# plt.title('0 x(t) axis tracking')
# plt.ylim((-5, 100))
#
# plt.subplot(322)
# plt.plot(doa.time_frames[0], track.track[0][:, 1], 'r', doa.time_frames[0], estimate_pos[0][:, 1], 'b')
# plt.title('0 y(t) axis tracking')
# plt.ylim((-5, 100))
#
# plt.subplot(323)
# plt.plot(doa.time_frames[1], track.track[1][:, 0], 'r',  doa.time_frames[1], estimate_pos[1][:, 0], 'b')
# plt.title('1 x(t) axis tracking')
# plt.ylim((-5, 100))
#
# plt.subplot(324)
# plt.plot(doa.time_frames[1], track.track[1][:, 1], 'r', doa.time_frames[1], estimate_pos[1][:, 1], 'b')
# plt.title('1 y(t) axis tracking')
# plt.ylim((-5, 100))
#
# plt.subplot(325)
# plt.plot(doa.time_frames[2], track.track[2][:, 0], 'r',  doa.time_frames[2], estimate_pos[2][:, 0], 'b')
# plt.title('2 x(t) axis tracking')
# plt.ylim((-5, 100))
#
# plt.subplot(326)
# plt.plot(doa.time_frames[2], track.track[2][:, 1], 'r', doa.time_frames[2], estimate_pos[2][:, 1], 'b')
# plt.title('2 y(t) axis tracking')
# plt.ylim((-5, 100))
# plt.show()

# 2D plot
plt.plot(track.track[1][:, 0], track.track[1][:, 1], 'r', estimate_pos[1][:, 0], estimate_pos[1][:, 1], 'b')
plt.xlim((-5, 100))
plt.ylim((-5, 100))
plt.show()

# 1D plot
plt.figure(1)

plt.subplot(211)
plt.plot(doa.time_frames[1], track.track[1][:, 0], 'r',  doa.time_frames[1], estimate_pos[1][:, 0], 'b')
plt.title('1 x(t) axis tracking')
plt.ylim((-5, 100))

plt.subplot(212)
plt.plot(doa.time_frames[1], track.track[1][:, 1], 'r', doa.time_frames[1], estimate_pos[1][:, 1], 'b')
plt.title('1 y(t) axis tracking')
plt.ylim((-5, 100))

plt.show()
