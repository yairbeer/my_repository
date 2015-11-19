import numpy as np
import exps.exp_22102015.config_exp as cfg_exp
import exps.fn_exp as fn_exp
import functions as fn
import matplotlib.pyplot as plt
import exps.exp_22102015.track as track
import exps.exp_22102015.doa as doa
import itertools
import pandas as pd
from sklearn.linear_model import LinearRegression

__author__ = 'YBeer'

"""
Using minimum difference between meta positions
"""
estimate_pos = []
estimate_pos_max_probability = []
estimate_pos_kalman = []
point_prob_density = []

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

    # # Change NaN to last known position
    # pos = fn.remove_nan(pos)

    # Remove points from outside
    # pos = fn.remove_outside(pos)
    estimate_pos.append(pos)

    """
    calculate probability density of the point
    """
    point_prob = fn_exp.estimate_pos_prob(estimate_pos[i], track.aps, doa.ap_timed_kaplan[i])
    estimate_pos_max_probability.append(fn_exp.find_pos_prob(estimate_pos[i], track.aps, doa.ap_timed_kaplan[i]))
    # print np.hstack((estimate_pos[i], estimate_pos_max_probability[i]))
    point_prob_density.append(point_prob)

    # Kalman's filtering algorithm
    kalman_x_p = np.zeros(estimate_pos[i].shape)
    kalman_x_m = np.zeros(estimate_pos[i].shape)
    kalman_x_p[0, :] = estimate_pos[i][0, :]
    kalman_p_m = np.zeros(estimate_pos[i].shape)
    kalman_p_p = np.zeros(estimate_pos[i].shape)
    kalman_p_p[0, :] = np.ones((1, 2)) * cfg_exp.kalman_pos_q

    for j in range(1, estimate_pos[i].shape[0]):
        for k in range(estimate_pos[i].shape[1]):
            # if sample is in the relevant timeout
            if doa.timeout_matrix[j, k] <= cfg_exp.timeout:
                # if it's the first sample in the subset of frames in timeout
                if doa.timeout_matrix[j-1, k] > cfg_exp.timeout:
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
         track.track_time_int[0], estimate_pos_kalman[0][:, 0], 'k',
         track.track_time_int[0], estimate_pos_max_probability[0][:, 0], 'g')
plt.title('track 0 x(t) axis tracking')
plt.ylim((-5, 130))

plt.subplot(222)
plt.plot(track.track_time_int[0], track.track[0][:, 1], 'r', track.track_time_int[0], estimate_pos[0][:, 1], 'b',
         track.track_time_int[0], estimate_pos_kalman[0][:, 1], 'k',
         track.track_time_int[0], estimate_pos_max_probability[0][:, 1], 'g')
plt.title('track 0 y(t) axis tracking')
plt.ylim((-5, 100))

plt.subplot(223)
plt.plot(track.track_time_int[1], track.track[1][:, 0], 'r',  track.track_time_int[1], estimate_pos[1][:, 0], 'b',
         track.track_time_int[1], estimate_pos_kalman[1][:, 0], 'k',
         track.track_time_int[1], estimate_pos_max_probability[1][:, 0], 'g')
plt.title('track 1 x(t) axis tracking')
plt.ylim((-5, 130))

plt.subplot(224)
plt.plot(track.track_time_int[1], track.track[1][:, 1], 'r', track.track_time_int[1], estimate_pos[1][:, 1], 'b',
         track.track_time_int[1], estimate_pos_kalman[1][:, 1], 'k',
         track.track_time_int[1], estimate_pos_max_probability[1][:, 1], 'g')
plt.title('track 1 y(t) axis tracking')
plt.ylim((-5, 100))
plt.show()
#
# # 2D plot
# plt.figure(1)
# plt.subplot(211)
# plt.plot(track.track[0][:, 0], track.track[0][:, 1], 'r', estimate_pos[0][:, 0], estimate_pos[0][:, 1], 'b',
#          estimate_pos_kalman[0][:, 0], estimate_pos_kalman[0][:, 1], 'k')
# plt.title('track 0 pos tracking')
# plt.subplot(212)
# plt.plot(track.track[1][:, 0], track.track[1][:, 1], 'r', estimate_pos[1][:, 0], estimate_pos[1][:, 1], 'b',
#          estimate_pos_kalman[1][:, 0], estimate_pos_kalman[1][:, 1], 'k')
# plt.title('track 1 pos tracking')
# plt.show()
#
# pd.DataFrame(np.vstack((estimate_pos[0], estimate_pos[1]))).to_csv('estimated_pos_22102015.csv')
# pd.DataFrame(np.vstack((track.track[0], track.track[1]))).to_csv('pos_22102015.csv')

"""
RSME calculation
"""
print 'The positioning rsme is:'
for i in range(len(track.track)):
    rsme = (estimate_pos[i][:, 0] - track.track[i][:, 0]) ** 2 + (estimate_pos[i][:, 1] - track.track[i][:, 1]) ** 2
    rsme = rsme[~np.isnan(rsme)]
    rsme = np.sqrt(np.sum(rsme) / rsme.shape[0])
    print "The root mean square error position for track %f is %f" % (i, rsme)
    rsme = (estimate_pos_kalman[i][:, 0] - track.track[i][:, 0]) ** 2 + \
           (estimate_pos_kalman[i][:, 1] - track.track[i][:, 1]) ** 2
    rsme = rsme[~np.isnan(rsme)]
    rsme = np.sqrt(np.sum(rsme) / rsme.shape[0])
    print "The root mean square error Kalman filtered position for track %f is %f" % (i, rsme)

"""
RSE probability density function
"""
grid = np.arange(0, 52, 2)
fx = []
dxfx = []
rse = []
for i in range(len(track.track)):
    print 'track %d' % i
    rse_i = np.sqrt((estimate_pos_kalman[i][:, 0] - track.track[i][:, 0]) ** 2 +
                    (estimate_pos_kalman[i][:, 1] - track.track[i][:, 1]) ** 2)
    rse_i = rse_i[~np.isnan(rse_i)]
    total_measurements = rse_i.shape[0]
    fx_i = np.zeros((grid.shape)).astype('float')
    dxfx_i = np.zeros((grid.shape)).astype('float')
    for j in range(grid.shape[0]):
        fx_i[j] = np.sum((((grid[j] <= rse_i) * 1) * ((rse_i < (grid[j] + 2)) * 1)))
        dxfx_i[j] = np.sum(rse < (grid[j] + 2))
    fx_i /= total_measurements
    dxfx_i /= total_measurements
    rse.append(rse_i)
    fx.append(fx_i),
    dxfx.append(dxfx_i)

# plt.figure(1)
# plt.subplot(211)
# plt.plot(grid, fx[0], 'r',  grid, dxfx[0], 'b')
# plt.title('track 0 probability density / cumulative density')
# plt.grid(which='both')
#
# plt.subplot(212)
# plt.plot(grid, fx[1], 'r',  grid, dxfx[1], 'b')
# plt.title('track 1 probability density / cumulative density')
# plt.grid(which='both')
#
# plt.show()

"""
RSE VS # APs
"""
print doa.ap_timed_kaplan[i]
print ~np.isnan(doa.ap_timed_kaplan[i])
n_ants = []
for i in range(len(track.track)):
    n_ants_i = np.sum(~np.isnan(doa.ap_timed_kaplan[i]), axis=1)
    n_ants_i = n_ants_i[~np.isnan((estimate_pos_kalman[i][:, 0] - track.track[i][:, 0]) ** 2 +
                                  (estimate_pos_kalman[i][:, 1] - track.track[i][:, 1]) ** 2)]
    n_ants.append(n_ants_i)
# print n_ants

# plt.figure(1)
# plt.subplot(211)
# plt.plot(n_ants[0], rse[0], 'ro')
# plt.title('track 0 error VS. # of antennas')
# plt.grid(which='both')
#
# plt.subplot(212)
# plt.plot(n_ants[1], rse[1], 'ro')
# plt.title('track 1 error VS. # of antennas')
# plt.grid(which='both')
#
# plt.show()

n_ant_fit = []
for i in range(len(track.track)):
    n_ant_fit.append(LinearRegression())
    n_ant_fit[i].fit(n_ants[i].reshape(n_ants[i].shape[0], 1), rse[i])
    print 'track %d.' % i
    print '# antennas vs error coefficient is %f, the intercept for 0 atnennas is %f' % \
          (n_ant_fit[i].coef_, n_ant_fit[i].intercept_)
    print 'The goodness of fit is %f' % n_ant_fit[i].score(n_ants[i].reshape(n_ants[i].shape[0], 1), rse[i])

"""
probability density @ point VS error
"""

for i in range(len(track.track)):
    point_prob_density[i] = point_prob_density[i][(~np.isnan((estimate_pos_kalman[i][:, 0] -
                                                             track.track[i][:, 0]) ** 2 +
                                                             (estimate_pos_kalman[i][:, 1] -
                                                             track.track[i][:, 1]) ** 2))]

# plt.figure(1)
# plt.subplot(211)
# plt.plot(point_prob_density[0], rse[0], 'ro')
# plt.title('track 0 error VS. probability density of chosen point')
# plt.grid(which='both')
#
# plt.subplot(212)
# plt.plot(point_prob_density[1], rse[1], 'ro')
# plt.title('track 1 error VS. probability density of chosen point')
# plt.grid(which='both')
#
# plt.show()

prob_fit = []
for i in range(len(track.track)):
    prob_fit.append(LinearRegression())
    prob_fit[i].fit(point_prob_density[i].reshape(point_prob_density[i].shape[0], 1), rse[i])
    print 'track %d.' % i
    print 'probability vs error coefficient is %f, the intercept for 0 probability is %f' % \
          (prob_fit[i].coef_, prob_fit[i].intercept_)
    print 'The goodness of fit is %f' % prob_fit[i].score(point_prob_density[i].reshape(point_prob_density[i].shape[0],
                                                                                        1), rse[i])
"""
separate probability density by # of antennas
"""