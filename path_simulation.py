__author__ = 'YBeer'

import numpy as np
import config as cfg
import functions as fn
import matplotlib.pyplot as plt
import track_simulation as track
import itertools

# Grid dimensions in meters, resolution of 1 meter [[x_min, x_max], [y_min, y_max]]
boundaries = [[0, 100], [0, 100]]

"""
simulate track
"""

aps_vector_x = np.ones((track.duration, 1)) * track.aps[:, 0].transpose()
aps_vector_y = np.ones((track.duration, 1)) * track.aps[:, 1].transpose()
aps_vector_dir = np.ones((track.duration, 1)) * track.aps[:, 2].transpose()

track_x = np.ones((track.duration, track.aps.shape[0])) * track.track[:, 0].reshape((track.duration, 1))
track_y = np.ones((track.duration, track.aps.shape[0])) * track.track[:, 1].reshape((track.duration, 1))

# Calculate distance between APs to MS
aps_dist = ((aps_vector_x - track_x) ** 2 + (aps_vector_y - track_y) ** 2)

# remove AP if it is saturated
aps_sat = aps_dist <= cfg.r_sat_sqrd

# Calculate simulated global angles from APs to the MS
global_angle_sim = fn.find_global_angle(aps_vector_x, aps_vector_y, track_x, track_y)

# Calculate simulated local angles from APs to the MS
local_angle_sim = global_angle_sim - aps_vector_dir

# Add random error
local_angle_sim = fn.add_error(local_angle_sim)

"""
Get position, remember to remove sat, when positioning
"""
# Converting to predicted global angle
global_angle = local_angle_sim + aps_vector_dir

# Converting predicted angles into slopes
slopes = 1 / np.tan(np.radians(global_angle))

# Finding y intercept
y_intercept = aps_vector_y - slopes * aps_vector_x

# pairs of APs for crossing points
couples = list(itertools.combinations(range(track.aps.shape[0]), 2))
n_couples = len(couples)

x_cross = np.zeros((track.duration, n_couples))
y_cross = np.zeros((track.duration, n_couples))

dist0 = np.zeros((track.duration, n_couples))
dist1 = np.zeros((track.duration, n_couples))

angle0 = np.zeros((track.duration, n_couples))
angle1 = np.zeros((track.duration, n_couples))

sdmax = np.zeros((track.duration, n_couples))
sdmin = np.zeros((track.duration, n_couples))

remove_sat = fn.crossings_sat(aps_sat, couples, track.duration)
remove_same_slope = fn.crossings_same_slopes(slopes, couples, track.duration)

remove_not_valid = remove_sat * remove_same_slope

i = 0
for crossing in couples:

    # Calculating cross-section
    x_cross[:, i], y_cross[:, i] = fn.crossings(slopes, y_intercept, crossing)

    # Calculate distance between APs and cross point
    dist0[:, i], dist1[:, i] = fn.crossings_dist(track.aps, crossing, x_cross[:, i], y_cross[:, i])

    # Find angles from both APs
    angle0[:, i] = local_angle_sim[:, crossing[0]]
    angle1[:, i] = local_angle_sim[:, crossing[1]]

    # Calculate total SD
    sdmax[:, i], sdmin[:, i] = fn.add_sd(angle0[:, i], angle1[:, i], dist0[:, i], dist1[:, i])

    i += 1

# Calculate position_error(crossing) in order to find optimal weights
weights = fn.find_weights(sdmax)

# remove not valid weights
weights_valid = weights * remove_not_valid

# Calculate grid
pos = fn.estimate_xy(x_cross, y_cross, weights, remove_not_valid)

# Remove points from outside, still doesn't work
pos = fn.remove_outside(pos, boundaries)

# Holt's filtering algorithm
holt = np.zeros(pos.shape)
holt[0, :] = pos[0, :]
holt_trend = np.zeros(pos.shape)

for i in range(1, pos.shape[0]):
    holt[i, :] = (1 - cfg.alpha) * (holt[i-1, :] + holt_trend[i-1, :]) + cfg.alpha * pos[i, :]
    holt_trend[i, :] = cfg.trend * (holt[i, :] - holt[i-1, :]) + (1 - cfg.trend) * holt_trend[i-1, :]

# Calculate RMSE
pos_error = np.sqrt((pos[:, 0] - track.track[:, 0]) ** 2 + (pos[:, 1] - track.track[:, 1]) ** 2)
rsme = np.sum(pos_error) / track.duration
print rsme

# t = range(track.duration)
# plt.figure(1)
# plt.subplot(211)
# plt.plot(t, track.track[:, 0], 'r', t, pos[:, 0], 'b', t, holt[:, 0], 'g')
# plt.title('x(t), position - red, prediction - blue, filtered prediction - green')
#
# plt.subplot(212)
# plt.plot(t, track.track[:, 1], 'r', t, pos[:, 1], 'b', t, holt[:, 1], 'g')
# plt.title('y(t), position - red, prediction - blue, filtered prediction - green')
# plt.show()
#
# # show track on 2D
# plt.plot(track.aps[:, 0], track.aps[:, 1], 'ro', track.track[:, 0], track.track[:, 1], 'r',
#          pos[:, 0], pos[:, 1], 'b', holt[:, 0], holt[:, 1], 'g')
# plt.title('2D position - red, 2D predictions - blue, 2D filtered predictions - green')
# plt.show()

# Eitan's graphs
t = range(track.duration)
plt.figure(1)
plt.subplot(211)
plt.plot(t, track.track[:, 0], 'r')
plt.title('x(t), position - red')

plt.subplot(212)
plt.plot(t, track.track[:, 1], 'r')
plt.title('y(t), position - red')
plt.show()

plt.figure(1)
plt.subplot(211)
plt.plot(t, track.track[:, 0], 'r', t, holt[:, 0], 'g')
plt.title('x(t), position - red, filtered prediction - green')

plt.subplot(212)
plt.plot(t, track.track[:, 1], 'r', t, holt[:, 1], 'g')
plt.title('y(t), position - red, filtered prediction - green')
plt.show()

# show track on 2D
plt.plot(track.aps[:, 0], track.aps[:, 1], 'ro', track.track[:, 0], track.track[:, 1], 'r')
plt.title('2D position - red')
plt.show()

plt.plot(track.aps[:, 0], track.aps[:, 1], 'ro', track.track[:, 0], track.track[:, 1], 'r', holt[:, 0], holt[:, 1], 'g')
plt.title('2D position - red, 2D filtered predictions - green')
plt.show()
