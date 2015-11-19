import numpy as np
import config as cfg
import functions as fn
import matplotlib.pyplot as plt
import track_simulation as track
import itertools
from sklearn.ensemble import GradientBoostingRegressor
from scipy import signal

__author__ = 'YBeer'

"""
Estimating position + ellipse of error over time
"""
# Grid dimensions in meters, resolution of 1 meter [[x_min, x_max], [y_min, y_max]]
boundaries = [[0, 100], [0, 100]]

# Creating grid
# [0: x, 1: y]
grid = []
for i in range(boundaries[0][0], boundaries[0][1], cfg.res):
    for j in range(boundaries[1][0], boundaries[1][1], cfg.res):
        grid.append([])
        grid[-1].append(i)
        grid[-1].append(j)

grid = np.array(grid)

aps_vector_x = np.ones((grid.shape[0], 1)) * track.aps[:, 0].transpose()
aps_vector_y = np.ones((grid.shape[0], 1)) * track.aps[:, 1].transpose()
aps_vector_dir = np.ones((grid.shape[0], 1)) * track.aps[:, 2].transpose()

grid_x = np.ones((grid.shape[0], track.aps.shape[0])) * grid[:, 0].reshape((grid.shape[0], 1))
grid_y = np.ones((grid.shape[0], track.aps.shape[0])) * grid[:, 1].reshape((grid.shape[0], 1))

# single repeat
pos_error_x = np.ones((grid.shape[0], cfg.N + 1))
pos_error_y = np.ones((grid.shape[0], cfg.N + 1))

"""
train covariance dataset
"""
print 'training covariance'
for k in range(cfg.N):
    """
    Simulate
    """

    # Calculate distance between APs to MS
    aps_dist = ((aps_vector_x - grid_x) ** 2 + (aps_vector_y - grid_y) ** 2)

    # remove AP if it is saturated
    aps_sat = aps_dist <= cfg.r_sat_sqrd

    # Calculate global angles from APs to the MS
    global_angle_sim = fn.find_global_angle(aps_vector_x, aps_vector_y, grid_x, grid_y)

    # Calculate local angles from APs to the MS
    local_angle_sim = global_angle_sim - aps_vector_dir

    # Add random error
    local_angle_sim = fn.add_error(local_angle_sim)

    """
    Get grid, remember to remove sat, when positioning
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

    x_cross = np.zeros((grid.shape[0], n_couples))
    y_cross = np.zeros((grid.shape[0], n_couples))

    dist0 = np.zeros((grid.shape[0], n_couples))
    dist1 = np.zeros((grid.shape[0], n_couples))

    angle0 = np.zeros((grid.shape[0], n_couples))
    angle1 = np.zeros((grid.shape[0], n_couples))

    sdmax = np.zeros((grid.shape[0], n_couples))
    sdmin = np.zeros((grid.shape[0], n_couples))

    remove_sat = fn.crossings_sat(aps_sat, couples, grid.shape[0])
    remove_same_slope = fn.crossings_same_slopes(slopes, couples, grid.shape[0])

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

    # # Remove points from outside, still doesn't work
    # pos = fn.remove_outside(pos, boundaries)

    # Save estimated error
    pos_error_x[:, k] = (pos[:, 0] - grid[:, 0])
    pos_error_y[:, k] = (pos[:, 1] - grid[:, 1])

    print k


# find covariance coefs
pos_error_xx = np.sum(pos_error_x * pos_error_x, axis=1) / cfg.N
pos_error_yy = np.sum(pos_error_y * pos_error_y, axis=1) / cfg.N
pos_error_xy = np.sum(pos_error_x * pos_error_y, axis=1) / cfg.N

"""
Ploting the heatmap, blur heatmaps - boxcar window filter
"""
w = signal.get_window('boxcar', 7)
w /= np.sum(w)

# xx
heatmap_data = pos_error_xx
Z = heatmap_data.reshape([(boundaries[1][1] - boundaries[1][0] - 1)/cfg.res + 1,
                          (boundaries[0][1] - boundaries[0][0]-1)/cfg.res + 1])
Z = signal.sepfir2d(Z, w, w)

pos_error_xx = Z.reshape((grid.shape[0]))

# yy
heatmap_data = pos_error_yy
Z = heatmap_data.reshape([(boundaries[1][1] - boundaries[1][0] - 1)/cfg.res + 1,
                          (boundaries[0][1] - boundaries[0][0]-1)/cfg.res + 1])
Z = signal.sepfir2d(Z, w, w)

pos_error_yy = Z.reshape((grid.shape[0]))

# plot xy
heatmap_data = pos_error_xy
Z = heatmap_data.reshape([(boundaries[1][1] - boundaries[1][0] - 1)/cfg.res + 1,
                          (boundaries[0][1] - boundaries[0][0]-1)/cfg.res + 1])
Z = signal.sepfir2d(Z, w, w)

pos_error_xy = Z.reshape((grid.shape[0]))

gbr_xx = GradientBoostingRegressor()
gbr_yy = GradientBoostingRegressor()
gbr_xy = GradientBoostingRegressor()

gbr_xx.fit(local_angle_sim, pos_error_xx)
gbr_yy.fit(local_angle_sim, pos_error_yy)
gbr_xy.fit(local_angle_sim, pos_error_xy)


"""
Simulate track
"""

print 'simulating track over time'

aps_vector_x = np.ones((track.track.shape[0], 1)) * track.aps[:, 0].transpose()
aps_vector_y = np.ones((track.track.shape[0], 1)) * track.aps[:, 1].transpose()
aps_vector_dir = np.ones((track.track.shape[0], 1)) * track.aps[:, 2].transpose()

track_x = track.track[:, 0].reshape((track.track.shape[0], 1))
track_y = track.track[:, 1].reshape((track.track.shape[0], 1))

# Calculate distance between APs to MS
aps_dist = ((aps_vector_x - track_x) ** 2 + (aps_vector_y - track_y) ** 2)

# remove AP if it is saturated
aps_sat = aps_dist <= cfg.r_sat_sqrd

# Calculate global angles from APs to the MS
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

print 'get position and covariance coefficients'
# Remove points from outside, still doesn't work
pos = fn.remove_outside(pos, boundaries)

# Holt's filtering algorithm
holt = np.zeros(pos.shape)
holt[0, :] = pos[0, :]
holt_trend = np.zeros(pos.shape)

for i in range(1, pos.shape[0]):
    holt[i, :] = (1 - cfg.alpha) * (holt[i-1, :] + holt_trend[i-1, :]) + cfg.alpha * pos[i, :]
    holt_trend[i, :] = cfg.trend * (holt[i, :] - holt[i-1, :]) + (1 - cfg.trend) * holt_trend[i-1, :]

sqrt_vectorize = np.vectorize(np.sqrt)
s_xx_pred = gbr_xx.predict(local_angle_sim)
s_xx_pred = sqrt_vectorize(s_xx_pred)
s_yy_pred = gbr_yy.predict(local_angle_sim)
s_yy_pred = sqrt_vectorize(s_yy_pred)
s_xy_pred = gbr_xy.predict(local_angle_sim)

"""
Getting error ellipse
"""
print 'get ellipse coefficients'
el_w, el_angle = fn.cov_coefs_to_ellipse(s_xx_pred, s_yy_pred, s_xy_pred)

"""
plotting ellipse
"""
# center of ellipse
x_center = holt[:, 0]
y_center = holt[:, 1]

# longer radius
a = el_w[:, 0]

# shorter radius
b = el_w[:, 1]

# rotation
rot_rad = el_angle / 180 * np.pi

# calculate of degrees
theta = range(0, 360 + 1, 360 / cfg.el_res)
# convert to radians
theta = map(lambda x: float(x) / 180 * np.pi, theta)
theta = np.array(theta)

for i in range(track.track.shape[0]):
    # find r for every angle
    r = np.sqrt(1 / (np.cos(theta) ** 2 / a[i] ** 2 + np.sin(theta) ** 2 / b[i] ** 2))

    # x and y coordinates of ellipse before rotations
    x_ellipse = r * np.cos(theta)
    y_ellipse = r * np.sin(theta)

    ellipse = np.hstack((x_ellipse.reshape((x_ellipse.shape[0], 1)), y_ellipse.reshape((y_ellipse.shape[0], 1))))

    # rotation operators
    rot_matrix = np.array([[np.cos(float(rot_rad[i])), np.sin(float(rot_rad[i]))],
                           [-(np.sin(float(rot_rad[i]))), np.cos(float(rot_rad[i]))]])

    # rotate ellipse
    ellipse = np.dot(ellipse, rot_matrix)

    # move to center
    ellipse = ellipse + np.hstack((x_center[i] * np.ones((ellipse.shape[0], 1)),
                                  y_center[i] * np.ones((ellipse.shape[0], 1))))

    # ploting
    plt.plot(ellipse[:, 0], ellipse[:, 1], 'r', track.track[i, 0], track.track[i, 1], 'ro', holt[i, 0], holt[i, 1], 'go')
    plt.xlim((-5, 105))
    plt.ylim((-5, 105))
    plt.show()

# #  Calculate RMSE
# pos_error = np.sqrt((pos[:, 0] - track.track[:, 0]) ** 2 + (pos[:, 1] - track.track[:, 1]) ** 2)
# rsme = np.sum(pos_error) / track.duration

