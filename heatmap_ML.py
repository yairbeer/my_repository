__author__ = 'YBeer'

import numpy as np
import config as cfg
import functions as fn
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingRegressor

# Placing and directing APs, direction is the AP main direction. phi = 0 -> y+, phi = 90 -> x+. like a compass
# [0: x, 1: y, 2: direction]
aps = np.array([[-1, -1, 45],
                [-1, 101, 135],
                [101, 101, 225],
                [101, -1, 315]])

# Grid dimesions in meters, resolution of 1 meter [[x_min, x_max], [y_min, y_max]]
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

aps_vector_x = np.ones((grid.shape[0], 1)) * aps[:, 0].transpose()
aps_vector_y = np.ones((grid.shape[0], 1)) * aps[:, 1].transpose()
aps_vector_dir = np.ones((grid.shape[0], 1)) * aps[:, 2].transpose()
track_x = np.ones((grid.shape[0], aps.shape[0])) * grid[:, 0].reshape((grid.shape[0], 1))
track_y = np.ones((grid.shape[0], aps.shape[0])) * grid[:, 1].reshape((grid.shape[0], 1))

"""
CV: 1/4
"""
cv_n = 4

# CV list
local_list = []

print 'simulate dataset'
for k in range(cv_n):
    """
    Simulate
    """

    # Calculate distance between APs to MS
    aps_dist = ((aps_vector_x - track_x) ** 2 + (aps_vector_y - track_y) ** 2)

    # remove AP if it is saturated
    aps_sat = aps_dist <= cfg.r_sat_sqrd

    # Calculate global angles from APs to the MS
    global_angle_sim = fn.find_global_angle(aps_vector_x, aps_vector_y, track_x, track_y)

    # Calculate local angles from APs to the MS
    local_angle_sim = global_angle_sim - aps_vector_dir
    # Add random error
    local_angle_sim = fn.add_error(local_angle_sim)
    local_list.append(local_angle_sim)


# single repeat
pos_error_x = np.ones((grid.shape[0], cv_n + 1))
pos_error_y = np.ones((grid.shape[0], cv_n + 1))

print 'create CV'

for i in range(cv_n):
    test_index = i
    train_index = []
    for j in range(cv_n):
        if j != i:
            train_index.append(j)
    """
    estimate position via random forest
    """
    # create train set
    train = np.vstack((local_list[train_index[0]], local_list[train_index[1]], local_list[train_index[2]]))
    # train machine learning
    rf_x = GradientBoostingRegressor()
    rf_y = GradientBoostingRegressor()
    rf_x.fit(train, np.repeat(grid[:, 0], cv_n - 1))
    rf_y.fit(train, np.repeat(grid[:, 1], cv_n - 1))

    # create test set
    test = local_list[test_index]

    # predict
    pos_x = rf_x.predict(test)
    pos_y = rf_y.predict(test)

    # Save estimated error
    pos_error_x[:, i] = (pos_x - grid[:, 0])
    pos_error_y[:, i] = (pos_y - grid[:, 1])

    print i

"""
Evaluating errors
"""
# mean across columns
pos_error_x_std = np.std(pos_error_x, axis=1)
pos_error_y_std = np.std(pos_error_y, axis=1)

pos_error = pos_error_x_std**2 + pos_error_y_std**2
vectorized_sqrt = np.vectorize(np.sqrt)
pos_error = vectorized_sqrt(pos_error)

print np.average(pos_error)
heatmap_data = pos_error

"""
Ploting the heatmap
"""
X = range(boundaries[0][0], boundaries[0][1], cfg.res)
Y = range(boundaries[1][0], boundaries[1][1], cfg.res)
Z = heatmap_data.reshape([(boundaries[1][1] - boundaries[1][0] - 1)/cfg.res + 1,
                          (boundaries[0][1] - boundaries[0][0]-1)/cfg.res + 1])
V = range(31)

# Plot heatmap
CS = plt.contourf(X, Y, Z, V)
plt.colorbar(CS, orientation='vertical', shrink=0.8)
plt.show()

# add another layer of regression between GBR and weighted average

# summary
# reference weighted average: 11.83
# default RF: avg STD = 16+
# n_estimators=30: avg STD = 14.1634376044
# add weighted average position: avg STD = 14.170320007, doesn't help

# Default GradientBoostingRegressor: avg STD = 11.259650598
