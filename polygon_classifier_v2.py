__author__ = 'YBeer'

import numpy as np
import config as cfg
import functions as fn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import itertools
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

classifier = GradientBoostingClassifier()

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

# single repeat
pos_error_x = np.ones((grid.shape[0], cfg.N + 1))
pos_error_y = np.ones((grid.shape[0], cfg.N + 1))

for k in range(cfg.N):
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
    couples = list(itertools.combinations(range(aps.shape[0]), 2))
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
        dist0[:, i], dist1[:, i] = fn.crossings_dist(aps, crossing, x_cross[:, i], y_cross[:, i])
    
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

"""
Create polygon
"""
print 'create polygon'
# polygon limits include edges
p_lims = [[30, 45], [50, 70]]

# polygon dimensions
polygon = np.ones((grid.shape[0], 1))

polygon = ((p_lims[0][0] <= grid[:, 0]) * 1) * ((p_lims[0][1] >= grid[:, 0]) * 1) * \
          ((p_lims[1][0] <= grid[:, 1]) * 1) * ((p_lims[1][1] >= grid[:, 1]) * 1)

# # plot polygon
# heatmap_data = polygon
# X = range(boundaries[0][0], boundaries[0][1], cfg.res)
# Y = range(boundaries[1][0], boundaries[1][1], cfg.res)
# Z = heatmap_data.reshape([(boundaries[1][1] - boundaries[1][0] - 1)/cfg.res + 1,
#                           (boundaries[0][1] - boundaries[0][0]-1)/cfg.res + 1]).transpose()
#
# CS = plt.contourf(X, Y, Z)
# plt.colorbar(CS, orientation='vertical', shrink=0.8)
# plt.title('polygon')
# plt.show()

"""
estimating classifier, CV = 1/4
"""
cv_n = 4

# polygon
print 'classifing'
cv_n = 4
kf = KFold(polygon.shape[0], n_folds=cv_n, shuffle=True)
X = np.hstack((pos, local_angle_sim))
for train_index, test_index in kf:
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = polygon[train_index].ravel(), polygon[test_index].ravel()

    # train machine learning
    classifier.fit(X_train, y_train)

    # predict
    poly_pred = classifier.predict_proba(X_test)[:, 1]

    # Save estimated error
    print roc_auc_score(y_test, poly_pred)

# final fit
classifier.fit(X, polygon.ravel())
"""
find chance to be inside polygon for each point
"""

print 'create test dataset'
# Calculate local angles from APs to the MS
local_angle_sim = global_angle_sim - aps_vector_dir

# Add random error
local_angle_sim = fn.add_error(local_angle_sim)

# Converting to predicted global angle
global_angle = local_angle_sim + aps_vector_dir

# Converting predicted angles into slopes
slopes = 1 / np.tan(np.radians(global_angle))

# Finding y intercept
y_intercept = aps_vector_y - slopes * aps_vector_x

# pairs of APs for crossing points
couples = list(itertools.combinations(range(aps.shape[0]), 2))
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
    dist0[:, i], dist1[:, i] = fn.crossings_dist(aps, crossing, x_cross[:, i], y_cross[:, i])

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

# plot probability in polygon(x,y)
X = np.hstack((pos, local_angle_sim))
prob_poly = classifier.predict_proba(X)[:, 1]
heatmap_data = prob_poly
X = range(boundaries[0][0], boundaries[0][1], cfg.res)
Y = range(boundaries[1][0], boundaries[1][1], cfg.res)
Z = heatmap_data.reshape([(boundaries[1][1] - boundaries[1][0] - 1)/cfg.res + 1,
                          (boundaries[0][1] - boundaries[0][0]-1)/cfg.res + 1]).transpose()
# Z = signal.sepfir2d(Z, w, w)
# V = range(0, 101, 1)

CS = plt.contourf(X, Y, Z)
plt.colorbar(CS, orientation='vertical', shrink=0.8)
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((p_lims[0][0], p_lims[1][0]), p_lims[0][1] - p_lims[0][0], p_lims[1][1] - p_lims[1][0],
                                alpha=1, facecolor='none'))
plt.title('polygon')
plt.show()
