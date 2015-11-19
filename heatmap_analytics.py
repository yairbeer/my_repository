__author__ = 'YBeer'

import numpy as np
import config as cfg
import functions as fn
import matplotlib.pyplot as plt
import math
import copy
import sklearn.linear_model as lm

# single repeat
pos_error = []
sd_max = []
sd_min = []
sd_tot = []

for k in range(cfg.N):
    for j in range(len(cfg.position)):
        # The mobile station position
        mobile = [cfg.position[j][0], cfg.position[j][1]]

        # Calculate distance between APs to MS
        # [0: x, 1: y, 2: direction, 3: distance]
        aps_dist = copy.deepcopy(cfg.aps)
        n_ap = len(aps_dist)
        for i in range(n_ap):
            aps_dist[i].append(math.sqrt((aps_dist[i][0] - mobile[0]) ** 2 + (aps_dist[i][1] - mobile[1]) ** 2))

        # remove AP if it is saturated
        aps = fn.remove_sat(aps_dist)

        # Find valid crossings
        # [0: ap_1, 1: ap_2]
        ap_cross = fn.find_crossings(aps)

        # Number of access points
        n_ap = len(aps)

        # Calculate global angles from APs to the MS
        # [0: x, 1: y, 2: direction, 3: distance, 4: global_angle]
        aps = fn.global_angle(aps, mobile)

        # Calculate local angles from APs to the MS
        # [0: x, 1: y, 2: direction, 3: distance, 4: global_angle, 5: local_angle]
        for i in range(n_ap):
            aps[i].append(aps[i][4] - aps[i][2])

        # Add random error
        # [0: x, 1: y, 2: direction, 3: distance, 4: global_angle, 5: predicted_local_angle]
        for i in range(n_ap):
            if cfg.min_angle < float(aps[i][5]) < cfg.max_angle:
                aps[i][5] += np.random.normal(loc=0, scale=cfg.std)
            else:
                aps[i][5] = np.random.uniform(cfg.min_angle, cfg.max_angle)

        # Converting to predicted global angle
        # [0: x, 1: y, 2: direction, 3: distance, 4: global_angle, 5: predicted_local_angle, 6: predicted_global_angle]
        for i in range(n_ap):
            aps[i].append(aps[i][2] + aps[i][5])

        # Converting predicted angles into slopes
        # [0: x, 1: y, 2: direction, 3: distance, 4: global_angle, 5: predicted_local_angle, 6: predicted_global_angle]
        # [7: predicted_slope]
        for i in range(n_ap):
            aps[i].append(1/math.tan(math.radians(aps[i][6])))

        # Finding y intercept
        # [0: x, 1: y, 2: direction, 3: distance, 4: global_angle, 5: predicted_local_angle, 6: predicted_global_angle]
        # [7: predicted_slope, 8: y_intercept]
        for i in range(n_ap):
            aps[i].append(aps[i][1] - aps[i][7] * aps[i][0])

        # Calculating cross-section
        # [0: ap_1, 1: ap_2, 2: cross_x, 3: cross_y]
        ap_cross = fn.crossings(ap_cross, aps)

        # Calculate distance between APs and cross point
        # [0: ap_1, 1: ap_2, 2: cross_x, 3: cross_y, 4: dist_1, 5: dist_2]
        ap_cross = fn.crossings_dist(ap_cross, aps)

        # Find angles from both APs
        # [0: ap_1, 1: ap_2, 2: cross_x, 3: cross_y, 4: dist_1, 5: dist_2, 6: angle_1, 7: angle_2]
        ap_cross = fn.crossings_angles(ap_cross, aps)

        # Calculate total SD
        # [0: ap_1, 1: ap_2, 2: cross_x, 3: cross_y, 4: dist_1, 5: dist_2, 6: angle_1, 7: angle_2, 8: SD_max, 9: SD_min]
        ap_cross = fn.add_sd(ap_cross)

        # Calculate position_error(crossing) in order to find optimal weights
        # [0: ap_1, 1: ap_2, 2: cross_x, 3: cross_y, 4: dist_1, 5: dist_2, 6: angle_1, 7: angle_2, 8: SD_max, 9: SD_min]
        # [10: pos_error]

        for i in range(len(ap_cross)):
            ap_cross[i].append(math.sqrt((ap_cross[i][2] - mobile[0]) ** 2 +
                                         (ap_cross[i][3] - mobile[1]) ** 2))
            # Save parameters
            pos_error.append(ap_cross[i][-1])
            sd_max.append(ap_cross[i][8])
            sd_min.append(ap_cross[i][9])
            sd_tot.append(math.sqrt(ap_cross[i][8]**2 + ap_cross[i][9]**2))

    print k

# Plot pos_error(sd_max)
plt.plot(sd_max, pos_error, 'ro')
plt.xlim((0, 1000))
plt.ylim((0, 200))
plt.show()

# Plot pos_error(sd_min)
plt.plot(sd_min, pos_error, 'ro')
plt.xlim((0, 1000))
plt.ylim((0, 2000))
plt.show()

# Plot pos_error(sd_tot)
plt.plot(sd_tot, pos_error, 'ro')
plt.xlim((0, 1000))
plt.ylim((0, 200))
plt.show()

# create linear regressions
sd_matrix_max = []
sd_matrix_min = []
sd_matrix_tot = []

for i in range(len(sd_max)):
    sd_matrix_max.append([sd_max[i]])
    sd_matrix_min.append([sd_min[i]])
    sd_matrix_tot.append([sd_tot[i]])

learn = lm.LinearRegression()
learn.fit(X=sd_matrix_max, y=pos_error)
print "pos_error(sd_max) r^2 = ", learn.score(X=sd_matrix_max, y=pos_error)

learn.fit(X=sd_matrix_min, y=pos_error)
print "pos_error(sd_min) r^2 = ", learn.score(X=sd_matrix_min, y=pos_error)

learn.fit(X=sd_matrix_tot, y=pos_error)
print "pos_error(sd_tot) r^2 = ", learn.score(X=sd_matrix_tot, y=pos_error)

# linear fir results
# pos_error(sd_max) r^2 = 0.753334464249
# pos_error(sd_min) r^2 = 0.998478434936
# pos_error(sd_tot) r^2 = 0.753334443653