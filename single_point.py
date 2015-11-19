__author__ = 'YBeer'

import numpy as np
import config as cfg
import functions as fn
import matplotlib.pyplot as plt
import math

# The mobile station position [0:x, 1:y]
mobile = [40, 20]

# Calculate distance between APs to MS
# [0: x, 1: y, 2: direction, 3: distance]
n_ap = len(cfg.aps)
for i in range(n_ap):
    cfg.aps[i].append(math.sqrt((cfg.aps[i][0] - mobile[0]) ** 2 + (cfg.aps[i][1] - mobile[1]) ** 2))

# remove AP if it is saturated
aps = fn.remove_sat(cfg.aps)

# Number of access points
n_ap = len(aps)

# Find valid crossings
# [0: ap_1, 1: ap_2]
ap_cross = fn.find_crossings(aps)

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
    aps[i].append(aps[i][2] + cfg.aps[i][5])

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
ap_cross = fn.max_sd(ap_cross)

# Calculate position_error(crossing) in order to find optimal weights
# [0: ap_1, 1: ap_2, 2: cross_x, 3: cross_y, 4: dist_1, 5: dist_2, 6: angle_1, 7: angle_2, 8: SD_max, 9: SD_min]
# [10: pos_error]
for i in range(len(ap_cross)):
    ap_cross[i].append(math.sqrt((ap_cross[i][2] - mobile[0]) ** 2 +
                                 (ap_cross[i][3] - mobile[1]) ** 2))
for row in ap_cross:
    print row

# Calculate mean estimated point
print np.mean(fn.column_to_list(ap_cross, 2)), np.mean(fn.column_to_list(ap_cross, 3))

# Plot position and crossings
plt.plot(mobile[0], mobile[1], 'ro', fn.column_to_list(ap_cross, 2), fn.column_to_list(ap_cross, 3), 'bo')
plt.title('1 mobile station')
plt.xlim(0, cfg.x_max-1)
plt.ylim(0, cfg.y_max-1)
plt.show()

# # Plot position and crossings
# plt.plot(fn.column_to_list(ap_cross, 8), fn.column_to_list(ap_cross, 10), 'bo')
# plt.title('1 mobile station')
# plt.show()
