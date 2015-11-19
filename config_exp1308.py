__author__ = 'YBeer'

import numpy as np

# Grid dimesions in meters, resolution of 1 meter
x_max = 101
y_max = 101

# RSSI limits
rssi_max = -48
rssi_min = -70

# Minimum distance
r_sat_sqrd = 1

# standard error in degrees
std = .5

# Resolution of samples
res = 1

# Creating grid
# [0: x, 1: y]
position = []
for i in range(0, x_max, res):
    for j in range(0, y_max, res):
        position.append([])
        position[-1].append(i)
        position[-1].append(j)

# number of repeats
N = 4

# angle limits
min_angle = -65
max_angle = 65

# remove crosspoint if dist is bigger than N-dist
N_dists = 1

# number of AP_remove for averaging
N_APremove = 5

# number of active crossings
N_cross = 9
cross_fac = 0.8

# timed path
alpha = 0.2
trend = 0.2

# filtered MAC address
mac = 'f02765408a10'

# time between different time windows in seconds
time_step = 1

# ellipse samples
el_res = 40