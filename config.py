__author__ = 'YBeer'

import numpy as np


# Minimum distance
r_sat_sqrd = 900

# standard error in degrees
std = 7.5

# Resolution of samples
res = 1

# number of repeats
N = 4

# angle limits
min_angle = -60
max_angle = 60

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

# ellipse samples
el_res = 40

# rssi limits
min_rssi = -70
max_rssi = -50
