import numpy as np

__author__ = 'YBeer'

# timed path
alpha = 0.2
trend = 0.2

# filtered MAC address
mac = 'f02765408a10'

# time between different time windows in seconds
time_step = 1000

# ellipse samples
el_res = 40

# margin in time frames for debugging
margin = 200

# DOA sd
doa_sd = np.radians(7.5)

# delay per antenna
# plot[3, 1, 2, 0]
delay = [-110000, 27000, -18000, 0]

# heading error per antenna
heading = [0, -8, -7, 5]

# calibration noise to add for the calibration
cal_noise = 2.0

# calibration samples for each angle
cal_rep = 200

# Kalman's constants
# measurement SD
kalman_r = 20
kalman_phi = 3
# info SD
kalman_q = 3

# positioning
kalman_pos_r = 10
kalman_pos_phi = 1
# info SD
kalman_pos_q = 1

# timeout in time_windows (seconds)
timeout = 5
