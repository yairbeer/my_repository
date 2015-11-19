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

# delay per antenna
# [1, 3, 2, 0]
delay = [-40000, 0, -20000, 0]

# heading error per antenna
heading = [0, 0, -10, 10]

# calibration noise to add for the calibration
cal_noise = 2.0

# calibration samples for each angle
cal_rep = 200

# Kalman's constants
# measurement SD
kalman_r = 13.61074923
kalman_phi = 1
# info SD
kalman_q = 0.74267305
