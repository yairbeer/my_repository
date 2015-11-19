__author__ = 'YBeer'
import pandas as pd
import config_exp1308 as cfg_exp
import numpy as np
import time
import functions as fn
import matplotlib.pyplot as plt

# Placing and directing APs, direction is the AP main direction. phi = 0 -> y+, phi = 90 -> x+. like a compass
# [0: x, 1: y, 2: direction]
aps = np.array([[34.81163956, 32.10330261, 97],     # Elor's miniPC - maybe Avshalom's, left
                [34.81225465, 32.10278758, 13],   # Yair's miniPC, bottom
                [34.8128402, 32.10298426, 296],    # Laptop, right
                [34.81274286, 32.1036658, 221],   # Avshalom's miniPC - maybe Elor's, top
                [34.81235563, 32.1032189, 0]])     # trash

x_min = np.min(aps[:, 0])
y_min = np.min(aps[:, 1])
coor_to_m = 100000

aps[:, 0] = (aps[:, 0] - x_min) * coor_to_m
aps[:, 1] = (aps[:, 1] - y_min) * coor_to_m

valid_ants = [0, 1]


def parse_drone_time(date):
    pattern = '%Y-%m-%d--%H:%M:%S'
    input_int = date[:-2]
    input_float = float(date[-2:])
    input_int = time.mktime(time.strptime(input_int, pattern))
    return float(input_int + input_float)

parse_drone_time_vectorized = np.vectorize(parse_drone_time)

track = pd.DataFrame.from_csv('DJIFlightRecord_2015-08-13.csv')
track['latitude'] = (track['latitude'] - y_min) * coor_to_m
track['longitude'] = (track['longitude'] - x_min) * coor_to_m
track['datetime(utc)'] = '2015-08-13--16:' + track['datetime(utc)']

track['datetime(utc)'] = parse_drone_time_vectorized(track['datetime(utc)'])
t_0 = track['datetime(utc)'][0]
track['datetime(utc)'] = track['datetime(utc)'] - t_0

# Building time frames
time_start = 0
time_stop = int(np.max(track['datetime(utc)']))
time_frames = range(time_start, time_stop, cfg_exp.time_step)

track_position = fn.divide_to_slots(np.array(track[['datetime(utc)', 'longitude', 'latitude']]), time_frames)

# # show track on 2D
# plt.plot(aps[:4, 0], aps[:4, 1], 'ro', aps[4, 0], aps[4, 1], 'go',
#          track['longitude'], track['latitude'], 'r')
# plt.show()
