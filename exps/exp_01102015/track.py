import pandas as pd
import config_exp as exp
import numpy as np
import time
import functions as fn
import csv
import matplotlib.pyplot as plt
import glob

__author__ = 'YBeer'

"""
read APs parameters
"""
# Placing and directing APs, direction is the AP main direction. phi = 0 -> y+, phi = 90 -> x+. like a compass
aps_raw = pd.DataFrame.from_csv('ap_pos.csv')
heading = np.array(aps_raw)[-1, :2]
aps_raw = np.array(aps_raw)[:-1, :]

x_min = np.min(aps_raw[:, 0])
y_min = np.min(aps_raw[:, 1])
coor_to_m = 100000

aps_raw[:, 0] = (aps_raw[:, 0] - x_min) * coor_to_m
aps_raw[:, 1] = (aps_raw[:, 1] - y_min) * coor_to_m

heading[0] = (heading[0] - x_min) * coor_to_m
heading[1] = (heading[1] - y_min) * coor_to_m

aps = np.ones((aps_raw.shape[0], 3)) * 1.0

# aps: [x, y, direction]
aps[:, :2] = aps_raw[:, :2]
deltax = heading[0]-aps[:, 0]
deltay = heading[1]-aps[:, 1]
aps_global_angle = np.arctan(deltax / deltay) * 180 / np.pi
# fix for negative dy
neg_dy_fix = (deltay < 0) * 180
aps_global_angle += neg_dy_fix
# fix for negative dx positive dy
neg_dx_pos_dy_fix = ((deltay > 0) * (deltax < 0)) * 360
aps_global_angle += neg_dx_pos_dy_fix
aps[:, 2] = aps_global_angle + exp.heading


def parse_drone_time(date):
    pattern = '%m/%d/%Y %I:%M:%S %p'
    output = time.mktime(time.strptime(date, pattern))
    return output

parse_drone_time_vectorized = np.vectorize(parse_drone_time)

"""
read tracks
"""
track_list = glob.glob('FlightRecords\\drone_2015-10-01*')
track = []
track_time = []
for filename in track_list:
    track_raw = []
    with open(filename, 'rb') as csvfile:
        filereader = csv.reader(csvfile, delimiter=';', quotechar='\n')
        for row in filereader:
            track_raw.append(row)
    csvfile.close()
    track_raw = track_raw[1:]
    # for i in range(4):
    #     for j in range(len(track_raw[i])):
    #         print j, track_raw[i][j]
    track_raw = np.array(track_raw)
    track_date = track_raw[:, 24]
    track_time.append(np.array(track_raw[:, 29].astype('float')))
    track_raw = track_raw[:, 86:88].astype('float')

    track_new = np.ones((track_raw.shape[0], 2))
    track_new[:, 0] = (track_raw[:, 1] - x_min) * coor_to_m
    track_new[:, 1] = (track_raw[:, 0] - y_min) * coor_to_m

    track_time_new = parse_drone_time_vectorized(track_date)
    t_0 = track_time_new[0]

    track_df = pd.DataFrame(np.hstack((track_new, track_time_new.reshape((track_new.shape[0], 1)))),
                            columns=['x', 'y', 'time'])
    track_df = track_df.groupby(track_df['time']).mean()
    track_df = np.array(track_df)
    track.append(track_df[:, :2])

    # show track on 2D
    # start = []
    # plt.plot(aps[:, 0], aps[:, 1], 'ro', track_new[:, 0], track_new[:, 1], 'r', heading[0], heading[1], 'go')
    # plt.show()


"""
Get real DOA from each antenna
"""

# Converting to predicted global angle
doa_true = []
for i in range(len(track)):
    # aps: [x, y, direction]
    deltax = np.repeat(track[i][:, 0].reshape(track[i].shape[0], 1), aps_raw.shape[0], axis=1) - \
             np.repeat(aps[:, 0].reshape(1, aps_raw.shape[0]), track[i].shape[0], axis=0)
    deltay = np.repeat(track[i][:, 1].reshape(track[i].shape[0], 1), aps_raw.shape[0], axis=1) - \
             np.repeat(aps[:, 1].reshape(1, aps_raw.shape[0]), track[i].shape[0], axis=0)
    aps_global_angle = np.arctan(deltax / deltay) * 180 / np.pi

    ap_direction = np.repeat(aps[:, 2].reshape((1, aps.shape[0])), track[i].shape[0], axis=0)

    # fix for negative dy
    neg_dy_fix = (deltay < 0) * 180
    aps_global_angle += neg_dy_fix
    # fix for negative dx positive dy
    neg_dx_pos_dy_fix = ((deltay > 0) * (deltax < 0)) * 360
    aps_global_angle += neg_dx_pos_dy_fix
    aps_local_angle = aps_global_angle - ap_direction
    aps_local_angle = (aps_local_angle + 180) % 360 - 180
    # converting to local
    doa_true.append(aps_local_angle)
