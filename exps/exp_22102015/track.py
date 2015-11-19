import pandas as pd
import config_exp as exp
import numpy as np
import time
import functions as fn
import csv
import matplotlib.pyplot as plt
import glob

__author__ = 'YBeer'


def parse_drone_time(date):
    pattern = '%m/%d/%Y %I:%M:%S %p'
    output = time.mktime(time.strptime(date, pattern))
    return output

parse_drone_time_vectorized = np.vectorize(parse_drone_time)

"""
read tracks
"""
track_list = glob.glob('FlightRecords\\drone_2015-10-22*')
track = []
track_time = []
track_time_int = []
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
    track_new[:, 0] = track_raw[:, 1]
    track_new[:, 1] = track_raw[:, 0]

    track_time_new = parse_drone_time_vectorized(track_date)
    t_0 = track_time_new[0]

    track_df = pd.DataFrame(np.hstack((track_new, track_time_new.reshape((track_new.shape[0], 1)))),
                            columns=['x', 'y', 'time'])
    track_df = track_df.groupby(track_df['time']).mean()
    track_time_int.append(np.array(track_df.index.values) * 1000)
    # fixing drone time error from iphone
    track_time_int[-1] -= (track_time_int[-1][0] - track_time[-1][0])
    track_df = np.array(track_df)
    track.append(track_df)

    # # show track on 2D
    # start = []
    # plt.plot(aps[:, 0], aps[:, 1], 'ro', track_new[:, 0], track_new[:, 1], 'r', heading[0], heading[1], 'go')
    # plt.show()
