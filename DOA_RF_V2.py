__author__ = 'YBeer'

import csv
import numpy
import functions
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

dataset = []
dataset_angle = []

# read database from file
with open('new_dataset_v3_dev.csv', 'rb') as csvfile:
    experiment_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in experiment_reader:
        row = map(lambda x: float(x), row)
        dataset.append(row)

# read data results
with open('new_dataset_angle_v3_dev.csv', 'rb') as csvfile:
    result_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in result_reader:
        dataset_angle.append(float(row[0]))
dataset = functions.remove_noise_dev(dataset)

dataset_angle_rssi = []
# read angle average rssi from file
# in order to calculate distance
with open('angle_avg_rssi.csv', 'rb') as csvfile:
    experiment_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in experiment_reader:
        row = map(lambda x: float(x), row)
        dataset_angle_rssi.append(row)


# Fitting to RF
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1, min_samples_leaf=1,
                             max_features=3, criterion="gini", min_samples_split=2)
clf.fit(dataset, dataset_angle)

# creating predicted test set angles
test_prediction = clf.predict(dataset)
test_prediction = numpy.ndarray.tolist(test_prediction)

# Getting the model data
model_angle = []
model_time = []
model_data = []

file_name = 'doa_with_without_window'
# read model data
with open(file_name + '_data.csv', 'rb') as csvfile:
    experiment_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in experiment_reader:
        row = map(lambda x: float(x), row)
        model_data.append(row)

# read model time of packet arrival
with open(file_name + '_time.csv', 'rb') as csvfile:
    experiment_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in experiment_reader:
        model_time.append((float(row[0])))

# read model known angle
with open(file_name + '_angle.csv', 'rb') as csvfile:
    experiment_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in experiment_reader:
        model_angle.append(float(row[0]))

# Arranging model data
model_rssi = map(lambda x: max(x), model_data)
model_data = functions.arrangeData(model_data)
model_data = functions.remove_noise_dev(model_data)

# Predicting basic model result
model_prediction = clf.predict(model_data)
model_prediction = numpy.ndarray.tolist(model_prediction)

# convert log power to linear
model_data = functions.log2lin_model(model_data)
dataset_angle_rssi = functions.log2lin_data(dataset_angle_rssi)
# Find distance between prediction to measurement
model_dist_angle = functions.calc_dist_lin(model_prediction, model_data, dataset_angle_rssi)

# Convert RSSI distance to predicted standard deviation
model_predicted_weight = functions.calc_weights_lin(model_dist_angle, model_prediction)

# Filtering bad RSSIs
model_angle_filtered = []
model_time_filtered = []
model_data_filtered = []
model_prediction_filtered = []
model_weight_filtered = []

for i in range(len(model_angle)):
    # Bad power
    if Constants.min_rssi < model_rssi[i] < Constants.max_rssi:
        # Bad V_minus_H
        if model_data[i][8] > -10:
            # Bad RSSI distance
            if model_dist_angle[i] < Constants.dist_TH:
                model_data_filtered.append(model_data[i])
                model_time_filtered.append(model_time[i])
                model_angle_filtered.append(float(model_angle[i]))
                model_prediction_filtered.append(float(model_prediction[i]))
                model_weight_filtered.append(float(model_predicted_weight[i]))

# Building time frames
time_start = Constants.time_window
time_stop = int(max(model_time))
time_frames = range(time_start, time_stop, Constants.time_step)

# Removing empty time slots
time_frame_time = []
for i in range(len(time_frames)):
    cur_frame_prediction = []
    for j in range(len(model_time_filtered)):
        if time_frames[i] - Constants.time_window < model_time_filtered[j] < time_frames[i]:
            time_frame_time.append(time_frames[i])
            break

# Building average angle for time window
time_frame_angle = []
for i in range(len(time_frame_time)):
    cur_frame_angle = []
    for j in range(len(model_time_filtered)):
        if time_frame_time[i] - Constants.time_window < model_time_filtered[j] < time_frame_time[i]:
            cur_frame_angle.append(model_angle_filtered[j])
    time_frame_angle.append(float(sum(cur_frame_angle)) / len(cur_frame_angle))

# Building average prediction for time window with SD removal
time_frame_prediction_filtered = []
time_frame_prediction_sd = []
for i in range(len(time_frame_time)):
    cur_frame_prediction = []
    cur_frame_weight = []
    for j in range(len(model_time_filtered)):
        # Building subsets
        if time_frame_time[i] - Constants.time_window < model_time_filtered[j] < time_frame_time[i]:
            cur_frame_prediction.append(model_prediction_filtered[j])
            cur_frame_weight.append(model_weight_filtered[j])
    cur_frame_prediction_mean = numpy.average(a=cur_frame_prediction, weights=cur_frame_weight)
    cur_frame_prediction_sd = numpy.std(cur_frame_prediction)
    # filter predictions with far from the mean
    cur_frame_prediction_filtered = []
    cur_frame_weight_filtered = []
    for j in range(len(cur_frame_prediction)):
        if abs(cur_frame_prediction[j] - cur_frame_prediction_mean) < 25:
            cur_frame_prediction_filtered.append(cur_frame_prediction[j])
            cur_frame_weight_filtered.append(cur_frame_weight[j])
    if len(cur_frame_prediction_filtered) > 0:
        time_frame_prediction_filtered.append(numpy.average(cur_frame_prediction_filtered,
                                                            weights=cur_frame_weight_filtered))
    else:
        time_frame_prediction_filtered.append(cur_frame_prediction_mean)
    time_frame_prediction_sd.append(cur_frame_prediction_sd)

# Holt's filtering algorithm
holt_doa = [time_frame_prediction_filtered[0]]
holt_trend = [0]
for i in range(1, len(time_frame_time)):
    holt_doa.append((1 - Constants.alpha) * (holt_doa[-1] + holt_trend[-1]) +
                    Constants.alpha * time_frame_prediction_filtered[i])
    holt_trend.append(Constants.trend * (holt_doa[-1] - holt_doa[-2]) + (1 - Constants.trend) * holt_trend[-1])

time_frame_error = []
time_frame_error_holt = []
for i in range(len(time_frame_time)):
    time_frame_error.append(time_frame_angle[i] - time_frame_prediction_filtered[i])
    time_frame_error_holt.append(time_frame_angle[i] - holt_doa[i])

print numpy.std(time_frame_error)
print numpy.std(time_frame_error_holt)

# # Filtering false sudden DOA changes that aren't logical
# time_frame_prediction_filtered_sudden = list(time_frame_prediction_filtered)
# time_frame_prediction_filtered_sudden = functions.filter_sudden(time_frame_prediction_filtered_sudden)

# Plotting angle and prediction
x = range(len(model_angle_filtered))
plt.plot(x, model_prediction_filtered, 'ro', x, model_angle_filtered, 'bo')
plt.ylabel('time slots: Predictions(t) + Angle(t)')
plt.grid(True)
plt.show()

# Plotting angle and prediction in time frames
plt.plot(time_frame_time, time_frame_angle, 'ro', time_frame_time, time_frame_prediction_filtered, 'bo',
         time_frame_time, holt_doa, 'go')
plt.ylabel('time slots: Predictions(t) + Angle(t)')
plt.grid(True)
plt.show()

# Calculate errors
angle_error = []
for i in range(len(model_dist_angle)):
    angle_error.append(model_prediction[i] - model_angle[i])

# Build a SD for each N 0 to 15
dist_std = functions.calc_dist_sd_lin(model_dist_angle, angle_error)
print dist_std

# Plotting angle and SD
x_model = numpy.arange(1.0 / (2*Constants.dist_fac), float(len(dist_std)) /
                       Constants.dist_fac + 1.0 / (2*Constants.dist_fac), 1.0 / Constants.dist_fac)
plt.plot(x_model, dist_std, 'ro')
plt.ylabel('distance: checking SD')
plt.title(functions.debug_portions(model_dist_angle, angle_error))
plt.grid(True)
plt.show()

# # Build a SD for each N 0 to 10 for each antenna
# model_dist_angle_ant = []
# for i in range(len(model_prediction)):
#     model_dist_angle_ant.append([])
#     for j in range(len(dataset_angle_rssi)):
#         if model_prediction[i] == dataset_angle_rssi[j][0]:
#             for k in range(8):
#                 model_dist_angle_ant[i].append(abs(model_data[i][k] - dataset_angle_rssi[j][k+1]))
#
# dist_err_ant = []
# [dist_err_ant.append([]) for i in range(8)]
# for i in range(8):
#     [dist_err_ant[i].append([]) for j in range(int(Constants.dist_TH * Constants.dist_fac))]
#
# for i in range(len(model_dist_angle_ant)):
#     for j in range(8):
#         floor_val = int(numpy.floor(model_dist_angle_ant[i][j] * Constants.dist_fac))
#         if floor_val > float(Constants.dist_TH) * Constants.dist_fac - 1:
#             floor_val = int(float(Constants.dist_TH) * Constants.dist_fac - 1)
#         dist_err_ant[j][floor_val].append(angle_error[i])
#
# dist_std_ant = []
# for i in range(8):
#     dist_std_ant.append([])
#     for j in range(int(Constants.dist_TH * Constants.dist_fac)):
#         dist_std_ant[i].append(numpy.std(dist_err_ant[i][j]))
#
# dist_err_ant_size = []
# for i in range(len(dist_err_ant)):
#     dist_err_ant_size.append(map(lambda x: len(x), dist_err_ant[i]))
#
# for i in range(len(dist_std_ant)):
#     print i
#     print dist_std_ant[i]
#     print dist_err_ant_size[i]
#
# print range(int(Constants.dist_TH * Constants.dist_fac))
