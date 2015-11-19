__author__ = 'YBeer'

import Constants
import numpy


def filter_sudden(predictions):
    # Input: 4 channel rssi array. Output: index of the 2nd power ant
    for i in range(1, len(predictions)):
        if abs(predictions[i] - predictions[i-1]) > 15:
            if predictions[i] - predictions[i-1] > 15:
                predictions[i] = predictions[i-1] + 15
            if predictions[i] - predictions[i-1] < 15:
                predictions[i] = predictions[i-1] - 15
    return predictions


def arrangeData(rssi):
    # Arranging init
    arranged_data = []

    for row in rssi:
        # split to V and H pol
        v_pol = row[0:4]
        h_pol = row[4:8]

        # get max RSSI in each polarization
        rssi_v = max(v_pol)
        rssi_h = max(h_pol)

        # get V maximum RSSI - H maximum RSSI
        v_minus_h_RSSI = rssi_v - rssi_h

        # normalize antennas
        for i in range(4):
            v_pol[i] = v_pol[i] - rssi_v
            h_pol[i] = h_pol[i] - rssi_h

        # build arrange row
        arranged_data.append(v_pol + h_pol + [v_minus_h_RSSI])

    return arranged_data

# change RSSI below the TH to the TH
def remove_noise_dev(rssi):
    for i in range(len(rssi)):
        for j in range(8):
            if rssi[i][j] <= Constants.ant_thresh:
                rssi[i][j] = Constants.ant_thresh

    return rssi

def calc_dist_v1(model_predictions, model_data, angle_data_base):
    model_dist_angle = []
    for i in range(len(model_predictions)):
        for j in range(len(angle_data_base)):
            dist_angle_i = 0
            if model_predictions[i] == angle_data_base[j][0]:
                for k in range(8):
                    dist_angle_i += (model_data[i][k] - angle_data_base[j][k+1])**2
                dist_angle_i = numpy.sqrt(dist_angle_i)
                model_dist_angle.append(dist_angle_i)
                break
    return model_dist_angle

def calc_weights_v1(model_dist_angle, model_prediction):
    model_predicted_weight = []
    for i in range(len(model_prediction)):
        model_predicted_SD = (model_dist_angle[i] * 2 + 20)
        model_predicted_weight.append(1/model_predicted_SD)
    return model_predicted_weight

def calc_dist_sd_v1(model_dist_angle, angle_error):
    # Build a SD for each N 0 to 15
    dist_err = []
    [dist_err.append([]) for i in range(15)]
    for i in range(len(model_dist_angle)):
        floor_val = int(numpy.floor(model_dist_angle[i]))
        if floor_val > 14:
            floor_val = 14
        dist_err[floor_val].append(angle_error[i])

    dist_std = []
    for i in range(15):
        dist_std.append(numpy.std(dist_err[i]))
    return dist_std

def log2lin_model(model_data):
    model_data_lin = []
    for i in range(len(model_data)):
        model_data_lin.append(map(lambda x: 10**(x/10), model_data[i][:8]) + [model_data[i][8]])
    return model_data_lin

def log2lin_data(angle_data):
    angle_data_lin = []
    for i in range(len(angle_data)):
        angle_data_lin.append([angle_data[i][0]] + map(lambda x: 10**(x/10), angle_data[i][1:]))
    return angle_data_lin

def calc_dist_lin(model_predictions, model_data, angle_data_base):
    model_dist_angle = []
    for i in range(len(model_predictions)):
        for j in range(len(angle_data_base)):
            dist_angle_i = 0
            if model_predictions[i] == angle_data_base[j][0]:
                for k in range(8):
                    dist_angle_i += (abs(model_data[i][k] - angle_data_base[j][k+1])) ** Constants.dist_metric
                dist_angle_i= (dist_angle_i / 8) **(1.0 / Constants.dist_metric)
                model_dist_angle.append(dist_angle_i)
                break
    return model_dist_angle

def calc_weights_lin(model_dist_angle, model_prediction):
    model_predicted_weight = []
    for i in range(len(model_dist_angle)):
        model_predicted_SD = (model_dist_angle[i] * 50 + 25)
        model_predicted_weight.append(1/(model_predicted_SD**2))
    return model_predicted_weight

def calc_dist_sd_lin(model_dist_angle, angle_error):
    # Build a SD for each 0.1 units of distance
    dist_err = []
    [dist_err.append([]) for i in range(int(Constants.dist_TH * Constants.dist_fac))]
    for i in range(len(model_dist_angle)):
        floor_val = int(numpy.floor(model_dist_angle[i] * Constants.dist_fac))
        if floor_val > float(Constants.dist_TH) * Constants.dist_fac - 1:
            floor_val = int(float(Constants.dist_TH) * Constants.dist_fac - 1)
        dist_err[floor_val].append(angle_error[i])
    print map(lambda x: len(x), dist_err)

    dist_std = []
    for i in range(int(Constants.dist_TH * Constants.dist_fac)):
        dist_std.append(numpy.std(dist_err[i]))
    return dist_std

def debug_portions(model_dist_angle, angle_error):
    # Build a SD for each 0.1 units of distance
    dist_err = []
    [dist_err.append([]) for i in range(int(Constants.dist_TH * Constants.dist_fac))]
    for i in range(len(model_dist_angle)):
        floor_val = int(numpy.floor(model_dist_angle[i] * Constants.dist_fac))
        if floor_val > float(Constants.dist_TH) * Constants.dist_fac - 1:
            floor_val = int(float(Constants.dist_TH) * Constants.dist_fac - 1)
        dist_err[floor_val].append(angle_error[i])
    return map(lambda x: len(x), dist_err)
