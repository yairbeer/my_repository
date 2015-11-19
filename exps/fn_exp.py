import numpy as np
import exps.exp_01102015.config_exp as cfg_exp
import config
from scipy.optimize import minimize

__author__ = 'YBeer'

"""
General use
"""


def create_ellipse(center, axis, rotation_angle):
    # res is the resolution of sampling. better be a factor of 360
    res = 40

    # center of ellipse
    x_center = center[0]
    y_center = center[1]

    # longer radius
    a = np.sqrt(axis[0])
    # shorter radius
    b = np.sqrt(axis[1])

    # rotation
    rot = rotation_angle[0]
    rot = 90 - rot
    rot %= 360
    rot_rad = float(rot) / 180 * np.pi

    # calculate of degrees
    theta = range(0, 360 + 360 / res, 360 / res)
    # convert to radians
    theta = map(lambda x: float(x) / 180 * np.pi, theta)
    theta = np.array(theta)

    # find r for every angle
    r = np.sqrt(1 / (np.cos(theta) ** 2 / a ** 2 + np.sin(theta) ** 2 / b ** 2))

    # x and y coordinates of ellipse before rotations
    x_ellipse = r * np.cos(theta)
    y_ellipse = r * np.sin(theta)
    ellipse = np.hstack((x_ellipse.reshape((x_ellipse.shape[0], 1)), y_ellipse.reshape((y_ellipse.shape[0], 1))))

    # rotation operators
    rot_matrix = np.array([[np.cos(rot_rad), np.sin(rot_rad)], [-(np.sin(rot_rad)), np.cos(rot_rad)]])

    # rotate ellipse
    ellipse = np.dot(ellipse, rot_matrix)

    # move to center
    ellipse = ellipse + np.array([x_center, y_center])

    # ploting
    return ellipse

"""
Calibration
"""


def cal_sd(cal_data, cal_ant):
    ant = [config.min_angle]
    while ant[-1] < config.max_angle:
        ant.append(ant[-1] + 2.5)
    sd_ant = np.zeros((len(ant), 8))
    for i, angle in enumerate(ant):
        valid_angle = (angle == cal_ant)
        cur_sd_ant = cal_data.loc[valid_angle]
        cur_sd_ant = np.array(cur_sd_ant)[:, :8]
        sd_ant[i, :] = np.std(cur_sd_ant, axis=0)
    return sd_ant


def find_pattern(cal_data, cal_ant):
    ant = [config.min_angle]
    while ant[-1] < config.max_angle:
        ant.append(ant[-1] + 2.5)
    ant_pat = np.zeros((len(ant), 9))
    for i, angle in enumerate(ant):
        valid_angle = (angle == cal_ant)
        cur_ant = cal_data.loc[valid_angle]
        cur_ant = np.array(cur_ant)
        ant_pat[i, :] = np.mean(cur_ant, axis=0)
    max_ant = np.max(ant_pat, axis=1)
    ant_pat -= max_ant.reshape((len(ant), 1))
    return ant_pat

"""
DOA
"""


def timed_predictions(preds, pred_times, time_frames):
    timed_pred = np.ones((len(time_frames), len(preds)))
    timed_std = np.ones((len(time_frames), len(preds)))
    for i in range(len(time_frames)):
        for j in range(len(preds)):
            cur_frame_pred = []
            cur_frame_weight = []
            for k in range(len(preds[j])):
                # Building subsets
                if time_frames[i] < pred_times[j][k] < (time_frames[i] + cfg_exp.time_step):
                    cur_frame_pred.append(float(preds[j][k]))
                    cur_frame_weight.append(1)
            # initializing with a value that isn't a legal outcome
            cur_frame_mean = float('nan')
            cur_frame_sd = float('nan')
            if cur_frame_pred:
                cur_frame_mean = np.average(a=cur_frame_pred, weights=cur_frame_weight)
                cur_frame_sd = np.std(cur_frame_pred)
            cur_frame_pred_filtered = []
            cur_frame_weight_filtered = []
            for k in range(len(cur_frame_pred)):
                if abs(cur_frame_pred[k] - cur_frame_mean) < 25:
                    cur_frame_pred_filtered.append(cur_frame_pred[k])
                    cur_frame_weight_filtered.append(cur_frame_weight[k])
            if cur_frame_pred_filtered:
                timed_pred[i, j] = np.average(cur_frame_pred_filtered, weights=cur_frame_weight_filtered)
                timed_std[i, j] = cur_frame_sd
            else:
                if cur_frame_pred:
                    cur_frame_mean = np.average(cur_frame_pred, weights=cur_frame_weight)
                timed_pred[i, j] = cur_frame_mean
                timed_std[i, j] = cur_frame_sd
    return timed_pred, timed_std


"""
Positioning functions.
Same as simulation but works for each line separately,
because not all the antennas are active every time.
"""


def find_crossing(preds):
    ap_list = []
    for i in range(preds.shape[0]):
        if not np.isnan(preds[i]):
            ap_list.append(i)
    return ap_list


# Find crossing points
def crossings(slopes, y_intercept, cur_aps):
    x_tmp = (y_intercept[cur_aps[1]] - y_intercept[cur_aps[0]]) / (slopes[cur_aps[0]] - slopes[cur_aps[1]])

    y_tmp = x_tmp * slopes[cur_aps[0]] + y_intercept[cur_aps[0]]
    return x_tmp, y_tmp


# Calculate predicted distance
def crossings_dist(cur_aps, crossing, prelim_pos):
    dist0 = np.sqrt((cur_aps[crossing[0], 0] - prelim_pos[0]) ** 2 + (cur_aps[crossing[0], 1] - prelim_pos[1]) ** 2)
    dist1 = np.sqrt((cur_aps[crossing[1], 0] - prelim_pos[0]) ** 2 + (cur_aps[crossing[1], 1] - prelim_pos[1]) ** 2)
    return dist0, dist1


def add_sd(angle0, angle1, dist0, dist1):
    # Calculate angle between the access points
    angle0 %= 180
    angle1 %= 180

    angle_difference = np.min((np.abs(angle0 - angle1), np.abs(180 - np.abs(angle0 - angle1))))

    # Using that the SD is proportional to the distance from each AP.
    cur_sd = np.zeros((len(range(0, 180, 5)), 1))
    for i, x in enumerate(range(0, 180, 5)):
        cur_sd[i] = 1/(np.abs(np.cos(np.radians(x)) / dist0) +
                       np.abs(np.cos(np.radians((x + angle_difference))) / dist1))
    cur_sd_max = np.max(cur_sd)
    cur_sd_min = np.min(cur_sd)
    return cur_sd_max, cur_sd_min


def sd_eigen(angle0, angle1, dist0, dist1):
    # Using that the SD is proportional to the distance from each AP.
    cur_sd = np.zeros((len(range(0, 180, 1)), 1))
    for i, x in enumerate(range(0, 180, 1)):
        cur_sd[i] = 1/(np.abs(np.sin(np.radians(x - angle0)) / dist0) +
                       np.abs(np.sin(np.radians((x - angle1))) / dist1))
    sd_eigen_max = np.max(cur_sd) ** 2
    sd_max_angle = np.argmax(cur_sd)
    sd_eigen_min = cur_sd[sd_max_angle - len(range(0, 180, 1))/2] ** 2
    sd_max_angle = range(0, 180, 1)[sd_max_angle]
    sd_min_angle = (sd_max_angle - 90) % 180

    return np.array([sd_eigen_max, sd_eigen_min]), np.array([sd_max_angle, sd_min_angle])


def sd_covar(eigen_values, sd_eigen_angles):
    # change from north clockwise system to x anti clockwise system
    sd_eigen_angles = 90 - sd_eigen_angles
    sd_eigen_angles %= 360
    angle = np.radians(sd_eigen_angles)

    # eigenvector matrix
    covariance_matrix = np.array([[eigen_values[0], 0], [0, eigen_values[1]]])

    # construct eigen vectors matrix
    vec_matrix = np.array([[np.cos(angle[0]), np.sin(angle[0])], [np.cos(angle[1]), np.sin(angle[1])]])

    covariance_matrix = np.dot(np.dot(vec_matrix, covariance_matrix), np.linalg.inv(vec_matrix))
    covariance_params = [np.sqrt(covariance_matrix[0, 0]), np.sqrt(covariance_matrix[1, 1]),
                         (covariance_matrix[0, 1] /
                          (np.sqrt(covariance_matrix[0, 0]) * np.sqrt(covariance_matrix[1, 1])))]
    return covariance_params


def find_weights(sdmax):
    return 1 / sdmax


def estimate_xy(x_cross, y_cross, weights):
    if x_cross.shape[0] > 1:
        x = np.average(x_cross, weights=weights)
        y = np.average(y_cross, weights=weights)
    else:
        x = x_cross
        y = y_cross
    return [x, y]


def estimate_xy_covar(meta_pos, covar_params):
    def prob_density(cur_pos):
        log_prob = 0
        for k in range(len(meta_pos)):
            log_prob += prob_density_single(cur_pos, meta_pos[k], covar_params[k])
        # multiplying by -1 in order to find minimum
        return -1 * log_prob

    def prob_density_single(cur_pos, meta_pos_tmp, covar_par_tmp):
        cur_x = cur_pos[0]
        cur_y = cur_pos[1]
        meta_x = meta_pos_tmp[0]
        meta_y = meta_pos_tmp[1]
        s_xx = covar_par_tmp[0]
        s_yy = covar_par_tmp[1]
        cor = covar_par_tmp[2]
        z_f = (cur_x - meta_x) ** 2 / s_xx ** 2 + (cur_y - meta_y) ** 2 / s_yy ** 2 - \
              (2 * cor * (cur_x - meta_x) * (cur_y - meta_y)) / (s_xx * s_yy)
        prob_single = 1 / (2 * np.pi * s_xx * s_yy * np.sqrt(1 - cor ** 2)) * np.exp(-z_f / (2 * (1 - cor ** 2)))
        prob_single = np.log(prob_single)
        return prob_single

    if len(meta_pos) <= 1:
        return meta_pos[0]
    meta_pos_x0 = np.array(meta_pos)
    meta_pos_x0 = np.mean(meta_pos_x0, axis=0).tolist()
    opt_xy = minimize(prob_density, meta_pos_x0, method='nelder-mead')
    return opt_xy.x


def estimate_pos_prob(point, aps, doa_list):
    doa_sd = np.radians(4.5)
    # single repeat
    ap_direction = aps[:, 2]
    gaussian = np.ones((doa_list.shape[0], 1))
    for i_in in range(doa_list.shape[0]):
        cur_x = point[i_in, 0]
        cur_y = point[i_in, 1]
        # Converting to predicted global angle
        global_doa = np.radians(ap_direction + doa_list[i_in, :])
        prob = 1
        for j_in in range(doa_list.shape[1]):
            if ~np.isnan(global_doa[j_in]):
                deltax = (cur_x - aps[j_in, 0]) * 1.0
                deltay = (cur_y - aps[j_in, 1]) * 1.0

                global_angle = np.arctan(deltax / deltay)
                if deltay < 0:
                    global_angle += np.pi
                else:
                    if deltax < 0:
                        global_angle += 2 * np.pi

                relative_angle = np.min([np.abs(global_angle - global_doa[j_in]),
                                         np.abs(2 * np.pi - np.abs(global_angle - global_doa[j_in]))])

                gaussian[i_in] *= np.exp(-1 * relative_angle ** 2 / doa_sd)
    return gaussian


def find_pos_prob(coarse_point, aps, doa_list):
    def single_point_prob(cur_pos):
        cur_x = cur_pos[0]
        cur_y = cur_pos[1]

        gaussian = 1
        for j_in in range(doa_list.shape[1]):
            if ~np.isnan(pred_ga[j_in]):
                deltax = (cur_x - aps[j_in, 0]) * 1.0
                deltay = (cur_y - aps[j_in, 1]) * 1.0

                global_angle = np.arctan(deltax / deltay)
                global_angle += np.pi * (deltay < 0) * 1.0
                global_angle += 2 * np.pi * (deltay > 0) * (deltax < 0) * 1.0

                # print 'dx ', deltax, 'dy', deltay, 'ga', global_angle
                relative_angle = np.min([np.abs(global_angle - pred_ga[j_in]),
                                         np.abs(2 * np.pi - np.abs(global_angle - pred_ga[j_in]))])
                gaussian *= np.exp(-1 * relative_angle ** 2 / doa_sd)
        return -1 * gaussian

    doa_sd = np.radians(4.5)

    fine_point = np.ones(coarse_point.shape)
    for i_in in range(doa_list.shape[0]):
        # Converting to predicted global angle
        ap_direction = aps[:, 2]

        # Converting to predicted global angle
        pred_ga = np.radians(ap_direction + doa_list[i_in, :])
        pred_ga += (pred_ga < 0) * 2.0 * np.pi
        pred_ga += (pred_ga > (2 * np.pi)) * (-2.0) * np.pi

        x0 = coarse_point[i_in, :]

        # print x0, single_prob(x0)
        res = minimize(single_point_prob, x0, method='Nelder-Mead', options={'disp': False})
        fine_point[i_in, :] = res.x
    print fine_point
    return fine_point
