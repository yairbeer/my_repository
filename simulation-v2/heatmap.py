import numpy as np
import config as cfg
import functions as fn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import itertools

__author__ = 'YBeer'


class AccessPoint(object):
    def __init__(self, ap_params):
        self.x = ap_params[0]
        self.y = ap_params[1]
        self.heading = ap_params[2]


class Grid(object):
    def __init__(self, mini, maxi, resolution):
        x = y = np.arange(mini, maxi + 1, resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.resolution = resolution
        self.max = maxi
        self.min = mini


class SimulatedDoa(object):
    def __init__(self, ap_cls, x_cells, y_cells):
        self.ap = ap_cls
        dx = x_cells - self.ap.x
        dy = y_cells - self.ap.y
        g_angle = global_angle_calc(dx, dy)

        self.doa = g_angle - self.ap.heading
        self.doa = (self.doa + 180) % 360 - 180
        self.doa_noised = None
        self.doa_grid = None
        self.doa_grid_res = None

    def add_noise(self, noise_level):
        self.doa_noised = self.doa + np.random.normal(0, noise_level, self.doa.shape)

    def add_grid(self, grid):
        dx = grid.X - self.ap.x
        dy = grid.Y - self.ap.y
        g_angle = global_angle_calc(dx, dy)
        self.doa_grid = g_angle - self.ap.heading
        self.doa_grid = (self.doa_grid + 180) % 360 - 180
        self.doa_grid_res = grid.resolution


def global_angle_calc(deltax, deltay):
    g_angle = np.arctan(deltax / deltay) * 180 / np.pi
    # fix for negative dy
    g_angle += (deltay < 0) * 180
    # fix for negative dx positive dy
    g_angle += ((deltay > 0) * (deltax < 0)) * 360
    return g_angle


# def prob_density_calc_max(doa_array, sd):
#     x_est = np.ones(doa_array[0].doa.shape)
#     y_est = np.ones(doa_array[0].doa.shape)
#
#     def relative_angle_calc(angle1, angle2):
#         relative_angle1 = np.abs(angle1 - angle2)
#         relative_angle2 = 360 - np.abs(angle1 - angle2)
#         return np.minimum(relative_angle1, relative_angle2)
#
#     for i_in in range(doa_array[0].doa.shape[0]):
#         for j_in in range(doa_array[0].doa.shape[1]):
#             prob_mat = np.ones(x_est.shape)
#             for k_in in range(doa_array.shape[0]):
#                 prob_mat *= np.exp(-1 * relative_angle_calc(doa_array[k_in].doa,
#                                                             doa_array[k_in].doa_noised[i_in, j_in]) ** 2 / sd)
#             dens_pos = np.argmax(prob_mat)
#             y_est[i_in, j_in] = dens_pos / 101
#
#     return x_est, y_est


def prob_density_grid_max(doa_array, sd):
    x_est = np.ones(doa_array[0].doa.shape)
    y_est = np.ones(doa_array[0].doa.shape)
    grid_n_rows = doa_array[0].doa_grid.shape[0]

    def relative_angle_calc(angle1, angle2):
        relative_angle1 = np.abs(angle1 - angle2)
        relative_angle2 = 360 - np.abs(angle1 - angle2)
        return np.minimum(relative_angle1, relative_angle2)

    for i_in in range(doa_array[0].doa.shape[0]):
        for j_in in range(doa_array[0].doa.shape[1]):
            prob_mat = np.zeros(doa_array[0].doa_grid.shape)
            for k_in in range(doa_array.shape[0]):
                prob_mat += (-1 * relative_angle_calc(doa_array[k_in].doa_grid,
                                                      doa_array[k_in].doa_noised[i_in, j_in]) ** 2 / sd)
            dens_pos = np.argmax(prob_mat)
            x_est[i_in, j_in] = (dens_pos % grid_n_rows) *
            y_est[i_in, j_in] = dens_pos / grid_n_rows
    return x_est, y_est


def prob_density_calc_neldermead(doa_array, sd):
    def relative_angle_calc(angle1, angle2):
        relative_angle1 = np.abs(angle1 - angle2)
        relative_angle2 = 360 - np.abs(angle1 - angle2)
        return np.minimum(relative_angle1, relative_angle2)

    def local_density(location):
        prob = 1
        for k_in in range(doa_array.shape[0]):
            dx = location[0] - doa_array[k_in].ap.x
            dy = location[1] - doa_array[k_in].ap.y
            prob *= np.exp(-1 * relative_angle_calc(global_angle_calc(dx, dy),
                                                    doa_array[k_in].doa_noised[i_in, j_in]) ** 2 / sd)
        return -prob

    x_est = np.ones(doa_array[0].doa.shape)
    y_est = np.ones(doa_array[0].doa.shape)

    for i_in in range(doa_array[0].doa.shape[0]):
        for j_in in range(doa_array[0].doa.shape[1]):
            res = minimize(local_density, [50, 50], method='Nelder-Mead',
                           # options={'disp': True}
                           )
            x_est[i_in, j_in] = res.x[0]
            y_est[i_in, j_in] = res.x[1]

    return x_est, y_est


def mat_fun_calc(fun, matrix_list):
    answer = np.ones(matrix_list[0].shape)
    for i_in in range(answer.shape[0]):
        for j_in in range(answer.shape[1]):
            cur_pos_est = []
            for k_in in range(len(matrix_list)):
                cur_pos_est.append(matrix_list[k_in][i_in, j_in])
            answer[i_in, j_in] = fun(cur_pos_est)
    return answer


# Placing and directing APs, direction is the AP main direction. phi = 0 -> y+, phi = 90 -> x+. like a compass
# [0: x, 1: y, 2: direction]
aps_raw = np.array([[-1, -1, 45.0],
                   [-1, 101, 135.0],
                   [101, 101, 225.0],
                   [101, -1, 315.0]])

# Grid dimesions in meters, resolution of 1 meter [[x_min, x_max], [y_min, y_max]]
boundaries = [[0, 100], [0, 100]]

# Creating grid
x = y = np.arange(0, 100 + 1)
X, Y = np.meshgrid(x, y)

# init APs
print 'initalizing APs'
ap_arr = []
for i in range(aps_raw.shape[0]):
    ap_arr.append(AccessPoint(aps_raw[i, :]))
ap_arr = np.array(ap_arr)

# init DOA
print 'initalizing true DOAs'
noise = 7.5
doa_arr = []
for i in range(len(ap_arr)):
    doa_arr.append(SimulatedDoa(ap_arr[i], X, Y))
    doa_arr[-1].add_noise(noise)
doa_arr = np.array(doa_arr)

# init grid
print 'initalizing coarse grid calculation'
grid = Grid(0, 100, 10)
for i in range(len(ap_arr)):
    doa_arr[i].add_grid(grid)

# monte-carlo
n_rep = 10
x_est_err_list = y_est_err_list = []
print 'Starting Monte-Carlo simulation with %d repetitions' % n_rep
for repetition in range(n_rep):
    print 'repetition # %d' % repetition
    # add noise
    for i in range(len(ap_arr)):
        doa_arr[i].add_noise(noise)

    # find position
    # grid search
    x_est, y_est = prob_density_grid_max(doa_arr, noise)
    # x_est, y_est = prob_density_calc_neldermead(doa_arr, noise)

    # find

    x_est_err_list.append(x_est - X)
    y_est_err_list.append(y_est - Y)

# calculate functions
print 'finding standard deviation'
sd_x = mat_fun_calc(np.std, x_est_err_list)
sd_y = mat_fun_calc(np.std, y_est_err_list)
sd_tot = np.sqrt(sd_x ** 2 + sd_y ** 2)

# plot
origin = 'lower'
V = range(16)
plt.contourf(X, Y, sd_tot, V, cmap=plt.cm.hot, origin=origin)
plt.colorbar(orientation='vertical', shrink=0.8)
plt.show()
