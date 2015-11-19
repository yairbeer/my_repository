import numpy as np
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

__author__ = 'WiBeer'

meta_pos = [(67.304266735100981, 61.387056814579829),
            (114.66810784724017, 35.900820056350021),
            (64.002972049606342, 43.379951670537139)]
meta_pos = np.array(meta_pos)
covar_params = [[85.493596980626904, 49.9620344765721, 0.11975183791071622],
                [149.33614593545667, 268.47252310731949, 0.99343128630067423],
                [61.066361563665794, 42.551928286162358, 0.078463933061815233]]
covar_params = np.array(covar_params)
print meta_pos
print covar_params


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
    z_f = (cur_x - meta_x) ** 2 / s_xx ** 2 + \
          (cur_y - meta_y) ** 2 / s_yy ** 2 - \
          (2 * cor * (cur_x - meta_x) * (cur_y - meta_y)) / (s_xx * s_yy)
    prob_single = 1 / (2 * np.pi * s_xx * s_yy * np.sqrt(1 - cor ** 2)) * np.exp(-z_f / (2 * (1 - cor ** 2)))
    prob_single = np.log(prob_single)
    return prob_single

fig = plt.figure()
X = np.arange(20, 200)
Y = np.arange(20, 100)
X, Y = np.meshgrid(X, Y)
Z = np.ones(X.shape)
for i in range(X.shape[0]):
    for j in range(Y.shape[1]):
        Z[i, j] = prob_density([X[i, j], Y[i, j]])
print Z

plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.show()
