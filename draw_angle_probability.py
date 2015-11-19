import numpy as np
import matplotlib.pyplot as plt

sd = 8 * np.pi /180
aps = np.array([[-1, -1, 45],
                [-1, 101, 135],
                [101, 101, 225],
                [101, -1, 315]])

doa = np.array([+45, +27.5, -27.5, -45])

# single repeat
ap_direction = aps[:, 2]

# Converting to predicted global angle
global_doa = np.radians(ap_direction + doa)


def single_ap_prob(cur_pos):
    cur_x = cur_pos[0]
    cur_y = cur_pos[1]

    deltax = (cur_x - aps[i, 0]) * 1.0
    deltay = (cur_y - aps[i, 1]) * 1.0

    global_angle = np.arctan(deltax / deltay)
    if deltay < 0:
        global_angle += np.pi
    else:
        if deltax < 0:
            global_angle += 2 * np.pi

    # print 'dx ', deltax, 'dy', deltay, 'ga', global_angle
    relative_angle = np.abs(global_angle - global_doa[i])
    gaussian = np.exp(-1 * relative_angle ** 2 / sd)
    return gaussian

x = y = np.arange(0, 101)
X, Y = np.meshgrid(x, y)

origin = 'lower'

Zs = []
for i in range(aps.shape[0]):
    gaussian_prob = np.zeros(X.shape)
    for j in range(X.shape[0]):
        for k in range(X.shape[1]):
            gaussian_prob[j, k] = single_ap_prob([X[j, k], Y[j, k]])
    Zs.append(gaussian_prob)

Z = 1
for i in range(len(Zs)):
    Z *= Zs[i]

CS = plt.contourf(X, Y, Z, 10,
                  # [-1, -0.1, 0, 0.1],
                  # alpha=0.5,
                  cmap=plt.cm.hot,
                  origin=origin)
plt.show()

"""
Find 1st and 2nd moments: integral(x*f(x))
"""
dens_pos = np.argmax(Z)
dens_i = dens_pos / 101
dens_j = dens_pos % 101
print dens_pos, dens_i, dens_j, Z[dens_i, dens_j]
