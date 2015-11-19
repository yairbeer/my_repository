import numpy as np
from scipy.optimize import minimize


def single_prob(cur_pos):
    cur_x = cur_pos[0]
    cur_y = cur_pos[1]

    gaussian = 1
    for i in range(aps.shape[0]):
        deltax = (cur_x - aps[i, 0]) * 1.0
        deltay = (cur_y - aps[i, 1]) * 1.0

        global_angle = np.arctan(deltax / deltay)
        global_angle += np.pi * (deltay < 0) * 1.0
        global_angle += 2 * np.pi * (deltay > 0) * (deltax < 0) * 1.0

        # print 'dx ', deltax, 'dy', deltay, 'ga', global_angle
        relative_angle = global_angle - sim_ga_noised[i]
        gaussian *= np.exp(-1 * relative_angle ** 2 / sd)
    return -1 * gaussian

sd = 7.5 * np.pi / 180

aps = np.array([[-1, -1, np.pi / 4],
                [-1, 101, 3 * np.pi / 4],
                [101, 101, 5 * np.pi / 4],
                [101, -1, 7 * np.pi / 4]])
"""
simulate position
"""
x = y = np.arange(0, 101)
X, Y = np.meshgrid(x, y)

sim_pos = [50, 100]

deltax = deltay = np.ones((X.shape[0], X.shape[1]))
# Calculate global angles from APs to the MS
deltax = (sim_pos[0] - aps[:, 0]) * 1.0
deltay = (sim_pos[1] - aps[:, 1]) * 1.0

sim_ga = np.arctan(deltax / deltay)
sim_ga += np.pi * (deltay < 0) * 1.0
sim_ga += 2 * np.pi * (deltay > 0) * (deltax < 0) * 1.0

"""
monte carlo simulation
"""
repetitions = 1
error_x = np.zeros((repetitions))
error_y = np.zeros((repetitions))

# Calculate local angles from APs to the MS
sim_la = [np.radians(-27.5), np.radians(-45), np.radians(45), np.radians(27.5)]
for i_rep in range(repetitions):
    # single repeat
    ap_direction = aps[:, 2]

    # Converting to predicted global angle
    sim_ga = ap_direction + sim_la

    # Add noise
    sim_ga_noised = (sim_ga + np.random.normal(0, sd, aps.shape[0]))
    sim_ga_noised += (sim_ga_noised < 0) * 2.0 * np.pi
    sim_ga_noised += (sim_ga_noised > (2 * np.pi)) * (-2.0) * np.pi

    # print global_doa
    Zs = []
    x0 = [50, 100]

    # print x0, single_prob(x0)
    res = minimize(single_prob, x0, method='Nelder-Mead', options={'disp': False})
    # print res.x, single_prob(res.x)
    error_x[i_rep] = (res.x[0] - sim_pos[0])
    error_y[i_rep] = (res.x[1] - sim_pos[1])

print 'Systematic X error: ', np.mean(error_x), 'Systematic Y error: ', np.mean(error_y)
print 'X\'s sd: ', np.std(error_x), 'Y\'s sd: ', np.std(error_y)
print 'total sd: ', np.sqrt(np.std(error_x) ** 2 + np.std(error_y) ** 2)
