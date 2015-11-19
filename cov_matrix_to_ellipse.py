__author__ = 'YBeer'

import numpy as np

cov = np.array([[3, 5], [5, 3]])

# constant for deciding the ellipse size, how much time is correct
confidence_const = np.e

w, v = np.linalg.eig(np.array(cov))
angle = 180 / np.pi * np.arctan(v[1, 0] / v[0, 0])

print w
print v
print angle

w = abs(w)
w = 2 * np.sqrt(confidence_const * w)
print w
