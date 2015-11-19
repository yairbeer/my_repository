__author__ = 'YBeer'

"""
drawing an ellipse
"""

import numpy as np
import matplotlib.pyplot as plt

# res is the resolution of sampling. better be a factor of 360
res = 40

# center of ellipse
x_center = 10
y_center = 20

# longer radius
a = 10
# shorter radius
b = 5

# rotation
rot = 30
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
plt.plot(ellipse[:, 0], ellipse[:, 1], 'r')
plt.show()
