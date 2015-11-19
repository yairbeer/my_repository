__author__ = 'YBeer'

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

someX, someY = 0.5, 0.5
fig, ax = plt.subplots()
plt.plot(range(10))
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((someX - .1, someY - .1), 5, 5, alpha=1, facecolor='none'))
plt.show()
