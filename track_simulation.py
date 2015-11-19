import numpy as np
import matplotlib.pyplot as plt

# Placing and directing APs, direction is the AP main direction. phi = 0 -> y+, phi = 90 -> x+. like a compass
# [0: x, 1: y, 2: direction]
aps = np.array([[-1, -1, 45],
                [-1, 101, 135],
                [101, 101, 225],
                [101, -1, 315]])

# straight line (5,5) -> (50, 50)
duration1 = 46
track = []
for i in range(duration1):
    track.append([])
    track[-1].append(5 + i)
    track[-1].append(5 + i)

# straight line (50, 50) -> (14, 50)
duration2 = 36
for i in range(duration2):
    track.append([])
    track[-1].append(50 - (i + 1))
    track[-1].append(50)


# half circle clockwise (14, 50) -> (86, 50)
duration3 = 90
for i in range(duration3):
    track.append([])
    track[-1].append(50 - 36*np.cos(np.pi*(i + 1)/duration3))
    track[-1].append(50 + 36*np.sin(np.pi*(i + 1)/duration3))

# straight line (86, 50) -> (65, 50)
duration4 = 21
for i in range(duration4):
    track.append([])
    track[-1].append(86 - (i + 1))
    track[-1].append(50)

# half circle clockwise (65, 50) -> (35, 50)
duration5 = 36
for i in range(duration5):
    track.append([])
    track[-1].append(50 + 15*np.cos(np.pi*(i + 1)/duration5))
    track[-1].append(50 - 15*np.sin(np.pi*(i + 1)/duration5))

track = np.array(track)

duration = duration1 + duration2 + duration3 + duration4 + duration5

# # show track on 2D
# plt.plot(aps[:, 0], aps[:, 1], 'ro', track[:, 0], track[:, 1], 'g')
# plt.show()
