import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pickle
import IPython
data = np.load('data.npy')
info = pickle.load(open('info.pkl', 'r'))
positions = info['obj_positions']
fails = info['fail_positions']
sups = info['sup_viol_positions']

good_positions = []
warn_positions = []
bad_positions = []
both_positions = []

for t in range(len(positions)):
    pos = positions[t]
    if fails[t] and sups[t]:
        both_positions.append(pos)
    elif fails[t] and not sups[t]:
        bad_positions.append(pos)
    elif not fails[t] and sups[t]:
        warn_positions.append(pos)
    else:
        good_positions.append(pos)

good_positions = np.array(good_positions).reshape(len(good_positions), 3)
warn_positions = np.array(warn_positions).reshape(len(warn_positions), 3)
bad_positions = np.array(bad_positions).reshape(len(bad_positions), 3)
both_positions = np.array(both_positions).reshape(len(both_positions), 3)
colors = ['green', 'blue', 'yellow', 'red']
positions = [good_positions, warn_positions, both_positions, bad_positions]

IPython.embed()

fig = plt.figure()
ax = fig.gca()

for color, data in zip(colors, positions):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    ax.scatter(x, y, color=color)
ax.legend()
plt.show()

