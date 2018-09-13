import numpy as np
import os
import matplotlib.pyplot as plt

from path_manager import path_root


path_to_npy_final_arrays = os.path.join(path_root, 'step_3')
# Load blocks data

global_no_aei = np.load(os.path.join(path_to_npy_final_arrays, 'steps_and_costs.npy'))
global_aei    = np.load(os.path.join(path_to_npy_final_arrays, 'steps_and_costs_aei.npy'))

# give each sector nicer names:

steps_no_aei_pyramid_1 = global_no_aei[0, 0, :]
steps_aei_pyramid_1 = global_aei[0, 0, :]

steps_no_aei_pyramid_2 = global_no_aei[0, 1, :]
steps_aei_pyramid_2 = global_aei[0, 1, :]

steps_no_aei_pyramid_3 = global_no_aei[0, 2, :]
steps_aei_pyramid_3 = global_aei[0, 2, :]

costs_no_aei_pyramid_1 = global_no_aei[1, 0, :]
costs_aei_pyramid_1 = global_aei[1, 0, :]

costs_no_aei_pyramid_2 = global_no_aei[1, 1, :]
costs_aei_pyramid_2 = global_aei[1, 1, :]

costs_no_aei_pyramid_3 = global_no_aei[1, 2, :]
costs_aei_pyramid_3 = global_aei[1, 2, :]


steps_diff_pyramid_1 = steps_no_aei_pyramid_1 - steps_aei_pyramid_1
costs_diff_pyramid_1 = costs_no_aei_pyramid_1 - costs_aei_pyramid_1

steps_diff_pyramid_2 = steps_no_aei_pyramid_2 - steps_aei_pyramid_2
costs_diff_pyramid_2 = costs_no_aei_pyramid_2 - costs_aei_pyramid_2

steps_diff_pyramid_3 = steps_no_aei_pyramid_3 - steps_aei_pyramid_3
costs_diff_pyramid_3 = costs_no_aei_pyramid_3 - costs_aei_pyramid_3

fig = plt.figure()

ax1 = fig.add_subplot(231)
ax1.hist(costs_diff_pyramid_1, 30, facecolor='red', alpha=0.5)
ax1.set_xlabel('costs_diff_pyramid_1')
ax1.grid(True)

ax2 = fig.add_subplot(232)
ax2.hist(costs_diff_pyramid_2, 30, facecolor='red', alpha=0.5)
ax2.set_xlabel('costs_diff_pyramid_2')
ax2.grid(True)

ax3 = fig.add_subplot(233)
ax3.hist(costs_diff_pyramid_3, 30, facecolor='red', alpha=0.5)
ax3.set_xlabel('costs_diff_pyramid_3')
ax3.grid(True)

ax4 = fig.add_subplot(234)
ax4.hist(steps_diff_pyramid_1, 30, facecolor='green', alpha=0.5)
ax4.set_xlabel('steps_diff_pyramid_1')
ax4.grid(True)

ax5 = fig.add_subplot(235)
ax5.hist(steps_diff_pyramid_2, 30, facecolor='green', alpha=0.5)
ax5.set_xlabel('steps_diff_pyramid_2')
ax5.grid(True)

ax6 = fig.add_subplot(236)
ax6.hist(steps_diff_pyramid_3, 30, facecolor='green', alpha=0.5)
ax6.set_xlabel('steps_diff_pyramid_3')
ax6.grid(True)

plt.show()
