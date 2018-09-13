import numpy as np
import matplotlib.pyplot as plt

import os

from z_utils.fields import Field

from z_utils.path_manager import path_to_results_folder

from visualizer.fields_at_the_window import see_field


fullpath = os.path.join(path_to_results_folder, 'figures')
filename_1 = os.path.join(fullpath, 'case1.pdf')
filename_random = os.path.join(fullpath, 'case_random.pdf')


# Introduce some functions
def f_vcon(t, x):
    # real eigenvalue both positive or both negative: stable node.
    t = float(t); x = [float(z) for z in x]
    sigma = 0.2
    I = 0.99
    w = 0.9
    alpha = 0.04
    tx, ty = -0, -0.5

    return alpha * (x[1]) + tx, alpha * ( -1 * sigma * x[1] + I + w * np.cos(x[0]) ) + ty


shape = (25, 25, 1, 1, 2)

# generate fields from functions
field_1 = Field.generate_zero(shape=shape)

for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            field_1.field[i, j, 0, 0, :] = f_vcon(1, [i, j])


see_field(field_1, scale=1, input_color='r', fig_tag=1, annotate='', annotate_position=[-4, 1])

plt.show()
