import numpy as np
import matplotlib.pyplot as plt
import os

from utils.fields import Field
from visualizer.fields_at_the_window import see_field
from utils.path_manager import path_to_results_folder

# Linear SVF:
fullpath = os.path.join(path_to_results_folder, 'figures')
filename_1 = os.path.join(fullpath, 'case1.pdf')
filename_random = os.path.join(fullpath, 'case_random.pdf')

# Introduce some functions
def f_1(t, x):
    # real eigenvalue both positive or both negative: stable node.
    t = float(t); x = [float(z) for z in x]
    a, b, c, d = 2.5, -2, 2, -2
    alpha = 0.2
    return alpha * (a*x[0] + b*x[1] + 5), alpha * (c*x[0] + d*x[1] + 5)


def f_2(t, x):
    # real eigenvalue opposite signs: saddle node.
    t = float(t); x = [float(z) for z in x]
    a, b, c, d = 1, 0, 0, -1
    alpha = 0.2
    return alpha * (a*x[0] + b*x[1] - 10), alpha * (c*x[0] + d*x[1] + 10)


def f_3(t, x):
    # complex (conjugate) eigenvalue: spirals.
    t = float(t); x = [float(z) for z in x]
    a, b, c, d = 0, -1, 1, 0
    alpha = 0.2
    return alpha * (a*x[0] + b*x[1] + 10), alpha * (c*x[0] + d*x[1] - 10)


def f_3_bis(t, x):
    # complex (conjugate) eigenvalue: spirals.
    t = float(t); x = [float(z) for z in x]
    a, b, c, d = 1.5, 1, -1, -1.7
    alpha = 0.2
    return alpha * (a*x[0] + b*x[1] - 25), alpha * (c*x[0] + d*x[1] + 25)


def f_4(t, x):
    t = float(t); x = [float(z) for z in x]
    a, b, c, d = 0.8, -3, 3, -0.7
    alpha = 0.05
    return alpha * (a*x[0] + b*x[1] + 25), alpha * (c*x[0] + d*x[1] - 20)


# generate fields from functions
field_1 = Field.generate_zero(shape=(20, 20, 1, 1, 2))
field_2 = Field.generate_zero(shape=(20, 20, 1, 1, 2))
field_3 = Field.generate_zero(shape=(20, 20, 1, 1, 2))
field_4 = Field.generate_zero(shape=(20, 20, 1, 1, 2))

field_rand = Field.generate_random_smooth(shape=(20, 20, 1, 1, 2))

for i in range(0, 20):
        for j in range(0, 20):
            field_1.field[i, j, 0, 0, :] = f_1(1, [i, j])
            field_2.field[i, j, 0, 0, :] = f_2(1, [i, j])
            field_3.field[i, j, 0, 0, :] = f_3(1, [i, j])
            field_4.field[i, j, 0, 0, :] = f_4(1, [i, j])


see_field(field_1, subtract_id=False, scale=1, input_color='k', fig_tag=1, annotate='', annotate_position=[-4, 1])

see_field(field_2, subtract_id=False, scale=1, input_color='k', fig_tag=2, annotate='', annotate_position=[-4, 1])
see_field(field_3, subtract_id=False, scale=1, title_input='SE(2) generated SVF',
          input_color='k', fig_tag=3, annotate='', annotate_position=[-4, 1])
plt.savefig(filename_1, dpi=400,  format='pdf')

see_field(field_4, subtract_id=False, scale=1, input_color='k', fig_tag=4, annotate='', annotate_position=[-4, 1])
see_field(field_rand, subtract_id=False, scale=1, title_input='Gaussian generated SVF', input_color='k', fig_tag=5,
          annotate='', annotate_position=[-4, 1])
plt.savefig(filename_random, dpi=400,  format='pdf')



plt.show()
