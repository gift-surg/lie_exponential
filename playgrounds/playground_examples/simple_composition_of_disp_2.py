import numpy as np
import matplotlib.pyplot as plt

from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from visualizer.fields_at_the_window import see_field


shape = (6, 6, 1, 1, 2)

sigma_init = 4
sigma_gaussian_filter = 2

svf_f   = SVF.generate_zero(shape=shape)
svf_g   = SVF.generate_zero(shape=shape)


def function_f(t, x):
    t = float(t); x = [float(y) for y in x]
    return np.array([0, 0.3])


def function_g(t, x):
    t = float(t); x = [float(y) for y in x]
    return np.array([0, -0.3])


for x in range(0, 6):
    for y in range(0, 6):
        svf_f.field[x, y, 0, 0, :] = function_f(1, [x, y])
        svf_g.field[x, y, 0, 0, :] = function_g(1, [x, y])


f_o_g = SDISP.composition(svf_f, svf_g)
g_o_f = SDISP.composition(svf_g, svf_f)


# sfv_0 is provided in Lagrangian coordinates!

see_field(svf_f, fig_tag=1, input_color='b')
see_field(svf_g, fig_tag=1, input_color='r', title_input='2 vector fields: f blue, g red')


see_field(svf_f, fig_tag=2, input_color='b')
see_field(svf_g, fig_tag=2, input_color='r')
see_field(f_o_g, fig_tag=2, input_color='g', title_input='composition (f o g) in green')


see_field(svf_f, fig_tag=3, input_color='b')
see_field(svf_g, fig_tag=3, input_color='r')
see_field(g_o_f, fig_tag=3, input_color='g', title_input='composition (g o f) in green')


plt.show()


