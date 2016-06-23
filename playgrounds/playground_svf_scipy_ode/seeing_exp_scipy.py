
import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.integrate import ode

from utils.fields import Field
from transformations.s_disp import SDISP
from transformations.s_vf import SVF

from transformations.se2_a import se2_g

from visualizer.fields_comparisons import see_overlay_of_n_fields, \
    see_2_fields_separate_and_overlay, see_n_fields_separate, see_n_fields_special


### compute matrix of transformations: ###
domain = (15, 15)

x_c = 7
y_c = 7
theta = np.pi/8

passepartout = 3

tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c
m_0 = se2_g.se2_g(theta, tx, ty)
dm_0 = se2_g.log(m_0)

print dm_0.get_matrix
print m_0.get_matrix


### generate subsequent vector fields ###

# svf
svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))
# displacement, I am subtracting the id to have a displacement and not a deformation.
sdisp_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))

print type(svf_0)
print type(sdisp_0)

# Initialize the displacement field that will be computed using the integral curves.
disp_computed =  svf_0.exponential_scipy(verbose=True, passepartout=passepartout)


if 1:
    title_input_l = ['Sfv Input',
                     'Ground Output',
                     'Vode integrator']

    list_fields_of_field = [[svf_0], [sdisp_0], [svf_0, sdisp_0, disp_computed]]
    list_colors = ['r', 'b', 'r', 'b', 'm']

    see_n_fields_special(list_fields_of_field,
                         fig_tag=50,
                         row_fig=1, col_fig=3,
                         colors_input=list_colors,
                         titles_input=title_input_l,
                         zoom_input=[0, 20, 0, 20], sample=(1, 1),
                         window_title_input='matrix, random generated',
                         legend_on=False)


plt.show()
