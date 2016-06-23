"""
Integration with scipy for one matrix-generated svf.
"""
import numpy as np
import matplotlib.pyplot as plt

from transformations.s_disp import SDISP
from transformations.s_vf import SVF

from transformations.se2_a import se2_g

from visualizer.fields_comparisons import see_n_fields_special
from visualizer.fields_and_integral_curves import see_overlay_of_n_fields_and_flow

### compute matrix of transformations: ###
domain = (10, 10)  # (20, 20)

x_c = 10
y_c = 10
theta = np.pi/8

tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c
m_0 = se2_g.se2_g(theta, tx, ty)
dm_0 = se2_g.log(m_0)

passepartout = 3


# Initialize the displacement field that will be computed using the integral curves.
methods_vode = ['bdf', 'adams']
max_steps    = [10]#, 15, 20, 25, 30]

print dm_0.get_matrix
print m_0.get_matrix


# Modulate over the number of steps:
for steps in max_steps:

    print '--------------------'
    print 'results for steps : ' + str(steps)
    print '--------------------'
    ### generate subsequent vector fields ###
    # svf
    svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))
    # displacement, I am subtracting the id to have a displacement and not a deformation.
    sdisp_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))

    print type(svf_0)
    print type(sdisp_0)

    spline_interpolation_order = 3

    #
    sdisp_ss      = svf_0.exponential(algorithm='ss', s_i_o=spline_interpolation_order)
    sdisp_ss_pa   = svf_0.exponential(algorithm='ss_pa', s_i_o=spline_interpolation_order)
    sdisp_euler   = svf_0.exponential(algorithm='euler', s_i_o=spline_interpolation_order)
    sdisp_mid_p   = svf_0.exponential(algorithm='midpoint', s_i_o=spline_interpolation_order)
    sdisp_euler_m = svf_0.exponential(algorithm='euler_mod', s_i_o=spline_interpolation_order)
    sdisp_rk4     = svf_0.exponential(algorithm='rk4', s_i_o=spline_interpolation_order)
    #
    disp_scipy =  svf_0.exponential_scipy(method=methods_vode[1], max_steps=steps, verbose=False,
                                          passepartout=passepartout, return_integral_curves=True)

    disp_computed   = disp_scipy[0]
    integral_curves = disp_scipy[1]

    print 'spam'
    print type(integral_curves)
    print type(integral_curves[0])

    error = (disp_computed - sdisp_0).norm(passe_partout_size=3)

    print type(sdisp_ss)
    print type(sdisp_ss_pa)
    print type(sdisp_euler)
    print type(sdisp_euler_m)
    print type(sdisp_rk4)

    print '--------------------'
    print "Norm of the svf:"
    print svf_0.norm(passe_partout_size=4)

    print '--------------------'
    print "Norm of the displacement field:"
    print sdisp_0.norm(passe_partout_size=4)

    print '--------------------'
    print "Norm of the errors:"
    print '--------------------'
    print '|ss - disp|        = ' + str((sdisp_ss - sdisp_0).norm(passe_partout_size=passepartout))
    print '|ss_pa - disp|     = ' + str((sdisp_ss_pa - sdisp_0).norm(passe_partout_size=passepartout))
    print '|euler - disp|     = ' + str((sdisp_euler - sdisp_0).norm(passe_partout_size=passepartout))
    print '|midpoint - disp|  = ' + str((sdisp_mid_p - sdisp_0).norm(passe_partout_size=passepartout))
    print '|euler_mod - disp| = ' + str((sdisp_euler_m - sdisp_0).norm(passe_partout_size=passepartout))
    print '|rk4 - disp|       = ' + str((sdisp_rk4 - sdisp_0).norm(passe_partout_size=passepartout))
    print '----------  Error scipy ----------------------'
    print '|vode - disp| = ' + str(error)
    print '--------------------------------'

    print
    print
    print

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

    if 1:
        fields_list_0 = [svf_0, sdisp_0, disp_computed]

        see_overlay_of_n_fields_and_flow(fields_list_0,
                                         integral_curves,
                                         fig_tag=14,
                                         input_color=['r', 'b', 'm'],
                                         input_label=None,
                                         see_tips=False,
                                         list_of_alpha_for_obj=[0.8, 0.5, 0.5],
                                         alpha_for_integral_curves=0.5)

plt.show()

