"""
Test of the numerical exponential map.

"""

import numpy as np
import matplotlib.pyplot as plt

from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from transformations.se2_a import se2_g

from visualizer.fields_at_the_window import see_field
from visualizer.fields_comparisons import see_overlay_of_n_fields, \
    see_2_fields_separate_and_overlay, see_n_fields_separate, see_n_fields_special


if __name__ == "__main__":

    # Tollerance: a numerical method passes the test if the error is below toll
    toll = 0.04

    # set the domain and the passepartout
    domain = (14, 14)
    passepartout = 3
    # create parameter of the transformation in se(2)_a
    x_c = 7
    y_c = 7
    theta = np.pi/8
    tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
    ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c
    # create the controlled transformation
    m_0 = se2_g.se2_g(theta, tx, ty)
    dm_0 = se2_g.log(m_0)
    # generate subsequent vector fields and displacement field:
    svf_0   = SVF.generate_from_matrix(domain, dm_0, affine=np.eye(4))
    sdisp_0 = SDISP.generate_from_matrix(domain, m_0, affine=np.eye(4), subtract_id=True)

    # compute the displacement using the numerical methods:
    spline_interpolation_order = 3

    sdisp_ss      = svf_0.exponential(algorithm='ss', s_i_o=spline_interpolation_order)
    sdisp_ss_pa   = svf_0.exponential(algorithm='ss_pa', s_i_o=spline_interpolation_order)
    sdisp_euler   = svf_0.exponential(algorithm='euler', s_i_o=spline_interpolation_order)
    sdisp_mid_p   = svf_0.exponential(algorithm='midpoint', s_i_o=spline_interpolation_order)
    sdisp_euler_m = svf_0.exponential(algorithm='euler_mod', s_i_o=spline_interpolation_order)
    sdisp_heun    = svf_0.exponential(algorithm='heun', s_i_o=spline_interpolation_order)
    sdisp_heun_m  = svf_0.exponential(algorithm='heun_mod', s_i_o=spline_interpolation_order)
    sdisp_rk4  = svf_0.exponential(algorithm='rk4', s_i_o=spline_interpolation_order)

    print '|ss - disp|        = ' + str((sdisp_ss - sdisp_0).norm(passe_partout_size=passepartout))
    print '|ss_pa - disp|     = ' + str((sdisp_ss_pa - sdisp_0).norm(passe_partout_size=passepartout))
    print '|euler - disp|     = ' + str((sdisp_euler - sdisp_0).norm(passe_partout_size=passepartout))
    print '|midpoint - disp|  = ' + str((sdisp_mid_p - sdisp_0).norm(passe_partout_size=passepartout))
    print '|euler_mod - disp| = ' + str((sdisp_euler_m - sdisp_0).norm(passe_partout_size=passepartout))
    print '|heun - disp|      = ' + str((sdisp_heun - sdisp_0).norm(passe_partout_size=passepartout))
    print '|heun_mod - disp|  = ' + str((sdisp_heun_m - sdisp_0).norm(passe_partout_size=passepartout))
    print '|rk4 - disp|       = ' + str((sdisp_rk4 - sdisp_0).norm(passe_partout_size=passepartout))

    assert (sdisp_ss - sdisp_0).norm(passe_partout_size=passepartout) < toll
    assert (sdisp_ss_pa - sdisp_0).norm(passe_partout_size=passepartout) < toll
    assert (sdisp_euler - sdisp_0).norm(passe_partout_size=passepartout) < toll
    assert (sdisp_mid_p - sdisp_0).norm(passe_partout_size=passepartout) < toll
    assert (sdisp_euler_m - sdisp_0).norm(passe_partout_size=passepartout) < toll
    assert (sdisp_heun - sdisp_0).norm(passe_partout_size=passepartout) < toll
    assert (sdisp_heun_m - sdisp_0).norm(passe_partout_size=passepartout) < toll

if 0:
    ### compute matrix of transformations:
    domain = (14, 14)

    x_c = 7
    y_c = 7
    theta = np.pi/8

    tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
    ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

    passepartout = 3

    m_0 = se2_g.se2_g(theta, tx, ty)
    dm_0 = se2_g.log(m_0)


    ### generate subsequent vector fields
    svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))

    # This provides the displacement since I am subtracting the id
    sdisp_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))

    print type(svf_0)
    print type(sdisp_0)

    ### compute exponential with different available methods:

    spline_interpolation_order = 3

    sdisp_ss      = svf_0.exponential(algorithm='ss', spline_interpolation_order=spline_interpolation_order)
    sdisp_ss_pa   = svf_0.exponential(algorithm='ss_pa', spline_interpolation_order=spline_interpolation_order)
    sdisp_euler   = svf_0.exponential(algorithm='euler', spline_interpolation_order=spline_interpolation_order)
    #sdisp_ser     = svf_0.exponential(algorithm='series', spline_interpolation_order=spline_interpolation_order)
    sdisp_mid_p   = svf_0.exponential(algorithm='midpoint', spline_interpolation_order=spline_interpolation_order)
    sdisp_euler_m = svf_0.exponential(algorithm='euler_mod', spline_interpolation_order=spline_interpolation_order)
    sdisp_rk4     = svf_0.exponential(algorithm='rk4', spline_interpolation_order=spline_interpolation_order)

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
    #print '|svf - disp|       = ' + str((svf_0 - sdisp_0).norm(passe_partout_size=4))
    print '|ss - disp|        = ' + str((sdisp_ss - sdisp_0).norm(passe_partout_size=passepartout))
    print '|ss_pa - disp|     = ' + str((sdisp_ss_pa - sdisp_0).norm(passe_partout_size=passepartout))
    print '|euler - disp|     = ' + str((sdisp_euler - sdisp_0).norm(passe_partout_size=passepartout))
    print '|midpoint - disp|  = ' + str((sdisp_mid_p - sdisp_0).norm(passe_partout_size=passepartout))
    print '|euler_mod - disp| = ' + str((sdisp_euler_m - sdisp_0).norm(passe_partout_size=passepartout))
    print '|rk4 - disp|       = ' + str((sdisp_rk4 - sdisp_0).norm(passe_partout_size=passepartout))

    print
    print
    print
    print

    print '--------------------'
    print "Norm of the errors:"

    ### Plot:

    fields_list = [svf_0, sdisp_0, sdisp_ss,   sdisp_ss_pa,   sdisp_euler,
                   sdisp_mid_p,   sdisp_euler_m,   sdisp_rk4]
    title_input = ['svf_0', 'sdisp_0', 'sdisp_ss', 'sdisp_ss_pa', 'sdisp_euler', 'disp_mid_p', 'disp_euler_m', 'disp_rk4']
    subtract_id = [False, False] + ([False, ] * (len(fields_list) - 2))
    input_color = ['r', 'b', 'g', 'c', 'm', 'k', 'b', 'g', 'c']

    if 0:
        # See svf and disp in separate fields.
        see_field(svf_0, fig_tag=0,  title_input="svf", input_color='r')
        see_field(sdisp_0, fig_tag=1, title_input="disp", input_color='b')

    if 0:
        see_field(sdisp_ss, fig_tag=4)
        see_field(sdisp_ss_pa, fig_tag=5)
        see_field(sdisp_euler, fig_tag=6)
        #see_field(sdisp_ser, fig_tag=7)
        see_field(sdisp_mid_p, fig_tag=8)
        see_field(sdisp_euler_m, fig_tag=9)
        see_field(sdisp_rk4, fig_tag=10)

    if 0:
        see_n_fields_separate(fields_list, fig_tag=12, title_input=title_input, input_color=input_color)

    if 0:
        see_overlay_of_n_fields([svf_0, sdisp_0], fig_tag=13,
                                input_color=['r', 'b'],
                                input_label=None
                                )

    if 0:
        see_overlay_of_n_fields(fields_list, fig_tag=14,
                                input_color=input_color,
                                input_label=None)

    if 0:
        see_2_fields_separate_and_overlay(svf_0, sdisp_0, fig_tag=3)


    if 1:
        title_input_l = ['Sfv Input',
                         'Ground Output',
                         'Scaling and Squaring',
                         'Polyaffine Scal. and Sq.',
                         'Euler method',
                         'Midpoint Method',
                         'Euler Modif Method',
                         'Runge Kutta 4']

        list_fields_of_field = [[svf_0], [sdisp_0]]
        list_colors = ['r', 'b']
        for third_field in fields_list[2:]:
            list_fields_of_field += [[svf_0, sdisp_0, third_field]]
            list_colors += ['r', 'b', 'm']

        see_n_fields_special(list_fields_of_field, fig_tag=50,
                             colors_input=list_colors,
                             titles_input=title_input_l,
                             sample=(1, 1),
                             zoom_input=[0, 14, 0, 14],
                             window_title_input='matrix, random generated',
                             legend_on=False)

    if 0:
        title_input_l = ['Scaling and Squaring',
                         'Polyaffine Scaling and Squaring',
                         'Euler method',
                         'Series',
                         'Midpoint Method',
                         'Euler Modif Method',
                         'Runge Kutta 4']

        for k, third_field in enumerate(fields_list[2::]):
            see_overlay_of_n_fields([svf_0, sdisp_0, third_field], fig_tag=20 + k,
                                    title_input=title_input_l[k],
                                    input_color=['r', 'b', 'm'],
                                    window_title_input='matrix, random generated')


    plt.show()


