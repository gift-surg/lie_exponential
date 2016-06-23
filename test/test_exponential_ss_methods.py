"""
Module to test the scaling and squaring-based methods.
Some tests are visual 
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from transformations.se2_a import se2_g

from visualizer.fields_comparisons import see_n_fields_special


def visual_assessment_method_one_se2():
    """
    This test is for visual assessment only.

    Aimed to test the prototyping of the computation of the exponential map
    with some methods.

    (Nothing is saved in external folder.)
    """

    ##############
    # controller #
    ##############

    domain = (14, 14)

    x_c = 7
    y_c = 7
    theta = np.pi/4

    tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
    ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

    passepartout = 2
    spline_interpolation_order = 3

    res_time = np.zeros(4)

    #########
    # model #
    #########

    m_0 = se2_g.se2_g(theta, tx, ty)
    dm_0 = se2_g.log(m_0)

    ### generate subsequent vector fields

    svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))
    sdisp_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))

    print type(svf_0)
    print type(sdisp_0)

    ### compute exponential with different available methods:

    start = time.time()
    sdisp_ss      = svf_0.exponential(algorithm='ss',        s_i_o=spline_interpolation_order)
    res_time[0] = (time.time() - start)

    start = time.time()
    sdisp_ss_pa   = svf_0.exponential(algorithm='ss_pa',     s_i_o=spline_interpolation_order)
    res_time[1] = (time.time() - start)

    start = time.time()
    sdisp_ss_pa_m = svf_0.exponential(algorithm='ss_pa_mod',     s_i_o=spline_interpolation_order)
    res_time[2] = (time.time() - start)

    ########
    # view #
    ########

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
    print '|ss_pa_m - disp|   = ' + str((sdisp_ss_pa_m - sdisp_0).norm(passe_partout_size=passepartout))

    print

    print '--------------------'
    print "Computational Times: "
    print '--------------------'

    print ' ss        = ' + str(res_time[0])
    print ' ss_pa     = ' + str(res_time[1])
    print ' ss_pa_m   = ' + str(res_time[2])

    fields_list = [svf_0, sdisp_0, sdisp_ss,   sdisp_ss_pa,   sdisp_ss_pa_m]

    if 1:
        title_input_l = ['Sfv Input',
                         'Ground Output',
                         'Scaling and Squaring',
                         'Poly Scal. and Sq.',
                         'Poly Scal. and Sq. mod']

        list_fields_of_field = [[svf_0], [sdisp_0]]
        list_colors = ['r', 'b']
        for third_field in fields_list[2:]:
            list_fields_of_field += [[svf_0, sdisp_0, third_field]]
            list_colors += ['r', 'b', 'm']

        see_n_fields_special(list_fields_of_field, fig_tag=50,
                             row_fig=2,
                             col_fig=3,
                             input_figsize=(10, 7),
                             colors_input=list_colors,
                             titles_input=title_input_l,
                             sample=(1, 1),
                             zoom_input=[0, 14, 0, 14],
                             window_title_input='matrix, random generated',
                             legend_on=False)

    plt.show()


visual_assessment_method_one_se2()
