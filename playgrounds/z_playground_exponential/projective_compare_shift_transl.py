"""
Generate SVF from element of projective Lie groups and compare the results.
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy.core.cache import clear_cache
import copy

from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from z_utils.projective_algebras import get_random_hom_a_matrices

from visualizer.fields_comparisons import see_overlay_of_n_fields
from visualizer.graphs_and_stats_new import plot_error_linewise
from visualizer.fields_at_the_window import triptych_image_quiver_image, see_field


if __name__ == "__main__":

    clear_cache()

    #### Data settings:

    random_seed = 10

    if random_seed > 0:
        np.random.seed(random_seed)

    x_1, y_1, z_1 = 21, 21, 1

    # point to select for tests:
    x, y, z = 5, 5, 0

    if z_1 == 1:
        domain = (x_1, y_1)
        shape = list(domain) + [1, 1, 2]
    else:
        domain = (x_1, y_1, z_1)
        shape = list(domain) + [1, 3]

    d = len(domain)

    if d == 2:
        x_c = x_1/2
        y_c = y_1/2
        z_c = 1

    elif d == 3:
        x_c = x_1/2
        y_c = y_1/2
        z_c = z_1/2
        w_c = 1

    scale_factor =  1./(np.max(domain)*10)
    sigma_in = 1
    special = False

    random_before = np.random.randint(0, 100)

    hom_attributes = [d, scale_factor, sigma_in, special]
    h_a_transl, h_g_transl = get_random_hom_a_matrices(d=hom_attributes[0],
                                              scale_factor=hom_attributes[1],
                                              sigma=hom_attributes[2],
                                              special=hom_attributes[3])

    if random_seed > 0:
        np.random.seed(random_seed)

    random_after_reset = np.random.randint(0, 100)

    np.testing.assert_array_equal(random_before, random_after_reset)
    print random_before, random_after_reset

    hom_attributes = [d, scale_factor, sigma_in, special]
    h_a_shift, h_g_shift = get_random_hom_a_matrices(d=hom_attributes[0],
                                              scale_factor=hom_attributes[1],
                                              sigma=hom_attributes[2],
                                              special=hom_attributes[3])



    #h_a_shift[-1, :] = np.abs(h_a_shift[-1, :])


    print 'h_a_transl = '
    print h_a_transl

    print 'h_g_transl = '
    print h_g_transl

    print 'h_a_shift = '
    print h_a_shift

    print 'h_g_shift = '
    print h_g_shift

    #np.testing.assert_almost_equal(h_a_transl, h_a_shift, decimal=3)
    #np.testing.assert_almost_equal(h_g_transl, h_g_shift, decimal=3)

    svf_h_transl = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a_transl)
    disp_h_ground_transl = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g_transl)

    svf_h_shift = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a_shift)
    disp_h_ground_shift = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g_shift)

    disp_h_ss_transl = svf_h_transl.exponential(algorithm='ss', input_num_steps=5)
    disp_h_ss_shift = svf_h_shift.exponential(algorithm='ss', input_num_steps=5)

    ###########################
    #### Print some results ###
    ###########################

    print 'TRANSLATION'

    print '\nsvf at x, y, z ' + str([x, y, z])
    print svf_h_transl.field[x, y, z, 0, :]

    print '\nanalytic solution from the computed SDISP, displacement'
    print disp_h_ground_transl.field[x, y, z, 0, :]

    print '\nExp scaling and squaring on deformation (in displacement coordinates):'
    print disp_h_ss_transl.field[x, y, z, 0, :]

    print 'SHIFT'

    print '\nsvf at x, y, z ' + str([x, y, z])
    print svf_h_shift.field[x, y, z, 0, :]

    print '\nanalytic solution from the computed SDISP, displacement'
    print disp_h_ground_shift.field[x, y, z, 0, :]

    print '\nExp scaling and squaring on deformation (in displacement coordinates):'
    print disp_h_ss_shift.field[x, y, z, 0, :]

    ############################
    #### Plotting parameters ###
    ############################

    see_field(svf_h_shift, fig_tag=42, input_color='r')
    see_field(svf_h_transl, fig_tag=43, input_color='m')

    see_field(svf_h_shift, fig_tag=45, input_color='r')
    see_field(svf_h_transl, fig_tag=45, input_color='m')

    plt.show()
