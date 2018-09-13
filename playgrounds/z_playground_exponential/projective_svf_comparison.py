"""
Generate SVF from element of projective Lie groups and compare the results.
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy.core.cache import clear_cache

from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from z_utils.projective_algebras import get_random_hom

from visualizer.fields_comparisons import see_overlay_of_n_fields
from visualizer.graphs_and_stats_new import plot_error_linewise
from visualizer.fields_at_the_window import triptych_image_quiver_image, see_field
from visualizer.fields_comparisons import see_n_fields_separate


if __name__ == "__main__":

    clear_cache()

    #### Data settings:

    random_seed = 18

    if random_seed > 0:
        np.random.seed(random_seed)

    x_1, y_1, z_1 = 30, 30, 1

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


    if random_seed > 0:
        np.random.seed(random_seed)

    scale_factor =  1./(np.max(domain)*10)
    sigma_in = 3
    special = False

    hom_attributes = [[x_c, y_c, z_c], 'diag', scale_factor, sigma_in, special]

    h_a, h_g = get_random_hom(d=d, scale_factor=hom_attributes[2],
                                              sigma=hom_attributes[3],)

    ### DIAG

    if random_seed > 0:
        np.random.seed(random_seed)

    scale_factor =  1./(np.max(domain)*10)
    sigma_in = 3
    special = False

    hom_attributes = [[x_c, y_c, z_c], 'diag', scale_factor, sigma_in, special]
    h_a_diag, h_g_diag = get_random_hom(d=d,
                                              center=hom_attributes[0],
                                              random_kind=hom_attributes[1],
                                              scale_factor=hom_attributes[2],
                                              sigma=hom_attributes[3],
                                              special=hom_attributes[4])

    ### SKEW

    if random_seed > 0:
        np.random.seed(random_seed)

    scale_factor =  1./(np.max(domain)*10)
    sigma_in = 3
    special = False

    hom_attributes = [[x_c, y_c, z_c], 'skew', scale_factor, sigma_in, special]
    h_a_skew, h_g_skew = get_random_hom(d=d,
                                              center=hom_attributes[0],
                                              random_kind=hom_attributes[1],
                                              scale_factor=hom_attributes[2],
                                              sigma=hom_attributes[3],
                                              special=hom_attributes[4])

    ### TRANSL

    if random_seed > 0:
        np.random.seed(random_seed)

    scale_factor =  1./(np.max(domain)*10)
    sigma_in = 3
    special = False

    hom_attributes = [[x_c, y_c, z_c], 'transl', scale_factor, sigma_in, special]
    h_a_transl, h_g_transl = get_random_hom(d=d,
                                              center=hom_attributes[0],
                                              random_kind=hom_attributes[1],
                                              scale_factor=hom_attributes[2],
                                              sigma=hom_attributes[3],
                                              special=hom_attributes[4])

    ### SHIFT

    if random_seed > 0:
        np.random.seed(random_seed)

    scale_factor =  1./(np.max(domain)*10)
    sigma_in = 3
    special = False

    hom_attributes = [[x_c, y_c, z_c], 'shift', scale_factor, sigma_in, special]
    h_a_shift, h_g_shift = get_random_hom(d=d,
                                          center=hom_attributes[0],
                                          random_kind=hom_attributes[1],
                                          scale_factor=hom_attributes[2],
                                          sigma=hom_attributes[3],
                                          special=hom_attributes[4])

    print 'h_a_diag = '
    print h_a_diag

    print 'h_g_diag = '
    print h_g_diag

    print 'h_a_skew = '
    print h_a_skew

    print 'h_g_skew = '
    print h_g_skew

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

    svf_h = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)
    disp_h_ground = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)

    svf_h_diag = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a_diag)
    disp_h_ground_diag = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g_diag)

    svf_h_skew = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a_skew)
    disp_h_ground_skew = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g_skew)

    svf_h_transl = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a_transl)
    disp_h_ground_transl = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g_transl)

    svf_h_shift = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a_shift)
    disp_h_ground_shift = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g_shift)

    ############################
    #### Plotting parameters ###
    ############################

    see_n_fields_separate([svf_h_diag, svf_h_skew, svf_h_shift, svf_h_transl],
                          input_color=['r', 'r', 'r', 'r'],
                          row_fig=2,
                          col_fig=2,
                          fig_tag=10,
                          input_figsize=(10, 10),
                          title_input=['translate svf diag',
                                       'translate svf skew',
                                       'translate svf shift',
                                       'translate svf transl'])

    see_field(svf_h, fig_tag=40, input_color='r', title_input='svf before the translation')

    #see_field(svf_h_diag, fig_tag=40, input_color='r', title_input='diag svf')
    #see_field(svf_h_skew, fig_tag=41, input_color='r', title_input='skew svf')
    #see_field(svf_h_shift, fig_tag=42, input_color='r', title_input='shift svf')
    #see_field(svf_h_transl, fig_tag=43, input_color='r', title_input='transl svf')

    #see_field(svf_h_shift, fig_tag=45, input_color='r')
    #see_field(svf_h_transl, fig_tag=45, input_color='m')

    plt.show()
