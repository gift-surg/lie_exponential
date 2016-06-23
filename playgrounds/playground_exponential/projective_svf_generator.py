"""
Generate SVF from element of projective Lie groups and compare the results.
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy.core.cache import clear_cache
import copy
from scipy.linalg import expm

from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from utils.projective_algebras import get_random_hom_a_matrices, get_random_hom_matrices, ProjectiveGroup, ProjectiveAlgebra

from visualizer.fields_comparisons import see_overlay_of_n_fields
from visualizer.graphs_and_stats_new import plot_error_linewise
from visualizer.fields_at_the_window import triptych_image_quiver_image, see_field


if __name__ == "__main__":

    clear_cache()

    #### Data settings:

    random_seed = 0

    if random_seed > 0:
        np.random.seed(random_seed)

    x_1, y_1, z_1 = 21, 21, 1

    # point to select for tests:
    x, y, z = 10, 10, 0

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
        center = [x_c, y_c]

    elif d == 3:
        x_c = x_1/2
        y_c = y_1/2
        z_c = z_1/2
        w_c = 1
        center = [x_c, y_c, z_c]

    #### Generate a random hom:
    scale_factor =  1./(np.max(domain)*10)
    hom_attributes = [center, scale_factor, 1, False]

    '''
    # Groups and group centered
    hom_g = ProjectiveGroup.randomgen(d=d,
                                      center=None,
                                      scale_factor=hom_attributes[1],
                                      sigma=hom_attributes[2],
                                      special=hom_attributes[3])

    hom_g_centered = hom_g.centered(center)

    # corresponding in the algebra:
    hom_a = hom_g.logaritmicate()
    hom_a_centered = hom_g_centered.logaritmicate()
    '''


    h_a, h_g = get_random_hom_a_matrices(d=d,
                                          scale_factor=hom_attributes[1],
                                          sigma=hom_attributes[2],
                                          special=hom_attributes[3])

    print
    print 'h algebra = '
    print h_a
    print
    print 'H group = '
    print h_g
    print
    print 'exp : '
    print expm(h_a)

    svf_hom_a = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)
    disp_hom_g = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)

    ### generate the same with a numerical method
    disp_hom_g_ss = svf_hom_a.exponential(algorithm='ss', input_num_steps=5)

    ###########################
    #### Print some results ###
    ###########################

    print '\nsvf at x, y, z ' + str([x, y, z])
    print svf_hom_a.field[x, y, z, 0, :]

    print '\nanalytic solution from the computed SDISP, displacement'
    print disp_hom_g.field[x, y, z, 0, :]

    print '\nExp scaling and squaring on deformation (in displacement coordinates):'
    print disp_hom_g_ss.field[x, y, z, 0, :]

    ############################
    #### Plotting parameters ###
    ############################

    # IF 3d select a slice to be plotted
    if d == 2:
        z_slice = 0
    elif d == 3:
        z_slice = z_c
    else:
        raise IOError

    error_high   = np.zeros([domain[0]])
    error_centre = np.zeros([domain[0]])
    error_low    = np.zeros([domain[0]])

    y_high   = domain[1] - 1
    y_centre = domain[1]/2
    y_low    = +1

    for x in range(domain[0]):

        error_high[x]   = np.linalg.norm(disp_hom_g.field[x, y_high, z_slice, 0, :] -
                                         disp_hom_g_ss.field[x, y_high, z_slice, 0, :])
        error_centre[x] = np.linalg.norm(disp_hom_g.field[x, y_centre, z_slice, 0, :] -
                                         disp_hom_g_ss.field[x, y_centre, z_slice, 0, :])
        error_low[x]    = np.linalg.norm(disp_hom_g.field[x, y_low, z_slice, 0, :] -
                                         disp_hom_g_ss.field[x, y_low, z_slice, 0, :])

    plot_error_linewise([error_low, error_centre, error_high],
                        [y_low, y_centre, y_high],
                        ['y = ' + str(y_high),
                         'y = ' + str(y_centre),
                         'y = ' + str(y_low)],
                        log_scale=True,
                        additional_field=svf_hom_a.field[:],
                        axial_quote=z_slice,
                        input_parameters=[x_1, y_1, z_1] + hom_attributes)

    ### Select a slice to be plotted
    if d == 3:
        svf_h_slice         = SVF.from_array(svf_hom_a.field[:, :, z_c:(z_c+1), :, :2])
        disp_h_ground_slice = SDISP.from_array(disp_hom_g.field[:, :, z_c:(z_c+1), :, :2])
        disp_h_ss_slice     = SDISP.from_array(disp_hom_g_ss.field[:, :, z_c:(z_c+1), :, :2])
    else:
        svf_h_slice         = SVF.from_array(svf_hom_a.field[:])
        disp_h_ground_slice = SDISP.from_array(disp_hom_g.field[:])
        disp_h_ss_slice     = SDISP.from_array(disp_hom_g_ss.field[:])

    see_overlay_of_n_fields([svf_h_slice, disp_h_ss_slice, disp_h_ground_slice],
                            input_color=('r', 'b', 'm'),
                            subtract_id=[False, False, False], scale=1)

    see_field(svf_h_slice, fig_tag=42, input_color='r')


    plt.show()
