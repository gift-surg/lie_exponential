"""
Generate SVF from element of projective Lie groups and compare the results.
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy.core.cache import clear_cache
import copy
from scipy.linalg import expm

from utils.projective_algebras import get_random_hom

from transformations.s_vf import SVF
from transformations.s_disp import SDISP
from utils.image import Image

from utils.aux_functions import grid_generator
from utils.helper import generate_position_from_displacement
from utils.resampler import NearestNeighbourResampler

from visualizer.fields_comparisons import see_overlay_of_n_fields
from visualizer.graphs_and_stats_new import plot_error_linewise
from visualizer.fields_at_the_window import triptych_image_quiver_image
from visualizer.fields_at_the_window import see_2_fields, see_field


if __name__ == "__main__":

    clear_cache()

    #### Data settings:

    random_seed = 10

    if random_seed > 0:
        np.random.seed(random_seed)

    x_1, y_1, z_1 = 30, 30, 1

    # point to select for tests:
    x, y, z = 10, 10, 1

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

        projective_center = [x_c, y_c, z_c]

    elif d == 3:
        x_c = x_1/2
        y_c = y_1/2
        z_c = z_1/2
        w_c = 1

        projective_center = [x_c, y_c, z_c, w_c]

    scale_factor = 1./(np.max(domain)*10)

    hom_attributes = [projective_center, 'diag', scale_factor, 5, False]
    h_a, h_g = get_random_hom(d=d,
                              center=hom_attributes[0],
                              kind=hom_attributes[1],
                              scale_factor=hom_attributes[2],
                              sigma=hom_attributes[3],
                              in_psl=hom_attributes[4])

    print hom_attributes

    print 'h = '
    print h_a

    print 'H = '
    print h_g

    ### generate the corresponding svf (sanity check to see the invariance under linear translation)

    h_a_inv = -1 * h_a

    svf_h          = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)
    svf_h_inv      = (-1) * SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)

    svf_h_inv_comp = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a_inv)

    ### Check first inverse generation equivalence:

    np.testing.assert_array_almost_equal(svf_h_inv.field, svf_h_inv_comp.field, decimal=8)

    ### Generate displacements

    sdisp_h = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain,
                                                          input_exp_h=h_g)
    sdisp_h_inv = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain,
                                                              input_exp_h=np.linalg.inv(h_g))

    ## compose them
    sdisp_composition_1 = SDISP.composition(sdisp_h, sdisp_h_inv, s_i_o=3)
    sdisp_composition_2 = SDISP.composition(sdisp_h_inv, sdisp_h, s_i_o=3)

    ## see if their composition is a zero.
    testing = False

    pp = 2
    zero_disp = SDISP.generate_zero(shape)

    print
    print "h o h_inv = " + str(np.linalg.norm(sdisp_composition_1.field - zero_disp.field))
    print "h_inv o h = " + str(np.linalg.norm(sdisp_composition_2.field - zero_disp.field))
    print

    if testing:
        np.testing.assert_array_almost_equal(sdisp_composition_1.field[pp:-pp, pp:-pp, 0, 0, :],
                                             zero_disp.field[pp:-pp, pp:-pp, 0, 0, :],
                                             decimal=0)
        np.testing.assert_array_almost_equal(sdisp_composition_2.field[pp:-pp, pp:-pp, 0, 0, :],
                                             zero_disp.field[pp:-pp, pp:-pp, 0, 0, :],
                                             decimal=0)


    ## plot

    print '\nEND'
    see_field(sdisp_h, fig_tag=4, input_color='b')
    see_field(sdisp_h_inv, fig_tag=4, input_color='r')
    see_field(sdisp_composition_1, fig_tag=4, input_color='g',
              title_input='exp(v) composed with exp(-v) (positive blue, negative red, composition green)')

    see_field(sdisp_h, fig_tag=5, input_color='b')
    see_field(sdisp_h_inv, fig_tag=5, input_color='r')
    see_field(sdisp_composition_1, fig_tag=5, input_color='g',
              title_input='exp(-v) composed with exp(v) (positive blue, negative red, composition green)')

    plt.show()


    '''
    # Generate displacement ground truth (for sanity check)
    sdisp_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))
    sdisp_0_inv = SDISP.generate_from_matrix(domain, np.linalg.inv(m_0.get_matrix) - np.eye(3), affine=np.eye(4))

    # Sanity check: composition of the ground truth, must be very close to the identity field.
    sdisp_o_sdisp_inv_ground = SDISP.composition(sdisp_0, sdisp_0_inv, s_i_o=s_i_o)
    sdisp_inv_o_sdisp_ground = SDISP.composition(sdisp_0_inv, sdisp_0, s_i_o=s_i_o)

    zero_disp = SDISP.generate_zero(shape)
    np.testing.assert_array_almost_equal(sdisp_o_sdisp_inv_ground.field, zero_disp.field, decimal=0)
    np.testing.assert_array_almost_equal(sdisp_inv_o_sdisp_ground.field, zero_disp.field, decimal=0)
    '''



    '''
    disp_h_ground = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)

    disp_h_ss = svf_h.exponential(algorithm='ss', input_num_steps=5)

    ###########################
    #### Print some results ###
    ###########################

    print '\nsvf at x, y, z'
    print svf_h.field[x, y, z, 0, :]

    print '\nanalytic solution from the computed SDISP, displacement'
    print disp_h_ground.field[x, y, z, 0, :]

    print '\nExp scaling and squaring on deformation (in displacement coordinates):'
    print disp_h_ss.field[x, y, z, 0, :]

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

        error_high[x]   = np.linalg.norm(disp_h_ground.field[x, y_high, z_slice, 0, :] -
                                         disp_h_ss.field[x, y_high, z_slice, 0, :])
        error_centre[x] = np.linalg.norm(disp_h_ground.field[x, y_centre, z_slice, 0, :] -
                                         disp_h_ss.field[x, y_centre, z_slice, 0, :])
        error_low[x]    = np.linalg.norm(disp_h_ground.field[x, y_low, z_slice, 0, :] -
                                         disp_h_ss.field[x, y_low, z_slice, 0, :])

    plot_error_linewise([error_low, error_centre, error_high],
                        [y_low, y_centre, y_high],
                        ['y = ' + str(y_high),
                         'y = ' + str(y_centre),
                         'y = ' + str(y_low)],
                        log_scale=True,
                        additional_field=svf_h.field[:],
                        axial_quote=z_slice,
                        input_parameters=[x_1, y_1, z_1] + [is_centered, in_psl, with_hack, is_scaled])

    ### Select a slice to be plotted
    if d == 3:
        svf_h_slice         = SVF.from_array(svf_h.field[:, :, z_c:(z_c+1), :, :2])
        disp_h_ground_slice = SDISP.from_array(disp_h_ground.field[:, :, z_c:(z_c+1), :, :2])
        disp_h_ss_slice     = SDISP.from_array(disp_h_ss.field[:, :, z_c:(z_c+1), :, :2])
    else:
        svf_h_slice         = SVF.from_array(svf_h.field[:])
        disp_h_ground_slice = SDISP.from_array(disp_h_ground.field[:])
        disp_h_ss_slice     = SDISP.from_array(disp_h_ss.field[:])

    see_overlay_of_n_fields([svf_h_slice, disp_h_ss_slice, disp_h_ground_slice],
                            input_color=('r', 'b', 'm'),
                            subtract_id=[False, False, False], scale=1)

    ### Select appropriate slices ###

    # generate the grid and copy as target
    grid_array = grid_generator(x_size=x_1,
                                y_size=y_1,
                                x_step=10,
                                y_step=10,
                                line_thickness=1)
    source_grid_im = Image.from_array(grid_array)

    zeros_array = np.zeros_like(grid_array)
    target_grid_im = Image.from_array(zeros_array)

    ### RESAMPLING!!!
    # Convert to position from displacement, select in the slice if in 3d:
    if d == 2:
        displacement_slice = SDISP.from_array(disp_h_ground.field[:])
    elif d == 3:
        displacement_slice = SDISP.from_array(disp_h_ground.field[:, :, z_slice:(z_slice+1), :, :(d-1)])
    else:
        raise IOError

    deformation = generate_position_from_displacement(displacement_slice)

    csr = NearestNeighbourResampler()
    csr.resample(source_grid_im, deformation, target_grid_im)

    triptych_image_quiver_image(source_grid_im.field,
                                disp_h_ground.field,
                                target_grid_im.field,
                                interval_svf=1)

    plt.show()
    '''

    '''
    # TODO: check step by step how to generate the inverse displacement
    # Generate displacement ground truth (for sanity check)
    sdisp_0 = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)
    sdisp_0_inv = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g_inv)
    # Sanity check: composition of the ground truth, must be very close to the identity field.
    sdisp_o_sdisp_inv_ground = SDISP.composition(sdisp_0, sdisp_0_inv, s_i_o=s_i_o)
    sdisp_inv_o_sdisp_ground = SDISP.composition(sdisp_0_inv, sdisp_0, s_i_o=s_i_o)

    zero_disp = SDISP.generate_zero(shape)
    np.testing.assert_array_almost_equal(sdisp_o_sdisp_inv_ground.field, zero_disp.field, decimal=0)
    np.testing.assert_array_almost_equal(sdisp_inv_o_sdisp_ground.field, zero_disp.field, decimal=0)
    '''