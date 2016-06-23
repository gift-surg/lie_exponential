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
from utils.image import Image

from utils.aux_functions import grid_generator
from utils.helper import generate_position_from_displacement
from utils.resampler import NearestNeighbourResampler

from visualizer.fields_comparisons import see_overlay_of_n_fields
from visualizer.graphs_and_stats_new import plot_error_linewise
from visualizer.fields_at_the_window import triptych_image_quiver_image


if __name__ == "__main__":

    clear_cache()

    #### Data settings:

    random_seed = 0

    if random_seed > 0:
        np.random.seed(random_seed)

    x_1, y_1, z_1 = 51, 51, 1

    # point to select for tests:
    x, y, z = 5, 5, 0

    if z_1 == 1:
        domain = (x_1, y_1)
        shape = list(domain) + [1, 1, 2]
    else:
        domain = (x_1, y_1, z_1)
        shape = list(domain) + [1, 3]

    d = len(domain)

    is_centered = True
    in_psl      = False
    with_hack   = False
    is_scaled   = True
    '''

    ### Generate matrix:
    h = np.random.randn(d+1,  d+1)

    if with_hack is True:
        n = np.max(h)
        h[d, :-1] = [n] * d

    print 'h = '
    print h

    if is_scaled:
        scale_factor = 1/float(np.max(domain)*10)
        h = copy.deepcopy(scale_factor * h)

    if is_centered:
        if d == 2:
            x_c = x_1/2
            y_c = y_1/2
            z_c = 1

            h[0, 2] = ((1 - h[0, 0]) * x_c - h[0, 1] * y_c) / float(z_c)
            h[1, 0] = ((1 - h[1, 1]) * y_c - h[1, 2] * z_c) / float(x_c)
            h[2, 1] = ((1 - h[2, 2]) * z_c - h[2, 0] * x_c) / float(y_c)

        elif d == 3:
            x_c = x_1/2
            y_c = y_1/2
            z_c = z_1/2
            w_c = 1

            h[0, 3] = ((1 - h[0, 0]) * x_c - h[0, 1] * y_c - h[0, 2] * z_c) / float(w_c)
            h[1, 0] = ((1 - h[1, 1]) * y_c - h[1, 2] * z_c - h[1, 3] * w_c) / float(x_c)
            h[2, 1] = ((1 - h[2, 2]) * z_c - h[2, 3] * w_c - h[2, 0] * x_c) / float(y_c)
            h[3, 2] = ((1 - h[3, 3]) * w_c - h[3, 0] * x_c - h[3, 1] * y_c) / float(z_c)

        else:
            raise IOError

    if 1:  # Twice the process:

        if is_scaled:
            scale_factor = 1/float(100000)
            h = copy.deepcopy(scale_factor * h)

        if is_centered:
            if d == 2:
                x_c = x_1/2
                y_c = y_1/2
                z_c = 1

                h[0, 2] = ((1 - h[0, 0]) * x_c - h[0, 1] * y_c) / float(z_c)
                h[1, 0] = ((1 - h[1, 1]) * y_c - h[1, 2] * z_c) / float(x_c)
                h[2, 1] = ((1 - h[2, 2]) * z_c - h[2, 0] * x_c) / float(y_c)

            elif d == 3:
                x_c = x_1/2
                y_c = y_1/2
                z_c = z_1/2
                w_c = 1

                h[0, 3] = ((1 - h[0, 0]) * x_c - h[0, 1] * y_c - h[0, 2] * z_c) / float(w_c)
                h[1, 0] = ((1 - h[1, 1]) * y_c - h[1, 2] * z_c - h[1, 3] * w_c) / float(x_c)
                h[2, 1] = ((1 - h[2, 2]) * z_c - h[2, 3] * w_c - h[2, 0] * x_c) / float(y_c)
                h[3, 2] = ((1 - h[3, 3]) * w_c - h[3, 0] * x_c - h[3, 1] * y_c) / float(z_c)

            else:
                raise IOError


    if in_psl:
        h[d, d] = -1 * np.sum(np.diagonal(h)[:-1])
    '''

    '''
    h = np.array([[3.45559391e-03,   1.45725883e-03,   7.25481688e-04,   2.48590416e+01],
                           [9.99117980e-01,  -1.94313172e-05,   9.21039653e-04,  -4.89706281e-04],
                           [-4.92876982e-04,   9.97246982e-01,   3.26170225e-03,  -3.95192812e-04],
                           [-2.63231246e-03,   1.49563981e-03,   4.12091152e-02,  -1.81106325e-03]])


    h = np.array([[ -3.07318842e-02,   3.70236155e-03,   2.57408903e+01,  -3.80158044e-02],
                             [  1.02298491e+00,  -1.60662818e-02,  -1.08934384e-02,  -9.56423115e-03],
                             [ -2.83419473e-02,  6.05854745e-01,   3.75621262e-02,  -1.25182874e-03],
                             [ -2.06521443e-02,  -2.76797866e-02,   1.46778856e-01,   6.61542591e-03]])
    '''

    sigma = 1
    scale_factor = 1/float(x_1*5)
    h = sigma*np.random.randn(d+1,  d+1)
    h = scale_factor * h
    h[-1, :] = np.abs(h[-1, :])


    h_g = expm(h)

    h_g = expm(h)

    print 'h = '
    print h

    print 'H = '
    print h_g

    ### generate the corresponding svf (sanity check to see the invariance under linear translation)
    svf_h = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h)
    ### generate the corresponding disp
    disp_h_ground = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)
    ### generate the same with a numerical method
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
                        input_parameters=None)

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