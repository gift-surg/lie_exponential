import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import time
import os
import copy
from sympy.core.cache import clear_cache
import pickle

from utils.projective_algebras import get_random_hom_a_matrices
from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from utils.path_manager import path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables

from visualizer.graphs_and_stats_new import plot_custom_bar_chart_with_error

from aaa_general_controller import methods_t_s

"""
Module for the computation of the error of the exponential map.
Svf involved is one 2d SVF generated with matrix in se2_a.
"""

def id_eulerian(omega, t=1):
    """
    :param omega: discretized domain of the vector field
    :param t: number of timepoints
    :return: identity vector field of given domain and timepoints, in Eulerian coordinates.
    """
    omega = list(omega)
    v_shape = omega + [1, 1, 2]
    id_vf = np.zeros(v_shape)

    if d == 2:
        x = range(v_shape[0])
        y = range(v_shape[1])
        gx, gy = np.meshgrid(x, y, indexing='ij')

        id_vf[..., 0, :, 0] = np.repeat(gx, t).reshape(omega + [t])
        id_vf[..., 0, :, 1] = np.repeat(gy, t).reshape(omega + [t])

    elif d == 3:
        x = range(v_shape[0])
        y = range(v_shape[1])
        z = range(v_shape[2])
        gx, gy, gz = np.meshgrid(x, y, z, indexing='ij')

        id_vf[..., :, 0] = np.repeat(gx, t).reshape(omega + [t])
        id_vf[..., :, 1] = np.repeat(gy, t).reshape(omega + [t])
        id_vf[..., :, 2] = np.repeat(gz, t).reshape(omega + [t])

    return id_vf


def see_field(input_vf,
              anatomical_plane='axial',
              h_slice=0, sample=(1, 1),
              window_title_input='quiver',
              title_input='2d vector field',
              long_title=False,
              fig_tag=1,
              scale=1,
              subtract_id=False,
              input_color='b',
              annotate=None, annotate_position=(1, 1)):

    id_field = id_eulerian(input_vf.shape[:2])

    fig = plt.figure(fig_tag)
    ax0 = fig.add_subplot(111)
    fig.canvas.set_window_title(window_title_input)

    input_field_copy = copy.deepcopy(input_vf)

    if subtract_id:
        input_field_copy -= id_field

    if anatomical_plane == 'axial':
        ax0.quiver(id_field[::sample[0], ::sample[1], h_slice, 0, 0],
                   id_field[::sample[0], ::sample[1], h_slice, 0, 1],
                   input_field_copy[::sample[0], ::sample[1], h_slice, 0, 0],
                   input_field_copy[::sample[0], ::sample[1], h_slice, 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, scale=scale, scale_units='xy', units='xy',
                   angles='xy')
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')

    elif anatomical_plane == 'sagittal':
        ax0.quiver(id_field[::sample[0], h_slice, ::sample[1], 0, 0],
                   id_field[::sample[0], h_slice, ::sample[1], 0, 1],
                   input_field_copy[::sample[0], h_slice, ::sample[1], 0, 0],
                   input_field_copy[::sample[0], h_slice, ::sample[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale,
                   scale_units='xy')

    elif anatomical_plane == 'coronal':
        ax0.quiver(id_field[h_slice, ::sample[0], ::sample[1], 0, 0],
                   id_field[h_slice, ::sample[0], ::sample[1], 0, 1],
                   input_field_copy[h_slice, ::sample[0], ::sample[1], 0, 0],
                   input_field_copy[h_slice, ::sample[0], ::sample[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale,
                   scale_units='xy')
    else:
        raise TypeError('Anatomical_plane must be axial, sagittal or coronal')

    ax0.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.set_axisbelow(True)

    if long_title:
        ax0.set_title(title_input + ', ' + str(anatomical_plane) + ' plane, slice ' + str(h_slice))
    else:
        ax0.set_title(title_input)

    if annotate is not None:
        ax0.text(annotate_position[0], annotate_position[1], annotate)

    plt.axes().set_aspect('equal', 'datalim')


if __name__ == "__main__":

    clear_cache()

    random_seed = 0

    if random_seed > 0:
        np.random.seed(random_seed)

    s_i_o = 3
    pp = 2

    # Parameters SVF:
    x_1, y_1, z_1 = 30, 30, 1

    in_psl = False

    if z_1 == 1:
        d = 2
        domain = (x_1, y_1)
        shape = list(domain) + [1, 1, 2]

        # center of the homography
        x_c = x_1 / 2
        y_c = y_1 / 2
        z_c = 1

        projective_center = [x_c, y_c, z_c]

    else:
        d = 3
        domain = (x_1, y_1, z_1)
        shape = list(domain) + [1, 3]

        # center of the homography
        x_c = x_1 / 2
        y_c = y_1 / 2
        z_c = z_1 / 2
        w_c = 1

        projective_center = [x_c, y_c, z_c, w_c]

    # import methods from external file aaa_general_controller
    methods = methods_t_s

    indexes_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
    num_method_considered    = len(indexes_methods_considered)

    names_method_considered  = [methods[j][0] for j in indexes_methods_considered]
    steps_methods_considered = [methods[j][2] for j in indexes_methods_considered]

    ###########################
    ### Model: computations ###
    ###########################

    print '---------------------'
    print 'Computations started!'
    print '---------------------'

    # init structures
    errors = np.zeros(num_method_considered)
    res_time = np.zeros(num_method_considered)

    # generate matrices homography
    scale_factor = 1. / (np.max(domain) * 10)
    hom_attributes = [d, scale_factor, 1, in_psl]

    h_a, h_g = get_random_hom_a_matrices(d=hom_attributes[0],
                                         scale_factor=hom_attributes[1],
                                         sigma=hom_attributes[2],
                                         special=hom_attributes[3])

    '''
    h_a = np.array([[  9.85981526e-01,   1.78819744e-02,  -3.86350073e-02],
                    [ -2.09596248e-04,   1.00200981e+00,  -1.80021390e-02],
                    [  6.63778964e-03,   2.71371314e-03,   9.06484972e-01]])

    h_g = expm(h_a)
    '''

    # generate SVF
    svf_0 = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)
    disp_0 = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)

    see_field(svf_0.field, input_color='r')
    see_field(disp_0.field, input_color='b')

    plt.show()
