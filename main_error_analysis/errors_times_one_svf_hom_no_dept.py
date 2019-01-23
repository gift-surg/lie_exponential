import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import time
from scipy.linalg import expm
import os
import copy
from sympy.core.cache import clear_cache
import pickle


def get_random_hom_a_matrices(d=2, scale_factor=None, sigma=1.0, special=False):
    """
    :param d: dimension of the homography in pgl by default or in psl
    :param scale_factor: scale factor of the homography
    :param sigma: sigma for the random values of the initial matrix.
    :param special: if the homography is in psl (True) or in pgl (False, default)
    :return: [h_g, h_a]random homography (in the GROUP) and the corresponding in the algebra h_g = expm(h_a)
    """
    h_al = sigma * np.random.randn(d + 1, d + 1)

    if scale_factor is not None:
        h_al = scale_factor * h_al

    if special:
        h_al[0, 0] = -1 * np.sum(np.diagonal(h_al)[1:])

    h_gp = expm(h_al)

    return h_al, h_gp


def generate_id(shape):
    """
    shape must have the form (x,y,z,t,d).
    The field is in matrix coordinates. The values of the domain is
    0 <= x < shape[0]
    0 <= y < shape[1]
    0 <= z < shape[2]  (optional if greater than 1)
    and d must have the appropriate dimension ALWAYS related to matrix coordinates.

    Parameters
    -------------
    :param shape: shape of a standard image (4 dim, [x,y,z,t]) or vector field (5 dim, [x,y,z,t,d]).
    :return:
    """

    if not len(shape) == 5:
        raise IOError("shape must be of the standard form (x,y,z,t,d) of len 5.")

    domain = [shape[j] for j in range(3) if shape[j] > 1]
    dim = len(domain)
    time_points = 1

    if not dim == shape[4]:
        raise IOError("To have the identity, shape must be of the standard form (x,y,z,t,d) "
                      "with d corresponding to the dimension.")

    if dim == 2:
        x = range(shape[0])
        y = range(shape[1])
        gx, gy = np.meshgrid(x, y)
        gx, gy = gx.T, gy.T

        id_field = np.zeros(list(gx.shape) + [1, time_points, 2])
        id_field[..., 0, :, 0] = np.repeat(gx, time_points).reshape(domain + [time_points])
        id_field[..., 0, :, 1] = np.repeat(gy, time_points).reshape(domain + [time_points])

    elif dim == 3:
        x = range(shape[0])
        y = range(shape[1])
        z = range(shape[2])
        gx, gy, gz = np.meshgrid(x, y, z)
        gx, gy, gz = gy, gx, gz  # swap!

        id_field = np.zeros(list(domain) + [time_points, 3])
        id_field[..., :, 0] = np.repeat(gx, time_points).reshape(domain + [time_points])
        id_field[..., :, 1] = np.repeat(gy, time_points).reshape(domain + [time_points])
        id_field[..., :, 2] = np.repeat(gz, time_points).reshape(domain + [time_points])

    else:
        raise IOError("Dimensions allowed: 2, 3")

    return id_field


def to_homogeneous(vf):
    """
    Adds the homogeneous coordinates to the given affine field.
    It changes the provided data structure instead of creating a new one.

    :return: field in homogeneous coordinates.

    NOTE: it works only for vector fields, so for fields with one more coordinate.
    If the given vector field was already affine nothing happen.
    """

    slice_shape = list(vf.shape[:])
    slice_shape[4] = 1

    vf = np.append(vf, np.ones(slice_shape), axis=4)

    return vf


def generate_from_projective_matrix_algebra(input_vol_ext, input_h):
    """
    See formula to generate these type of field.
    :param input_vol_ext:
    :param input_h: projective matrix in an algebra
    :return:
    """
    d = len(input_vol_ext)
    np.testing.assert_array_equal(input_h.shape, [d+1, d+1])

    if d == 2:

        idd = generate_id(shape=list(input_vol_ext) + [1, 1, d])
        vf = np.zeros(list(input_vol_ext) + [1, 1, d])

        idd = to_homogeneous(idd)

        x_intervals, y_intervals = input_vol_ext
        for i in range(x_intervals):
            for j in range(y_intervals):
                vf[i, j, 0, 0, 0] = input_h[0, :].dot(idd[i, j, 0, 0, :]) - i * input_h[2, :].dot(idd[i, j, 0, 0, :])
                vf[i, j, 0, 0, 1] = input_h[1, :].dot(idd[i, j, 0, 0, :]) - j * input_h[2, :].dot(idd[i, j, 0, 0, :])

        return vf

    elif d == 3:

        idd = generate_id(shape=list(input_vol_ext) + [1, d])
        vf = np.zeros(shape=list(input_vol_ext) + [1, d])

        to_homogeneous(idd)

        x_intervals, y_intervals, z_intervals = input_vol_ext
        for i in range(x_intervals):
            for j in range(y_intervals):
                for k in range(z_intervals):
                    vf[i, j, k, 0, 0] = input_h[0, :].dot(idd[i, j, k, 0, :]) - i * input_h[3, :].dot(idd[i, j, k, 0, :])
                    vf[i, j, k, 0, 1] = input_h[1, :].dot(idd[i, j, k, 0, :]) - j * input_h[3, :].dot(idd[i, j, k, 0, :])
                    vf[i, j, k, 0, 2] = input_h[2, :].dot(idd[i, j, k, 0, :]) - k * input_h[3, :].dot(idd[i, j, k, 0, :])

        return vf

    else:
        raise TypeError("Dimensions allowed: 2 or 3")


def generate_from_projective_matrix_group(input_vol_ext, input_exp_h):
    """
    See formula to generate these type of field.
    :param input_vol_ext:
    :param input_exp_h: projective matrix in a group
    :return:
    """
    d = len(input_vol_ext)
    np.testing.assert_array_equal(input_exp_h.shape, [d+1, d+1])

    if d == 2:

        vf = np.zeros(shape=list(input_vol_ext) + [1, 1, d])

        x_intervals, y_intervals = input_vol_ext
        for i in range(x_intervals):
            for j in range(y_intervals):

                s = input_exp_h.dot(np.array([i, j, 1]))[:]
                if abs(s[2]) > 1e-5:
                    # subtract the id to have the result in displacement coordinates
                    vf[i, j, 0, 0, :] = (s[0:2]/float(s[2])) - np.array([i, j])

        return vf

    elif d == 3:

        vf = np.zeros(shape=list(input_vol_ext) + [1, d])

        x_intervals, y_intervals, z_intervals = input_vol_ext
        for i in range(x_intervals):
            for j in range(y_intervals):
                for k in range(z_intervals):

                    s = input_exp_h.dot(np.array([i, j, k, 1]))[:]
                    if abs(s[3]) > 1e-5:
                        vf[i, j, k, 0, :] = (s[0:3]/float(s[3])) - np.array([i, j, k])

        return vf

    else:
        raise TypeError("Dimensions allowed: 2 or 3")


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
    x_1, y_1, z_1 = 50, 50, 1

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

    print '---------------------'
    print 'Computations started!'
    print '---------------------'

    # generate matrices homography
    scale_factor = 1. / (np.max(domain) * 8)
    hom_attributes = [d, scale_factor, 1, in_psl]

    h_a, h_g = get_random_hom_a_matrices(d=hom_attributes[0],
                                         scale_factor=hom_attributes[1],
                                         sigma=hom_attributes[2],
                                         special=hom_attributes[3])


    # generate SVF
    svf_0 = generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)
    disp_0 = generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)

    see_field(svf_0, input_color='r')
    see_field(disp_0, input_color='b')

    plt.show()