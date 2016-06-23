import numpy as np
import copy
import matplotlib.pyplot as plt

from utils.fields import Field


def see_field(input_obj,
              anatomical_plane='axial',
              h_slice=0, sample=(1, 1),
              window_title_input='quiver',
              title_input= '2d vector field',
              long_title=False,
              fig_tag=1, scale=1,
              subtract_id=False,
              input_color='b',
              annotate=None, annotate_position=(1, 1)):

    if not len(input_obj.shape) == 5:
        raise TypeError('Wrong input size for a field')

    if not (input_obj.dim == input_obj.shape[4] == 2 or input_obj.dim == input_obj.shape[4] == 3):
            raise TypeError('See field 2d works only for 2d to 2d fields.')

    id_field = input_obj.__class__.generate_id_from_obj(input_obj)

    fig = plt.figure(fig_tag)
    ax0 = fig.add_subplot(111)
    fig.canvas.set_window_title(window_title_input)

    input_field_copy = copy.deepcopy(input_obj)

    if subtract_id:
        input_field_copy -= id_field

    if anatomical_plane == 'axial':
        ax0.quiver(id_field.field[::sample[0], ::sample[1], h_slice, 0, 0],
                   id_field.field[::sample[0], ::sample[1], h_slice, 0, 1],
                   input_field_copy.field[::sample[0], ::sample[1], h_slice, 0, 0],
                   input_field_copy.field[::sample[0], ::sample[1], h_slice, 0, 1],
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

    return fig


def see_2_fields(input_obj_0, input_obj_1,
                 anatomical_plane_0='axial', anatomical_plane_1='axial',
                 h_slice_0=0, h_slice_1=0,
                 sample_0=(1, 1), sample_1=(1, 1),
                 window_title_input='quiver 2 screens',
                 title_input_0='Vector field', title_input_1='Vector field',
                 long_title_0=False, long_title_1=False,
                 fig_tag=1, scale_0=1, scale_1=1,
                 subtract_id_0=False, subtract_id_1=False, input_color='b'):
    """

    :param input_obj_0:
    :param input_obj_1:
    :param anatomical_plane_0:
    :param anatomical_plane_1:
    :param h_slice_0:
    :param h_slice_1:
    :param sample_0:
    :param sample_1:
    :param window_title_input:
    :param title_input_0:
    :param title_input_1:
    :param long_title_0:
    :param long_title_1:
    :param fig_tag:
    :param scale_0:
    :param scale_1:
    :param subtract_id_0:
    :param subtract_id_1:
    :param input_color:
    :return:
    """

    """
    if not len(input_obj_0.shape) == 5 and len(input_obj_1.shape) == 5:
        raise TypeError('Wrong input size for a field')

    if not (input_obj_0.dim == input_obj_0.shape[4] == 2 or input_obj_0.dim == input_obj_0.shape[4] == 3):
            raise TypeError('First input elements: see 2 fields works only for 2d to 2d Fields or children.')

    if not (input_obj_1.dim == input_obj_1.shape[4] == 2 or input_obj_1.dim == input_obj_1.shape[4] == 3):
            raise TypeError('First input elements: see 2 fields works only for 2d to 2d Fields or children.')
    """
    id_field_0 = input_obj_0.__class__.generate_id_from_obj(input_obj_0)  # other option is casting with Field()
    id_field_1 = input_obj_1.__class__.generate_id_from_obj(input_obj_1)

    input_field_0 = copy.deepcopy(input_obj_0)
    input_field_1 = copy.deepcopy(input_obj_1)

    if subtract_id_0:
        input_field_0 -= id_field_0

    if subtract_id_1:
        input_field_1 -= id_field_1

    fig = plt.figure(fig_tag)
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    fig.canvas.set_window_title(window_title_input)

    # figure 0
    if anatomical_plane_0 == 'axial':
        ax0.quiver(id_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   id_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   input_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   input_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')

    elif anatomical_plane_0 == 'sagittal':
        ax0.quiver(id_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   id_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   input_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   input_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')

    elif anatomical_plane_0 == 'coronal':
        ax0.quiver(id_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   id_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   input_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   input_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
    else:
        raise TypeError('anatomical_plane_0 must be axial, sagittal or coronal')

    if long_title_0:
        ax0.set_title(title_input_0 + ', ' + str(anatomical_plane_0) + ' plane, slice ' + str(h_slice_0))
    else:
        ax0.set_title(title_input_0)

    ax0.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.set_axisbelow(True)

    # figure 1
    if anatomical_plane_1 == 'axial':
        ax1.quiver(id_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   id_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   input_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   input_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')

    elif anatomical_plane_1 == 'sagittal':
        ax1.quiver(id_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
                   id_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
                   input_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
                   input_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')

    elif anatomical_plane_1 == 'coronal':
        ax1.quiver(id_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   id_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   input_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   input_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')
    else:
        raise TypeError('anatomical_plane_1 must be axial, sagittal or coronal')

    if long_title_1:
        ax1.set_title(title_input_1 + ', ' + str(anatomical_plane_1) + ' plane, slice ' + str(h_slice_1))
    else:
        ax1.set_title(title_input_1)

    ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.set_axisbelow(True)

    fig.set_tight_layout(True)


def see_field_subregion(input_obj,
                        anatomical_plane='axial',
                        h_slice=0, sample=(1, 1),
                        subregion=([0, 20], [0, 20]),
                        window_title_input='quiver',
                        title_input= 'Vector field',
                        long_title=False,
                        fig_tag=1, scale=1,
                        subtract_id=False):
    """
    Fields and children visualizer.
    :param input_obj: print a slice of a 2d or 3d svf
    :param anatomical_plane: if 2d is axial, with slice = 0
    :param h_slice: the slice of the plane we want to plot
    :param sample: does not plot all the vertical lines but a subspace
    :param subregion: subregion of the initial domain.
    :param long_title: add plane and slice information to the title
    :param fig_tag: change the tag of the figure for each new instance!
    The default will create problems for multiple images in the same module!
    :return: plot the quiver of the svf in the sub-region we are interested in.
    """
    if not len(input_obj.shape) == 5:
        raise TypeError('Wrong input size for a field')

    if not (input_obj.dim == input_obj.shape[4] == 2 or input_obj.dim == input_obj.shape[4] == 3):
            raise TypeError('See field 2d works only for 2d to 2d fields.')

    id_field = input_obj.__class__.generate_id_from_obj(input_obj)

    input_field_copy = copy.deepcopy(input_obj)

    if subtract_id:
        input_field_copy -= id_field

    fig = plt.figure(fig_tag)
    ax0 = fig.add_subplot(111)
    fig.canvas.set_window_title(window_title_input)

    if anatomical_plane == 'axial':
        ax0.quiver(id_field.field[subregion[0][0]:subregion[0][1]:sample[0],
                            subregion[1][0]:subregion[1][1]:sample[1],
                            h_slice, 0, 0],
                   id_field.field[subregion[0][0]:subregion[0][1]:sample[0],
                            subregion[1][0]:subregion[1][1]:sample[1],
                            h_slice, 0, 1],
                   input_field_copy.field[subregion[0][0]:subregion[0][1]:sample[0],
                                    subregion[1][0]:subregion[1][1]:sample[1],
                                    h_slice, 0, 0],
                   input_field_copy.field[subregion[0][0]:subregion[0][1]:sample[0],
                                    subregion[1][0]:subregion[1][1]:sample[1],
                                    h_slice, 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')

    elif anatomical_plane == 'sagittal':
        ax0.quiver(id_field.field[subregion[0][0]:subregion[0][1]:sample[0],
                            h_slice,
                            subregion[1][0]:subregion[1][1]:sample[1], 0, 0],
                   id_field.field[subregion[0][0]:subregion[0][1]:sample[0],
                            h_slice,
                            subregion[1][0]:subregion[1][1]:sample[1], 0, 1],
                   input_field_copy.field[subregion[0][0]:subregion[0][1]:sample[0],
                                    h_slice,
                                    subregion[1][0]:subregion[1][1]:sample[1], 0, 0],
                   input_field_copy.field[subregion[0][0]:subregion[0][1]:sample[0],
                                    h_slice,
                                    subregion[1][0]:subregion[1][1]:sample[1], 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')

    elif anatomical_plane == 'coronal':
        ax0.quiver(id_field.field[h_slice,
                                  subregion[0][0]:subregion[0][1]:sample[0],
                                  subregion[1][0]:subregion[1][1]:sample[1], 0, 0],
                   id_field.field[h_slice,
                                  subregion[0][0]:subregion[0][1]:sample[0],
                                  subregion[1][0]:subregion[1][1]:sample[1], 0, 1],
                   input_field_copy.field[h_slice,
                                          subregion[0][0]:subregion[0][1]:sample[0],
                                          subregion[1][0]:subregion[1][1]:sample[1], 0, 0],
                   input_field_copy.field[h_slice,
                                          subregion[0][0]:subregion[0][1]:sample[0],
                                          subregion[1][0]:subregion[1][1]:sample[1], 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')

    else:
        raise TypeError('Anatomical_plane must be axial, sagittal or coronal')

    if long_title:
        ax0.set_title(title_input + ', ' + str(anatomical_plane) + ' plane, slice ' + str(h_slice))
    else:
        ax0.set_title(title_input)

    fig.set_tight_layout(True)
    return fig


###  Methods for jacobian 2d ###


def see_jacobian_of_a_field_2d(input_jac,
                               sample=(1, 1),
                               window_title_input='quiver',
                               title_input= '2d vector field',
                               fig_tag=1, scale=1,
                               subtract_id=False):
    """
    Fields visualizer.
    :param sample: does not plot all the vertical lines but a subspace
    :return: plot the quiver of the svf in the sub-region we are interested in.
    """

    if not len(input_jac.shape) == 5:
        raise TypeError('Wrong input size for a field')

    if not input_jac.dim == 2 and input_jac.shape[4] == 4:
            raise TypeError('See field 2d works only for 2d to 4d fields (as 2d jacobian).')

    # Creating new sliced fields (jacobian is ordered column major):
    half_shape_jac_0 = list(input_jac.shape[:])
    half_shape_jac_0[4] = 2
    jac_f1 = Field.generate_id(shape=half_shape_jac_0)  # elem 0 and 2
    jac_f2 = Field.generate_id(shape=half_shape_jac_0)  # elem 1 and 3

    jac_f1.field[..., 0] = copy.deepcopy(input_jac.field[..., 0])
    jac_f1.field[..., 1] = copy.deepcopy(input_jac.field[..., 2])

    jac_f2.field[..., 0] = copy.deepcopy(input_jac.field[..., 1])
    jac_f2.field[..., 1] = copy.deepcopy(input_jac.field[..., 3])

    id_field = Field.generate_id_from_obj(jac_f1)

    # copy to subtract the identity without modifying the original copy:
    jac_f1_c = copy.deepcopy(jac_f1)
    jac_f2_c = copy.deepcopy(jac_f2)

    if subtract_id:
        jac_f1_c -= id_field
        jac_f2_c -= id_field

    fig = plt.figure(fig_tag)
    ax_f1 = fig.add_subplot(121)
    ax_f2 = fig.add_subplot(122)
    fig.canvas.set_window_title(window_title_input)

    ax_f1.quiver(id_field.field[::sample[0], ::sample[1], 0, 0, 0],
                     id_field.field[::sample[0], ::sample[1], 0, 0, 1],
                     jac_f1_c.field[::sample[0], ::sample[1], 0, 0, 0],
                     jac_f1_c.field[::sample[0], ::sample[1], 0, 0, 1],
                     color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')
    ax_f2.quiver(id_field.field[::sample[0], ::sample[1], 0, 0, 0],
                     id_field.field[::sample[0], ::sample[1], 0, 0, 1],
                     jac_f2_c.field[::sample[0], ::sample[1], 0, 0, 0],
                     jac_f2_c.field[::sample[0], ::sample[1], 0, 0, 1],
                     color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')

    ax_f1.set_title(title_input + ', df_1/dx vs df_1/dy')
    ax_f2.set_title(title_input + ', df_2/dx vs df_2/dy')

    fig.set_tight_layout(True)
    return fig


def see_2_jacobian_of_2_fields_2d(input_jac_0, input_jac_1,
                                  anatomical_plane_0='axial', anatomical_plane_1='axial',
                                  h_slice_0=0, h_slice_1=0,
                                  sample_0=(1, 1), sample_1=(1, 1),
                                  window_title_input='quiver 2 screens',
                                  title_input_0= 'Vector field',title_input_1= 'Vector field',
                                  long_title_0=False, long_title_1=False,
                                  fig_tag=1, scale_0=1, scale_1=1,
                                  subtract_id_0=False, subtract_id_1=False):
    """
    Visualiser for a couple of 2d jacobian vector fields (from 2d to 4d vector fields).
    4d fields are divided into 2 2d fields for its visualisation, as done in see_jacobian_of_a_field_2d
    for only one jacobian field.
    """

    # Test if the input are meaningful:
    if not len(input_jac_0.shape) == len(input_jac_1.shape) == 5:
        raise TypeError('Wrong input size for the given fields')

    if not (input_jac_0.dim == input_jac_1.dim == 2 and input_jac_0.shape[4] == input_jac_0.shape[4] == 4):
        raise TypeError('See field 2d works only for a couple of 2d to 4d fields (as 2d jacobian).')

    # Creating new sliced fields (jacobian is ordered column major):
    half_shape_jac_0 = list(input_jac_0.shape[:])
    half_shape_jac_0[4] = 2
    jac_0_f1 = Field.generate_id(shape=half_shape_jac_0)  # elem 0 and 2
    jac_0_f2 = Field.generate_id(shape=half_shape_jac_0)  # elem 1 and 3

    half_shape_jac_1 = list(input_jac_1.shape[:])
    half_shape_jac_1[4] = 2
    jac_1_f1 = Field.generate_id(shape=half_shape_jac_1)  # elem 0 and 2
    jac_1_f2 = Field.generate_id(shape=half_shape_jac_1)  # elem 1 and 3

    jac_0_f1.field[..., 0] = copy.deepcopy(input_jac_0.field[..., 0])
    jac_0_f1.field[..., 1] = copy.deepcopy(input_jac_0.field[..., 2])

    jac_0_f2.field[..., 0] = copy.deepcopy(input_jac_0.field[..., 1])
    jac_0_f2.field[..., 1] = copy.deepcopy(input_jac_0.field[..., 3])

    jac_1_f1.field[..., 0] = copy.deepcopy(input_jac_1.field[..., 0])
    jac_1_f1.field[..., 1] = copy.deepcopy(input_jac_1.field[..., 2])

    jac_1_f2.field[..., 0] = copy.deepcopy(input_jac_1.field[..., 1])
    jac_1_f2.field[..., 1] = copy.deepcopy(input_jac_1.field[..., 3])

    id_field_0 = Field.generate_id_from_obj(jac_0_f1)
    id_field_1 = Field.generate_id_from_obj(jac_1_f1)

    # copy to subtract the identity without modifying the original copy:
    jac_0_f1_c = copy.deepcopy(jac_0_f1)
    jac_0_f2_c = copy.deepcopy(jac_0_f2)
    jac_1_f1_c = copy.deepcopy(jac_1_f1)
    jac_1_f2_c = copy.deepcopy(jac_1_f2)

    if subtract_id_0:
        jac_0_f1_c -= id_field_0
        jac_0_f2_c -= id_field_0

    if subtract_id_1:
        jac_1_f1_c -= id_field_1
        jac_1_f2_c -= id_field_1

    # Initialize figure:
    fig = plt.figure(fig_tag)
    ax0 = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)

    fig.canvas.set_window_title(window_title_input)

    # figure 0: jac 0 f1 and f2
    if anatomical_plane_0 == 'axial':
        ax0.quiver(id_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   id_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   jac_0_f1_c.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   jac_0_f1_c.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
        ax1.quiver(id_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   id_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   jac_0_f2_c.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   jac_0_f2_c.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')

    elif anatomical_plane_0 == 'sagittal':
        ax0.quiver(id_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   id_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   jac_0_f1_c.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   jac_0_f1_c.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
        ax1.quiver(id_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
                   id_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
                   jac_0_f2_c.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
                   jac_0_f2_c.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')

    elif anatomical_plane_0 == 'coronal':
        ax0.quiver(id_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   id_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   jac_0_f1_c.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   jac_0_f1_c.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
        ax1.quiver(id_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   id_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   jac_0_f2_c.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   jac_0_f2_c.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')
    else:
        raise TypeError('anatomical_plane_0 must be axial, sagittal or coronal')

    if long_title_0:
        ax0.set_title(title_input_0 + 'df_1/dx vs df_1/dy' + ', ' + str(anatomical_plane_0) + ' plane, slice ' + str(h_slice_0))
        ax1.set_title(title_input_0 + 'df_2/dx vs df_2/dy' + ', ' + str(anatomical_plane_1) + ' plane, slice ' + str(h_slice_1))
    else:
        ax0.set_title(title_input_0 + 'df_1/dx vs df_1/dy')
        ax1.set_title(title_input_0 + 'df_2/dx vs df_2/dy')

    # figure 2: jac 1 f1 and f2
    if anatomical_plane_0 == 'axial':
        ax2.quiver(id_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   id_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   jac_1_f1_c.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   jac_1_f1_c.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
        ax3.quiver(id_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   id_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   jac_1_f2_c.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   jac_1_f2_c.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')

    elif anatomical_plane_0 == 'sagittal':
        ax2.quiver(id_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   id_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   jac_1_f1_c.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   jac_1_f1_c.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
        ax3.quiver(id_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
                   id_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
                   jac_1_f2_c.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
                   jac_1_f2_c.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')

    elif anatomical_plane_0 == 'coronal':
        ax2.quiver(id_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   id_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   jac_1_f1_c.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   jac_1_f1_c.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
        ax3.quiver(id_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   id_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   jac_1_f2_c.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   jac_1_f2_c.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')
    else:
        raise TypeError('anatomical_plane_0 must be axial, sagittal or coronal')

    if long_title_1:
        ax2.set_title(title_input_1 + 'df_1/dx vs df_1/dy' + ', ' + str(anatomical_plane_0) + ' plane, slice ' + str(h_slice_0))
        ax3.set_title(title_input_1 + 'df_2/dx vs df_2/dy' + ', ' + str(anatomical_plane_1) + ' plane, slice ' + str(h_slice_1))
    else:
        ax2.set_title(title_input_1 + 'df_1/dx vs df_1/dy')
        ax3.set_title(title_input_1 + 'df_2/dx vs df_2/dy')

    fig.set_tight_layout(True)
    return fig


def triptych_image_quiver_image(image_1,
                                def_field,
                                image_2,
                                fig_tag=5,
                                input_fig_size=(15, 5),
                                window_title_input='triptych',
                                interval_svf=1):

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    ax_1 = plt.subplot(131)
    ax_1.imshow(image_1, cmap='Greys',  interpolation='nearest', origin='lower')
    ax_1.axes.xaxis.set_ticklabels([])
    ax_1.axes.yaxis.set_ticklabels([])
    ax_1.set_xlabel('(grid)', fontdict=font)
    ax_1.set_aspect('equal')

    ax_2 = plt.subplot(132)
    x, y = np.meshgrid(np.arange(def_field.shape[0]), np.arange(def_field.shape[1]))
    ax_2.quiver(y[::interval_svf, ::interval_svf],
                x[::interval_svf, ::interval_svf],
                def_field[::interval_svf, ::interval_svf, 0, 0, 0],
                def_field[::interval_svf, ::interval_svf, 0, 0, 1], scale=1, scale_units='xy')
    ax_2.axes.xaxis.set_ticklabels([])
    ax_2.axes.yaxis.set_ticklabels([])
    ax_2.set_xlabel('(transformation)', fontdict=font)
    ax_2.set_aspect('equal')

    ax_3 = plt.subplot(133)
    ax_3.imshow(image_2, cmap='Greys',  interpolation='none', origin='lower')
    ax_3.axes.xaxis.set_ticklabels([])
    ax_3.axes.yaxis.set_ticklabels([])
    ax_3.set_xlabel('(transformed grid)', fontdict=font)
    ax_3.set_aspect('equal')

    return fig


def triptych_quiver_quiver_quiver(vector_field_1,
                                  vector_field_2,
                                  vector_field_3,
                                  fig_tag=5,
                                  input_fig_size=(15, 5),
                                  window_title_input='triptych'):

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    ax_1 = plt.subplot(131)
    x1, y1 = np.meshgrid(np.arange(vector_field_1.shape[0]), np.arange(vector_field_1.shape[1]))
    ax_1.quiver(y1,
                x1,
                vector_field_1[:, :, 0, 0, 0],
                vector_field_1[:, :, 0, 0, 1], scale=1, scale_units='xy')
    ax_1.axes.xaxis.set_ticklabels([])
    ax_1.axes.yaxis.set_ticklabels([])
    ax_1.set_xlabel('(a)', fontdict=font)
    ax_1.set_aspect('equal')

    ax_2 = plt.subplot(132)
    x2, y2 = np.meshgrid(np.arange(vector_field_2.shape[0]), np.arange(vector_field_2.shape[1]))
    ax_2.quiver(y2,
                x2,
                vector_field_2[:, :, 0, 0, 0],
                vector_field_2[:, :, 0, 0, 1], scale=1, scale_units='xy')
    ax_2.axes.xaxis.set_ticklabels([])
    ax_2.axes.yaxis.set_ticklabels([])
    ax_2.set_xlabel('(b)', fontdict=font)
    ax_2.set_aspect('equal')

    ax_3 = plt.subplot(133)
    x3, y3 = np.meshgrid(np.arange(vector_field_3.shape[0]), np.arange(vector_field_3.shape[2]))
    ax_3.quiver(y3,
                x3,
                vector_field_3[:, 0, :, 0, 0],
                vector_field_3[:, 0, :, 0, 1], scale=1, scale_units='xy')
    ax_3.axes.xaxis.set_ticklabels([])
    ax_3.axes.yaxis.set_ticklabels([])
    ax_3.set_xlabel('(c)', fontdict=font)
    ax_3.set_aspect('equal')

    return fig


def quadrivium_quiver(vector_field_1,
                      vector_field_2,
                      vector_field_3,
                      vector_field_4,
                      fig_tag=5,
                      input_fig_size=(18, 4),
                      window_title_input='triptych'):

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    ax_1 = plt.subplot(141)
    x1, y1 = np.meshgrid(np.arange(vector_field_1.shape[0]), np.arange(vector_field_1.shape[1]))
    ax_1.quiver(y1,
                x1,
                vector_field_1[:, :, 0, 0, 0],
                vector_field_1[:, :, 0, 0, 1], scale=1, scale_units='xy')
    ax_1.axes.xaxis.set_ticklabels([])
    ax_1.axes.yaxis.set_ticklabels([])
    ax_1.set_xlabel('(a)', fontdict=font)
    ax_1.set_aspect('equal')

    ax_2 = plt.subplot(142)
    x2, y2 = np.meshgrid(np.arange(vector_field_2.shape[0]), np.arange(vector_field_2.shape[1]))
    ax_2.quiver(y2,
                x2,
                vector_field_2[:, :, 0, 0, 0],
                vector_field_2[:, :, 0, 0, 1], scale=1, scale_units='xy')
    ax_2.axes.xaxis.set_ticklabels([])
    ax_2.axes.yaxis.set_ticklabels([])
    ax_2.set_xlabel('(b)', fontdict=font)
    ax_2.set_aspect('equal')

    ax_3 = plt.subplot(143)
    x3, y3 = np.meshgrid(np.arange(vector_field_3.shape[0]), np.arange(vector_field_3.shape[1]))
    ax_3.quiver(y3,
                x3,
                vector_field_3[:, :, 0, 0, 0],
                vector_field_3[:, :, 0, 0, 1], scale=1, scale_units='xy')
    ax_3.axes.xaxis.set_ticklabels([])
    ax_3.axes.yaxis.set_ticklabels([])
    ax_3.set_xlabel('(c)', fontdict=font)
    ax_3.set_aspect('equal')

    ax_4 = plt.subplot(144)
    x4, y4 = np.meshgrid(np.arange(vector_field_4.shape[0]), np.arange(vector_field_4.shape[2]))

    print x4.size, y4.size

    ax_4.quiver(y4,
                x4,
                vector_field_4[:, 0, :, 0, 0],
                vector_field_4[:, 0, :, 0, 1], scale=1, scale_units='xy')
    ax_4.axes.xaxis.set_ticklabels([])
    ax_4.axes.yaxis.set_ticklabels([])
    ax_4.set_xlabel('(d)', fontdict=font)
    ax_4.set_aspect('equal')

    return fig


def triptych_quiver_quiver_image(vector_field_1,
                                 vector_field_2,
                                 image_1,
                                 fig_tag=5,
                                 input_fig_size=(15, 5),
                                 window_title_input='triptych'):

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    ax_1 = plt.subplot(131)
    x1, y1 = np.meshgrid(np.arange(vector_field_1.shape[0]), np.arange(vector_field_1.shape[1]))
    ax_1.quiver(y1,
                x1,
                vector_field_1[:, :, 0, 0, 0],
                vector_field_1[:, :, 0, 0, 1], scale=1, scale_units='xy')
    ax_1.axes.xaxis.set_ticklabels([])
    ax_1.axes.yaxis.set_ticklabels([])
    ax_1.set_xlabel('(a)', fontdict=font)
    ax_1.set_aspect('equal')

    ax_2 = plt.subplot(132)
    x2, y2 = np.meshgrid(np.arange(vector_field_2.shape[0]), np.arange(vector_field_2.shape[1]))
    ax_2.quiver(y2,
                x2,
                vector_field_2[:, :, 0, 0, 0],
                vector_field_2[:, :, 0, 0, 1], scale=1, scale_units='xy')
    ax_2.axes.xaxis.set_ticklabels([])
    ax_2.axes.yaxis.set_ticklabels([])
    ax_2.set_xlabel('(b)', fontdict=font)
    ax_2.set_aspect('equal')

    ax_3 = plt.subplot(133)
    ax_3.imshow(image_1, cmap='Greys',  interpolation='nearest', origin='lower')
    ax_3.axes.xaxis.set_ticklabels([])
    ax_3.axes.yaxis.set_ticklabels([])
    ax_3.set_xlabel('(c)', fontdict=font)
    ax_3.set_aspect('equal')

    return fig
