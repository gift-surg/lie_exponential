"""
Visualizer for fields and integral curves, when the
integrator method exponential_scipy with return_integral_curves=True
is provided.
"""
import numpy as np
import copy
import matplotlib.pyplot as plt

from utils.fields import Field


def see_overlay_of_n_fields_and_flow(list_of_obj,
                                     list_of_integral_curves,
                                     list_of_alpha_for_obj=None,
                                     alpha_for_integral_curves=None,
                                     anatomical_plane='axial',
                                     h_slice=0, sample=(1, 1),
                                     window_title_input='quiver',
                                     title_input= '2d vector field',
                                     long_title=False,
                                     fig_tag=1, scale=1,
                                     subtract_id=None,
                                     input_color=('r', 'b'),
                                     input_label=None,
                                     see_tips=False):
    """
    :param list_of_obj: list of fields or children, or None.
    :param anatomical_plane:
    :param h_slice:
    :param sample:
    :param window_title_input:
    :param title_input:
    :param long_title:
    :param fig_tag:
    :param scale:
    :param subtract_id:
    :return:
    """

    fig = plt.figure(fig_tag)
    ax0 = fig.add_subplot(111)
    fig.canvas.set_window_title(window_title_input)

    if subtract_id is None:
        subtract_id = [False, ] * len(list_of_obj)

    if list_of_alpha_for_obj is None:
        list_of_alpha_for_obj = [0.8, ] * len(list_of_obj)
    if alpha_for_integral_curves is None:
        alpha_for_integral_curves = 0.8

    for num_obj, input_obj in enumerate(list_of_obj):

        if input_obj is not None:

            if not len(input_obj.shape) == 5:
                raise TypeError('Wrong input size for a the field' + str(input_obj))

            if not (input_obj.dim == input_obj.shape[4] == 2 or input_obj.dim == input_obj.shape[4] == 3):
                raise TypeError('See field 2d works only for 2d to 2d fields.')

            id_field = input_obj.__class__.generate_id_from_obj(input_obj)
            input_field_copy = copy.deepcopy(input_obj)

            if subtract_id[num_obj]:
                input_field_copy -= id_field

        if anatomical_plane == 'axial':
            q = ax0.quiver(id_field.field[::sample[0], ::sample[1], h_slice, 0, 0],
                           id_field.field[::sample[0], ::sample[1], h_slice, 0, 1],
                           input_field_copy.field[::sample[0], ::sample[1], h_slice, 0, 0],
                           input_field_copy.field[::sample[0], ::sample[1], h_slice, 0, 1],
                           color=input_color[num_obj], linewidths=0.01, width=0.03, scale=scale,
                           scale_units='xy', units='xy', angles='xy',
                           alpha=list_of_alpha_for_obj[num_obj])

        elif anatomical_plane == 'sagittal':
            q = ax0.quiver(id_field[::sample[0], h_slice, ::sample[1], 0, 0],
                           id_field[::sample[0], h_slice, ::sample[1], 0, 1],
                           input_field_copy[::sample[0], h_slice, ::sample[1], 0, 0],
                           input_field_copy[::sample[0], h_slice, ::sample[1], 0, 1],
                           color=input_color[num_obj], linewidths=1, units='xy', angles='xy',
                           scale=scale, scale_units='xy',
                           alpha=list_of_alpha_for_obj[num_obj])

        elif anatomical_plane == 'coronal':
            q = ax0.quiver(id_field[h_slice, ::sample[0], ::sample[1], 0, 0],
                           id_field[h_slice, ::sample[0], ::sample[1], 0, 1],
                           input_field_copy[h_slice, ::sample[0], ::sample[1], 0, 0],
                           input_field_copy[h_slice, ::sample[0], ::sample[1], 0, 1],
                           color=input_color[num_obj], linewidths=1, units='xy', angles='xy',
                           scale=scale, scale_units='xy',
                           alpha=list_of_alpha_for_obj[num_obj])
        else:
            raise TypeError('Anatomical_plane must be axial, sagittal or coronal')

        # add the integral curves:
        for fl in list_of_integral_curves:
            ax0.plot(fl[:, 0], fl[:, 1], color='m', lw=1, alpha=alpha_for_integral_curves)
            if see_tips:
                ax0.plot(fl[0, 0], fl[0, 1], 'go', alpha=0.3)
                ax0.plot(fl[fl.shape[0]-1, 0], fl[fl.shape[0]-1, 1], 'mo', alpha=0.5)


        if input_label is not None:
            # TODO to add a legend in some painful way...!
            ax0.quiverkey(q, 1.2, 0.515, 2, input_label[num_obj], coordinates='data')

        ax0.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax0.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax0.set_axisbelow(True)

        if long_title:
            ax0.set_title(title_input + ', ' + str(anatomical_plane) + ' plane, slice ' + str(h_slice))
        else:
            ax0.set_title(title_input)

    fig.set_tight_layout(True)