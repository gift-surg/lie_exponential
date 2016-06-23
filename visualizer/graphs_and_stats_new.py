import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

####################
# 8 custom methods #
####################


def plot_custom_bar_chart_with_error(input_data,
                                     input_names=None,
                                     fig_tag=1,
                                     input_fig_size=(9, 7),
                                     titles=('bar plot', 'field'),
                                     window_title_input='bar plot',
                                     color_bar='b',
                                     kind=None,
                                     additional_field=None,
                                     input_parameters=None,
                                     log_scale=False,
                                     add_extra_numbers=None):

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    fig.canvas.set_window_title(window_title_input)

    ax_bar = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
    if additional_field is not None:
        ax_field   = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)

    index = np.arange(len(input_data))
    bar_width = 0.35

    # bar plot
    ax_bar.bar(index, list(input_data), bar_width,
                     color=color_bar)

    ax_bar.set_title(titles[0])

    ax_bar.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_bar.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_bar.set_axisbelow(True)

    ax_bar.set_xlabel('Methods')
    ax_bar.set_ylabel('Error (pixel)')
    if log_scale:
        ax_bar.set_yscale('log')
        ax_bar.set_ylabel('Error - log scale - (pixel)')

    ax_bar.set_xlim(0 - bar_width, len(input_data) - bar_width)

    if input_names is not None:
        ax_bar.set_xticks(index, minor=False)
        xtick_names = plt.setp(ax_bar, xticklabels=input_names)
        plt.setp(xtick_names, rotation=45, fontsize=12)

    ax_bar.grid(True)

    # fig.text(0.5, 0.04, 'Methods', ha='center')
    # fig.text(0.01, 0.5, 'Errors', va='center', rotation='vertical')

    # right side of the figure:
    # Quiver
    if additional_field is not None:
        ax_field.set_title(titles[1])
        X, Y = np.meshgrid(np.arange(additional_field.shape[0]), np.arange(additional_field.shape[1]))
        ax_field.quiver(Y,
                        X,
                        additional_field[:, :, 0, 0, 0],
                        additional_field[:, :, 0, 0, 1],
                        color='r', linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')

    # Annotate computational time:
    if add_extra_numbers is not None:
        y_val_a, y_val_b = ax_bar.get_ylim()
        for i in range(len(input_data)):
            ax_bar.text(i + bar_width/2, 0.85*(y_val_b - y_val_a), str(np.around(add_extra_numbers[i], decimals=9)),
                        horizontalalignment='center', size='small',
                        color='k', rotation=90)

    # Text on the figure customise this part for the need!
    # 6 options 'one_SE2', 'multi_SE2', 'one_GAUSS', 'multi_GAUSS', 'one_REALI', 'multi_REALI'
    if kind is not None:

        dom = tuple([int(j) for j in input_parameters[:3]])
        fig.text(.78, .80, r'Domain = ' + str(dom))

        if kind == 'one_SE2':
            fig.text(.765, .85,  r'SE(2) generated SVF: ')
            fig.text(.78, .75, r'$\theta = $ ' + str(input_parameters[3]))
            fig.text(.78, .70, r'$t_x = $ ' + str(input_parameters[4]))
            fig.text(.78, .65, r'$t_y = $ ' + str(input_parameters[5]))

        elif kind == 'one_HOM':
            fig.text(.765, .85,  r'HOM generated SVF: ')
            fig.text(.78, .75, r'center: ' + str(input_parameters[3]))
            fig.text(.78, .70, r'kind: ' + str(input_parameters[4]))
            fig.text(.78, .65, r'scale_factor: ' + str(input_parameters[5]))
            fig.text(.78, .60, r'sigma: ' + str(input_parameters[6]))
            fig.text(.78, .55, r'in_psl: ' + str(input_parameters[7]))

        elif kind == 'one_GAUSS':
            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'$\sigma_i = $ ' + str(input_parameters[3]))
            fig.text(.78, .70, r'$\sigma_g = $ ' + str(input_parameters[4]))
            fig.text(.78, .60, r'Ground truth, steps ')
            fig.text(.78, .55, str(input_parameters[5]) + ' ' + str(input_parameters[6]))

        elif kind == 'one_REAL':
            fig.text(.78, .85, r'id element: ' + str(input_parameters[3]))
            fig.text(.78, .60, r'Ground truth method ')
            fig.text(.78, .55, str(input_parameters[4]))

        else:
            raise Warning('Kind not recognized.')

    fig.set_tight_layout(True)
    return fig


def plot_custom_boxplot(input_data,
                        input_names=None,
                        fig_tag=1,
                        input_fig_size=(11, 7.5),
                        x_axis_label='Methods',
                        y_axis_label='Error (pixel)',
                        input_titles=('Error', 'field'),
                        window_title_input='boxplot plot',
                        kind=None,
                        additional_field=None,
                        input_parameters=None,
                        log_scale=False,
                        annotate_mean=True,
                        add_extra_annotation=None):
    """

    :param input_data: list of lists, one for each block!
    :param input_names:
    :param fig_tag:
    :param x_axis_label:
    :param y_axis_label:
    :param input_fig_size:
    :param input_titles:
    :param window_title_input:
    :param kind:
    :param additional_field:
    :param input_parameters:
    :param log_scale:
    :param annotate_mean:
    :param add_extra_annotation:
    :return:
    """

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    if input_parameters is None:
        ax_box = plt.subplot(111)
    else:
        ax_box = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
        if additional_field is not None:
            ax_field   = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)

    num_boxes = len(input_data)

    index_boxes = np.arange(1, num_boxes+1)

    bp = ax_box.boxplot(input_data, notch=False, patch_artist=False,  sym='+', vert=1, whis=1.5)

    # set the colors:
    plt.setp(bp['boxes'], color='blue')
    plt.setp(bp['whiskers'], color='blue')
    plt.setp(bp['fliers'], color='red', marker='+')

    ax_box.set_title(input_titles[0])

    ax_box.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_box.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_box.set_axisbelow(True)

    ax_box.set_xlabel(x_axis_label, fontdict=font, labelpad=18)
    ax_box.set_ylabel(y_axis_label, fontdict=font, labelpad=10)
    if log_scale:
        ax_box.set_yscale('log')
        ax_box.set_ylabel(y_axis_label + ' log-scale')

    # ax_box.set_xlim(0 - 0.5, num_boxes + 0.5)

    if input_names is not None:
        ax_box.set_xticks(index_boxes, minor=False)
        xtick_names = plt.setp(ax_box, xticklabels=input_names)
        plt.setp(xtick_names, rotation=45, fontsize=12)

    #ax_box.grid(True)

    # right side of the figure:
    # Quiver
    if additional_field is not None:
        ax_field.set_title(input_titles[1])
        xx, yy = np.meshgrid(np.arange(additional_field.shape[0]), np.arange(additional_field.shape[1]))
        ax_field.quiver(yy,
                        xx,
                        additional_field[:, :, 0, 0, 0],
                        additional_field[:, :, 0, 0, 1],
                        color='r', linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')

    # Annotate mean
    mu = [np.mean(input_data[i]) for i in range(len(input_data))]
    colors_num = ['green', 'green', 'green', 'green', 'green']

    if annotate_mean:
        y_val = ax_box.get_ylim()[1]
        for i in range(len(mu)):
            ax_box.text(i + 0.775, y_val - y_val * 0.1, str(np.around(mu[i], decimals=9)),
                        horizontalalignment='center', size='small',
                        color=colors_num[i % 5], rotation=90)

    if add_extra_annotation is not None:
        y_val = ax_box.get_ylim()[1]
        for i in range(len(add_extra_annotation)):
            ax_box.text(i + 1.225, y_val - y_val * 0.1, str(np.around(add_extra_annotation[i], decimals=9)),
                        horizontalalignment='center', size='small',
                        color='k', rotation=90)

    # Text on the figure customise this part for the need!
    # 6 options 'one_SE2', 'multi_SE2', 'one_GAUSS', 'multi_GAUSS', 'one_REALI', 'multi_REALI'
    if kind is not None and input_parameters is not None:

        dom = tuple([int(j) for j in input_parameters[:3]])
        fig.text(.78, .80, r'Domain = ' + str(dom))

        if kind == 'multiple_SE2':
            fig.text(.765, .85,  r'SE(2) generated SVF: ')
            fig.text(.78, .75, r'number of samples: ' + str(int(input_parameters[3])))
            fig.text(.78, .70, str(np.round(input_parameters[4], 3)) +
                     r'$ \leq \theta \leq $ ' +
                     str(np.round(input_parameters[5], 3)))
            fig.text(.78, .65, str(np.round(input_parameters[3], 3)) +
                     r'$ \leq t_x \leq $ ' +
                     str(np.round(input_parameters[7], 3)))
            fig.text(.78, .60, str(np.round(input_parameters[5], 3)) +
                     r'$ \leq t_y \leq $ ' +
                     str(np.round(input_parameters[9], 3)))

        elif kind == 'multiple_HOM':
            fig.text(.765, .85,  r'HOM generated SVF: ')
            fig.text(.78, .75, r'center: ' + str(input_parameters[3]))
            fig.text(.78, .70, r'kind: ' + str(input_parameters[4]))
            fig.text(.78, .65, r'scale_factor: ' + str(input_parameters[5]))
            fig.text(.78, .60, r'sigma: ' + str(input_parameters[6]))
            fig.text(.78, .55, r'in_psl: ' + str(input_parameters[7]))
            fig.text(.78, .50, r'number of samples: ' + str(int(input_parameters[8])))

        elif kind == 'multiple_GAUSS':
            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'number of samples =  ' + str(input_parameters[3]))
            fig.text(.78, .70, r'$\sigma_i$ = ' + str(input_parameters[4]))
            fig.text(.78, .65, r'$\sigma_g$ = ' + str(input_parameters[5]))
            fig.text(.78, .60, r'Ground truth, steps: ')
            fig.text(.78, .57, str(input_parameters[6]) + ' ' + str(input_parameters[7]))

        elif kind == 'multiple_REAL':
            fig.text(.765, .85,  r'Real Data: ')
            fig.text(.78, .70, r'SFVs id string:')
            fig.text(.78, .65, str(input_parameters[3]))
            fig.text(.78, .60, r'Ground truth method ')
            fig.text(.78, .55, str(input_parameters[4]))

        else:
            raise Warning('Kind not recognized.')

    fig.set_tight_layout(True)
    return fig


def plot_custom_step_versus_error_single(list_steps,
                                         matrix_of_lines,  # errors ordered row-major
                                         label_lines,
                                         fig_tag=2,
                                         input_parameters=None,
                                         additional_field=None,
                                         window_title_input='errors',
                                         titles=('iterations vs. error', 'Field'),
                                         x_axis_label='number of steps',
                                         y_axis_label='Error',
                                         kind=None,
                                         input_fig_size=(9, 7),
                                         input_colors=None,
                                         input_line_style=None,
                                         input_marker=None,
                                         log_scale=False,
                                         additional_vertical_line=None,
                                         legend_location='upper right',
                                         ):

    assert len(list_steps) == matrix_of_lines.shape[1]

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    ax_graph = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
    if additional_field is not None:
        ax_svf   = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)

    if input_colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        input_colors = [colors[j % len(colors)] for j in range(len(list_steps))]
    if input_marker is None:
        input_marker = ['.', ] * len(list_steps)
    if input_line_style is None:
        input_line_style = ['-', ] * len(list_steps)

    for j in range(matrix_of_lines.shape[0]):
        ax_graph.plot(list_steps, matrix_of_lines[j, :],
                      color=input_colors[j],
                      linestyle=input_line_style[j],
                      marker=input_marker[j],
                      label=label_lines[j])

    ax_graph.set_title(titles[0])
    ax_graph.legend(loc=legend_location, shadow=False)

    ax_graph.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.set_axisbelow(True)

    ax_graph.set_xlabel(x_axis_label, fontdict=font, labelpad=18)
    ax_graph.set_ylabel(y_axis_label, fontdict=font, labelpad=10)
    if log_scale:
        ax_graph.set_yscale('log')
        ax_graph.set_ylabel(y_axis_label + ' log-scale')

    if additional_vertical_line is not None:
        # print vertical lines:
        xa, xb, ya, yb = list(ax_graph.axis())
        ax_graph.plot([additional_vertical_line, additional_vertical_line], [ya, yb], 'k--', lw=0.5, color='0.3')
        ax_graph.text(additional_vertical_line + 0.2, (yb - ya)/2., r'automatic = '+str(additional_vertical_line))

    # ax_graph.set_xlim(0 - 0.5, num_boxes + 0.5)

    # right side of the figure:
    # Quiver
    if additional_field is not None:
        ax_svf.set_title(titles[1])
        xx, yy = np.meshgrid(np.arange(additional_field.shape[0]), np.arange(additional_field.shape[1]))
        ax_svf.quiver(yy,
                      xx,
                      additional_field[:, :, 0, 0, 0],
                      additional_field[:, :, 0, 0, 1],
                      color='r', linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')

    # Text on the figure customise this part for the need!
    # 6 options 'one_SE2', 'multi_SE2', 'one_GAUSS', 'multi_GAUSS', 'one_REALI', 'multi_REALI'
    if kind is not None and input_parameters is not None:

        dom = tuple([int(j) for j in input_parameters[:3]])
        fig.text(.78, .80, r'Domain = ' + str(dom))

        if kind == 'one_SE2':
            fig.text(.765, .85,  r'SE(2) generated SVF: ')
            fig.text(.78, .75, r'$\theta = $ ' + str(input_parameters[3]))
            fig.text(.78, .70, r'$t_x = $ ' + str(input_parameters[4]))
            fig.text(.78, .65, r'$t_y = $ ' + str(input_parameters[5]))

        if kind == 'one_HOM':
            fig.text(.765, .85,  r'HOM generated SVF: ')
            fig.text(.78, .75, r'center: ' + str(input_parameters[3]))
            fig.text(.78, .70, r'kind: ' + str(input_parameters[4]))
            fig.text(.78, .65, r'scale_factor: ' + str(input_parameters[5]))
            fig.text(.78, .60, r'sigma: ' + str(input_parameters[6]))
            fig.text(.78, .55, r'in_psl: ' + str(input_parameters[7]))

        elif kind == 'one_GAUSS':
            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'$\sigma_i = $ ' + str(input_parameters[3]))
            fig.text(.78, .70, r'$\sigma_g = $ ' + str(input_parameters[4]))
            if len(input_parameters) > 5:
                fig.text(.745, .65, r'Ground truth method, steps: ')
                fig.text(.78, .60, str(input_parameters[5]) + ' ' + str(input_parameters[6]))

        elif kind == 'one_REAL':
            fig.text(.765, .85,  r'Real data: ')
            fig.text(.78, .75, r'id svf:')
            fig.text(.78, .70, str(input_parameters[3]))
            if len(input_parameters) > 5:
                fig.text(.745, .65, r'Ground truth method, steps: ')
                fig.text(.78, .60, str(input_parameters[4]) + ' ' + str(input_parameters[5]))

        else:
            raise Warning('Kind not recognized.')

    fig.set_tight_layout(True)
    return fig


def plot_custom_step_versus_error_multiple(list_steps,
                                             matrix_of_lines_means,  # errors ordered row-major
                                             label_lines,
                                             y_error=None,
                                             fig_tag=2,
                                             input_parameters=None,
                                             additional_field=None,
                                             window_title_input='errors',
                                             titles=('iterations vs. error', 'Field'),
                                             x_axis_label='number of steps',
                                             y_axis_label='Error',
                                             kind=None,
                                             input_fig_size=(9, 7),
                                             input_colors=None,
                                             input_line_style=None,
                                             input_marker=None,
                                             log_scale=False,
                                             additional_vertical_line=None,
                                             legend_location='upper right',
                                             ):

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    if input_parameters is None:
        ax_graph = plt.subplot(111)
    else:
        ax_graph = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
        if additional_field is not None:
            ax_svf   = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)

    if input_colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        input_colors = [colors[j % len(colors)] for j in range(len(list_steps))]
    if input_marker is None:
        input_marker = ['.', ] * len(list_steps)
    if input_line_style is None:
        input_line_style = ['-', ] * len(list_steps)

    for j in range(matrix_of_lines_means.shape[0]):
        if y_error is None:
            ax_graph.errorbar(list_steps, matrix_of_lines_means[j, :],
                              color=input_colors[j],
                              linestyle=input_line_style[j],
                              marker=input_marker[j],
                              label=label_lines[j])
        else:
            if len(y_error) == 2:
                ax_graph.errorbar(list_steps, matrix_of_lines_means[j, :],
                                    yerr=[y_error[0][j], y_error[1][j]],
                                    color=input_colors[j],
                                    linestyle=input_line_style[j],
                                    marker=input_marker[j],
                                    label=label_lines[j])
            else:
                ax_graph.errorbar(list_steps, matrix_of_lines_means[j, :],
                                    yerr=y_error[j],
                                    color=input_colors[j],
                                    linestyle=input_line_style[j],
                                    marker=input_marker[j],
                                    label=label_lines[j])

    ax_graph.set_title(titles[0])
    ax_graph.legend(loc=legend_location, shadow=False)

    ax_graph.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.set_axisbelow(True)

    ax_graph.set_xlabel(x_axis_label, fontdict=font, labelpad=18)
    ax_graph.set_ylabel(y_axis_label, fontdict=font, labelpad=10)
    if log_scale:
        ax_graph.set_yscale('log', nonposy='mask')
        ax_graph.set_ylabel(y_axis_label + ' log-scale')

    if additional_vertical_line is not None:
        # print vertical lines:
        xa, xb, ya, yb = list(ax_graph.axis())
        ax_graph.plot([additional_vertical_line, additional_vertical_line], [ya, yb], 'k--', lw=0.5, color='0.3')
        ax_graph.text(additional_vertical_line + 0.2, (yb - ya)/2., r'automatic = '+str(additional_vertical_line))

    # ax_graph.set_xlim(0 - 0.5, num_boxes + 0.5)

    # right side of the figure:
    # Quiver
    if additional_field is not None and input_parameters is not None:

        ax_svf.set_title(titles[1])
        xx, yy = np.meshgrid(np.arange(additional_field.shape[0]), np.arange(additional_field.shape[1]))
        ax_svf.quiver(yy,
                      xx,
                      additional_field[:, :, 0, 0, 0],
                      additional_field[:, :, 0, 0, 1],
                      color='r', linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')

    # Text on the figure customise this part for the need!
    # 6 options 'one_SE2', 'multi_SE2', 'one_GAUSS', 'multi_GAUSS', 'one_REALI', 'multi_REALI'
    if kind is not None and input_parameters is not None:

        dom = tuple([int(j) for j in input_parameters[:3]])
        fig.text(.78, .80, r'Domain = ' + str(dom))

        if kind == 'multiple_SE2':
            fig.text(.765, .85,  r'SE(2) generated SVF: ')
            fig.text(.78, .75, r'$N = $ ' + str(int(input_parameters[3])))
            fig.text(.78, .70, str(np.round(input_parameters[4], 3)) +
                     r'$ \leq \theta \leq $ ' +
                     str(np.round(input_parameters[5], 3)))
            fig.text(.78, .65, str(np.round(input_parameters[3], 3)) +
                     r'$ \leq t_x \leq $ ' +
                     str(np.round(input_parameters[6], 3)))
            fig.text(.78, .60, str(np.round(input_parameters[5], 3)) +
                     r'$ \leq t_y \leq $ ' +
                     str(np.round(input_parameters[7], 3)))

        if kind == 'multiple_HOM':
            fig.text(.765, .85,  r'HOM generated SVF: ')
            fig.text(.78, .75, r'center: ' + str(input_parameters[3]))
            fig.text(.78, .70, r'kind: ' + str(input_parameters[4]))
            fig.text(.78, .65, r'scale_factor: ' + str(input_parameters[5]))
            fig.text(.78, .60, r'sigma: ' + str(input_parameters[6]))
            fig.text(.78, .55, r'in_psl: ' + str(input_parameters[7]))
            fig.text(.78, .75, r'$N = $ ' + str(int(input_parameters[8])))

        elif kind == 'multiple_GAUSS':
            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'N =' + str(input_parameters[3]))
            fig.text(.78, .60, r'$\sigma_i$ = ' + str(input_parameters[4]))
            fig.text(.78, .55, r'$\sigma_g$ = ' + str(input_parameters[5]))
            print len(input_parameters)

            fig.text(.78, .50, r'Ground truth method, steps: ')
            fig.text(.78, .45, str(input_parameters[6]) + ' ' + str(input_parameters[7]))

        elif kind == 'multiple_REAL':
            fig.text(.765, .85,  r'Real Data: ')
            fig.text(.78, .70, r'SFVs id string:')
            fig.text(.78, .65, str(input_parameters[3]))

            fig.text(.78, .60, r'Ground truth method ')
            fig.text(.78, .55, str(input_parameters[4]))

        elif kind == 'multiple_GAUSS_ic':
            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'N =' + str(input_parameters[3]))
            fig.text(.78, .60, r'$\sigma_i$ = ' + str(input_parameters[4]))
            fig.text(.78, .55, r'$\sigma_g$ = ' + str(input_parameters[5]))
            print len(input_parameters)

        elif kind == 'multiple_REAL_ic':
            fig.text(.765, .85,  r'Real Data: ')
            fig.text(.78, .70, r'SFVs id string:')
            fig.text(.78, .65, str(input_parameters[4]))


        else:
            raise Warning('Kind not recognized.')

    fig.set_tight_layout(True)
    return fig


def plot_custom_cluster(x_in, y_in,
                        fig_tag=11,
                        window_title_input='scatter plot',
                        input_titles=('Main window', 'secondary window'),
                        x_axis_label='time (s)',
                        y_axis_label='error (pixel)',
                        log_scale_x=False,
                        log_scale_y=False,
                        legend_location='upper right',
                        threshold=10,
                        input_fig_size=(12, 7),
                        clusters_labels=None,
                        clusters_colors=None,
                        clusters_markers=None,
                        kind=None,
                        input_parameters=None,
                        additional_field=None,
                        additional_passepartout_values=None):

    # adapt input if they are not lists.
    if not isinstance(x_in, list):
        x_in = [x_in]
    if not isinstance(y_in, list):
        y_in = [y_in]

    elements_per_clusters = [len(x_array) for x_array in x_in]
    number_of_clusters    = len(elements_per_clusters)

    if clusters_colors is None:
        clusters_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.8', 'b', 'g', 'r', 'c']
    if clusters_labels is None:
        clusters_labels = ['cluster' + str(i) for i in range(number_of_clusters)]
    if clusters_markers is None:
        clusters_markers = ['+' for _ in range(number_of_clusters)]

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    # Sanity check:
    assert len(x_in) == len(y_in)
    assert len(y_in) == number_of_clusters

    # Initialize figure:
    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)

    fig.canvas.set_window_title(window_title_input)

    if input_parameters is None:
        ax_scatter = plt.subplot(111)
    else:
        ax_scatter = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)

    if additional_field is not None:
        ax_field   = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)

    # Plot figure:
    for j in range(number_of_clusters):
        ax_scatter.scatter(x_in[j], y_in[j], c=clusters_colors[j], marker=clusters_markers[j],
                           label=clusters_labels[j])

    # Title and axis labels
    ax_scatter.set_title(input_titles[0])
    ax_scatter.legend(loc=legend_location, shadow=False)

    ax_scatter.set_xlabel(x_axis_label, fontdict=font, labelpad=18)
    ax_scatter.set_ylabel(y_axis_label, fontdict=font, labelpad=10)

    if log_scale_x:
        ax_scatter.set_xscale('log')
        ax_scatter.set_xlabel(x_axis_label + ' log-scale')

    if log_scale_y:
        ax_scatter.set_yscale('log')
        ax_scatter.set_ylabel(y_axis_label + ' log-scale')

    # Grid:
    ax_scatter.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_scatter.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_scatter.set_axisbelow(True)

    # right side of the figure:
    # Quiver
    if additional_field is not None:
        ax_field.set_title(input_titles[1])
        X, Y = np.meshgrid(np.arange(additional_field.shape[0]), np.arange(additional_field.shape[1]))
        ax_field.quiver(Y,
                        X,
                        additional_field[:, :, 0, 0, 0],
                        additional_field[:, :, 0, 0, 1],
                        color='r', linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')

    # Additional errorbar boxplot.
    if max(elements_per_clusters) > threshold:
        pass

    # kind options
    if kind is not None and input_parameters is not None:

        dom = tuple([int(j) for j in input_parameters[:3]])
        fig.text(.78, .80, r'Domain = ' + str(dom))

        if kind == 'multiple_SE2':
            fig.text(.765, .85,  r'SE(2) generated SVF: ')
            fig.text(.78, .75, r'$N = $ ' + str(int(input_parameters[3])))
            fig.text(.78, .70, str(np.round(input_parameters[4], 3)) +
                     r'$ \leq \theta \leq $ ' +
                     str(np.round(input_parameters[5], 3)))
            fig.text(.78, .65, str(np.round(input_parameters[3], 3)) +
                     r'$ \leq t_x \leq $ ' +
                     str(np.round(input_parameters[7], 3)))
            fig.text(.78, .60, str(np.round(input_parameters[5], 3)) +
                     r'$ \leq t_y \leq $ ' +
                     str(np.round(input_parameters[9], 3)))

        elif kind == 'multiple_GAUSS':
            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'$\sigma_i = $ ' + str(input_parameters[3]))
            fig.text(.78, .70, r'$\sigma_g = $ ' + str(input_parameters[4]))

        elif kind == 'multiple_REAL':
            pass

        else:
            raise Warning('Kind not recognized.')


def plot_custom_step_error(list_steps_number,
                           matrix_of_lines,  # errors ordered column-major
                           label_lines,
                           stdev=None,
                           additional_field=None,
                           window_title_input='errors',
                           titles=('iterations vs. error', 'Field'),
                           input_parameters=None,
                           fig_tag=2,
                           kind=None,
                           input_fig_size=(10, 6),
                           log_scale=False,
                           input_colors=None,
                           legend_location='upper right',
                           input_line_style='-',
                           input_marker='o'):

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    if input_parameters is None:
        ax_graph = plt.subplot(111)
    else:
        ax_graph = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
        if additional_field is not None:
            ax_svf   = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    fig.canvas.set_window_title(window_title_input)

    if input_colors is None:
        input_colors = ['r'] * matrix_of_lines.shape[0]

    # Graph
    for num_line in range(matrix_of_lines.shape[0]):  # number of methods

        if stdev is None:
            ax_graph.plot(list_steps_number, matrix_of_lines[num_line, :],
                          linestyle=input_line_style[num_line],
                          marker=input_marker[num_line],
                          label=label_lines[num_line],
                          color=input_colors[num_line])
        else:
            ax_graph.errorbar(list_steps_number, matrix_of_lines[num_line, :],
                              yerr=stdev[num_line, :],
                              color=input_colors[num_line],
                              linestyle=input_line_style[num_line],
                              marker=input_marker[num_line],
                              label=label_lines[num_line]
                              )

    ax_graph.set_title(titles[0])

    ax_graph.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.set_axisbelow(True)

    ax_graph.set_xlabel('steps number', fontdict=font)
    ax_graph.set_ylabel('error', fontdict=font)
    if log_scale:
        ax_graph.set_yscale('log')
        ax_graph.set_ylabel('error - log scale')

    xa, xb = ax_graph.get_xlim()
    ya, yb = ax_graph.get_ylim()
    ax_graph.set_xlim([xa, xb + 0.05*xb])
    ax_graph.set_ylim([ya-0.05*ya, yb])

    ax_graph.legend(loc=legend_location, shadow=False)

    # Quiver
    if additional_field is not None:
        ax_svf.set_title(titles[1])
        X, Y = np.meshgrid(np.arange(additional_field.shape[0]), np.arange(additional_field.shape[1]))
        ax_svf.quiver(Y,
                      X,
                      additional_field[:, :, 0, 0, 0],
                      additional_field[:, :, 0, 0, 1],
                      color='r',
                      linewidths=0.01, width=0.03, scale=1, scale_units='xy', units='xy', angles='xy')

    # Text on the figure customise this part for the need!
    if input_parameters is not None and kind is not None:

        dom = tuple([int(j) for j in input_parameters[:3]])
        fig.text(.78, .80, r'Domain = ' + str(dom))

        if kind == 'one_SE2':
            fig.text(.765, .85,  r'SE(2) generated SVF: ')
            fig.text(.78, .75, r'$\theta = $ ' + str(input_parameters[3]))
            fig.text(.78, .70, r'$t_x = $ ' + str(input_parameters[4]))
            fig.text(.78, .65, r'$t_y = $ ' + str(input_parameters[5]))
            fig.text(.78, .60, r'Number of steps considered: ')
            fig.text(.78, .55, str(int(input_parameters[6])))

        elif kind == 'one_HOM':
            fig.text(.765, .85,  r'HOM generated SVF: ')
            fig.text(.78, .75, r'center: ' + str(input_parameters[3]))
            fig.text(.78, .70, r'kind: ' + str(input_parameters[4]))
            fig.text(.78, .65, r'scale_factor: ' + str(input_parameters[5]))
            fig.text(.78, .60, r'sigma: ' + str(input_parameters[6]))
            fig.text(.78, .55, r'in_psl: ' + str(input_parameters[7]))

        elif kind == 'one_GAUSS':
            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'$\sigma_i = $ ' + str(input_parameters[3]))
            fig.text(.78, .70, r'$\sigma_g = $ ' + str(input_parameters[4]))
            fig.text(.78, .65, r'Number of steps considered: ')
            fig.text(.78, .60, str(input_parameters[5]))

        elif kind == 'one_REAL':
            fig.text(.765, .85,  r'Real Data: ')
            fig.text(.78, .70, r'id svf:')
            fig.text(.78, .65, str(input_parameters[3]))
            fig.text(.78, .60, r'Number of steps considered: ')
            fig.text(.78, .55, str(input_parameters[4]))

        elif kind == 'multiple_SE2':
            fig.text(.765, .85,  r'SE(2) generated SVF: ')
            fig.text(.78, .75, r'$N = $ ' + str(int(input_parameters[3])))
            fig.text(.78, .70, str(np.round(input_parameters[4], 3)) +
                     r'$ \leq \theta \leq $ ' +
                     str(np.round(input_parameters[5], 3)))
            fig.text(.78, .65, str(np.round(input_parameters[6], 3)) +
                     r'$ \leq t_x \leq $ ' +
                     str(np.round(input_parameters[7], 3)))
            fig.text(.78, .60, str(np.round(input_parameters[8], 3)) +
                     r'$ \leq t_y \leq $ ' +
                     str(np.round(input_parameters[9], 3)))
            fig.text(.78, .55, r'Steps considered: ' + str(input_parameters[7]))

        elif kind == 'multiple_HOM':
            fig.text(.765, .85,  r'HOM generated SVF: ')
            fig.text(.78, .75, r'center: ' + str(input_parameters[3]))
            fig.text(.78, .70, r'kind: ' + str(input_parameters[4]))
            fig.text(.78, .65, r'scale_factor: ' + str(input_parameters[5]))
            fig.text(.78, .60, r'sigma: ' + str(input_parameters[6]))
            fig.text(.78, .55, r'in_psl: ' + str(input_parameters[7]))
            fig.text(.78, .50, 'N = ' +  str(input_parameters[8]))
            fig.text(.78, .45, 'max steps = ' + str(input_parameters[9]))

        elif kind == 'multiple_GAUSS':
            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'$\sigma_i = $ ' + str(input_parameters[1]))
            fig.text(.78, .70, r'$\sigma_g = $ ' + str(input_parameters[2]))
            fig.text(.78, .65, r'Number of samples: ' + str(input_parameters[0]))
            fig.text(.78, .55, r'Steps considered: ' + str(input_parameters[3]))

        elif kind == 'multiple_REAL':
            fig.text(.78, .85, r'SFVs id string:')
            fig.text(.78, .80, str(input_parameters[3]))
            fig.text(.78, .60, r'Max number of steps: ')
            fig.text(.78, .55, str(input_parameters[4]))

        else:
            raise Warning('Kind not recognized.')

    fig.set_tight_layout(True)
    return fig


# ------------- Bossa figures: --------------


def plot_custom_bossa_figures_like_fov(time_matrix,
                                       error_matrix,  # errors ordered row-major
                                       label_lines,
                                       fig_tag=2,
                                       input_parameters=None,
                                       additional_field=None,
                                       window_title_input='errors',
                                       titles=('time vs. error (increasing field of views)', 'Field'),
                                       x_axis_label='Time(s)',
                                       y_axis_label='Error',
                                       kind=None,
                                       input_fig_size=(9, 7),
                                       input_colors=None,
                                       input_line_style=None,
                                       input_marker=None,
                                       log_scale=False,
                                       additional_vertical_line=None,
                                       legend_location='lower right',
                                       ):

    assert time_matrix.shape[0] == error_matrix.shape[0]
    assert time_matrix.shape[1] == error_matrix.shape[1]

    num_methods = time_matrix.shape[0]

    if kind == 'one_SE2':
        fov_list = [int(f) for f in input_parameters[3:]]
    elif kind == 'one_GAUSS':
        fov_list = [int(f) for f in input_parameters[4:]]
    else:
        raise Warning('Kind not recognized.')

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    ax_graph = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
    if additional_field is not None:
        ax_svf   = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)

    if input_colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        input_colors = [colors[c % len(colors)] for c in range(num_methods)]
    if input_marker is None:
        input_marker = ['.', ] * num_methods
    if input_line_style is None:
        input_line_style = ['-', ] * num_methods

    for met in range(num_methods):  # cycle over the method.
        ax_graph.plot(time_matrix[met, :], error_matrix[met, :],
                      color=input_colors[met],
                      linestyle=input_line_style[met],
                      marker=input_marker[met],
                      label=label_lines[met])

    ax_graph.set_title(titles[0])
    ax_graph.legend(loc=legend_location, shadow=False)

    ax_graph.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.set_axisbelow(True)

    ax_graph.set_xlabel(x_axis_label, fontdict=font, labelpad=18)
    ax_graph.set_ylabel(y_axis_label, fontdict=font, labelpad=10)
    if log_scale:
        ax_graph.set_yscale('log')
        ax_graph.set_ylabel(y_axis_label + ' log-scale')

    # right side of the figure:
    # Quiver
    if additional_field is not None:
        ax_svf.set_title(titles[1])
        xx, yy = np.meshgrid(np.arange(additional_field.shape[0]), np.arange(additional_field.shape[1]))
        ax_svf.quiver(yy,
                      xx,
                      additional_field[:, :, 0, 0, 0],
                      additional_field[:, :, 0, 0, 1],
                      color='r', linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')

        x_size, y_size = additional_field.shape[0:2]

        for fov in fov_list:
            val_fov = int(fov)
            ax_svf.add_patch(patches.Rectangle(
                             (x_size/2 - val_fov/2, y_size/2 - val_fov/2),   # (x,y)
                              val_fov,         # width
                              val_fov,         # height
                              fill=False   # remove background
                             ))

    # Text on the figure customise this part for the need!
    # 6 options 'one_SE2', 'multi_SE2', 'one_GAUSS', 'multi_GAUSS', 'one_REALI', 'multi_REALI'
    if kind is not None and input_parameters is not None:

        dom = tuple([int(j) for j in input_parameters[:3]])
        fig.text(.78, .80, r'Domain = ' + str(dom))

        if kind == 'one_SE2':
            fig.text(.765, .85,  r'SE(2) generated SVF: ')
            fig.text(.78, .75, r'$\theta = $ ' + str(input_parameters[3]))
            fig.text(.78, .70, r'$t_x = $ ' + str(input_parameters[4]))
            fig.text(.78, .65, r'$t_y = $ ' + str(input_parameters[5]))
            fig.text(.10, .88, r'FOVs: ' + str(input_parameters[6:]))

        elif kind == 'one_GAUSS':
            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'$\sigma_i = $ ' + str(input_parameters[5]))
            fig.text(.78, .70, r'$\sigma_g = $ ' + str(input_parameters[6]))
            fig.text(.75, .65, r'Ground truth method, steps: ')
            fig.text(.78, .60, str(input_parameters[3]) + ', ' + str(input_parameters[4]))

            fig.text(.18, .88, r'FOVs: ' + str(input_parameters[7:]))

        elif kind == 'one_REAL':
            fig.text(.765, .85,  r'Real data: ')
            fig.text(.78, .75, r'id svf:')
            fig.text(.78, .70, str(input_parameters[3]))
            fig.text(.78, .65, r'Ground truth method, steps: ')
            fig.text(.78, .60, str(input_parameters[3]))

        else:
            raise Warning('Kind not recognized.')

    fig.set_tight_layout(True)
    return fig


def plot_custom_bossa_figures_like_3(time_matrix,
                                       error_matrix,  # errors ordered row-major
                                       label_lines,
                                       fig_tag=2,
                                       input_parameters=None,
                                       additional_field=None,
                                       window_title_input='errors',
                                       titles=('time vs. error (increasing parameter)', 'Last Field'),
                                       x_axis_label='Time(s)',
                                       y_axis_label='Error',
                                       kind=None,
                                       input_fig_size=(9, 7),
                                       input_colors=None,
                                       input_line_style=None,
                                       input_marker=None,
                                       log_scale=False,
                                       additional_vertical_line=None,
                                       legend_location='lower right',
                                       ):

    assert time_matrix.shape[0] == error_matrix.shape[0]
    assert time_matrix.shape[1] == error_matrix.shape[1]

    num_methods = time_matrix.shape[0]

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    ax_graph = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
    if additional_field is not None:
        ax_svf   = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)

    if input_colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        input_colors = [colors[c % len(colors)] for c in range(num_methods)]
    if input_marker is None:
        input_marker = ['.', ] * num_methods
    if input_line_style is None:
        input_line_style = ['-', ] * num_methods

    for met in range(num_methods):  # cycle over the method.
        ax_graph.plot(time_matrix[met, :], error_matrix[met, :],
                      color=input_colors[met],
                      linestyle=input_line_style[met],
                      marker=input_marker[met],
                      label=label_lines[met])

    ax_graph.set_title(titles[0])
    ax_graph.legend(loc=legend_location, shadow=False)

    ax_graph.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.set_axisbelow(True)

    ax_graph.set_xlabel(x_axis_label, fontdict=font, labelpad=18)
    ax_graph.set_ylabel(y_axis_label, fontdict=font, labelpad=10)
    if log_scale:
        ax_graph.set_yscale('log')
        ax_graph.set_ylabel(y_axis_label + ' log-scale')

    # right side of the figure:
    # Quiver
    if additional_field is not None:
        ax_svf.set_title(titles[1])
        xx, yy = np.meshgrid(np.arange(additional_field.shape[0]), np.arange(additional_field.shape[1]))
        ax_svf.quiver(yy,
                      xx,
                      additional_field[:, :, 0, 0, 0],
                      additional_field[:, :, 0, 0, 1],
                      color='r', linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')

    # Text on the figure customise this part for the need!
    # 6 options 'one_SE2', 'multi_SE2', 'one_GAUSS', 'multi_GAUSS', 'one_REALI', 'multi_REALI'
    if kind is not None and input_parameters is not None:

        dom = tuple([int(j) for j in input_parameters[:3]])
        fig.text(.78, .80, r'Domain = ' + str(dom))

        if kind == 'one_SE2':

            ax_graph.set_title('time vs. error (increasing rotation angle)')

            fig.text(.765, .85,  r'SE(2) generated SVF: ')
            fig.text(.10, .88, r'$\theta$ = ' + str(input_parameters[2:]))

        elif kind == 'one_GAUSS':

            ax_graph.set_title('time vs. error (increasing sigma Gaussian filter)')

            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .72, r'$\sigma_i = $' + str(input_parameters[5]))

            fig.text(.75, .65, r'Ground truth, steps: ')
            fig.text(.78, .60, str(input_parameters[3]) + ', ' + str(input_parameters[4]))

            fig.text(.15, .88, r'$\sigma_g = $' + str(input_parameters[6:]))

        elif kind == 'one_REAL':
            fig.text(.765, .85,  r'Real Data: ')
            fig.text(.78, .75, r'id svf:')
            fig.text(.78, .70, str(input_parameters[0]))
            fig.text(.78, .65, r'Ground truth method: ')
            fig.text(.78, .60, str(input_parameters[0]))

        else:
            raise Warning('Kind not recognized.')

    fig.set_tight_layout(True)
    return fig


def plot_custom_time_error_steps(time_matrix,
                                 error_matrix,  # errors ordered row-major
                                 label_lines,
                                 y_error=None,
                                 fig_tag=2,
                                 input_parameters=None,
                                 additional_field=None,
                                 window_title_input='errors',
                                 titles=('mean time vs. mean error (increasing steps)', 'Field sample'),
                                 x_axis_label='Time(s)',
                                 y_axis_label='Error',
                                 kind=None,
                                 input_fig_size=(9, 7),
                                 input_colors=None,
                                 input_line_style=None,
                                 input_marker=None,
                                 x_log_scale=False,
                                 y_log_scale=False,
                                 legend_location='lower right',
                                 additional_data=None):

    assert time_matrix.shape[0] == error_matrix.shape[0]
    assert time_matrix.shape[1] == error_matrix.shape[1]

    num_methods = time_matrix.shape[0]

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    # Set the axis according to the inputs: (GOOD version!)
    ax_graph = plt.subplot2grid((3, 4), (0, 0), colspan=4, rowspan=3)
    if input_parameters is not None:
        ax_graph = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
    if additional_field is not None:
        ax_graph = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
        ax_svf = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)

    if input_colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        input_colors = [colors[c % len(colors)] for c in range(num_methods)]
    if input_marker is None:
        input_marker = ['.', ] * num_methods
    if input_line_style is None:
        input_line_style = ['-', ] * num_methods

    for met in range(num_methods):  # cycle over the method.
        if y_error is None:
            ax_graph.plot(time_matrix[met, :], error_matrix[met, :],
                          color=input_colors[met],
                          linestyle=input_line_style[met],
                          marker=input_marker[met],
                          label=label_lines[met])
        else:
            if len(y_error) == 2:

                ax_graph.errorbar(time_matrix[met, :], error_matrix[met, :],
                                  yerr=[y_error[0][met, :], y_error[1][met, :]],
                                  color=input_colors[met],
                                  linestyle=input_line_style[met],
                                  marker=input_marker[met],
                                  label=label_lines[met])
            else:
                ax_graph.errorbar(time_matrix[met, :], error_matrix[met, :],
                                  yerr=y_error[met, :],
                                  color=input_colors[met],
                                  linestyle=input_line_style[met],
                                  marker=input_marker[met],
                                  label=label_lines[met])

    ax_graph.set_title(titles[0])
    ax_graph.legend(loc=legend_location, shadow=False)

    ax_graph.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.set_axisbelow(True)

    ax_graph.set_xlabel(x_axis_label, fontdict=font, labelpad=18)
    ax_graph.set_ylabel(y_axis_label, fontdict=font, labelpad=10)

    if x_log_scale:
        ax_graph.set_xscale('log')
        ax_graph.set_xlabel(x_axis_label + ' log-scale')
    if y_log_scale:
        ax_graph.set_yscale('log')
        ax_graph.set_ylabel(y_axis_label + ' log-scale')

    # right side of the figure:
    # Quiver
    if additional_field is not None:
        ax_svf.set_title(titles[1])
        xx, yy = np.meshgrid(np.arange(additional_field.shape[0]), np.arange(additional_field.shape[1]))

        ax_svf.quiver(yy,
                      xx,
                      additional_field[:, :, additional_field.shape[2]/2, 0, 0],
                      additional_field[:, :, additional_field.shape[2]/2, 0, 1],
                      color='r', linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')
        ax_svf.set_aspect('equal')

    if kind is not None and input_parameters is not None:

        dom = tuple([int(j) for j in input_parameters[:3]])
        fig.text(.78, .80, r'Domain = ' + str(dom))

        if additional_data is not None:
            fig.text(.75, .50, r'Max norm: ' + str(additional_data[0]))
            fig.text(.75, .45, r'Mean norm: ' + str(additional_data[1]))

        if kind == 'one_SE2':
            fig.text(.765, .85,  r'SE(2) generated SVF: ')

            if isinstance(input_parameters[3], float):  # is the angle
                fig.text(.78, .75, r'$\theta = $ ' + str(input_parameters[3]))
            else:  # is the denominator of pi
                fig.text(.78, .75, r'$\theta = \pi /$ ' + str(input_parameters[3]))

            fig.text(.78, .70, r'$t_x = $ ' + str(input_parameters[4]))
            fig.text(.78, .65, r'$t_y = $ ' + str(input_parameters[5]))
            fig.text(.11, .18, r'Steps: ' + str(input_parameters[6:]))

        elif kind == 'one_HOM':

            fig.text(.765, .85, r'HOM generated SVF: ')
            fig.text(.78, .75, r'scale_factor: ' + str(input_parameters[3]))
            fig.text(.78, .70, r'sigma: ' + str(input_parameters[4]))
            fig.text(.78, .65, r'in_psl: ' + str(input_parameters[5]))

            fig.text(.11, .18, r'Steps: ' + str(input_parameters[6:]))

        elif kind == 'one_GAUSS':

            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'$\sigma_i = $ ' + str(input_parameters[3]))
            fig.text(.78, .70, r'$\sigma_g = $ ' + str(input_parameters[4]))

            fig.text(.75, .65, r'Ground method, steps: ')
            fig.text(.78, .60, str(input_parameters[5]) + ', ' + str(input_parameters[6]))

            fig.text(.15, .88, r'$Steps = $' + str(input_parameters[7:]))

        elif kind == 'one_REAL':

            fig.text(.78, .85, r'Real data')
            fig.text(.78, .75, 'id data:')
            fig.text(.78, .70, str(input_parameters[3]))
            fig.text(.78, .65, r'Ground method, steps: ')
            fig.text(.78, .60, str(input_parameters[4]) + ' ' + str(input_parameters[5]))

            fig.text(.15, .88, r'$Steps = $' + str(input_parameters[6:]))

        elif kind == 'multiple_SE2':

            fig.text(.765, .85,  r'SE(2) generated SVF: ')
            fig.text(.78, .75, r'$N = $ ' + str(int(input_parameters[3])))
            fig.text(.78, .70, str(np.round(input_parameters[4], 3)) +
                     r'$ \leq \theta \leq $ ' +
                     str(np.round(input_parameters[5], 3)))
            fig.text(.78, .65, str(np.round(input_parameters[6], 3)) +
                     r'$ \leq t_x \leq $ ' +
                     str(np.round(input_parameters[7], 3)))
            fig.text(.78, .60, str(np.round(input_parameters[8], 3)) +
                     r'$ \leq t_y \leq $ ' +
                     str(np.round(input_parameters[9], 3)))
            fig.text(.11, .15, r'$Steps = $' + str(input_parameters[10:]))

        elif kind == 'multiple_HOM':

            fig.text(.765, .85,  r'HOM generated SVF: ')
            fig.text(.78, .65, r'scale_factor: ' + str(input_parameters[3]))
            fig.text(.78, .60, r'sigma: ' + str(input_parameters[4]))
            fig.text(.78, .55, r'in_psl: ' + str(input_parameters[5]))
            fig.text(.78, .50, r'$N = $ ' + str(int(input_parameters[6])))
            fig.text(.11, .18, r'Steps: ' + str(input_parameters[7:]))

        elif kind == 'multiple_GAUSS':

            fig.text(.765, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'$\sigma_i = $ ' + str(input_parameters[4]))
            fig.text(.78, .70, r'$\sigma_g = $ ' + str(input_parameters[5]))
            fig.text(.78, .65, r'Number of samples: ' + str(input_parameters[3]))
            fig.text(.78, .55, r'Ground method, steps: ')
            fig.text(.78, .50, str(input_parameters[6]) + ' ' + str(input_parameters[7]))

            fig.text(.15, .88, r'$Steps = $' + str(input_parameters[8:]))

        elif kind == 'multiple_REAL':

            fig.text(.765, .85, r'Real data')
            fig.text(.78, .75, r'SFVs id string:')
            fig.text(.78, .70, str(input_parameters[3]))
            fig.text(.78, .65, r'Ground method, steps: ')
            fig.text(.78, .60, str(input_parameters[4]) + ' ' + str(input_parameters[5]))

            fig.text(.11, .15, r'$Steps = $' + str(input_parameters[6:]))

        elif kind == 'multiple_REAL_ic':

            fig.text(.765, .85, r'Real data')
            fig.text(.78, .75, r'SFVs id string:')
            fig.text(.78, .70, str(input_parameters[3]))

            fig.text(.11, .15, r'$Steps = $' + str(input_parameters[4:]))

        else:
            raise Warning('Kind not recognized.')

    fig.set_tight_layout(True)
    return fig


"""
To see errors at each level of the vector field. Created for homographies, searching for a computationally well-defined
homographies.
"""


def plot_error_linewise(error_lines,
                        quotes_lines,  # errors ordered row-major
                        label_lines,
                        fig_tag=2,
                        input_parameters=None,
                        additional_field=None,  # must be with the axial quote if 3d
                        axial_quote=0,
                        log_scale=False,
                        window_title_input='errors',
                        titles=('time vs. error (increasing field of views)', 'Field'),
                        legend_location='lower right',
                        ):

    num_lines = len(error_lines)  # list of error vectors

    fig = plt.figure(fig_tag, figsize=(8, 5), dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    ax_graph = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
    if additional_field is not None:
        ax_svf   = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['.', ] * len(colors)
    line_styles = ['-', ] * len(colors)

    for line_num in range(num_lines):  # cycle over the method.
        ax_graph.plot(range(len(error_lines[line_num])), error_lines[line_num],
                      color=colors[line_num],
                      linestyle=line_styles[line_num],
                      marker=markers[line_num],
                      label=label_lines[line_num])

    ax_graph.set_title('Error per position')
    ax_graph.legend(loc=legend_location, shadow=False)

    ax_graph.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.set_axisbelow(True)

    ax_graph.set_xlabel('Position (pixel/voxel)', fontdict=font, labelpad=18)
    ax_graph.set_ylabel('Error', fontdict=font, labelpad=10)
    if log_scale:
        ax_graph.set_yscale('log')
        ax_graph.set_ylabel('Error' + ' log-scale')

    # right side of the figure:
    # Quiver
    if additional_field is not None:
        ax_svf.set_title(titles[1])
        xx, yy = np.meshgrid(np.arange(additional_field.shape[0]), np.arange(additional_field.shape[1]))
        ax_svf.quiver(yy,
                      xx,
                      additional_field[:, :, axial_quote, 0, 0],
                      additional_field[:, :, axial_quote, 0, 1],
                      color='r', linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')

        x_size, y_size = additional_field.shape[0:2]

        for h_lines in range(len(quotes_lines)):
            ax_svf.axhline(y=quotes_lines[h_lines], xmin=0, xmax=x_size, c='b', linewidth=1, zorder=2)

    # Text on the figure customise this part for the need!
    # to show the content of the input_parameters:
    #
    if input_parameters is not None:

        dom = tuple([int(j) for j in input_parameters[:3]])
        fig.text(.78, .80, r'Domain = ' + str(dom))

        fig.text(.765, .85,  r'pgl(n) generated SVF: ')
        fig.text(.78, .75, r'center: ' + str(input_parameters[3]))
        fig.text(.78, .65, r'scale_factor: ' + str(input_parameters[4]))
        fig.text(.78, .60, r'sigma: ' + str(input_parameters[5]))
        fig.text(.78, .55, r'in_psl: ' + str(input_parameters[6]))

    fig.set_tight_layout(True)
    return fig


