import numpy as np
import matplotlib.pyplot as plt


def custom_boxplot(input_data, fig_tag=1,
                   x_labels=None,
                   x_axis_label='methods',
                   y_axis_label='errors',
                   add_boxes_size=False,
                   add_mean=True,
                   add_extra_numbers=False):
    """

    :param input_data:
    :param fig_tag:
    :param x_labels:
    :param x_axis_label:
    :param y_axis_label:
    :param add_boxes_size:
    :param add_mean:
    :param add_extra_numbers:
    :return:
    """
    
    if x_labels is None:
        x_labels = [str(a) for a in range(len(input_data))]

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
    # font_on_fig = {'family': 'serif', 'color': 'black','weight': 'normal', 'size': 12}

    ### begin figure: ###
    
    fig = plt.figure(fig_tag, figsize=(14, 7), dpi=80, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    
    # set title
    fig.canvas.set_window_title('boxplot')

    # adjust values of the internal image
    ax = fig.add_subplot(111)

    # write the data leaving the spaces for the interval subdivision
    bp = ax.boxplot(list(input_data), notch=False, patch_artist=False,  sym='+', vert=1, whis=1.5)

    # set the colors:
    plt.setp(bp['boxes'], color='blue')
    plt.setp(bp['whiskers'], color='blue')
    plt.setp(bp['fliers'], color='red', marker='+')

    # set axis labels
    ax.set_xlabel(x_axis_label, fontdict=font, labelpad=18)
    ax.set_ylabel(y_axis_label, fontdict=font, labelpad=10)

    # compute means and plot values above each boxplot:
    mu = [np.mean(input_data[i]) for i in range(len(input_data))]
    # box_sizes = [len(input_data[i]) for i in [0, 5, 10]]

    colors_num = ['green', 'green', 'green', 'green', 'green']
    # mean
    if add_mean:
        y_val = ax.get_ylim()[1]
        for i in range(len(mu)):
            ax.text(i + 1, y_val - y_val * 0.1, str(np.around(mu[i], decimals=9)),
                    horizontalalignment='center', size='small',
                    color=colors_num[i % 5])

    # add extra data:
    if add_extra_numbers is not False:
        y_val = ax.get_ylim()[1]
        for i in range(len(mu)):
            ax.text(i + 1, y_val - y_val * 0.2, str(np.around(add_extra_numbers[i], decimals=9)),
                    horizontalalignment='center', size='small',
                    color='k')

    # box size
    if add_boxes_size:
        ax.text(1, 0.4, 'Boxes size: ' + str(len(input_data[0])),
                horizontalalignment='center', size='small', color='k')

    xtick_names = plt.setp(ax, xticklabels=x_labels)
    plt.setp(xtick_names, rotation=45, fontsize=12)


def custom_n_boxplots(input_data,
                      fig_tag=1,
                      n_row=1,
                      n_col=1,
                      x_labels=None,
                      x_axis_labels=None,
                      y_axis_labels=None,
                      canvas_title='boxplot',
                      add_boxes_size=False,
                      add_mean=True):
    """

    :param input_data:
    :param fig_tag:
    :param n_row:
    :param n_col:
    :param x_labels:
    :param x_axis_labels:
    :param y_axis_labels:
    :param canvas_title:
    :param add_boxes_size:
    :param add_mean:
    :return:
    """
    n = len(input_data)

    if x_labels is None:
        # generate labels of the boxes with numbers
        x_labels = []
        for i in range(n):
            x_labels += [str(a) for a in range(len(input_data[i]))]

    if x_axis_labels is None:
        x_axis_labels = ['methods'] * n

    if y_axis_labels is None:
        y_axis_labels = ['errors'] * n

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 12}
    # font_on_fig = {'family': 'serif', 'color': 'black','weight': 'normal', 'size': 12}

    ### begin figure: ###

    fig = plt.figure(fig_tag, figsize=( 4.5* n_col, 4.2 * n_row), dpi=90, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    fig.canvas.set_window_title(canvas_title)

    for i in range(n):
        ax = fig.add_subplot(n_row, n_col, i + 1)

        # write the data leaving the spaces for the interval subdivision
        bp = ax.boxplot(list(input_data[i]), notch=False, patch_artist=False,  sym='+', vert=1, whis=1.5)

        # set the colors:
        plt.setp(bp['boxes'],    color='blue')
        plt.setp(bp['whiskers'], color='blue')
        plt.setp(bp['fliers'],   color='red', marker='+')

        # set axis labels
        ax.set_xlabel(x_axis_labels[i], fontdict=font, labelpad=18)
        ax.set_ylabel(y_axis_labels[i], fontdict=font, labelpad=10)

        # Add boxes label
        xtickNames = plt.setp(ax, xticklabels=x_labels[i])
        plt.setp(xtickNames, rotation=45, fontsize=12)

        # add mean
        if add_mean:
            mu = [np.mean(input_data[i][j]) for j in range(7)]

            colors_num = ['red', 'green', 'blue'] * (len(mu))
            # mean
            y_val = ax.get_ylim()[1]
            for k in range(len(mu)):
                ax.text(k + 1, y_val - y_val * 0.1, str(np.around(mu[k], decimals=5)),
                        horizontalalignment='center', size='small',
                        color=colors_num[k % 5])

    fig.set_tight_layout(True)


def plot_steps_vs_multiple_lines(list_steps_number,
                                 matrix_of_lines,  # errors ordered column-major
                                 label_lines,
                                 additional_field=None,
                                 window_title_input='errors',
                                 titles=('iterations vs. error', 'Field'),
                                 input_parameters=(0, 0, 0),
                                 fig_tag=2,
                                 input_fig_size=(11.5, 6),
                                 log_scale=False,
                                 input_colors=None,
                                 additional_vertical_line=None,
                                 legend_location='upper right',
                                 input_line_style='-', input_marker='o'):

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    fig.canvas.set_window_title(window_title_input)

    ax_graph = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
    ax_svf   = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)

    # Graph
    for num_line in range(matrix_of_lines.shape[1]):  # number of methods
        if input_colors is None:
            ax_graph.plot(list_steps_number, matrix_of_lines[:, num_line], linestyle=input_line_style,
                          marker=input_marker,
                          label=label_lines[num_line])
        else:
            ax_graph.plot(list_steps_number, matrix_of_lines[:, num_line], linestyle=input_line_style,
                          marker=input_marker,
                          label=label_lines[num_line],
                          color=input_colors[num_line])
    ax_graph.set_title(titles[0])

    ax_graph.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_graph.set_axisbelow(True)

    ax_graph.set_xlabel('steps number')
    ax_graph.set_ylabel('error')
    if log_scale:
        ax_graph.set_yscale('log')
        ax_graph.set_ylabel('error - log scale')

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
                      #linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')

    if additional_vertical_line is not None:
        # print vertical lines:
        xa, xb, ya, yb = list(ax_graph.axis())
        ax_graph.plot([additional_vertical_line, additional_vertical_line], [ya, yb], 'k--', lw=0.5, color='0.3')
        ax_graph.text(additional_vertical_line + 0.2, (yb - ya)/2., r'automatic = '+str(additional_vertical_line))

    # Text on the figure customise this part for the need!
    if input_parameters is not None:
        fig.text(.75, .8,  r'Parameters: ')
        if len(input_parameters) == 3:
            fig.text(.75, .85,  r'SE(2) generated SVF: ')
            fig.text(.78, .75, r'$\theta = $ ' + str(input_parameters[0]))
            fig.text(.78, .70, r'$t_x = $ ' + str(input_parameters[1]))
            fig.text(.78, .65, r'$t_y = $ ' + str(input_parameters[2]))
        elif len(input_parameters) == 2:
            fig.text(.75, .85,  r'Gauss generated SVF: ')
            fig.text(.78, .75, r'$\sigma_i = $ ' + str(input_parameters[0]))
            fig.text(.78, .70, r'$\sigma_g = $ ' + str(input_parameters[1]))
        else:
            print '(plot_steps_vs_multiple_lines: No additional Text Added)'

    fig.set_tight_layout(True)
    return fig


def plot_multiple_graphs(list_steps_number,
                         matrix_of_input,  # errors ordered column-major
                         num_row=2, num_col=4,
                         window_title_input='errors',
                         methods_titles=None,
                         fig_tag=2,
                         log_scale=False,
                         input_colors_per_methods=None):
    """
    Comparing different methods for each value.
    print different lines over the same nodes.


    :param list_steps_number: x values of each graph
    :param matrix_of_input:   (x, y, z) x:steps, y:methods, z:subjects
    :param window_title_input:
    :param methods_titles:  titles (or name) of each method
    :param fig_tag:
    :return: multiple graph. Each graph corresponds to a method, and we see the value of this methods for different
    subjects.

    Subplot 1 has the behaviour of all of the subjects for each step, for the method 1
    Subplot 2 has the behaviour of all of the subjects for each step, for the method 2
    ...
    Subplot n has the behaviour of all of the subjects for each step, for the method n

    Subplots are divided in num_row and num_col in the figure!
    """

    fig = plt.figure(fig_tag, figsize=(13, 5.8), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    fig.canvas.set_window_title(window_title_input)

    # Sanity input check:
    if not len(list_steps_number) == matrix_of_input.shape[0]:
        raise IOError('Error in the input data!')
    if not len(methods_titles) == num_col*num_row:
        raise IOError('Incompatible input data!')

    # Manage titles:
    if methods_titles is None:
        methods_titles = ['method ' + str(j) for j in range(num_col*num_row)]

    # Manage colors:
    if input_colors_per_methods is None:
        input_colors_per_methods = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.8']

    # Graph
    for j in range(matrix_of_input.shape[1]):  # number of methods corresponding to the graph

        ax_method_j = plt.subplot(num_row, num_col, j+1)

        for k in range(matrix_of_input.shape[2]):

            ax_method_j.plot(list_steps_number, matrix_of_input[:, j, k], color=input_colors_per_methods[k])

        ax_method_j.set_title(methods_titles[j])

        ax_method_j.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax_method_j.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax_method_j.set_axisbelow(True)

        ax_method_j.set_xlabel('steps number')
        ax_method_j.set_ylabel('error')
        if log_scale:
            ax_method_j.set_yscale('log')
            ax_method_j.set_ylabel('error - log scale')

        # ax_method_j.legend(loc=legend_location, shadow=False)

    fig.set_tight_layout(True)
    return fig


def plot_splitted_bar_chart(input_data, y_intervals=((0, 1), (2, 3)), title_input=None):
    """

    :param input_data:
    :param intervals:
    :param title_input:
    :return:
    """

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5), dpi=120)

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)

    index = np.arange(len(input_data[1]))
    bar_width = 0.35

    r_11 = ax1.bar(index, list(input_data[0, :]), bar_width,
                     color='b',
                     label=r'$\varphi \circ \varphi^{-1}$')

    r_12 = ax1.bar(index + bar_width, list(input_data[1, :]), bar_width,
                     color='r',
                     label=r'$\varphi \circ \varphi^{-1}$')

    r_21 = ax2.bar(index, list(input_data[0, :]), bar_width,
                     color='b',
                     label=r'$\varphi \circ \varphi^{-1}$')

    r_22 = ax2.bar(index + bar_width, list(input_data[1, :]), bar_width,
                     color='r',
                     label=r'$\varphi \circ \varphi^{-1}$')

    ax1.set_ylim(y_intervals[1][0], y_intervals[1][1])  # outliers only
    ax2.set_ylim(y_intervals[0][0], y_intervals[0][1])  # most of the data

    ax1.set_xlim(0 - bar_width, len(input_data[1]))
    ax2.set_xlim(0 - bar_width, len(input_data[1]))

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax1.grid(True)
    ax2.grid(True)

    f.text(0.5, 0.04, 'Methods', ha='center')
    f.text(0.01, 0.5, 'Errors', va='center', rotation='vertical')

    ax1.set_title(r'Inverse consistency errors: $| \varphi \circ \varphi^{-1} - I |$')
    if title_input is not None and len(title_input) == input_data.shape[1]:
        plt.xticks(index + bar_width, title_input)
    ax1.legend(loc=2)

    f.subplots_adjust(hspace=0.03)

    #plt.tight_layout()
