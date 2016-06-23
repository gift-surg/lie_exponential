import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

##################
# custom methods #
##################


def triptych_graphs(time_matrix_se2,
                    error_matrix_se2,
                    label_lines_se2,
                    input_colors_se2,
                    input_line_style_se2,
                    input_marker_se2,
                    legend_location_se2,
                    #
                    time_matrix_gauss,
                    error_matrix_gauss,
                    label_lines_gauss,
                    input_colors_gauss,
                    input_line_style_gauss,
                    input_marker_gauss,
                    legend_location_gauss,
                    #
                    time_matrix_real,
                    error_matrix_real,
                    label_lines_real,
                    input_colors_real,
                    input_line_style_real,
                    input_marker_real,
                    legend_location_real,
                    #
                    fig_tag=108):

    fig = plt.figure(fig_tag, figsize=(14, 5), dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.15)

    font_top = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
    font_bl = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13}
    legend_prop = {'size': 12}

    fig.canvas.set_window_title('triptych_results')

    num_methods_se2   = time_matrix_se2.shape[0]
    num_methods_gauss = time_matrix_gauss.shape[0]
    num_methods_real  = time_matrix_real.shape[0]

    ax_1 = plt.subplot(131)

    for met in range(num_methods_se2):  # cycle over the method.
        ax_1.plot(time_matrix_se2[met, :], error_matrix_se2[met, :],
                      color=input_colors_se2[met],
                      linestyle=input_line_style_se2[met],
                      marker=input_marker_se2[met],
                      label=label_lines_se2[met])

    ax_1.set_title('(a)', fontdict=font_top)
    ax_1.legend(loc=legend_location_se2, shadow=False, prop=legend_prop)

    ax_1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_1.set_axisbelow(True)

    ax_1.set_xscale('log')
    ax_1.set_yscale('log')
    ax_1.set_xlabel('Time - log scale (sec)', fontdict=font_bl, labelpad=5)
    ax_1.set_ylabel('Error - log scale (voxel) ', fontdict=font_bl, labelpad=5)

    ax_2 = plt.subplot(132)

    for met in range(num_methods_gauss):  # cycle over the method.
        ax_2.plot(time_matrix_gauss[met, :], error_matrix_gauss[met, :],
                      color=input_colors_gauss[met],
                      linestyle=input_line_style_gauss[met],
                      marker=input_marker_gauss[met],
                      label=label_lines_gauss[met])

    ax_2.set_title('(b)', fontdict=font_top)
    ax_2.legend(loc=legend_location_gauss, shadow=False, prop=legend_prop)

    ax_2.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_2.set_axisbelow(True)

    ax_2.set_xscale('log')
    ax_2.set_yscale('log')
    #ax_2.set_xlabel('Time - log scale (sec)', fontdict=font_bl, labelpad=5)
    #ax_2.set_ylabel('Error - log scale (voxel) ', fontdict=font_bl, labelpad=5)

    ax_3 = plt.subplot(133)

    for met in range(num_methods_real):  # cycle over the method.
        ax_3.plot(time_matrix_real[met, :], error_matrix_real[met, :],
                      color=input_colors_real[met],
                      linestyle=input_line_style_real[met],
                      marker=input_marker_real[met],
                      label=label_lines_real[met])

    ax_3.set_title('(c)', fontdict=font_top)
    ax_3.legend(loc=legend_location_real, shadow=False, prop=legend_prop)

    ax_3.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_3.set_axisbelow(True)

    ax_3.set_xscale('log')
    ax_3.set_yscale('log')
    #ax_3.set_xlabel('Time - log scale (sec)', fontdict=font_bl, labelpad=5)
    #ax_3.set_ylabel('Error - log scale (voxel) ', fontdict=font_bl, labelpad=5)

    return fig


def quadrivium_graphs(time_matrix_se2,
                        error_matrix_se2,
                        label_lines_se2,
                        input_colors_se2,
                        input_line_style_se2,
                        input_marker_se2,
                        legend_location_se2,
                        #
                        time_matrix_hom,
                        error_matrix_hom,
                        label_lines_hom,
                        input_colors_hom,
                        input_line_style_hom,
                        input_marker_hom,
                        legend_location_hom,
                        #
                        time_matrix_gauss,
                        error_matrix_gauss,
                        label_lines_gauss,
                        input_colors_gauss,
                        input_line_style_gauss,
                        input_marker_gauss,
                        legend_location_gauss,
                        #
                        time_matrix_real,
                        error_matrix_real,
                        label_lines_real,
                        input_colors_real,
                        input_line_style_real,
                        input_marker_real,
                        legend_location_real,
                        #
                        fig_tag=108):

    fig = plt.figure(fig_tag, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.1, right=0.96, top=0.95, bottom=0.1)

    font_top = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
    font_bl = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13}
    legend_prop = {'size': 12}

    fig.canvas.set_window_title('quadrivium_results')

    num_methods_se2   = time_matrix_se2.shape[0]
    num_methods_hom   = time_matrix_hom.shape[0]
    num_methods_gauss = time_matrix_gauss.shape[0]
    num_methods_real  = time_matrix_real.shape[0]

    ax_1 = plt.subplot(221)

    for met in range(num_methods_se2):  # cycle over the method.
        ax_1.plot(time_matrix_se2[met, :], error_matrix_se2[met, :],
                      color=input_colors_se2[met],
                      linestyle=input_line_style_se2[met],
                      marker=input_marker_se2[met],
                      label=label_lines_se2[met])

    ax_1.set_title('(a)', fontdict=font_top)
    ax_1.legend(loc=legend_location_se2, shadow=False, prop=legend_prop)

    ax_1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_1.set_axisbelow(True)

    ax_1.set_xscale('log')
    ax_1.set_yscale('log')
    #ax_1.set_xlabel('Time - log scale (sec)', fontdict=font_bl, labelpad=5)
    #ax_1.set_ylabel('Error - log scale (voxel) ', fontdict=font_bl, labelpad=5)

    ax_2 = plt.subplot(222)

    for met in range(num_methods_hom):  # cycle over the method.
        ax_2.plot(time_matrix_hom[met, :], error_matrix_hom[met, :],
                      color=input_colors_hom[met],
                      linestyle=input_line_style_hom[met],
                      marker=input_marker_hom[met],
                      label=label_lines_hom[met])

    ax_2.set_title('(b)', fontdict=font_top)
    ax_2.legend(loc=legend_location_hom, shadow=False, prop=legend_prop)

    ax_2.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_2.set_axisbelow(True)

    ax_2.set_xscale('log')
    ax_2.set_yscale('log')
    #ax_2.set_xlabel('Time - log scale (sec)', fontdict=font_bl, labelpad=5)
    #ax_2.set_ylabel('Error - log scale (voxel) ', fontdict=font_bl, labelpad=5)

    ax_3 = plt.subplot(223)

    for met in range(num_methods_gauss):  # cycle over the method.
        ax_3.plot(time_matrix_gauss[met, :], error_matrix_gauss[met, :],
                      color=input_colors_gauss[met],
                      linestyle=input_line_style_gauss[met],
                      marker=input_marker_gauss[met],
                      label=label_lines_gauss[met])

    ax_3.set_title('(c)', fontdict=font_top)
    ax_3.legend(loc=legend_location_gauss, shadow=False, prop=legend_prop)

    ax_3.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_3.set_axisbelow(True)

    ax_3.set_xscale('log')
    ax_3.set_yscale('log')
    ax_3.set_xlabel('Time - log scale (sec)', fontdict=font_bl, labelpad=5)
    ax_3.set_ylabel('Error - log scale (voxel) ', fontdict=font_bl, labelpad=5)

    ax_4 = plt.subplot(224)

    for met in range(num_methods_real):  # cycle over the method.
        ax_4.plot(time_matrix_real[met, :], error_matrix_real[met, :],
                      color=input_colors_real[met],
                      linestyle=input_line_style_real[met],
                      marker=input_marker_real[met],
                      label=label_lines_real[met])

    ax_4.set_title('(d)', fontdict=font_top)
    ax_4.legend(loc=legend_location_real, shadow=False, prop=legend_prop)

    ax_4.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_4.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_4.set_axisbelow(True)

    ax_4.set_xscale('log')
    ax_4.set_yscale('log')
    #ax_4.set_xlabel('Time - log scale (sec)', fontdict=font_bl, labelpad=5)
    #ax_4.set_ylabel('Error - log scale (voxel) ', fontdict=font_bl, labelpad=5)

    return fig
    pass


def single_graph(time_matrix,
                 error_matrix,
                 label_lines,
                 input_colors,
                 input_line_style,
                 input_marker,
                 legend_location,
                 fig_tag=109,
                 input_title=''):

    fig = plt.figure(fig_tag, figsize=(8, 5), dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.12)

    font_top = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
    font_bl = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
    legend_prop = {'size': 12}

    fig.canvas.set_window_title('single_graph_results.pdf')

    num_methods   = time_matrix.shape[0]

    ax = plt.subplot(111)

    for met in range(num_methods):  # cycle over the method.
        ax.plot(time_matrix[met, :], error_matrix[met, :],
                      color=input_colors[met],
                      linestyle=input_line_style[met],
                      marker=input_marker[met],
                      label=label_lines[met])

    ax.set_title(input_title, fontdict=font_top)
    ax.legend(loc=legend_location, shadow=False, prop=legend_prop)

    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Mean time - log scale (sec)', fontdict=font_bl, labelpad=5)
    ax.set_ylabel('Mean error - log scale (voxel) ', fontdict=font_bl, labelpad=5)

    return fig


def plot_ic_and_sa(list_steps_ic,
                    means_error_ic,
                    input_colors_ic,
                    input_line_style_ic,
                    input_marker_ic,
                    label_lines_ic,
                    legend_location_ic,
                    #
                    list_steps_sa,
                    means_error_sa,
                    input_colors_sa,
                    input_line_style_sa,
                    input_marker_sa,
                    label_lines_sa,
                    legend_location_sa,
                    #
                    y_error_ic=None,
                    y_error_sa=None,
                    fig_tag=110):

    horizontal = True
    if horizontal:
        fig = plt.figure(fig_tag, figsize=(13, 5.5), dpi=100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.12)
        ax_ic = plt.subplot(121)
        ax_sa = plt.subplot(122)

    else:
        fig = plt.figure(fig_tag, figsize=(5.5, 8), dpi=100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.08)
        ax_ic = plt.subplot(211)
        ax_sa = plt.subplot(212)

    font_top = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
    font_bl = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
    legend_prop = {'size': 12}

    fig.canvas.set_window_title('ic_and_sa.pdf')

    ######
    ###### Left figure Inverse Consistency
    ######

    for j in range(means_error_ic.shape[0]):
        if y_error_ic is None:
            ax_ic.errorbar(list_steps_ic, means_error_ic[j, :],
                              color=input_colors_ic[j],
                              linestyle=input_line_style_ic[j],
                              marker=input_marker_ic[j],
                              label=label_lines_ic[j])
        else:
            if len(y_error_ic) == 2:
                ax_ic.errorbar(list_steps_ic, means_error_ic[j, :],
                                    yerr=[y_error_ic[0][j], y_error_ic[1][j]],
                                    color=input_colors_ic[j],
                                    linestyle=input_line_style_ic[j],
                                    marker=input_marker_ic[j],
                                    label=label_lines_ic[j])
            else:
                ax_ic.errorbar(list_steps_ic, means_error_ic[j, :],
                                    yerr=y_error_ic[j],
                                    color=input_colors_ic[j],
                                    linestyle=input_line_style_ic[j],
                                    marker=input_marker_ic[j],
                                    label=label_lines_ic[j])

    #ax_ic.set_title('Inverse consistency', fontdict=font_top)
    ax_ic.legend(loc=legend_location_ic, shadow=False, prop=legend_prop)

    ax_ic.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_ic.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_ic.set_axisbelow(True)

    ax_ic.set_xlabel('Number of steps', fontdict=font_bl, labelpad=5)
    ax_ic.set_ylabel('Inverse consistency error (log-scale)', fontdict=font_bl, labelpad=5)
    ax_ic.set_yscale('log', nonposy='clip')

    ######
    ###### Right figure Scalar associativity
    ######

    num_methods_sa = means_error_sa.shape[0]

    for met in range(num_methods_sa):  # cycle over the method.
        if y_error_sa is None:
            ax_sa.plot(list_steps_sa, means_error_sa[met, :],
                          color=input_colors_sa[met],
                          linestyle=input_line_style_sa[met],
                          marker=input_marker_sa[met],
                          label=label_lines_sa[met])
        else:
            if len(y_error_sa) == 2:
                ax_sa.errorbar(list_steps_sa, means_error_sa[met, :],
                                  yerr=[y_error_sa[0][met, :], y_error_sa[1][met, :]],
                                  color=input_colors_sa[met],
                                  linestyle=input_line_style_sa[met],
                                  marker=input_marker_sa[met],
                                  label=label_lines_sa[met])
            else:
                ax_sa.errorbar(list_steps_sa, means_error_sa[met, :],
                                  yerr=y_error_sa[met, :],
                                  color=input_colors_sa[met],
                                  linestyle=input_line_style_sa[met],
                                  marker=input_marker_sa[met],
                                  label=label_lines_sa[met])

    #ax_sa.set_title('Scalar associativity', fontdict=font_top)
    ax_sa.legend(loc=legend_location_sa, shadow=False, prop=legend_prop)

    ax_sa.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_sa.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_sa.set_axisbelow(True)

    ax_sa.set_xlabel('Number of steps', fontdict=font_bl, labelpad=5)
    ax_sa.set_ylabel('Scalar associativity error (log-scale)', fontdict=font_bl, labelpad=5)

    ax_sa.set_yscale('log')

    return fig


def plot_ic_sa_and_se(list_steps_ic,
                        means_error_ic,
                        input_colors_ic,
                        input_line_style_ic,
                        input_marker_ic,
                        label_lines_ic,
                        legend_location_ic,
                        #
                        list_steps_sa,
                        means_error_sa,
                        input_colors_sa,
                        input_line_style_sa,
                        input_marker_sa,
                        label_lines_sa,
                        legend_location_sa,
                        #
                        list_steps_se,
                        means_lines_se,
                        input_colors_se,
                        input_line_style_se,
                        input_marker_se,
                        label_lines_se,
                        legend_location_se,
                        #
                        y_error_ic=None,
                        y_error_sa=None,
                        y_error_se=None,
                        fig_tag=110):

    horizontal = True
    if horizontal:
        fig = plt.figure(fig_tag, figsize=(14, 5), dpi=100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12)
        ax_ic = plt.subplot(131)
        ax_sa = plt.subplot(132)
        ax_se = plt.subplot(133)

    else:
        fig = plt.figure(fig_tag, figsize=(5.5, 8), dpi=100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.08)
        ax_ic = plt.subplot(311)
        ax_sa = plt.subplot(312)
        ax_se = plt.subplot(313)

    font_top = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
    font_bl = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
    legend_prop = {'size': 12}

    fig.canvas.set_window_title('ic_and_sa.pdf')

    ######
    ###### Left figure Inverse Consistency
    ######

    for j in range(means_error_ic.shape[0]):
        if y_error_ic is None:
            ax_ic.errorbar(list_steps_ic, means_error_ic[j, :],
                              color=input_colors_ic[j],
                              linestyle=input_line_style_ic[j],
                              marker=input_marker_ic[j],
                              label=label_lines_ic[j])
        else:
            if len(y_error_ic) == 2:
                ax_ic.errorbar(list_steps_ic, means_error_ic[j, :],
                                    yerr=[y_error_ic[0][j], y_error_ic[1][j]],
                                    color=input_colors_ic[j],
                                    linestyle=input_line_style_ic[j],
                                    marker=input_marker_ic[j],
                                    label=label_lines_ic[j])
            else:
                ax_ic.errorbar(list_steps_ic, means_error_ic[j, :],
                                    yerr=y_error_ic[j],
                                    color=input_colors_ic[j],
                                    linestyle=input_line_style_ic[j],
                                    marker=input_marker_ic[j],
                                    label=label_lines_ic[j])

    #ax_ic.set_title('Inverse consistency', fontdict=font_top)
    ax_ic.legend(loc=legend_location_ic, shadow=False, prop=legend_prop)

    ax_ic.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_ic.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_ic.set_axisbelow(True)

    ax_ic.set_xlabel('Number of steps', fontdict=font_bl, labelpad=5)
    ax_ic.set_ylabel('Mean inverse consistency error (log-scale)', fontdict=font_bl, labelpad=5)
    ax_ic.set_yscale('log', nonposy='clip')

    ######
    ###### Central figure Scalar associativity
    ######

    num_methods_sa = means_error_sa.shape[0]

    for met in range(num_methods_sa):  # cycle over the method.
        if y_error_sa is None:
            ax_sa.plot(list_steps_sa, means_error_sa[met, :],
                          color=input_colors_sa[met],
                          linestyle=input_line_style_sa[met],
                          marker=input_marker_sa[met],
                          label=label_lines_sa[met])
        else:
            if len(y_error_sa) == 2:
                ax_sa.errorbar(list_steps_sa, means_error_sa[met, :],
                                  yerr=[y_error_sa[0][met, :], y_error_sa[1][met, :]],
                                  color=input_colors_sa[met],
                                  linestyle=input_line_style_sa[met],
                                  marker=input_marker_sa[met],
                                  label=label_lines_sa[met])
            else:
                ax_sa.errorbar(list_steps_sa, means_error_sa[met, :],
                                  yerr=y_error_sa[met, :],
                                  color=input_colors_sa[met],
                                  linestyle=input_line_style_sa[met],
                                  marker=input_marker_sa[met],
                                  label=label_lines_sa[met])

    #ax_sa.set_title('Scalar associativity', fontdict=font_top)
    ax_sa.legend(loc=legend_location_sa, shadow=False, prop=legend_prop)

    ax_sa.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_sa.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_sa.set_axisbelow(True)

    ax_sa.set_xlabel('Number of steps', fontdict=font_bl, labelpad=5)
    ax_sa.set_ylabel('Mean scalar associativity error (log-scale)', fontdict=font_bl, labelpad=5)
    ax_sa.set_yscale('log')

    ######
    ###### Right figure Step error
    ######

    num_methods_se = means_lines_se.shape[0]

    # Graph
    for num_line in range(means_lines_se.shape[0]):  # number of methods

        if y_error_se is None:
            ax_se.plot(list_steps_se, means_lines_se[num_line, :],
                          linestyle=input_line_style_se[num_line],
                          marker=input_marker_se[num_line],
                          label=label_lines_se[num_line],
                          color=input_colors_se[num_line])
        else:
            if len(y_error_se) == 2:
                ax_se.errorbar(list_steps_se, means_lines_se[num_line, :],
                                  yerr=[y_error_se[0][num_line, :], y_error_se[1][num_line, :]],
                                  linestyle=input_line_style_se[num_line],
                                  marker=input_marker_se[num_line],
                                  label=label_lines_se[num_line],
                                  color=input_colors_se[num_line]
                                  )

            else:
                ax_se.errorbar(list_steps_se, means_lines_se[num_line, :],
                                  yerr=y_error_sa[num_line, :],
                                  linestyle=input_line_style_se[num_line],
                                  marker=input_marker_se[num_line],
                                  label=label_lines_se[num_line],
                                  color=input_colors_se[num_line]
                                  )

    ax_se.legend(loc=legend_location_ic, shadow=False, prop=legend_prop)

    ax_se.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_se.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_se.set_axisbelow(True)

    ax_se.set_xlabel('Number of steps', fontdict=font_bl, labelpad=5)
    ax_se.set_ylabel('Mean step-wise error (log-scale)', fontdict=font_bl, labelpad=5)
    ax_se.set_yscale('log', nonposy='clip')

    return fig


