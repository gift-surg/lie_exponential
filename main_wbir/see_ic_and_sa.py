import numpy as np
import os
from matplotlib import pyplot as plt
import pickle
from tabulate import tabulate

from utils.path_manager import path_to_sharing_folder
from visualizer.graphs_and_stats_new import plot_custom_step_versus_error_multiple, plot_custom_time_error_steps
from visualizer.graphs_and_stats_new_2 import plot_ic_and_sa


if __name__ == "__main__":

    verbose = True
    plot_results_ic = False
    plot_results_sa = False
    plot_final_graph = True

    ###################################################
    # Collect information inverse consistency results #
    ###################################################

    # Load data inverse consistency

    errors_ic = np.load(os.path.join(path_to_sharing_folder, 'ic_errors_real_new.npy'))

    with open(os.path.join(path_to_sharing_folder, 'ic_parameters_real_new'), 'rb') as f:
            parameters_ic = pickle.load(f)

    #                            name        visualize  colour line-style  marker
    visualize_methods_ic = [['ss',        True,      'b',      '-',     '+'],
                            ['gss_aei',   True,      'b',      '--',     'x'],
                            ['gss_ei',    True,      'r',      '--',     '.'],
                            ['gss_rk4',   True,      'k',      '-',     '.'],
                            ['midpoint',  False,      'c',      '-',     '.'],
                            ['euler',     True,      'g',      '-',     '>'],
                            ['euler_mod', False,      'm',      '-',     '>'],
                            ['euler_aei', True,      'm',      '--',     '>'],
                            ['heun',      False,      'k',      '-',     '.'],
                            ['heun_mod',  False,      'k',      '--',     '.'],
                            ['rk4',       True,     'y',      '--',     'x']]

    index_methods_considered_ic = [j for j in range(len(visualize_methods_ic))
                                      if visualize_methods_ic[j][1] is True]
    num_method_considered_ic         = len(index_methods_considered_ic)
    names_method_considered_ic       = [visualize_methods_ic[j][0] for j in index_methods_considered_ic]
    color_methods_considered_ic      = [visualize_methods_ic[j][2] for j in index_methods_considered_ic]
    line_style_methods_considered_ic = [visualize_methods_ic[j][3] for j in index_methods_considered_ic]
    marker_method_considered_ic      = [visualize_methods_ic[j][4] for j in index_methods_considered_ic]

    list_steps_ic = parameters_ic[6:]

    if verbose:

        print '----------------------------------------------------------'
        print 'Inverse consistency error for multiple REAL generated SVF'
        print '----------------------------------------------------------'

        print '\nParameters that generate the SVF'
        print 'Subjects ids = ' + str(parameters_ic[3])
        print 'Svf dimension: ' + str(parameters_ic[:3])
        print 'List of steps considered:'
        print str(parameters_ic[6:])

        print '\n'
        print 'Methods and parameters:'
        print tabulate(visualize_methods_ic,
                       headers=['name', 'compute (True/False)', 'colour',  'line-style',   'marker'])
        print '\n'
        print 'List of the methods considered:'
        print names_method_considered_ic
        print '---------------------------------------------'

    ####################################################
    # Collect information scalar associativity results #
    ####################################################

    # Load data scalar associativity

    errors_sa = np.load(os.path.join(path_to_sharing_folder, 'exp_scalar_associativity_errors_real_new.npy'))

    with open(os.path.join(path_to_sharing_folder, 'exp_scalar_associativity_parameters_real_new'), 'rb') as f:
        parameters_sa = pickle.load(f)

    with open(os.path.join(path_to_sharing_folder, 'exp_scalar_associativity_methods_table_real_new'), 'rb') as f:
                methods_sa = pickle.load(f)

    #visualize_methods_sa = methods_sa[:]

    visualize_methods_sa = [['ss',        True,      'b',      '-',     '+'],
                            ['gss_aei',   True,      'b',      '--',     'x'],
                            ['gss_ei',    True,      'r',      '--',     '.'],
                            ['gss_rk4',   True,      'k',      '-',     '.'],
                            ['midpoint',  False,      'c',      '-',     '.'],
                            ['euler',     True,      'g',      '-',     '>'],
                            ['euler_mod', False,      'm',      '-',     '>'],
                            ['euler_aei', True,      'm',      '--',     '>'],
                            ['heun',      False,      'k',      '-',     '.'],
                            ['heun_mod',  False,      'k',      '--',     '.'],
                            ['rk4',       True,     'y',      '--',     'x']]

    index_methods_considered_sa = [j for j in range(len(visualize_methods_sa))
                                      if visualize_methods_sa[j][1] is True]

    num_method_considered_sa         = len(index_methods_considered_sa)
    names_method_considered_sa       = [visualize_methods_sa[j][0] for j in index_methods_considered_sa]
    color_methods_considered_sa      = [visualize_methods_sa[j][2] for j in index_methods_considered_sa]
    line_style_methods_considered_sa = [visualize_methods_sa[j][3] for j in index_methods_considered_sa]
    marker_method_considered_sa      = [visualize_methods_sa[j][4] for j in index_methods_considered_sa]

    list_steps_sa = parameters_sa[4:]

    ##############################################
    # Elaborate Data inverse consistency results #
    ##############################################

    means_errors_ic = np.mean(errors_ic, axis=2)
    percentile_25_ic = np.percentile(errors_ic, 25,  axis=2)
    percentile_75_ic = np.percentile(errors_ic, 75,  axis=2)

    selected_error_ic = np.array([means_errors_ic[i, :] for i in index_methods_considered_ic])
    selected_percentile_25_ic = np.array([percentile_25_ic[i, :] for i in index_methods_considered_ic])
    selected_percentile_75_ic = np.array([percentile_75_ic[i, :] for i in index_methods_considered_ic])

    if verbose:
        print '---------------------------------'
        print 'Inverse consistency results table of the mean for ' + str(parameters_ic[3]) + ' samples.'
        print '---------------------------------'

        results_by_column = [[names_method_considered_ic[j]] + list(selected_error_ic[j, :])
                             for j in range(num_method_considered_ic)]

        print tabulate(results_by_column, headers=[''] + list(list_steps_ic))

    if plot_results_ic:

        plot_custom_step_versus_error_multiple(list_steps_ic,
                                               selected_error_ic,  # means
                                               names_method_considered_ic,
                                               y_error=[percentile_25_ic, percentile_75_ic],  # std
                                               input_parameters=parameters_ic,
                                               fig_tag=201,
                                               log_scale=True,
                                               additional_vertical_line=None,
                                               additional_field=None,
                                               kind='multiple_REAL_ic',
                                               titles=('inverse consistency errors vs iterations', 'Fields like:'),
                                               input_marker=marker_method_considered_ic,
                                               input_colors=color_methods_considered_ic,
                                               input_line_style=line_style_methods_considered_ic
                                               )
        #plt.show()

    ###############################################
    # Elaborate Data scalar associativity results #
    ###############################################

    means_errors_sa = np.mean(errors_sa, axis=2)
    percentile_25_sa = np.percentile(errors_sa, 25,  axis=2)
    percentile_75_sa = np.percentile(errors_sa, 75,  axis=2)

    selected_error_sa = np.array([means_errors_sa[i, :] for i in index_methods_considered_sa])
    selected_percentile_25_sa = np.array([percentile_25_sa[i, :] for i in index_methods_considered_sa])
    selected_percentile_75_sa = np.array([percentile_75_sa[i, :] for i in index_methods_considered_sa])

    if verbose:
        print '---------------------------------'
        print 'Inverse consistency results table of the mean for ' + str(parameters_ic[3]) + ' samples.'
        print '---------------------------------'

        results_by_column = [[names_method_considered_sa[j]] + list(selected_error_sa[j, :])
                             for j in range(num_method_considered_sa)]

        print tabulate(results_by_column, headers=[''] + list(list_steps_sa))

    if plot_results_sa:

        steps_for_all = np.array(list(list_steps_sa) * num_method_considered_sa).reshape(selected_error_sa.shape)

        plot_custom_time_error_steps(steps_for_all,
                                     selected_error_sa,
                                     fig_tag=202,
                                     y_error=[selected_percentile_25_sa, selected_percentile_75_sa],
                                     label_lines=names_method_considered_sa,
                                     additional_field=None,
                                     kind='multiple_REAL_ic',
                                     titles=('Scalar associativity, percentile', 'Field sample'),
                                     x_log_scale=False,
                                     y_log_scale=True,
                                     input_parameters=parameters_sa,
                                     input_marker=marker_method_considered_sa,
                                     input_colors=color_methods_considered_sa,
                                     input_line_style=line_style_methods_considered_sa,
                                     legend_location='upper right')

    if plot_final_graph:
        plot_ic_and_sa(list_steps_ic,
                        selected_error_ic,
                        color_methods_considered_ic,
                        line_style_methods_considered_ic,
                        marker_method_considered_ic,
                        names_method_considered_ic,
                        'upper right',
                        #
                        list_steps_sa,
                        selected_error_sa,
                        color_methods_considered_sa,
                        line_style_methods_considered_sa,
                        marker_method_considered_sa,
                        names_method_considered_sa,
                        'upper right',
                        #
                        y_error_ic=None,
                        y_error_sa=None,
                        fig_tag=120)


    plt.show()
