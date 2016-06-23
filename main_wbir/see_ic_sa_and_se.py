import numpy as np
import os
from matplotlib import pyplot as plt
import pickle
from tabulate import tabulate

from utils.path_manager import path_to_sharing_folder
from visualizer.graphs_and_stats_new import plot_custom_step_versus_error_multiple, plot_custom_time_error_steps, plot_custom_step_error
from visualizer.graphs_and_stats_new_2 import plot_ic_sa_and_se


if __name__ == "__main__":

    verbose = True
    plot_results_ic = False
    plot_results_sa = False
    plot_results_se = False
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
                            ['ss_aei',   True,      'b',      '--',     'x'],
                            ['ss_ei',    True,      'r',      '--',     '.'],
                            ['ss_rk4',   True,      'k',      '-',     '.'],
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
                            ['ss_aei',   True,      'b',      '--',     'x'],
                            ['ss_ei',    True,      'r',      '--',     '.'],
                            ['ss_rk4',   True,      'k',      '-',     '.'],
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

    ######################################
    # Collect information stepwise error #
    ######################################

    # Load data stepwise error

    errors_se = np.load(os.path.join(path_to_sharing_folder, 'step_relative_errors_real_new.npy'))  # step_relative_errors_real_new

    with open(os.path.join(path_to_sharing_folder, 'step_relative_errors_transformation_parameters_real_new'), 'rb') as f:  ### ic_parameters_real_new
            parameters_se = pickle.load(f)

    print parameters_se

    print errors_se.shape

    #                            name        visualize  colour line-style  marker
    visualize_methods_se = [['ss',        True,      'b',      '-',     '+'],
                            ['ss_aei',   True,      'b',      '--',     'x'],
                            ['ss_ei',    True,      'r',      '--',     '.'],
                            ['ss_rk4',   True,      'k',      '-',     '.'],
                            ['midpoint',  False,      'c',      '-',     '.'],
                            ['euler',     True,      'g',      '-',     '>'],
                            ['euler_mod', False,      'm',      '-',     '>'],
                            ['euler_aei', True,      'm',      '--',     '>'],
                            ['heun',      False,      'k',      '-',     '.'],
                            ['heun_mod',  False,      'k',      '--',     '.'],
                            ['rk4',       True,     'y',      '--',     'x']]

    index_methods_considered_se = [j for j in range(len(visualize_methods_se))
                                      if visualize_methods_se[j][1] is True]
    num_method_considered_se         = len(index_methods_considered_se)
    names_method_considered_se       = [visualize_methods_se[j][0] for j in index_methods_considered_se]
    color_methods_considered_se      = [visualize_methods_se[j][2] for j in index_methods_considered_se]
    line_style_methods_considered_se = [visualize_methods_se[j][3] for j in index_methods_considered_se]
    marker_method_considered_se      = [visualize_methods_se[j][4] for j in index_methods_considered_se]

    max_steps_se = parameters_se[4]

    if verbose:

        print '----------------------------------------------------------'
        print 'Stepwise error for multiple REAL generated SVF'
        print '----------------------------------------------------------'

        print '\nParameters that generate the SVF'
        print 'string of subjects = ' + str(parameters_se[3])
        print 'domain = ' + str(parameters_se[:3])
        print 'max steps considered:'
        print str(parameters_se[4])

        print '\n'
        print 'Methods and parameters:'
        print tabulate(visualize_methods_se,
                       headers=['name', 'compute (True/False)', 'colour',  'line-style',   'marker'])
        print '\n'
        print 'List of the methods considered:'
        print names_method_considered_se
        print '---------------------------------------------'

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

    ##############################################
    # Elaborate Data Stepwise Error results #
    ##############################################

    means_errors_se = np.mean(errors_se, axis=2)
    percentile_25_se = np.percentile(errors_se, 25,  axis=2)
    percentile_75_se = np.percentile(errors_se, 75,  axis=2)

    selected_error_se = np.array([means_errors_se[i, :] for i in index_methods_considered_se])
    selected_percentile_25_se = np.array([percentile_25_se[i, :] for i in index_methods_considered_se])
    selected_percentile_75_se = np.array([percentile_75_se[i, :] for i in index_methods_considered_se])

    if verbose:
        print '---------------------------------'
        print 'Inverse consistency results table of the mean for ' + str(parameters_se[3]) + ' samples.'
        print '---------------------------------'

        results_by_column = [[names_method_considered_se[j]] + list(selected_error_se[j, :])
                             for j in range(num_method_considered_se)]

        print tabulate(results_by_column, headers=[''] + list(range(max_steps_se)))

    if plot_results_se:

        errors_mean_se = np.mean(errors_se, axis=2)

        plot_custom_step_error(range(1, max_steps_se-1),
                               errors_mean_se[:, 1:max_steps_se-1],  # here is the mean of the errors
                               names_method_considered_se,
                               input_parameters=parameters_se,
                               fig_tag=2,
                               kind='multiple_REAL',
                               log_scale=False,
                               input_colors=color_methods_considered_se,
                               window_title_input='step errors',
                               titles=('iterations vs. MEANS of the step-errors', ''),
                               additional_field=None,
                               legend_location='upper right',
                               input_line_style=line_style_methods_considered_se,
                               input_marker=marker_method_considered_se)

    ##################
    ## Plot 'em all! #
    ##################

    print len(range(0, max_steps_se))
    print selected_error_se.shape

    if plot_final_graph:
        plot_ic_sa_and_se(list_steps_ic,
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
                            range(3, max_steps_se+1),
                            selected_error_se[:,1:-1],
                            color_methods_considered_se,
                            line_style_methods_considered_se,
                            marker_method_considered_se,
                            names_method_considered_se,
                            'upper right',
                            #
                            y_error_ic=None,
                            y_error_sa=None,
                            y_error_se=None,
                            fig_tag=121)


    plt.show()
