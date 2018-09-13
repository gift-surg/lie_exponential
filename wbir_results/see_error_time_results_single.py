"""
Method aimed to produce the images from the data saved in the error_exponential package
in the main folder.
Figures are formatted to be proposed for the publication.
"""
import numpy as np
import os
from tabulate import tabulate
from sympy.core.cache import clear_cache
import matplotlib.pyplot as plt
import pickle

from path_manager import pfo_results
from visualizer.graphs_and_stats_new_2 import single_graph


if __name__ == "__main__":

    clear_cache()

    #######################
    # Collect information #
    #######################

    see_se2   = False
    see_hom   = True
    see_gauss = False
    see_real  = False

    ## Select which method must be visualized

    #                          name        visualize  colour    line-style marker
    visualize_methods_se2 = [['ss',        True,      'b',      '-',     '+'],
                             ['ss_aei',   True,      'b',      '--',     '+'],
                             ['ss_ei',    True,      'g',      '-',     '.'],
                             ['ss_rk4',   True,      'r',      '--',     '.'],
                             ['midpoint',  True,      'c',      '-',     '.'],
                             ['euler',     True,      'g',      '-',     '>'],
                             ['euler_mod', True,      'm',      '-',     '>'],
                             ['euler_aei', True,      'm',      '--',     '>'],
                             ['heun',      True,      'k',      '-',     '.'],
                             ['heun_mod',  True,      'k',      '--',     '.'],
                             ['rk4',       True,      'y',      '--',     'x']]

    all_methods = [visualize_methods_se2[j][0] for j in range(11)]

    index_methods_considered_se2 = [j for j in range(len(visualize_methods_se2))
                                    if visualize_methods_se2[j][1] is True]
    num_method_considered_se2         = len(index_methods_considered_se2)
    names_method_considered_se2       = [visualize_methods_se2[j][0] for j in index_methods_considered_se2]
    color_methods_considered_se2      = [visualize_methods_se2[j][2] for j in index_methods_considered_se2]
    line_style_methods_considered_se2 = [visualize_methods_se2[j][3] for j in index_methods_considered_se2]
    marker_method_considered_se2      = [visualize_methods_se2[j][4] for j in index_methods_considered_se2]

    #                          name        visualize  colour    line-style marker
    visualize_methods_hom = [['ss',        True,      'b',      '-',     '+'],
                             ['ss_aei',   True,      'b',      '--',     '+'],
                             ['ss_ei',    True,      'g',      '-',     '.'],
                             ['ss_rk4',   True,      'r',      '--',     '.'],
                             ['midpoint',  True,      'c',      '-',     '.'],
                             ['euler',     True,      'g',      '-',     '>'],
                             ['euler_mod', True,      'm',      '-',     '>'],
                             ['euler_aei', True,      'm',      '--',     '>'],
                             ['heun',      True,      'k',      '-',     '.'],
                             ['heun_mod',  True,      'k',      '--',     '.'],
                             ['rk4',       True,      'y',      '--',     'x']]

    all_methods = [visualize_methods_hom[j][0] for j in range(11)]

    index_methods_considered_hom = [j for j in range(len(visualize_methods_hom))
                                    if visualize_methods_hom[j][1] is True]
    num_method_considered_hom         = len(index_methods_considered_se2)
    names_method_considered_hom       = [visualize_methods_hom[j][0] for j in index_methods_considered_hom]
    color_methods_considered_hom      = [visualize_methods_hom[j][2] for j in index_methods_considered_hom]
    line_style_methods_considered_hom = [visualize_methods_hom[j][3] for j in index_methods_considered_hom]
    marker_method_considered_hom      = [visualize_methods_hom[j][4] for j in index_methods_considered_hom]

    #                            name        visualize  colour line-style  marker
    visualize_methods_gauss = [['ss',        True,      'b',      '-',     '+'],
                               ['gss_aei',   True,      'b',      '--',     '+'],
                               ['gss_ei',    True,      'g',      '-',     '.'],
                               ['gss_rk4',   True,      'r',      '--',     '.'],
                               ['midpoint',  True,      'c',      '-',     '.'],
                               ['euler',     True,      'g',      '-',     '>'],
                               ['euler_mod', True,      'm',      '-',     '>'],
                               ['euler_aei', True,      'm',      '--',     '>'],
                               ['heun',      True,      'k',      '-',     '.'],
                               ['heun_mod',  True,      'k',      '--',     '.'],
                               ['rk4',       True,     'y',      '--',     'x']]

    index_methods_considered_gauss = [j for j in range(len(visualize_methods_gauss))
                                      if visualize_methods_gauss[j][1] is True]
    num_method_considered_gauss         = len(index_methods_considered_gauss)
    names_method_considered_gauss       = [visualize_methods_gauss[j][0] for j in index_methods_considered_gauss]
    color_methods_considered_gauss      = [visualize_methods_gauss[j][2] for j in index_methods_considered_gauss]
    line_style_methods_considered_gauss = [visualize_methods_gauss[j][3] for j in index_methods_considered_gauss]
    marker_method_considered_gauss      = [visualize_methods_gauss[j][4] for j in index_methods_considered_gauss]

    #                          name         visualize  colour line-style  marker
    visualize_methods_real = [['ss',        True,      'b',      '-',     '+'],
                              ['gss_aei',   True,      'b',      '--',     '+'],
                              ['gss_ei',    True,      'g',      '-',     '.'],
                              ['gss_rk4',   True,      'r',      '--',     '.'],
                              ['midpoint',  True,      'c',      '-',     '.'],
                              ['euler',     True,      'g',      '-',     '>'],
                              ['euler_mod', True,      'm',      '-',     '>'],
                              ['euler_aei', True,      'm',      '--',     '>'],
                              ['heun',      True,      'k',      '-',     '.'],
                              ['heun_mod',  True,      'k',      '--',     '.'],
                              ['rk4',       True,     'y',      '--',     'x']]

    index_methods_considered_real = [j for j in range(len(visualize_methods_real))
                                      if visualize_methods_real[j][1] is True]
    num_method_considered_real         = len(index_methods_considered_real)
    names_method_considered_real       = [visualize_methods_real[j][0] for j in index_methods_considered_real]
    color_methods_considered_real      = [visualize_methods_real[j][2] for j in index_methods_considered_real]
    line_style_methods_considered_real = [visualize_methods_real[j][3] for j in index_methods_considered_real]
    marker_method_considered_real      = [visualize_methods_real[j][4] for j in index_methods_considered_real]

    ### Load results ###
    path_to_shared_results = os.path.join(pfo_results, 'sharing_folder')

    # se2 #
    suffix = 'se2'
    res_errors_se2            = np.load(os.path.join(path_to_shared_results, 'errors_' + suffix + '.npy'))
    res_times_se2             = np.load(os.path.join(path_to_shared_results, 'comp_time_' + suffix + '.npy'))
    res_exp_methods_param_se2 = np.load(os.path.join(path_to_shared_results, 'exp_methods_param_' + suffix + '.npy'))
    res_transf_param_se2      = np.load(os.path.join(path_to_shared_results, 'transformation_param_' + suffix + '.npy'))

    # HOM #
    res_errors_hom = np.load(os.path.join(path_to_shared_results, 'comp_time_hom_5.npy'))
    res_times_hom = np.load(os.path.join(path_to_shared_results, 'errors_hom_5.npy'))

    with open(os.path.join(path_to_shared_results, 'exp_methods_table_hom_5'), 'rb') as f:
        res_exp_methods_param_hom = pickle.load(f)

    with open(os.path.join(path_to_shared_results, 'exp_methods_param_hom_5'), 'rb') as f:
        res_transf_param_hom = pickle.load(f)

    # Gauss #
    suffix = 'gauss'
    res_errors_gauss            = np.load(os.path.join(path_to_shared_results, 'errors_' + suffix + '.npy'))
    res_times_gauss             = np.load(os.path.join(path_to_shared_results, 'comp_time_' + suffix + '.npy'))
    res_exp_methods_param_gauss = np.load(os.path.join(path_to_shared_results, 'exp_methods_param_' + suffix + '.npy'))
    res_transf_param_gauss      = np.load(os.path.join(path_to_shared_results, 'transformation_param_' + suffix + '.npy'))

    # Real #
    suffix = 'real'
    res_errors_real            = np.load(os.path.join(path_to_shared_results, 'errors_' + suffix + '.npy'))
    res_times_real             = np.load(os.path.join(path_to_shared_results, 'comp_time_' + suffix + '.npy'))
    res_exp_methods_param_real = np.load(os.path.join(path_to_shared_results, 'exp_methods_param_' + suffix + '.npy'))
    res_transf_param_real      = np.load(os.path.join(path_to_shared_results, 'transformation_param_' + suffix + '.npy'))

    suffix = ''

    print '------------'
    print 'Data loaded!'
    print '------------'

    ###################################
    # Elaborate and Print information #
    ###################################

    if see_se2:

        # se2 1 - Table methods, steps, plot color, line-style, marker.
        results_errors_by_slice_se2 = [[all_methods[j]] + [res_exp_methods_param_se2[j][2]] +
                                       [visualize_methods_se2[j][1]] +
                                       [visualize_methods_se2[j][2]] + [visualize_methods_se2[j][3]] +
                                       [visualize_methods_se2[j][4]]
                                       for j in range(11)]

        # se2 2 - message
        list_of_steps_se2 = [int(res_transf_param_se2[j]) for j in range(10, 24)]
        msg_se2 = "Se2 generated SVF" \
                  ".\n Dimension of each transformation: " + str(res_transf_param_se2[:3]) + \
                  ".\n Number of samples " + str(res_transf_param_se2[3]) + \
                  ".\n Interval theta: " + str(res_transf_param_se2[4]) + ", " + str(res_transf_param_se2[5]) + \
                  ".\n interval tx " + str(res_transf_param_se2[6]) + ", " + str(res_transf_param_se2[7]) + \
                  ".\n interval ty " + str(res_transf_param_se2[8]) + ", " + str(res_transf_param_se2[9]) + \
                  ".\n Steps considered " + str(list_of_steps_se2)

        # se2 3 - Remove the methods we do not want to visualize from the matrices:
        mean_error_se2 = np.mean(res_errors_se2, axis=2)
        selected_error_se2 = np.array([mean_error_se2[i, :] for i in index_methods_considered_se2])

        mean_time_se2 = np.mean(res_times_se2, axis=2)
        selected_time_se2 = np.array([mean_time_se2[i, :] for i in index_methods_considered_se2])

        selected_error_by_slice_se2 = [[names_method_considered_se2[j]] + list(selected_error_se2[j, :])
                                       for j in range(num_method_considered_se2)]

        selected_time_by_slice_se2 = [[names_method_considered_se2[j]] + list(selected_time_se2[j, :])
                                      for j in range(num_method_considered_se2)]

        ### se2 - 1 ###
        print '---------------------------------------------'
        print 'Se2: Controller'
        print '---------------------------------------------'
        print tabulate(results_errors_by_slice_se2,
                       headers=['method', 'default steps', 'plot (T/F)', 'colour', 'line-style', 'marker'])
        print '---------------------------------------------'

        # se2 - 2
        print
        print msg_se2
        print

        # se2 - 3
        print 'Errors: '
        print '---------------------------------------------'
        print tabulate(selected_error_by_slice_se2,
                       headers=['method'] + [str(j) for j in list_of_steps_se2])
        print '---------------------------------------------'
        print ''
        print 'Times: '
        print '---------------------------------------------'
        print tabulate(selected_time_by_slice_se2,
                       headers=['method'] + [str(j) for j in list_of_steps_se2])
        print '---------------------------------------------'
        print

        single_graph(selected_time_se2,
                     selected_error_se2,
                     names_method_considered_se2,
                     color_methods_considered_se2,
                     line_style_methods_considered_se2,
                     marker_method_considered_se2,
                     'upper right',
                     fig_tag=1,
                     input_title='')

    if see_hom:

        # se2 1 - Table methods, steps, plot color, line-style, marker.
        results_errors_by_slice_hom = [[all_methods[j]] + [res_exp_methods_param_hom[j][2]] +
                                       [visualize_methods_hom[j][1]] +
                                       [visualize_methods_hom[j][2]] + [visualize_methods_hom[j][3]] +
                                       [visualize_methods_hom[j][4]]
                                       for j in range(11)]

        # se2 2 - message
        list_of_steps_hom = res_transf_param_hom[8:]
        msg_hom = "HOM generated SVF" \
                  ".\n Dimension of each transformation: " + str(res_transf_param_hom[:3]) + \
                  ".\n Number of samples " + str(res_transf_param_hom[3]) + \
                  ".\n Interval theta: " + str(res_transf_param_hom[4]) + ", " + str(res_transf_param_hom[5]) + \
                  ".\n interval tx " + str(res_transf_param_hom[6]) + ", " + str(res_transf_param_hom[7]) + \
                  ".\n interval ty " + str(res_transf_param_hom[8]) + ", " + str(res_transf_param_hom[9]) + \
                  ".\n Steps considered " + str(list_of_steps_hom)

        # se2 3 - Remove the methods we do not want to visualize from the matrices:
        mean_error_hom = np.mean(res_errors_hom, axis=2)
        selected_error_hom = np.array([mean_error_hom[i, :] for i in index_methods_considered_hom])

        mean_time_hom = np.mean(res_times_hom, axis=2)
        selected_time_hom = np.array([mean_time_hom[i, :] for i in index_methods_considered_hom])

        selected_error_by_slice_hom = [[names_method_considered_hom[j]] + list(selected_error_hom[j, :])
                                       for j in range(num_method_considered_hom)]

        selected_time_by_slice_hom = [[names_method_considered_hom[j]] + list(selected_time_hom[j, :])
                                      for j in range(num_method_considered_hom)]

        ### se2 - 1 ###
        print '---------------------------------------------'
        print 'Se2: Controller'
        print '---------------------------------------------'
        print tabulate(results_errors_by_slice_hom,
                       headers=['method', 'default steps', 'plot (T/F)', 'colour', 'line-style', 'marker'])
        print '---------------------------------------------'

        # se2 - 2
        print
        print msg_hom
        print

        # se2 - 3
        print 'Errors: '
        print '---------------------------------------------'
        print tabulate(selected_error_by_slice_hom,
                       headers=['method'] + [str(j) for j in list_of_steps_hom])
        print '---------------------------------------------'
        print ''
        print 'Times: '
        print '---------------------------------------------'
        print tabulate(selected_time_by_slice_hom,
                       headers=['method'] + [str(j) for j in list_of_steps_hom])
        print '---------------------------------------------'
        print

        single_graph(selected_error_hom,
                     selected_time_hom,
                     names_method_considered_hom,
                     color_methods_considered_hom,
                     line_style_methods_considered_hom,
                     marker_method_considered_hom,
                     'upper right',
                     fig_tag=1,
                     input_title='')





    if see_gauss:

        # Gauss 1 - Table methods, steps, plot color, line-style, marker.
        results_errors_by_slice_gauss = [[all_methods[j]] + [res_exp_methods_param_gauss[j][2]] +
                                         [visualize_methods_gauss[j][1]] +
                                         [visualize_methods_gauss[j][2]] + [visualize_methods_gauss[j][3]] +
                                         [visualize_methods_gauss[j][4]]
                                         for j in range(11)]

        # Gauss 2 - message
        list_of_steps_gauss = res_transf_param_gauss[8:]
        msg_gauss = "Gauss generated SVF" \
                  ".\n Dimension of each transformation: " + str(res_transf_param_gauss[:3]) + \
                  ".\n Number of samples " + str(res_transf_param_gauss[3]) + \
                  ".\n Sigma init: " + str(res_transf_param_gauss[4]) + \
                  ".\n Sigma gaussian filter " + str(res_transf_param_gauss[5]) + \
                  ".\n Ground method, steps " + str(res_transf_param_gauss[6]) + ", " + str(res_transf_param_gauss[7]) + \
                  ".\n Steps considered " + str(list_of_steps_gauss)

        # Gauss 3 - Remove the methods we do not want to visualize from the matrices:
        mean_error_gauss = np.mean(res_errors_gauss, axis=2)
        selected_error_gauss = np.array([mean_error_gauss[i, :] for i in index_methods_considered_gauss])

        mean_time_gauss = np.mean(res_times_gauss, axis=2)
        selected_time_gauss = np.array([mean_time_gauss[i, :] for i in index_methods_considered_gauss])

        selected_error_by_slice_gauss = [[names_method_considered_gauss[j]] + list(selected_error_gauss[j, :])
                                         for j in range(num_method_considered_gauss)]

        selected_time_by_slice_gauss = [[names_method_considered_gauss[j]] + list(selected_time_gauss[j, :])
                                        for j in range(num_method_considered_gauss)]

        ### Gauss - 1 ###
        print '---------------------------------------------'
        print 'Gauss: Controller'
        print '---------------------------------------------'
        print tabulate(results_errors_by_slice_gauss,
                       headers=['method', 'default steps', 'plot (T/F)', 'colour', 'line-style', 'marker'])
        print '---------------------------------------------'

        # gauss - 2
        print
        print msg_gauss
        print

        # gauss - 3
        print 'Errors: '
        print '---------------------------------------------'
        print tabulate(selected_error_by_slice_gauss,
                       headers=['method'] + [str(j) for j in list_of_steps_gauss])
        print '---------------------------------------------'
        print ''
        print 'Times: '
        print '---------------------------------------------'
        print tabulate(selected_time_by_slice_gauss,
                       headers=['method'] + [str(j) for j in list_of_steps_gauss])
        print '---------------------------------------------'
        print

        single_graph(selected_time_gauss,
                     selected_error_gauss,
                     names_method_considered_gauss,
                     color_methods_considered_gauss,
                     line_style_methods_considered_gauss,
                     marker_method_considered_gauss,
                     'upper right',
                     fig_tag=2,
                     input_title='Gauss generated SVF')
    #

    if see_real:

        # Real 1 - Table methods, steps, plot color, line-style, marker.
        results_errors_by_slice_real = [[all_methods[j]] + [res_exp_methods_param_real[j][2]] +
                                         [visualize_methods_real[j][1]] +
                                         [visualize_methods_real[j][2]] + [visualize_methods_real[j][3]] +
                                         [visualize_methods_real[j][4]]
                                         for j in range(11)]

        # Real 2 - message
        list_of_steps_real = res_transf_param_real[6:]
        msg_real = "Real data generated SVF" \
                  ".\n Dimension of each transformation: " + str(res_transf_param_real[:3]) + \
                  ".\n Id samples " + str(res_transf_param_real[3]) + \
                  ".\n Ground method, steps " + str(res_transf_param_real[4]) + ", " + str(res_transf_param_real[5]) + \
                  ".\n Steps considered " + str(list_of_steps_real)

        # Real 3 - Remove the methods we do not want to visualize from the matrices:
        mean_error_real = np.mean(res_errors_real, axis=2)
        selected_error_real = np.array([mean_error_real[i, :] for i in index_methods_considered_real])

        mean_time_real = np.mean(res_times_real, axis=2)
        selected_time_real = np.array([mean_time_real[i, :] for i in index_methods_considered_real])

        selected_error_by_slice_real = [[names_method_considered_real[j]] + list(selected_error_real[j, :])
                                         for j in range(num_method_considered_real)]

        selected_time_by_slice_real = [[names_method_considered_real[j]] + list(selected_time_real[j, :])
                                        for j in range(num_method_considered_real)]

        ### Real - 1 ###
        print '---------------------------------------------'
        print 'Real: Controller'
        print '---------------------------------------------'
        print tabulate(results_errors_by_slice_real,
                       headers=['method', 'default steps', 'plot (T/F)', 'colour', 'line-style', 'marker'])
        print '---------------------------------------------'

        # Real - 2
        print
        print msg_real
        print

        # Real - 3
        print 'Errors: '
        print '---------------------------------------------'
        print tabulate(selected_error_by_slice_real,
                       headers=['method'] + [str(j) for j in list_of_steps_real])
        print '---------------------------------------------'
        print ''
        print 'Times: '
        print '---------------------------------------------'
        print tabulate(selected_time_by_slice_real,
                       headers=['method'] + [str(j) for j in list_of_steps_real])
        print '---------------------------------------------'

        single_graph(selected_time_real,
                     selected_error_real,
                     names_method_considered_real,
                     color_methods_considered_real,
                     line_style_methods_considered_real,
                     marker_method_considered_real,
                     'upper right',
                     fig_tag=3,
                     input_title='Real data SVF')

    ####################
    # Plot information #
    ####################

    plt.show()

