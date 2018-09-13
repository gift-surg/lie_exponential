"""
Method aimed to produce the images from the data saved in the error_exponential package
in the main folder.
Figures are formatted to be proposed for the publication.

Collects and shows data produced in time_vs_error_multiple_steps_multiple_xxx.py
 xxx = se2, gauss, real
"""

import numpy as np
import os
from tabulate import tabulate
from sympy.core.cache import clear_cache

from path_manager import pfo_results, pfo_notes_tables


if __name__ == "__main__":

    clear_cache()

    #######################
    # Collect information #
    #######################

    see_se2   = True
    see_gauss = False
    see_real  = False

    ### Load results ###
    path_to_shared_results = os.path.join(pfo_results, 'sharing_folder')

    # se2 #
    suffix = 'se2'
    res_errors_se2            = np.load(os.path.join(path_to_shared_results, 'errors_' + suffix + '.npy'))
    res_times_se2             = np.load(os.path.join(path_to_shared_results, 'comp_time_' + suffix + '.npy'))
    res_exp_methods_param_se2 = np.load(os.path.join(path_to_shared_results, 'exp_methods_param_' + suffix + '.npy'))
    res_transf_param_se2      = np.load(os.path.join(path_to_shared_results, 'transformation_param_' + suffix + '.npy'))

    # Gauss #
    suffix = 'gauss'
    res_errors_gauss            = np.load(os.path.join(path_to_shared_results, 'errors_' + suffix + '.npy'))
    res_times_gauss             = np.load(os.path.join(path_to_shared_results, 'comp_time_' + suffix + '.npy'))
    res_exp_methods_param_gauss = np.load(os.path.join(path_to_shared_results, 'exp_methods_param_' + suffix + '.npy'))
    res_transf_param_gauss    = np.load(os.path.join(path_to_shared_results, 'transformation_param_' + suffix + '.npy'))

    # Real #
    suffix = 'real'
    res_errors_real            = np.load(os.path.join(path_to_shared_results, 'errors_' + suffix + '.npy'))
    res_times_real             = np.load(os.path.join(path_to_shared_results, 'comp_time_' + suffix + '.npy'))
    res_exp_methods_param_real = np.load(os.path.join(path_to_shared_results, 'exp_methods_param_' + suffix + '.npy'))
    res_transf_param_real     = np.load(os.path.join(path_to_shared_results, 'transformation_param_' + suffix + '.npy'))

    suffix = ''

    print '------------'
    print 'Data loaded!'
    print '------------'

    #################################################
    # Elaborate and Print information and Table Se2 #
    #################################################

    if see_se2:

        ## Select which row cols must be visualized
        #                          name        keep row
        visualize_methods_se2 = [['ss',        True],
                                 ['gss_aei',   True],
                                 ['gss_ei',    True],
                                 ['gss_rk4',   True],
                                 ['midpoint',  True],
                                 ['euler',     True],
                                 ['euler_mod', True],
                                 ['euler_aei', True],
                                 ['heun',      True],
                                 ['heun_mod',  True],
                                 ['rk4',       True]]

        visualize_steps_se2 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40],   # Steps
                               [1, 1, 0, 0, 0, 0, 1, 0, 0, 1,  0,  1,  1,  0]]    # Keep column

        all_methods = [visualize_methods_se2[j][0] for j in range(11)]

        index_rows_se2 = [j for j in range(len(visualize_methods_se2))
                          if visualize_methods_se2[j][1] is True]
        num_rows_se2   = len(index_rows_se2)

        index_cols_se2 = [j for j in range(len(visualize_steps_se2[0]))
                          if visualize_steps_se2[1][j] == 1]
        num_cols_se2   = len(index_cols_se2)

        names_rows_se2       = [visualize_methods_se2[j][0] for j in index_rows_se2]
        names_cols_se2       = [visualize_steps_se2[0][j] for j in index_cols_se2]

        msg_se2 = "Se2 generated SVF" \
                  ".\n Dimension of each transformation: " + str(res_transf_param_se2[:3]) + \
                  ".\n Number of samples " + str(res_transf_param_se2[3]) + \
                  ".\n Interval theta: " + str(res_transf_param_se2[4]) + ", " + str(res_transf_param_se2[5]) + \
                  ".\n interval tx " + str(res_transf_param_se2[6]) + ", " + str(res_transf_param_se2[7]) + \
                  ".\n interval ty " + str(res_transf_param_se2[8]) + ", " + str(res_transf_param_se2[9]) + \
                  ".\n Steps considered " + str(names_cols_se2)

        # se2 3 - Remove the methods and steps, and round as we like

        decimals_error_se2 = 4
        decimals_time_se2 = 3

        mean_error_se2 = np.mean(res_errors_se2, axis=2)
        mean_time_se2 = np.mean(res_times_se2, axis=2)

        selected_errors_se2 = np.array([[np.round(mean_error_se2[i, j], decimals_error_se2) for j in list(index_cols_se2)]
                                                     for i in list(index_rows_se2)])
        selected_times_se2 = np.array([[np.round(mean_time_se2[i, j], decimals_time_se2) for j in list(index_cols_se2)]
                                                   for i in list(index_rows_se2)])

        selected_errors_to_tabulate_se2 = [[names_rows_se2[i]] + list(selected_errors_se2[i, :])
                                           for i in range(selected_errors_se2.shape[0])]
        selected_times_to_tabulate_se2  = [[names_rows_se2[i]] + list(selected_times_se2[i, :])
                                           for i in range(selected_times_se2.shape[0])]

        ### se2 - 1 ###
        print '---------------------------------------------'
        print 'Se2: Controller'
        print '---------------------------------------------'
        print 'Methods chosen: ' + str(names_rows_se2)
        print 'Steps chosen: ' + str(names_cols_se2)
        print '---------------------------------------------'

        # se2 - 2
        print
        print msg_se2
        print

        # se2 - 3
        print '--------'
        print 'Errors: '
        print '--------'
        print tabulate(selected_errors_to_tabulate_se2,
                       headers=['method'] + [str(j) for j in names_cols_se2])
        print ''
        print '-------'
        print 'Times: '
        print '-------'
        print tabulate(selected_times_to_tabulate_se2,
                       headers=['method'] + [str(j) for j in names_cols_se2])
        print

        assert selected_times_se2.shape[0] == selected_errors_se2.shape[0]
        assert selected_times_se2.shape[1] == selected_errors_se2.shape[1]

        # Shuffle data:selected_times_se2
        shuffled_data_se2 = np.zeros([selected_times_se2.shape[0], 2*selected_times_se2.shape[1]])
        header_caption_se2 = [' '] * (2*selected_times_se2.shape[1])

        for col in range(shuffled_data_se2.shape[1]):
            if col % 2 == 0:
                shuffled_data_se2[:, col] = selected_errors_se2[:, col/2]
                header_caption_se2[col] = names_cols_se2[col/2]
            else:
                shuffled_data_se2[:, col] = selected_times_se2[:, col/2]

        shuffled_times_error_to_tabulate_se2 = [[names_rows_se2[i]] +
                                                list(shuffled_data_se2[i, :])
                                                for i in range(shuffled_data_se2.shape[0])]

        print '--------------'
        print 'Errors/Times: '
        print '--------------'
        print tabulate(shuffled_times_error_to_tabulate_se2,
                       headers=['method'] + header_caption_se2)
        print

        # Save the table in latex format!
        f = open(os.path.join(pfo_notes_tables, 'se2.tex'), 'w')
        f.write(tabulate(shuffled_times_error_to_tabulate_se2,
                         headers=['method'] + header_caption_se2,
                         tablefmt="latex"))
        f.close()

    if see_gauss:
        ## Select which row cols must be visualized
        #                          name        keep row
        visualize_methods_gauss = [['ss',        True],
                                   ['gss_aei',   True],
                                   ['gss_ei',    True],
                                   ['gss_rk4',   True],
                                   ['midpoint',  True],
                                   ['euler',     True],
                                   ['euler_mod', True],
                                   ['euler_aei', True],
                                   ['heun',      True],
                                   ['heun_mod',  False],
                                   ['rk4',       False]]

        visualize_steps_gauss = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40],   # Steps
                                 [1, 1, 0, 0, 0, 0, 1, 0, 0, 1,  0,  1,  1,  0]]    # Keep column

        all_methods = [visualize_methods_gauss[j][0] for j in range(11)]

        index_rows_gauss = [j for j in range(len(visualize_methods_gauss))
                            if visualize_methods_gauss[j][1] is True]
        num_rows_gauss   = len(index_rows_gauss)

        index_cols_gauss = [j for j in range(len(visualize_steps_gauss[0]))
                            if visualize_steps_gauss[1][j] == 1]
        num_cols_gauss   = len(index_cols_gauss)

        names_rows_gauss       = [visualize_methods_gauss[j][0] for j in index_rows_gauss]
        names_cols_gauss       = [visualize_steps_gauss[0][j] for j in index_cols_gauss]

        msg_gauss = "Se2 generated SVF" \
                  ".\n Dimension of each transformation: " + str(res_transf_param_gauss[:3]) + \
                  ".\n Number of samples " + str(res_transf_param_gauss[3]) + \
                  ".\n Interval theta: " + str(res_transf_param_gauss[4]) + ", " + str(res_transf_param_gauss[5]) + \
                  ".\n interval tx " + str(res_transf_param_gauss[6]) + ", " + str(res_transf_param_gauss[7]) + \
                  ".\n interval ty " + str(res_transf_param_gauss[8]) + ", " + str(res_transf_param_gauss[9]) + \
                  ".\n Steps considered " + str(names_cols_gauss)

        # se2 3 - Remove the methods and steps, and round as we like

        decimals_error_gauss = 6
        decimals_time_gauss = 3

        mean_error_gauss = np.mean(res_errors_gauss, axis=2)
        mean_time_gauss = np.mean(res_times_gauss, axis=2)

        selected_errors_gauss = np.array([[np.round(mean_error_gauss[i, j], decimals_error_gauss) for j in list(index_cols_gauss)]
                                                     for i in list(index_rows_gauss)])
        selected_times_gauss = np.array([[np.round(mean_time_gauss[i, j], decimals_time_gauss) for j in list(index_cols_gauss)]
                                                   for i in list(index_rows_gauss)])

        selected_errors_to_tabulate_gauss = [[names_rows_gauss[i]] + list(selected_errors_gauss[i, :])
                                           for i in range(selected_errors_gauss.shape[0])]
        selected_times_to_tabulate_gauss  = [[names_rows_gauss[i]] + list(selected_times_gauss[i, :])
                                           for i in range(selected_times_gauss.shape[0])]

        ### Gauss - 1 ###
        print '---------------------------------------------'
        print 'Gauss: Controller'
        print '---------------------------------------------'
        print 'Methods chosen: ' + str(names_rows_gauss)
        print 'Steps chosen: ' + str(names_cols_gauss)
        print '---------------------------------------------'

        # Gauss - 2
        print
        print msg_gauss
        print

        # Gauss - 3
        print '--------'
        print 'Errors: '
        print '--------'
        print tabulate(selected_errors_to_tabulate_gauss,
                       headers=['method'] + [str(j) for j in names_cols_gauss])
        print ''
        print '-------'
        print 'Times: '
        print '-------'
        print tabulate(selected_times_to_tabulate_gauss,
                       headers=['method'] + [str(j) for j in names_cols_gauss])
        print

        assert selected_times_gauss.shape[0] == selected_errors_gauss.shape[0]
        assert selected_times_gauss.shape[1] == selected_errors_gauss.shape[1]

        # Shuffle data:selected_times_gauss
        shuffled_data_gauss = np.zeros([selected_times_gauss.shape[0], 2*selected_times_gauss.shape[1]])
        header_caption_gauss = [' '] * (2*selected_times_gauss.shape[1])

        for col in range(shuffled_data_gauss.shape[1]):
            if col % 2 == 0:
                shuffled_data_gauss[:, col] = selected_errors_gauss[:, col/2]
                header_caption_gauss[col] = names_cols_gauss[col/2]
            else:
                shuffled_data_gauss[:, col] = selected_times_gauss[:, col/2]

        shuffled_times_error_to_tabulate_gauss = [[names_rows_gauss[i]] +
                                                list(shuffled_data_gauss[i, :])
                                                for i in range(shuffled_data_gauss.shape[0])]

        print '--------------'
        print 'Errors/Times: '
        print '--------------'
        print tabulate(shuffled_times_error_to_tabulate_gauss,
                       headers=['method'] + header_caption_gauss)
        print

    if see_real:

        ## Select which row cols must be visualized
        #                          name        keep row
        visualize_methods_real = [['ss',        True],
                                  ['gss_aei',   True],
                                  ['gss_ei',    True],
                                  ['gss_rk4',   True],
                                  ['midpoint',  True],
                                  ['euler',     True],
                                  ['euler_mod', True],
                                  ['euler_aei', True],
                                  ['heun',      True],
                                  ['heun_mod',  True],
                                  ['rk4',       True]]

        visualize_steps_real = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30],   # Steps from main code
                                [1, 1, 0, 0, 0, 0, 1, 0, 0, 1,  0,  1,  1]]    # Keep column

        all_methods = [visualize_methods_real[j][0] for j in range(11)]

        index_rows_real = [j for j in range(len(visualize_methods_real))
                          if visualize_methods_real[j][1] is True]
        num_rows_real   = len(index_rows_real)

        index_cols_real = [j for j in range(len(visualize_steps_real[0]))
                          if visualize_steps_real[1][j] == 1]
        num_cols_real   = len(index_cols_real)

        names_rows_real       = [visualize_methods_real[j][0] for j in index_rows_real]
        names_cols_real       = [visualize_steps_real[0][j] for j in index_cols_real]

        msg_real = "Se2 generated SVF" \
                  ".\n Dimension of each transformation: " + str(res_transf_param_real[:3]) + \
                  ".\n Number of samples " + str(res_transf_param_real[3]) + \
                  ".\n Interval theta: " + str(res_transf_param_real[4]) + ", " + str(res_transf_param_real[5]) + \
                  ".\n interval tx " + str(res_transf_param_real[6]) + ", " + str(res_transf_param_real[7]) + \
                  ".\n interval ty " + str(res_transf_param_real[8]) + ", " + str(res_transf_param_real[9]) + \
                  ".\n Steps considered " + str(names_cols_real)

        # se2 3 - Remove the methods and steps, and round as we like

        decimals_error_real = 5
        decimals_time_real = 3

        mean_error_real = np.mean(res_errors_real, axis=2)
        mean_time_real = np.mean(res_times_real, axis=2)

        selected_errors_real = np.array([[np.round(mean_error_real[i, j], decimals_error_real) for j in list(index_cols_real)]
                                                     for i in list(index_rows_real)])
        selected_times_real = np.array([[np.round(mean_time_real[i, j], decimals_time_real) for j in list(index_cols_real)]
                                                   for i in list(index_rows_real)])

        selected_errors_to_tabulate_real = [[names_rows_real[i]] + list(selected_errors_real[i, :])
                                           for i in range(selected_errors_real.shape[0])]
        selected_times_to_tabulate_real  = [[names_rows_real[i]] + list(selected_times_real[i, :])
                                           for i in range(selected_times_real.shape[0])]

        ### Real - 1 ###
        print '---------------------------------------------'
        print 'Real: Controller'
        print '---------------------------------------------'
        print 'Methods chosen: ' + str(names_rows_real)
        print 'Steps chosen: ' + str(names_cols_real)
        print '---------------------------------------------'

        # Real - 2
        print
        print msg_real
        print

        # Real - 3
        print '--------'
        print 'Errors: '
        print '--------'
        print tabulate(selected_errors_to_tabulate_real,
                       headers=['method'] + [str(j) for j in names_cols_real])
        print ''
        print '-------'
        print 'Times: '
        print '-------'
        print tabulate(selected_times_to_tabulate_real,
                       headers=['method'] + [str(j) for j in names_cols_real])
        print

        assert selected_times_real.shape[0] == selected_errors_real.shape[0]
        assert selected_times_real.shape[1] == selected_errors_real.shape[1]

        # Shuffle data:selected_times_real
        shuffled_data_real = np.zeros([selected_times_real.shape[0], 2*selected_times_real.shape[1]])
        header_caption_real = [' '] * (2*selected_times_real.shape[1])

        for col in range(shuffled_data_real.shape[1]):
            if col % 2 == 0:
                shuffled_data_real[:, col] = selected_errors_real[:, col/2]
                header_caption_real[col] = names_cols_real[col/2]
            else:
                shuffled_data_real[:, col] = selected_times_real[:, col/2]

        shuffled_times_error_to_tabulate_real = [[names_rows_real[i]] +
                                                list(shuffled_data_real[i, :])
                                                for i in range(shuffled_data_real.shape[0])]

        print '--------------'
        print 'Errors/Times: '
        print '--------------'
        print tabulate(shuffled_times_error_to_tabulate_real,
                       headers=['method'] + header_caption_real)
        print
