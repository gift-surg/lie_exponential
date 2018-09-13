import os
import pickle
import time
from os.path import join as jph

import matplotlib.pyplot as plt
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from VECtorsToolkit.tools.transformations import se2
from VECtorsToolkit.tools.fields.generate_vf import generate_from_matrix
from VECtorsToolkit.tools.local_operations.lie_exponential import lie_exponential_scipy, lie_exponential
from VECtorsToolkit.tools.fields.queries import vf_norm

from controller import methods_t_s
from path_manager import pfo_results, pfo_notes_figures, pfo_notes_sharing
from visualizer.graphs_and_stats_new import plot_custom_boxplot, plot_custom_cluster

"""
Module for the computation of one 2d SVF generated with matrix of se2_a.
It compares the exponential computation with different methods for a number of
steps defined by the user.

"""


if __name__ == "__main__":

    clear_cache()

    ##################
    #   Controller   #
    ##################

    compute       = True
    verbose       = True
    save_external = True
    plot_results  = True

    #######################
    #   Path management   #
    #######################

    prefix_fn = 'exp_comparing_errors'
    kind      = 'SE2'
    number    = 'multiple'
    tag       = '_' + str(1)

    fin_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    fin_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    fin_csv_table_comp_time_output = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_cp_time'
    fin_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    fin_array_comp_time_output     = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_cp_time'
    fin_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'
    fin_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    fin_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    pfo_errors_times_results = jph(pfo_results, 'errors_times_results')

    os.system('mkdir -p {}'.format(pfo_errors_times_results))
    print("\nPath to results folder {}\n".format(pfo_errors_times_results))

    pfi_array_errors_output        = jph(pfo_errors_times_results, fin_array_errors_output + tag + '.npy')
    pfi_array_comp_time_output     = jph(pfo_errors_times_results, fin_array_comp_time_output + tag + '.npy')
    pfi_field                      = jph(pfo_errors_times_results, fin_field + tag + '.npy')
    pfi_transformation_parameters  = jph(pfo_errors_times_results, fin_transformation_parameters + tag + '.npy')
    pfi_numerical_method_table     = jph(pfo_errors_times_results, fin_numerical_methods_table + tag)
    pfi_figure_output              = jph(pfo_notes_figures, fin_figure_output + tag + '.pdf')
    pfi_csv_table_errors_output    = jph(pfo_notes_sharing, fin_csv_table_errors_output + '.csv')
    pfi_csv_table_comp_time_output = jph(pfo_notes_sharing, fin_csv_table_comp_time_output + '.csv')

    ####################
    #   Computations   #
    ####################

    if compute:  # or compute or load
        random_seed = 0

        if random_seed > 0:
            np.random.seed(random_seed)

        pp = 2     # passepartout
        s_i_o = 3  # spline interpolation order

        N = 5

        # Parameters SVF:
        x_1, y_1, z_1 = 20, 20, 5

        if z_1 == 1:
            omega = (x_1, y_1)
        else:
            omega = (x_1, y_1, z_1)

        interval_theta = (- np.pi / 8, np.pi / 8)
        epsilon        = np.pi / 12
        center         = (12, 13, 7, 13)  # where to locate the center of the random rotation

        parameters = [x_1, y_1, z_1,
                      N,                       # number of samples
                      interval_theta[0], interval_theta[1],  # interval of the rotations
                      center[0], center[1],      # interval of the center of the rotation x
                      center[2], center[3]]      # interval of the center of the rotation y

        # import methods from external file aaa_general_controller
        methods = methods_t_s

        index_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
        num_method_considered    = len(index_methods_considered)

        names_method_considered      = [methods[j][0] for j in range(len(methods)) if methods[j][1] is True]
        steps_methods_considered     = [methods[j][2] for j in range(len(methods)) if methods[j][1] is True]
        colour_methods_considered    = [methods[j][3] for j in range(len(methods)) if methods[j][1] is True]
        linestyle_methods_considered = [methods[j][4] for j in range(len(methods)) if methods[j][1] is True]
        markers_methods_considered   = [methods[j][5] for j in range(len(methods)) if methods[j][1] is True]

        ###########################
        #   Model: computations   #
        ###########################

        print '---------------------'
        print 'Computations started!'
        print '---------------------'

        errors = np.zeros([num_method_considered, N])  # Row: method, col: sampling
        res_time = np.zeros([num_method_considered, N])  # Row: method, col: sampling

        for s in range(N):  # sample

            # generate matrices
            m_0 = se2.se2g_randomgen_custom_center(interval_theta=interval_theta, interval_center=center,
                                                   epsilon_zero_avoidance=epsilon)
            dm_0 = se2.se2g_log(m_0)

            # Generate SVF
            svf_0        = generate_from_matrix(omega, dm_0.get_matrix, t=1, structure='algebra')
            disp_ground  = generate_from_matrix(omega, m_0.get_matrix, t=1, structure='group')

            for m in range(num_method_considered):  # method
                if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                    start = time.time()
                    disp_computed = lie_exponential_scipy(svf_0, integrator=names_method_considered[m],
                                                          max_steps=steps_methods_considered[m])
                    res_time[m] = (time.time() - start)

                else:
                    start = time.time()
                    disp_computed = lie_exponential(svf_0, algorithm=names_method_considered[m], s_i_o=s_i_o,
                                                    input_num_steps=steps_methods_considered[m])
                    res_time[m, s] = (time.time() - start)

                # compute error:
                errors[m, s] = vf_norm(disp_computed - disp_ground, passe_partout_size=pp, normalized=True)

            if verbose:

                results_by_column = [[met, err, tim]
                                     for met, err, tim
                                     in zip(names_method_considered, list(errors[:, s]), list(res_time[:, s]))]
                
                print '--------------------'
                print 'Sampling ' + str(s + 1) + '/' + str(N) + ' .'
                print '--------------------'
                print 'theta, tx, ty =    ' + str(m_0.get)
                print 'dtheta, dtx, dty = ' + str(dm_0.get)
                print '--------------------'
                print tabulate(results_by_column,
                               headers=['method', 'error', 'comp. time (sec)'])
                print '--------------------'

        ### Save data to folder ###
        np.save(pfi_array_errors_output,       errors)
        np.save(pfi_array_comp_time_output,    res_time)

        with open(pfi_transformation_parameters, 'wb') as f:
            pickle.dump(parameters, f)

        with open(pfi_numerical_method_table, 'wb') as f:
            pickle.dump(methods, f)

        print
        print '------------------------------------------'
        print 'Data computed and saved in external files!'
        print '------------------------------------------'

    else:
        errors     = np.load(pfi_array_errors_output)
        res_time   = np.load(pfi_array_comp_time_output)

        with open(pfi_transformation_parameters, 'rb') as f:
            parameters = pickle.load(f)

        with open(pfi_numerical_method_table, 'rb') as f:
            methods = pickle.load(f)

        print
        print '------------'
        print 'Data loaded!'
        print '------------'

        index_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
        num_method_considered    = len(index_methods_considered)

        names_method_considered       = [methods[j][0] for j in index_methods_considered]
        steps_methods_considered      = [methods[j][2] for j in index_methods_considered]
        colour_methods_considered     = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        markers_methods_considered     = [methods[j][5] for j in index_methods_considered]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print 'Error-bar and time for multiple se2 generated SVF'
        print '-------------------------------------------------'

        print '\nParameters of the transformation se2:'
        print 'Number of samples = ' + str(parameters[3])
        print 'domain = ' + str(parameters[:3])
        print 'interval theta = ' + str(parameters[4:6])
        print 'Omega, interval tx, ty = ' + str(parameters[6:])

        print '\n'
        print 'Methods and parameters:'
        print tabulate(methods,
                       headers=['name', 'compute (True/False)', 'num_steps'])
        print '\n'

        print 'List of the methods considered:'
        print names_method_considered
        print 'List of the steps of the methods considered'
        print steps_methods_considered

    ################################
    # Visualization and statistics #
    ################################
    
    mean_errors   = np.mean(errors, axis=1)
    mean_res_time = np.mean(res_time, axis=1)

    print mean_errors
    print len(mean_errors)

    results_by_column = [[met, err, tim]
                         for met, err, tim in zip(names_method_considered, list(mean_errors), list(mean_res_time))]

    print '\n'
    print 'Results and computational time:'
    print tabulate(results_by_column,
                   headers=['method', 'mean error', 'mean comp. time (sec)'])
    print '\n END'

    # plot results
    if plot_results:

        reordered_errors_for_plot = []
        reordered_times_for_plot = []
        for m in range(errors.shape[0]):
            reordered_errors_for_plot += [list(errors[m, :])]
            reordered_times_for_plot += [list(res_time[m, :])]

        # BOXPLOT custom

        plot_custom_boxplot(input_data=reordered_errors_for_plot,
                            input_names=names_method_considered,
                            fig_tag=11,
                            input_titles=('Error exponential map for multiple SE2-generated svf', 'field'),
                            kind='multiple_SE2',
                            window_title_input='bar_plot_multiple_se2',
                            additional_field=None,
                            log_scale=False,
                            input_parameters=parameters,

                            annotate_mean=True,
                            add_extra_annotation=mean_res_time)

        # SCATTER-PLOT custom

        plot_custom_cluster(reordered_errors_for_plot, reordered_times_for_plot,
                            fig_tag=22,
                            clusters_labels=names_method_considered,
                            clusters_colors=colour_methods_considered,
                            clusters_markers=markers_methods_considered)

        plt.show()

    ### Save figures in external folder ###

    if save_external:
        os.system('mkdir -p {}'.format(pfo_notes_figures))
        # Save table csv
        # np.savetxt(pfi_csv_table_errors_output, errors, delimiter=" & ")
        # np.savetxt(pfi_csv_table_comp_time_output, errors, delimiter=" & ")
        # Save image:
        plt.savefig(pfi_figure_output, format='pdf', dpi=400)
        print 'Figure ' + fin_figure_output + ' saved in the external folder ' + str(pfi_figure_output)
