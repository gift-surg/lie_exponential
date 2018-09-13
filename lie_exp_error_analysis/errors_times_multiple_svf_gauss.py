import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from VECtorsToolkit.tools.fields.generate_vf import generate_random
from VECtorsToolkit.tools.local_operations.lie_exponential import lie_exponential, lie_exponential_scipy
from VECtorsToolkit.tools.fields.queries import vf_norm

from controller import methods_t_s
from path_manager import pfo_notes_figures, pfo_notes_tables, pfo_results
from visualizer.graphs_and_stats_new import plot_custom_boxplot, plot_custom_cluster


"""
Module for the computation of the error of the exponential map.
Svf involved is more than one 2d SVF generated with a gaussian filter.
For this kind of svf there is no ground truth available. One of the numerical integrator can be chosen
as ground truth.
"""


if __name__ == "__main__":

    clear_cache()

    ##################
    ### Controller ###
    ##################

    compute = True  # or compute or load if computed before.
    verbose = True
    plot_results = True

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_errors'
    kind   = 'GAUSS'
    number = 'multiple'
    file_suffix  = '_' + str(1)

    fin_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    fin_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    fin_csv_table_comp_time_output = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_cp_time'
    fin_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    fin_array_comp_time_output     = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_cp_time'
    fin_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'
    fin_transf_parameters          = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    fin_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    pfo_error_time_results = os.path.join(pfo_results, 'errors_times_results')

    os.system('mkdir -p {}'.format(pfo_error_time_results))

    pfi_array_errors_output = os.path.join(pfo_error_time_results, fin_array_errors_output + file_suffix + '.npy')
    pfi_array_comp_time_output = os.path.join(pfo_error_time_results, fin_array_comp_time_output + file_suffix + '.npy')
    pfi_field = os.path.join(pfo_error_time_results, fin_field + file_suffix + '.npy')
    pfi_transformation_parameters = os.path.join(pfo_error_time_results, fin_transf_parameters + file_suffix + '.npy')
    pfi_numerical_method_table = os.path.join(pfo_error_time_results, fin_numerical_methods_table + file_suffix)

    # path to results external to the project:
    pfi_figure_output  = os.path.join(pfo_notes_figures, fin_figure_output + file_suffix + '.pdf')
    pfi_csv_table_errors_output = os.path.join(pfo_notes_tables, fin_csv_table_errors_output + '.csv')
    pfi_csv_table_comp_time_output = os.path.join(pfo_notes_tables, fin_csv_table_comp_time_output + '.csv')

    ####################
    ### Computations ###
    ####################

    if compute:

        random_seed = 0

        if random_seed > 0:
            np.random.seed(random_seed)

        s_i_o = 3
        pp = 2

        N = 50

        # Parameters SVF:
        x_1, y_1, z_1 = 60, 60, 1

        if z_1 == 1:
            omega = (x_1, y_1)
        else:
            omega = (x_1, y_1, z_1)

        sigma_init = 7
        sigma_gaussian_filter = 2

        # Numerical method whose result corresponds to the ground truth:
        ground_method = 'rk4'  # in the following table should be false.
        ground_method_steps = 10

        parameters = [x_1, y_1, z_1] + [N, sigma_init, sigma_gaussian_filter, ground_method, ground_method_steps]

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
        ### Model: computations ###
        ###########################

        print '---------------------'
        print 'Computations started!'
        print '---------------------'

        # init data
        errors = np.zeros([num_method_considered, N])  # Row: method, col: sampling
        res_time = np.zeros([num_method_considered, N])  # Row: method, col: sampling

        for s in range(N):  # sample
            # Generate svf
            svf_0 = generate_random(omega, parameters=(sigma_init, sigma_gaussian_filter))

            # compute the exponential with the selected GROUND TRUTH method:
            if ground_method == 'vode' or ground_method == 'lsoda':
                disp_silver_ground = lie_exponential_scipy(svf_0, integrator=ground_method,
                                                           max_steps=ground_method_steps)

            else:
                disp_silver_ground = lie_exponential(svf_0, algorithm=ground_method, s_i_o=s_i_o,
                                                     input_num_steps=ground_method_steps)

            for m in range(num_method_considered):
                if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                    start = time.time()
                    sdisp_0 = lie_exponential_scipy(svf_0, integrator=names_method_considered[m],
                                                    max_steps=steps_methods_considered[m])
                    res_time[m, s] = (time.time() - start)

                else:
                    start = time.time()
                    sdisp_0 = lie_exponential(svf_0, algorithm=names_method_considered[m], s_i_o=s_i_o,
                                              input_num_steps=steps_methods_considered[m])
                    res_time[m, s] = (time.time() - start)

                # compute error:
                errors[m, s] = vf_norm(sdisp_0 - disp_silver_ground, passe_partout_size=pp, normalized=True)

            if verbose:

                results_by_column = [[met, err, tim]
                                     for met, err, tim
                                     in zip(names_method_considered, list(errors[:, s]), list(res_time[:, s]))]

                print '--------------------'
                print 'Sampling ' + str(s + 1) + '/' + str(N) + ' .'
                print '--------------------'
                print 'sigma init =    ' + str(sigma_init)
                print 'sigma gaussian filter = ' + str(sigma_gaussian_filter)
                print '--------------------'
                print tabulate(results_by_column,
                               headers=['method', 'error', 'comp. time (sec)'])
                print '--------------------'

        ### Save data to folder ###
        np.save(pfi_array_errors_output, errors)
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
        errors       = np.load(pfi_array_errors_output)
        res_time     = np.load(pfi_array_comp_time_output)

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

        names_method_considered      = [methods[j][0] for j in range(len(methods)) if methods[j][1] is True]
        steps_methods_considered     = [methods[j][2] for j in range(len(methods)) if methods[j][1] is True]
        colour_methods_considered    = [methods[j][3] for j in range(len(methods)) if methods[j][1] is True]
        linestyle_methods_considered = [methods[j][4] for j in range(len(methods)) if methods[j][1] is True]
        markers_methods_considered   = [methods[j][5] for j in range(len(methods)) if methods[j][1] is True]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:
        # parameters = [x_1, y_1, z_1] + [N, sigma_init, sigma_gaussian_filter, ground_method, ground_method_steps]

        print 'Error-bar and time for multiple GAUSS generated SVF'
        print '---------------------------------------------'

        print '\nParameters of the transformation se2:'
        print 'domain = ' + str(parameters[:3])
        print 'number of samples = ' + str(parameters[3])
        print 'sigma_init, sigma_gaussian_filter = ' + str(parameters[4:6])
        print 'dummy ground truth and steps = ' + str(parameters[6:8])

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

    results_by_column = [[met, err, tim]
                         for met, err, tim in zip(names_method_considered, list(mean_errors), list(mean_res_time))]

    print '\n'
    print 'Results and computational time:'
    print tabulate(results_by_column,
                   headers=['method', 'mean error', 'mean comp. time (sec)'])
    print '\nEND'

    # plot results
    if plot_results:

        reordered_errors_for_plot = []
        reordered_times_for_plot = []
        for m in range(errors.shape[0]):
            reordered_errors_for_plot += [list(errors[m, :])]
            reordered_times_for_plot += [list(res_time[m, :])]

        # BOX-PLOT custom
        plot_custom_boxplot(input_data=reordered_errors_for_plot,
                            input_names=names_method_considered,
                            fig_tag=11,
                            input_titles=('Error exponential map for multiple SE2-generated svf', 'field'),
                            kind='multiple_GAUSS',
                            window_title_input='bar_plot_multiple_gauss',
                            additional_field=None,
                            log_scale=False,
                            input_parameters=parameters,
                            annotate_mean=True,
                            add_extra_annotation=mean_res_time)

        print len(reordered_errors_for_plot)
        print len(reordered_times_for_plot)

        plot_custom_cluster(reordered_errors_for_plot, reordered_times_for_plot,
                            fig_tag=42,
                            clusters_labels=names_method_considered,
                            clusters_markers=markers_methods_considered,
                            clusters_colors=colour_methods_considered)

        plt.show()
