import os
import pickle
import time
from os.path import join as jph

import matplotlib.pyplot as plt
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from VECtorsToolkit.tools.fields.generate_vf import generate_random
from VECtorsToolkit.tools.local_operations.lie_exponential import lie_exponential, lie_exponential_scipy
from VECtorsToolkit.tools.fields.queries import vf_norm

from controller import methods_t_s
from path_manager import pfo_results, pfo_notes_figures, pfo_notes_sharing
from visualizer.graphs_and_stats_new import plot_custom_time_error_steps

"""
Module aimed to compare computational time versus error for different steps of the exponential algorithm.
"""

if __name__ == "__main__":

    clear_cache()

    ##################
    ### Controller ###
    ##################

    compute       = True
    verbose       = True
    save_external = True
    plot_results  = True

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_time_vs_error_per_steps'
    kind = 'GAUSS'
    number = 'single'
    tag = '_' + str(1)

    fin_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    fin_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    fin_csv_table_comp_time_output = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_cp_time'
    fin_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    fin_array_comp_time_output     = str(prefix_fn) + '_' + str(number) + str(kind) + '_array_cp_time'
    fin_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    fin_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'
    fin_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    pfo_errors_times_results = jph(pfo_results, 'errors_times_results')

    os.system('mkdir -p {}'.format(pfo_errors_times_results))
    print("\nPath to results folder {}\n".format(pfo_errors_times_results))

    pfi_array_errors_output        = jph(pfo_errors_times_results, fin_array_errors_output + tag + '.npy')
    pfi_array_comp_time_output     = jph(pfo_errors_times_results, fin_array_comp_time_output + tag + '.npy')
    pfi_transformation_parameters  = jph(pfo_errors_times_results, fin_transformation_parameters + tag)
    pfi_field                      = jph(pfo_errors_times_results, fin_field + tag + '.npy')
    pfi_numerical_method_table     = jph(pfo_errors_times_results, fin_numerical_methods_table + tag)
    pfi_figure_output              = jph(pfo_notes_figures, fin_figure_output + tag + '.pdf')
    pfi_csv_table_errors_output    = jph(pfo_notes_sharing, fin_csv_table_errors_output + '.csv')
    pfi_csv_table_comp_time_output = jph(pfo_notes_sharing, fin_csv_table_comp_time_output + '.csv')

    ####################
    ### Computations ###
    ####################

    if compute:  # or compute or load

        random_seed = 0

        if random_seed > 0:
            np.random.seed(random_seed)

        pp = 2  # passepartout
        s_i_o = 3  # spline interpolation order

        # Different field of views:

        list_of_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40]

        list_of_steps_as_str = ''
        for i in list_of_steps:
            list_of_steps_as_str += str(i) + '_'

        num_of_steps_considered = len(list_of_steps)

        x_1, y_1, z_1 = 20, 20, 10

        if z_1 == 1:
            omega = (x_1, y_1)
        else:
            omega = (x_1, y_1, z_1)

        sigma_init = 5
        sigma_gaussian_filter = 2

        # Numerical method whose result corresponds to the ground truth:
        ground_method = 'rk4'  # in the following table should be false.
        ground_method_steps = 10

        # store transformation parameters into a list ( domain - sigmas - ground truths - steps )
        parameters = [x_1, y_1, z_1] + [sigma_init, sigma_gaussian_filter] + [ground_method, ground_method_steps] + \
                     list_of_steps

        # import methods from external file aaa_general_controller
        methods = methods_t_s

        index_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
        num_method_considered    = len(index_methods_considered)

        names_method_considered       = [methods[j][0] for j in index_methods_considered]
        color_methods_considered      = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        marker_method_considered      = [methods[j][5] for j in index_methods_considered]

        ###########################
        ### Model: computations ###
        ###########################

        print '---------------------'
        print 'Computations started!'
        print '---------------------'

        # init matrices:
        errors = np.zeros([num_method_considered, num_of_steps_considered])  # Row: method, col: sampling
        res_time = np.zeros([num_method_considered, num_of_steps_considered])  # Row: method, col: sampling

        # Generate SVF and displacement:
        svf_0 = generate_random(omega, parameters=(sigma_init, sigma_gaussian_filter))

        if ground_method == 'vode' or ground_method == 'lsoda':
            disp_silver_ground = lie_exponential_scipy(svf_0, integrator=ground_method, max_steps=ground_method_steps)

        else:
            disp_silver_ground = lie_exponential(svf_0, algorithm=ground_method, s_i_o=s_i_o,
                                                 input_num_steps=ground_method_steps)

        for step_index, step_input in enumerate(list_of_steps):

            for m in range(num_method_considered):  # method
                if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                    start = time.time()
                    sdisp_0 = lie_exponential_scipy(svf_0, integrator=names_method_considered[m], max_steps=step_input)
                    res_time[m, step_index] = (time.time() - start)

                else:
                    start = time.time()
                    sdisp_0 = lie_exponential(svf_0, algorithm=names_method_considered[m], s_i_o=s_i_o,
                                              input_num_steps=step_input)
                    res_time[m, step_index] = (time.time() - start)

                # compute error:
                errors[m, step_index] = vf_norm(sdisp_0 - disp_silver_ground, passe_partout_size=2, normalized=True)

            if verbose:
                results_by_column = [[met, err, tim]
                                     for met, err, tim
                                     in zip(names_method_considered,
                                            list(errors[:, step_index]),
                                            list(res_time[:, step_index]))]

                print '--------------------'
                print 'Stage ' + str(step_index + 1) + '/' + str(num_of_steps_considered) + ' .'
                print '--------------------'
                print 'sigma_i, sigma_g = ' + str(sigma_init) + '_' + str(sigma_gaussian_filter)
                print 'Number of steps at this stage step =    ' + str(step_input)
                print '--------------------'
                print tabulate(results_by_column,
                               headers=['method', 'error', 'comp. time (sec)'])
                print '--------------------'

        ### Save data to folder ###
        np.save(pfi_array_errors_output, errors)
        np.save(pfi_array_comp_time_output, res_time)
        np.save(pfi_field, svf_0)

        with open(pfi_transformation_parameters, 'wb') as f:
            pickle.dump(parameters, f)

        with open(pfi_numerical_method_table, 'wb') as f:
            pickle.dump(methods, f)

        print
        print '------------------------------------------'
        print 'Data computed and saved in external files!'
        print '------------------------------------------'

    else:  # if not compute then load

        errors = np.load(pfi_array_errors_output)
        res_time = np.load(pfi_array_comp_time_output)

        with open(pfi_transformation_parameters, 'rb') as f:
            parameters = pickle.load(f)

        with open(pfi_numerical_method_table, 'rb') as f:
            methods = pickle.load(f)

        print
        print '------------'
        print 'Data loaded!'
        print '------------'

        index_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
        num_method_considered = len(index_methods_considered)

        names_method_considered = [methods[j][0] for j in index_methods_considered]
        color_methods_considered = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        marker_method_considered = [methods[j][5] for j in index_methods_considered]

        list_of_steps = list(parameters[7:])
        num_of_steps_considered = len(list_of_steps)

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:
        print '\nParameters of the gauss generated SVF:'
        print 'domain = ' + str(parameters[:3])
        print 'sigma_init, sigma_gaussian_filter = ' + str(parameters[3:5])
        print 'ground method, ground method steps = ' + str(parameters[5:7])

        print '\n'
        print 'Methods and parameters:'
        print tabulate(methods,
                       headers=['name', 'compute (True/False)', 'num_steps'])
        print '\n'

        print 'You chose to compute ' + str(num_method_considered) + ' methods for ' + str(num_of_steps_considered) \
              + ' samples.'
        print 'List of the methods considered:'
        print names_method_considered
        print 'List of the steps of the methods considered'
        print list_of_steps

    ################################
    # Visualization and statistics #
    ################################

    results_by_column_error = [[names_method_considered[j]] + list(errors[j, :])
                               for j in range(num_method_considered)]

    results_by_column_time = [[names_method_considered[j]] + list(res_time[j, :])
                              for j in range(num_method_considered)]

    print '\n'
    print 'Results Errors per field of view:'
    print tabulate(results_by_column_error,
                   headers=[''] + list_of_steps)

    print '\n'
    print 'Results Computational time per view:'
    print tabulate(results_by_column_time,
                   headers=[''] + list_of_steps)
    print '\n'

    # plot results
    if plot_results:
        plot_custom_time_error_steps(res_time,
                                     errors,
                                     label_lines=names_method_considered,
                                     additional_field=svf_0,
                                     kind='one_GAUSS',
                                     x_log_scale=True,
                                     y_log_scale=True,
                                     input_parameters=parameters,
                                     input_marker=marker_method_considered,
                                     input_colors=color_methods_considered,
                                     input_line_style=line_style_methods_considered,
                                     legend_location='lower left')

        plt.show()
