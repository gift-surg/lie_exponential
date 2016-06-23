import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from tabulate import tabulate
from sympy.core.cache import clear_cache
import time
import pickle

from transformations.s_vf import SVF
from utils.path_manager import path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables
from visualizer.graphs_and_stats_new import plot_custom_time_error_steps
from aaa_general_controller import methods_t_s

"""
Module aimed to compare computational time versus error for different steps of the exponential algorithm.
"""

if __name__ == "__main__":

    clear_cache()

    ##################
    ### Controller ###
    ##################

    compute = True
    verbose = True
    save_external = True
    plot_results = True

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_time_vs_error_per_steps'
    kind = 'GAUSS'
    number = 'single'
    file_suffix  = '_' + str(1)

    filename_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    filename_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    filename_csv_table_comp_time_output = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_cp_time'

    filename_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    filename_array_comp_time_output     = str(prefix_fn) + '_' + str(number) + str(kind) + '_array_cp_time'

    filename_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    filename_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'

    filename_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    path_to_results_folder = os.path.join(path_to_results_folder, 'errors_times_results')

    fullpath_array_errors_output = os.path.join(path_to_results_folder,
                                                    filename_array_errors_output + file_suffix + '.npy')
    fullpath_array_comp_time_output = os.path.join(path_to_results_folder,
                                                       filename_array_comp_time_output + file_suffix + '.npy')
    fullpath_transformation_parameters = os.path.join(path_to_results_folder,
                                                      filename_transformation_parameters + file_suffix)
    fullpath_field = os.path.join(path_to_results_folder,
                                  filename_field + file_suffix + '.npy')
    fullpath_numerical_method_table = os.path.join(path_to_results_folder,
                                                   filename_numerical_methods_table + file_suffix)

    # path to results external to the project:
    fullpath_figure_output  = os.path.join(path_to_exp_notes_figures,
                                           filename_figure_output + file_suffix + '.pdf')
    fullpath_csv_table_errors_output = os.path.join(path_to_exp_notes_tables,
                                                    filename_csv_table_errors_output + '.csv')
    fullpath_csv_table_comp_time_output = os.path.join(path_to_exp_notes_tables,
                                                       filename_csv_table_comp_time_output + '.csv')

    ####################
    ### Computations ###
    ####################

    if compute:  # or compute or load

        random_seed = 0

        if random_seed > 0:
            np.random.seed(random_seed)

        pp = 2     # passepartout
        s_i_o = 3  # spline interpolation order

        # Different field of views:

        list_of_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40]

        list_of_steps_as_str = ''
        for i in list_of_steps:
            list_of_steps_as_str += str(i) + '_'

        num_of_steps_considered = len(list_of_steps)

        x_1, y_1, z_1 = 20, 20, 10

        if z_1 == 1:
            domain = (x_1, y_1)
            shape = list(domain) + [1, 1, 2]
        else:
            domain = (x_1, y_1, z_1)
            shape = list(domain) + [1, 3]

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
        svf_0   = SVF.generate_random_smooth(shape=shape,
                                             sigma=sigma_init,
                                             sigma_gaussian_filter=sigma_gaussian_filter)

        svf_as_array = copy.deepcopy(svf_0.field)

        if ground_method == 'vode' or ground_method == 'lsoda':
            disp_chosen_ground = svf_0.exponential_scipy(integrator=ground_method,
                                                         max_steps=ground_method_steps)

        else:
            disp_chosen_ground = svf_0.exponential(algorithm=ground_method,
                                                   s_i_o=s_i_o,
                                                   input_num_steps=ground_method_steps)

        svf_as_array = copy.deepcopy(svf_0.field)

        for step_index, step_input in enumerate(list_of_steps):

            for m in range(num_method_considered):  # method
                if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                    start = time.time()
                    disp_computed = svf_0.exponential_scipy(integrator=names_method_considered[m],
                                                                    max_steps=step_input)
                    res_time[m, step_index] = (time.time() - start)

                else:
                    start = time.time()
                    disp_computed = svf_0.exponential(algorithm=names_method_considered[m],
                                                              s_i_o=s_i_o,
                                                              input_num_steps=step_input)
                    res_time[m, step_index] = (time.time() - start)

                # compute error:
                errors[m, step_index] = (disp_computed - disp_chosen_ground).norm(passe_partout_size=2, normalized=True)

            if verbose:

                results_by_column = [[met, err, tim]
                                     for met, err, tim
                                     in zip(names_method_considered,
                                            list(errors[:, step_index]),
                                            list(res_time[:, step_index]))]

                print '--------------------'
                print 'Stage ' + str(step_index+1) + '/' + str(num_of_steps_considered) + ' .'
                print '--------------------'
                print 'sigma_i, sigma_g = ' + str(sigma_init) + '_' + str(sigma_gaussian_filter)
                print 'Number of steps at this stage step =    ' + str(step_input)
                print '--------------------'
                print tabulate(results_by_column,
                               headers=['method', 'error', 'comp. time (sec)'])
                print '--------------------'

        ### Save data to folder ###
        np.save(fullpath_array_errors_output,       errors)
        np.save(fullpath_array_comp_time_output,    res_time)
        np.save(fullpath_field, svf_as_array)

        with open(fullpath_transformation_parameters, 'wb') as f:
            pickle.dump(parameters, f)

        with open(fullpath_numerical_method_table, 'wb') as f:
            pickle.dump(methods, f)

        print
        print '------------------------------------------'
        print 'Data computed and saved in external files!'
        print '------------------------------------------'

    else:  # if not compute then load

        errors       = np.load(fullpath_array_errors_output)
        res_time     = np.load(fullpath_array_comp_time_output)
        svf_as_array = np.load(fullpath_field)

        with open(fullpath_transformation_parameters, 'rb') as f:
            parameters = pickle.load(f)

        with open(fullpath_numerical_method_table, 'rb') as f:
            methods = pickle.load(f)

        print
        print '------------'
        print 'Data loaded!'
        print '------------'

        index_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
        num_method_considered    = len(index_methods_considered)

        names_method_considered       = [methods[j][0] for j in index_methods_considered]
        color_methods_considered      = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        marker_method_considered      = [methods[j][5] for j in index_methods_considered]

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

    results_by_column_time  = [[names_method_considered[j]] + list(res_time[j, :])
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
                                           additional_field=svf_as_array,
                                           kind='one_GAUSS',
                                           x_log_scale=True,
                                           y_log_scale=True,
                                           input_parameters=parameters,
                                           input_marker=marker_method_considered,
                                           input_colors=color_methods_considered,
                                           input_line_style=line_style_methods_considered,
                                           legend_location='lower left')

        plt.show()
