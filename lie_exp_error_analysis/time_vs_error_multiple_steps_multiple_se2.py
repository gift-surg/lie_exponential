import copy
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from controller import methods_t_s, path_to_results_folder, path_to_exp_notes_figures, \
    path_to_exp_notes_tables
from visualizer.graphs_and_stats_new import plot_custom_time_error_steps

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
    save_external = False
    plot_results = True

    # The results, and additional information are loaded in see_error_time_results
    # with a simplified name. They are kept safe from other subsequent tests with the same code.
    save_for_sharing = False

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_time_vs_error_per_steps'
    kind = 'SE2'
    number = 'multiple'
    file_suffix  = '_' + str(1)  # 5 ok with 50 sample, 1 test

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

        list_of_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

        list_of_steps_as_str = ''
        for i in list_of_steps:
            list_of_steps_as_str += str(i) + '_'

        num_of_steps_considered = len(list_of_steps)

        # Parameters SVF

        N = 10

        x_1, y_1, z_1 = 60, 60, 60

        if z_1 == 1:
            domain = (x_1, y_1)
            shape = list(domain) + [1, 1, 2]
        else:
            domain = (x_1, y_1, z_1)
            shape = list(domain) + [1, 3]

        interval_theta = (- np.pi / 8, np.pi / 8)
        epsilon = np.pi / 12
        omega = (20, 40, 20, 40)  # where to locate the center of the random rotation on the axial plane

        parameters = [x_1, y_1, z_1,           # Domain
                      N,                       # number of samples
                      interval_theta[0], interval_theta[1],  # interval of the rotations
                      omega[0], omega[1],      # interval of the center of the rotation x
                      omega[2], omega[3]] + list_of_steps  # interval of the center of the rotation y

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

        # init matrices and data
        errors = np.zeros([num_method_considered, num_of_steps_considered, N])  # Row: method, col: sampling
        res_time = np.zeros([num_method_considered, num_of_steps_considered, N])  # Row: method, col: sampling
        svf_as_array = []

        for s in range(N):  # sample

            # generate matrices
            m_0 = se2_g.randomgen_custom_center(interval_theta=interval_theta,
                                                omega=omega,
                                                epsilon_zero_avoidance=epsilon)
            dm_0 = se2_g.log(m_0)

            svf_0   = SVF.generate_from_matrix(domain[:2], dm_0.get_matrix, affine=np.eye(4))

            # store the vector field
            svf_as_array = copy.deepcopy(svf_0.field)

            svf_0   = SVF.generate_from_matrix(domain[:2], dm_0.get_matrix, affine=np.eye(4))
            disp_0  = SDISP.generate_from_matrix(domain[:2], m_0.get_matrix - np.eye(3), affine=np.eye(4))

            for step_index, step_input in enumerate(list_of_steps):

                for m in range(num_method_considered):  # method
                    if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                        start = time.time()
                        disp_computed = svf_0.exponential_scipy(integrator=names_method_considered[m],
                                                                max_steps=step_input)
                        res_time[m, step_index, s] = (time.time() - start)

                    else:
                        start = time.time()
                        disp_computed = svf_0.exponential(algorithm=names_method_considered[m],
                                                          s_i_o=s_i_o,
                                                          input_num_steps=step_input)
                        res_time[m, step_index, s] = (time.time() - start)

                    # compute error:
                    errors[m, step_index, s] = (disp_computed - disp_0).norm(passe_partout_size=pp, normalized=True)

                if verbose:

                    results_errors_by_slice = [[names_method_considered[j]] + list(errors[j, :, s])
                                               for j in range(num_method_considered)]

                    results_times_by_slice = [[names_method_considered[j]] + list(res_time[j, :, s])
                                              for j in range(num_method_considered)]

                    print 'Sample ' + str(s + 1) + '/' + str(N) + '.'
                    print 'Errors: '
                    print '---------------------------------------------'
                    print tabulate(results_errors_by_slice,
                                   headers=['method'] + [str(j) for j in list_of_steps])
                    print '---------------------------------------------'
                    print ''
                    print 'Times: '
                    print '---------------------------------------------'
                    print tabulate(results_times_by_slice,
                                   headers=['method'] + [str(j) for j in list_of_steps])
                    print '---------------------------------------------'

        ### Save data to folder ###
        np.save(fullpath_array_errors_output,       errors)
        np.save(fullpath_array_comp_time_output,    res_time)
        np.save(fullpath_field,                     svf_as_array)

        with open(fullpath_transformation_parameters, 'wb') as f:
            pickle.dump(parameters, f)

        with open(fullpath_numerical_method_table, 'wb') as f:
            pickle.dump(methods, f)

        if save_for_sharing:

            path_to_sharing_folder = os.path.join(path_to_results_folder, 'sharing_folder')
            np.save(os.path.join(path_to_sharing_folder, 'errors_se2' + file_suffix), errors)
            np.save(os.path.join(path_to_sharing_folder, 'comp_time_se2' + file_suffix),    res_time)

            with open(os.path.join(path_to_sharing_folder, 'exp_methods_param_se2' + file_suffix), 'wb') as f:
                pickle.dump(parameters, f)

            with open(os.path.join(path_to_sharing_folder, 'exp_methods_table_se2' + file_suffix), 'wb') as f:
                pickle.dump(methods, f)

        print
        print '------------------------------------------'
        print 'Data computed and saved in external files!'
        print '------------------------------------------'

    else:
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

        list_of_steps = list(parameters[10:])
        num_of_steps_considered = len(list_of_steps)

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print '\nParameters of the generated multiple SVF:'
        print 'domain = ' + str(parameters[:3])
        print 'number of samples = ' + str(parameters[3])
        print 'interval theta = '  + str(parameters[4:6])
        print 'omega (centre location) = ' + str(parameters[6:10])

        print '\nMethods and parameters:'
        print tabulate(methods,
                       headers=['name', 'compute (True/False)', 'num_steps'])
        print '\n'

        print 'You chose to compute ' + str(num_method_considered) + ' methods for ' \
              + str(num_of_steps_considered) + ' steps.'
        print 'List of the methods considered:'
        print names_method_considered
        print 'steps chosen:'
        print list_of_steps

    ################################
    # Visualization and statistics #
    ################################

    results_by_column_error = [[names_method_considered[j]] + list(errors[j, :])
                               for j in range(num_method_considered)]

    results_by_column_time  = [[names_method_considered[j]] + list(res_time[j, :])
                               for j in range(num_method_considered)]

    mean_errors = np.mean(errors,  axis=2)
    stdev_errors = np.std(errors,  axis=2)
    mean_times  = np.mean(res_time, axis=2)

    if verbose:

        # Tabulate Errors
        mean_errors_by_column = [[names_method_considered[j]] + list(mean_errors[j, :])
                                 for j in range(num_method_considered)]

        print '\n'
        print 'Results Errors per steps of the numerical integrators:'
        print tabulate(mean_errors_by_column, headers=[''] + list_of_steps)

        # Tabulate Times
        mean_times_by_column = [[names_method_considered[j]] + list(mean_errors[j, :])
                                for j in range(num_method_considered)]

        print '\n'
        print 'Results Errors per steps of the numerical integrators:'

        print tabulate(mean_times_by_column, headers=[''] + list_of_steps)

        print '\n'

    # plot results
    if plot_results:

        plot_custom_time_error_steps(mean_times,
                                     mean_errors,
                                     label_lines=names_method_considered,
                                     additional_field=svf_as_array,
                                     kind='multiple_SE2',
                                     titles=('mean time vs. mean error (increasing steps)', 'Field sample'),
                                     x_log_scale=True,
                                     y_log_scale=True,
                                     input_parameters=parameters,
                                     input_marker=marker_method_considered,
                                     input_colors=color_methods_considered,
                                     input_line_style=line_style_methods_considered,
                                     legend_location='upper right')

        plt.show()
