import copy
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from controller import methods_t_s, path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables
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

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_time_vs_error_per_steps'
    kind   = 'SE2'
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

        # Parameters SVF

        x_1, y_1, z_1 = 60, 60, 60

        if z_1 == 1:
            domain = (x_1, y_1)
            shape = list(domain) + [1, 1, 2]
        else:
            domain = (x_1, y_1, z_1)
            shape = list(domain) + [1, 3]

        x_c = np.floor(x_1 / 2)
        y_c = np.floor(y_1 / 2)
        theta = np.pi / 8

        tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
        ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

        parameters = [x_1, y_1, z_1] + [theta, tx, ty] + list_of_steps

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
        m_0 = se2_g.se2_g(theta, tx, ty)
        dm_0 = se2_g.log(m_0)

        svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))
        disp_0  = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))

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
                errors[m, step_index] = (disp_computed - disp_0).norm(passe_partout_size=pp, normalized=True)

            if verbose:

                results_by_column = [[met, err, tim]
                                     for met, err, tim
                                     in zip(names_method_considered,
                                            list(errors[:, step_index]),
                                            list(res_time[:, step_index]))]

                print '--------------------'
                print 'Stage ' + str(step_index + 1) + '/' + str(num_of_steps_considered) + ' .'
                print '--------------------'
                print 'theta, tx, ty = ' + str(m_0.get)
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

    else:
        errors       = np.load(fullpath_array_errors_output)
        res_time     = np.load(fullpath_array_comp_time_output)
        parameters   = np.load(fullpath_transformation_parameters)
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

        list_of_steps = list(parameters[6:])
        num_of_steps_considered = len(list_of_steps)

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print parameters

        print '\nParameters of the se2 generated SVF:'
        print 'domain = ' + str(parameters[:3])
        print 'theta, tx, ty = ' + str(parameters[3:6])

        print '\nMethods and parameters of the integrators method:'
        print tabulate(methods,
                       headers=['name', 'compute (True/False)', 'num_steps'])
        print '\n'

        print 'You chose to compute ' + str(num_method_considered) + ' methods for ' +\
              str(num_of_steps_considered) + ' different steps.'
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
    print 'Results Errors per steps of the numerical integrators:'
    print tabulate(results_by_column_error,
                   headers=[''] + list_of_steps)

    print '\n'
    print 'Results Computational time per steps of the numerical integrators:'
    print tabulate(results_by_column_time,
                   headers=[''] + list_of_steps)
    print '\n'

    # plot results
    if plot_results:

        plot_custom_time_error_steps(res_time,
                                     errors,
                                     label_lines=names_method_considered,
                                     additional_field=svf_as_array,
                                     kind='one_SE2',
                                     x_log_scale=True,
                                     y_log_scale=True,
                                     input_parameters=parameters,
                                     input_marker=marker_method_considered,
                                     input_colors=color_methods_considered,
                                     input_line_style=line_style_methods_considered,
                                     legend_location='upper right')

        plt.show()
