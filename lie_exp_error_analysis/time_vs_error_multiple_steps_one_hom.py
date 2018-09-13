import copy
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
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
    save_external = True
    plot_results = True

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_time_vs_error_per_steps'
    kind = 'HOM'
    number = 'single'
    file_suffix  = '_' + str(4)  # 1 skew, 2 diag 51 (if z = 51)

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

        list_of_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

        list_of_steps_as_str = ''
        for i in list_of_steps:
            list_of_steps_as_str += str(i) + '_'

        num_of_steps_considered = len(list_of_steps)

        # Parameters SVF

        x_1, y_1, z_1 = 30, 30, 30

        x, y, z = 5, 5, 0

        in_psl = False

        if z_1 == 1:
            d = 2
            domain = (x_1, y_1)
            shape = list(domain) + [1, 1, 2]

            # center of the homography
            x_c = x_1 / 2
            y_c = y_1 / 2
            z_c = 1

            projective_center = [x_c, y_c, z_c]

        else:
            d = 3
            domain = (x_1, y_1, z_1)
            shape = list(domain) + [1, 3]

            # center of the homography
            x_c = x_1 / 2
            y_c = y_1 / 2
            z_c = z_1 / 2
            w_c = 1

            projective_center = [x_c, y_c, z_c, w_c]

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

        # generate matrices homography

        scale_factor = 1. / (np.max(domain) * 10)
        sigma = 1

        hom_attributes = [scale_factor, sigma, in_psl]

        ### Alternative construction of the homography, generated by the
        h_a = sigma * np.random.randn(d + 1, d + 1)
        h_a *= scale_factor
        h_a[-1, :] = np.abs(h_a[-1, :])

        h_g = expm(h_a)

        if verbose:
            print 'h = '
            print str(h_a)

            print 'H = '
            print str(h_g)

        svf_h = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)
        disp_h = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)

        #### some tests
        if verbose:
            print '\nsvf at x, y, z'
            print svf_h.field[x, y, z, 0, :]

            print '\nanalytic solution from the computed SDISP, displacement'
            print disp_h.field[x, y, z, 0, :]

            disp_h_ss = svf_h.exponential(algorithm='ss', input_num_steps=5)

            print '\nExp scaling and squaring on deformation (in displacement coordinates):'
            print disp_h_ss.field[x, y, z, 0, :]

        if d == 2:
            svf_as_array = copy.deepcopy(svf_h.field)
        elif d == 3:
            svf_as_array = copy.deepcopy(svf_h.field[:, :, z_c:(z_c + 1), :, :2])

        for step_index, step_input in enumerate(list_of_steps):

            for m in range(num_method_considered):  # method
                disp_computed = None
                if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                    start = time.time()
                    disp_computed = svf_h.exponential_scipy(integrator=names_method_considered[m],
                                                            max_steps=step_input)
                    res_time[m, step_index] = (time.time() - start)

                else:
                    start = time.time()
                    disp_computed = svf_h.exponential(algorithm=names_method_considered[m],
                                                      s_i_o=s_i_o,
                                                      input_num_steps=step_input)
                    res_time[m, step_index] = (time.time() - start)

                # compute error:
                errors[m, step_index] = (disp_computed - disp_h).norm(passe_partout_size=pp, normalized=True)

            if verbose:

                results_by_column = [[met, err, tim]
                                     for met, err, tim
                                     in zip(names_method_considered,
                                            list(errors[:, step_index]),
                                            list(res_time[:, step_index]))]

                print '--------------------'
                print 'Random generated homograpy: '
                print 'Stage ' + str(step_index + 1) + '/' + str(num_of_steps_considered) + ' .'
                print '--------------------'
                print '--------------------'
                print tabulate(results_by_column,
                               headers=['method', 'error', 'comp. time (sec)'])
                print '--------------------'

        # store transformation parameters into a list ( domain - steps )
        parameters = [x_1, y_1, z_1] + hom_attributes + list_of_steps

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

        print '\nParameters of the homograpy generated SVF:'
        print 'domain = ' + str(parameters[:3])
        print 'scale factor = ' + str(parameters[3])
        print 'sigma = ' + str(parameters[4])
        print 'in psl = ' + str(parameters[5])
        print

        print '\nMethods and parameters:'
        print tabulate(methods,
                       headers=['name', 'compute (True/False)', 'num_steps'])
        print '\n'

        print 'You chose to compute ' + str(num_method_considered) + ' methods for ' \
              + str(num_of_steps_considered) + ' steps.'
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
                                     kind='one_HOM',
                                     x_log_scale=True,
                                     y_log_scale=True,
                                     input_parameters=parameters,
                                     input_marker=marker_method_considered,
                                     input_colors=color_methods_considered,
                                     input_line_style=line_style_methods_considered,
                                     legend_location='upper right')

        plt.show()
