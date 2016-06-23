import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from tabulate import tabulate
from sympy.core.cache import clear_cache
import time
import pickle
from scipy.linalg import expm

from utils.projective_algebras import get_random_hom_a_matrices

from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from utils.path_manager import path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables, \
    path_to_sharing_folder

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

    compute = False
    verbose = True
    save_external = True
    plot_results = True

    #######################
    ### Path management ###
    #######################

    # The results, and additional information are loaded in see_error_time_results
    # with a simplified name. They are kept safe from other subsequent tests with the same code.
    save_for_sharing = True

    prefix_fn = 'exp_comparing_time_vs_error_per_steps'
    kind = 'HOM'
    number = 'multiple'
    file_suffix  = '_' + str(5)  # 1 skew, 2 diag 51, 3 none

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

        N = 50
        x_1, y_1, z_1 = 60, 60, 60

        in_psl = False

        random_seed = 0

        if random_seed > 0:
            np.random.seed(random_seed)

        pp = 2     # passepartout
        s_i_o = 3  # spline interpolation order

        # Different field of views:

        list_of_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

        list_of_steps_as_str = ''
        for i in list_of_steps:
            list_of_steps_as_str += str(i) + '_'

        num_of_steps_considered = len(list_of_steps)

        # Parameters SVF

        if z_1 == 1:
            d = 2
            domain = (x_1, y_1)
            shape = list(domain) + [1, 1, 2]

            # center of the homography
            x_c = x_1/2
            y_c = y_1/2
            z_c = 1

            projective_center = [x_c, y_c, z_c]

        else:
            d = 3
            domain = (x_1, y_1, z_1)
            shape = list(domain) + [1, 3]

            # center of the homography
            x_c = x_1/2
            y_c = y_1/2
            z_c = z_1/2
            w_c = 1

            projective_center = [x_c, y_c, z_c, w_c]

        # actual matrix in pgl(n) is generated below!

        ### import methods from external file aaa_general_controller
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
        errors = np.zeros([num_method_considered, num_of_steps_considered, N])  # Row: method, col: sampling
        res_time = np.zeros([num_method_considered, num_of_steps_considered, N])  # Row: method, col: sampling

        for s in range(N):  # sample

            scale_factor = 1./(np.max(domain)*10)
            sigma = 1
            hom_attributes = [scale_factor, sigma, in_psl]

            h_a, h_g = get_random_hom_a_matrices(d=d,
                                                  scale_factor=hom_attributes[0],
                                                  sigma=hom_attributes[1],
                                                  special=hom_attributes[2])

            h_a = sigma*np.random.randn(d+1,  d+1)
            h_a = scale_factor * h_a
            h_a[-1, :] = np.abs(h_a[-1, :])

            h_g = expm(h_a)

            print 'h = '
            print str(h_a)

            print 'H = '
            print str(h_g)

            svf_h = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)
            disp_h = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)

            if verbose:

                # point in the domain to perform some tests:
                x, y, z = 5, 5, 0

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
                svf_as_array = copy.deepcopy(svf_h.field[:, :, z_c:(z_c+1), :, :2])

            for step_index, step_input in enumerate(list_of_steps):

                for m in range(num_method_considered):  # method
                    disp_computed = None

                    if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                        start = time.time()
                        disp_computed = svf_h.exponential_scipy(integrator=names_method_considered[m],
                                                                        max_steps=step_input)
                        res_time[m, step_index, s] = (time.time() - start)

                    else:
                        start = time.time()
                        disp_computed = svf_h.exponential(algorithm=names_method_considered[m],
                                                                  s_i_o=s_i_o,
                                                                  input_num_steps=step_input)
                        res_time[m, step_index, s] = (time.time() - start)

                    # compute error:
                    errors[m, step_index, s] = (disp_computed - disp_h).norm(passe_partout_size=pp, normalized=True)

                if verbose:

                    results_errors_by_slice = [[names_method_considered[j]] + list(errors[j, :, s])
                                               for j in range(num_method_considered)]

                    results_times_by_slice = [[names_method_considered[j]] + list(res_time[j, :, s])
                                              for j in range(num_method_considered)]

                    print 'Sample ' + str(s+1) + '/' + str(N) + '.'
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

        parameters = [x_1, y_1, z_1] + hom_attributes + [N] + list_of_steps

        ### Save data to folder ###
        np.save(fullpath_array_errors_output,       errors)
        np.save(fullpath_array_comp_time_output,    res_time)
        np.save(fullpath_field, svf_as_array)

        with open(fullpath_transformation_parameters, 'wb') as f:
            pickle.dump(parameters, f)

        with open(fullpath_numerical_method_table, 'wb') as f:
            pickle.dump(methods, f)

        if save_for_sharing:

            np.save(os.path.join(path_to_sharing_folder, 'errors_hom_' + file_suffix), errors)
            np.save(os.path.join(path_to_sharing_folder, 'comp_time_hom_' + file_suffix),    res_time)

            with open(os.path.join(path_to_sharing_folder, 'exp_methods_param_hom_' + file_suffix), 'wb') as f:
                pickle.dump(parameters, f)

            with open(os.path.join(path_to_sharing_folder, 'exp_methods_table_hom_' + file_suffix), 'wb') as f:
                pickle.dump(methods, f)

            print 'data saved in sharing folder'
            print path_to_sharing_folder
            print

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

        list_of_steps = list(parameters[9:])
        num_of_steps_considered = len(list_of_steps)

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print '\nParameters of the homograpy generated SVF:'
        print 'domain = ' + str(parameters[:3])
        print 'center = ' + str(parameters[3])
        print 'kind = ' + str(parameters[4])
        print 'scale factor = ' + str(parameters[5])
        print 'sigma = ' + str(parameters[6])
        print 'in psl = ' + str(parameters[7])
        print 'number of samples = ' + str(parameters[8])

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

    percentile_25 = np.percentile(errors, 25,  axis=2)
    percentile_75 = np.percentile(errors, 75,  axis=2)

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

    print '\nEND'

    # plot results
    if plot_results:

        plot_custom_time_error_steps(mean_times,
                                     mean_errors,
                                     fig_tag=10,
                                     y_error=stdev_errors,
                                     label_lines=names_method_considered,
                                     additional_field=svf_as_array,
                                     kind='multiple_HOM',
                                     titles=('mean time vs. mean error (increasing steps), stdev', 'Field sample'),
                                     x_log_scale=True,
                                     y_log_scale=True,
                                     input_parameters=parameters,
                                     input_marker=marker_method_considered,
                                     input_colors=color_methods_considered,
                                     input_line_style=line_style_methods_considered,
                                     legend_location='lower right')

        plot_custom_time_error_steps(mean_times,
                                     mean_errors,
                                     fig_tag=11,
                                     y_error=[percentile_25, percentile_75],
                                     label_lines=names_method_considered,
                                     additional_field=svf_as_array,
                                     kind='multiple_HOM',
                                     titles=('mean time vs. mean error (increasing steps), 25-75 percentile',
                                             'Field sample'),
                                     x_log_scale=True,
                                     y_log_scale=True,
                                     input_parameters=parameters,
                                     input_marker=marker_method_considered,
                                     input_colors=color_methods_considered,
                                     input_line_style=line_style_methods_considered,
                                     legend_location='upper right')

        plt.show()
