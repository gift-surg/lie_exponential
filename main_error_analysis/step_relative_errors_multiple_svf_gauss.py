import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from tabulate import tabulate
from sympy.core.cache import clear_cache
import pickle

from transformations.s_vf import SVF
from utils.path_manager import path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables
from visualizer.graphs_and_stats_new import plot_custom_step_error
from aaa_general_controller import methods_t_s

"""
Study for the estimate of step error for the Scaling and squaring based and Taylor numerical methods.
Multiple Gauss-generated SVF.
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

    # The results, and additional information are loaded in see_error_time_results
    # with a simplified name. They are kept safe from other subsequent tests with the same code.
    save_for_sharing = True

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_step_relative_errors_'
    kind   = 'GAUSS'
    number = 'multiple'
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

        s_i_o = 3
        pp = 2

        # Parameters SVF:
        x_1, y_1, z_1 = 50, 50, 50

        if z_1 == 1:
            domain = (x_1, y_1)
            shape = list(domain) + [1, 1, 2]
        else:
            domain = (x_1, y_1, z_1)
            shape = list(domain) + [1, 3]

        sigma_init = 4
        sigma_gaussian_filter = 2

        # Number of sampled SVFs:
        N = 10

        # maximal number of consecutive steps where to compute the step-relative error
        max_steps = 20

        parameters = [x_1, y_1, z_1] + [N, sigma_init, sigma_gaussian_filter, max_steps]

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

        step_errors = np.zeros([num_method_considered, max_steps, N])  # Methods x Steps

        for s in range(N):  # sample

            svf_0   = SVF.generate_random_smooth(shape=shape,
                                                 sigma=sigma_init,
                                                 sigma_gaussian_filter=sigma_gaussian_filter)

            svf_as_array = copy.deepcopy(svf_0.field)

            sdisp_step_j_0 = []
            sdisp_step_j_1 = []

            for met in range(num_method_considered):

                if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':
                    disp_computed = svf_0.exponential_scipy(integrator=names_method_considered[met],
                                                            max_steps=1)

                else:
                    disp_computed = svf_0.exponential(algorithm=names_method_considered[met],
                                                      s_i_o=s_i_o,
                                                      input_num_steps=1)
                sdisp_step_j_0 += [disp_computed]

            for stp in range(2, max_steps):
                for met in range(num_method_considered):

                    if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':
                        disp_computed = svf_0.exponential_scipy(integrator=names_method_considered[met],
                                                                max_steps=stp)

                    else:
                        disp_computed = svf_0.exponential(algorithm=names_method_considered[met],
                                                          s_i_o=s_i_o,
                                                          input_num_steps=stp)

                    sdisp_step_j_1 += [disp_computed]
                    step_errors[met, stp - 1, s] = \
                        (sdisp_step_j_1[met] - sdisp_step_j_0[met]).norm(passe_partout_size=pp)

                sdisp_step_j_0 = copy.deepcopy(sdisp_step_j_1)
                sdisp_step_j_1 = []

                if verbose:
                    results_by_column = [[met, err] for met, err in zip(names_method_considered,
                                                                        list(step_errors[:, stp-1, s]))]

                    print 'Step-error for each method computed at ' + str(stp) + 'th. step. for the sample '\
                          + str(s) + '/' + str(N)
                    print '---------------------------------------------'
                    print tabulate(results_by_column,
                                   headers=['method', 'error'])
                    print '---------------------------------------------'

        # print the summary table
        if verbose:
            step_errors_mean = np.mean(step_errors, axis=2)
            print step_errors_mean.shape

            print 'Mean of the step-error for each method computed for ' + str(N) + ' samples '
            print '---------------------------------------------'
            print tabulate(step_errors_mean,
                           headers=['method'] + [str(a) for a in range(1, max_steps - 1)])
            print '---------------------------------------------'

        # Save the data
        np.save(fullpath_array_errors_output, step_errors)
        np.save(fullpath_field, svf_as_array)

        with open(fullpath_transformation_parameters, 'wb') as f:
            pickle.dump(parameters, f)

        with open(fullpath_numerical_method_table, 'wb') as f:
            pickle.dump(methods, f)

        if save_for_sharing:

            path_to_sharing_folder = os.path.join(path_to_results_folder, 'sharing_folder')
            np.save(os.path.join(path_to_sharing_folder, 'step_relative_errors_gauss'), step_errors)
            np.save(os.path.join(path_to_sharing_folder, 'step_relative_exp_methods_param_gauss'), methods_t_s)

            with open(os.path.join(path_to_sharing_folder,
                                   'step_relative_errors_transformation_parameters_gauss'), 'wb') as f:
                pickle.dump(parameters, f)

            with open(os.path.join(path_to_sharing_folder,
                                   'step_relative_errors_numerical_method_table_gauss'), 'wb') as f:
                pickle.dump(methods, f)

        print
        print '------------------------------------------'
        print 'Data computed and saved in external files!'
        print '------------------------------------------'

    else:

        step_errors = np.load(fullpath_array_errors_output)
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

        max_steps = parameters[6]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print 'Step-wise error for multiple Gauss generated SVF'
        print 'step-error is computed as norm(exp(svf,step+1) - exp(svf,step)) '
        print 'Up to step : ' + str(max_steps)

        print '---------------------------------------------'

        print 'Parameters that generate the SVF'
        print 'Number of samples = ' + str(parameters[3])
        print 'Svf shape = ' + str(parameters[:3])
        print 'sigma init, sigma gaussian filter = ' + str(parameters[4:6])

        print '\n'
        print 'Methods and parameters:'
        print tabulate(methods,
                       headers=['name', 'compute (True/False)', 'colour',  'line-style',   'marker'])
        print '\n'

        print 'You chose to visualize the step error for each of the considered method.'
        print 'step-error is computed as norm(exp(svf,step+1) - exp(svf,step)) '
        print 'List of the methods considered:'
        print names_method_considered
        print '---------------------------------------------'

    ############################
    ### Visualization method ###
    ############################

    if plot_results:

        step_errors_mean = np.mean(step_errors, axis=2)

        plot_custom_step_error(range(1, max_steps - 1),
                               step_errors_mean[:, 1:max_steps - 1],  # here is the mean of the errors
                               names_method_considered,
                               input_parameters=parameters,
                               fig_tag=2,
                               kind='multiple_GAUSS',
                               log_scale=False,
                               input_colors=color_methods_considered,
                               window_title_input='step errors',
                               titles=('iterations vs. MEANS of the step-errors', ''),
                               additional_field=None,
                               legend_location='upper right',
                               input_line_style='-')

        plt.show()
