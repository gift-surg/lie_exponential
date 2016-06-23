import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import time
import os
from sympy.core.cache import clear_cache
import pickle

from transformations.s_vf import SVF
from utils.path_manager import path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables
from visualizer.graphs_and_stats_new import plot_custom_boxplot, plot_custom_cluster
from aaa_general_controller import methods_t_s

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

    compute = True
    verbose = True
    save_external = False
    plot_results = True

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_errors'
    kind   = 'GAUSS'
    number = 'multiple'
    file_suffix  = '_' + str(1)

    filename_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    filename_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    filename_csv_table_comp_time_output = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_cp_time'
    filename_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    filename_array_comp_time_output     = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_cp_time'
    filename_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'
    filename_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    filename_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    path_to_results_folder = os.path.join(path_to_results_folder, 'errors_times_results')

    fullpath_array_errors_output = os.path.join(path_to_results_folder,
                                                filename_array_errors_output + file_suffix + '.npy')
    fullpath_array_comp_time_output = os.path.join(path_to_results_folder,
                                                   filename_array_comp_time_output + file_suffix + '.npy')
    fullpath_field = os.path.join(path_to_results_folder,
                                  filename_field + file_suffix + '.npy')
    fullpath_transformation_parameters = os.path.join(path_to_results_folder,
                                                      filename_transformation_parameters + file_suffix + '.npy')
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

        s_i_o = 3
        pp = 2

        N = 50

        # Parameters SVF:
        x_1, y_1, z_1 = 60, 60, 1

        if z_1 == 1:
            domain = (x_1, y_1)
            shape = list(domain) + [1, 1, 2]
        else:
            domain = (x_1, y_1, z_1)
            shape = list(domain) + [1, 3]

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
            svf_0   = SVF.generate_random_smooth(shape=shape,
                                                 sigma=sigma_init,
                                                 sigma_gaussian_filter=sigma_gaussian_filter)

            # compute the exponential with the selected GROUND TRUTH method:
            if ground_method == 'vode' or ground_method == 'lsoda':
                disp_chosen_ground = svf_0.exponential_scipy(integrator=ground_method,
                                                             max_steps=ground_method_steps)

            else:
                disp_chosen_ground = svf_0.exponential(algorithm=ground_method,
                                                       s_i_o=s_i_o,
                                                       input_num_steps=ground_method_steps)

            for m in range(num_method_considered):
                if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                    start = time.time()
                    disp_computed = svf_0.exponential_scipy(integrator=names_method_considered[m],
                                                            max_steps=steps_methods_considered[m])
                    res_time[m, s] = (time.time() - start)

                else:
                    start = time.time()
                    disp_computed = svf_0.exponential(algorithm=names_method_considered[m],
                                                      s_i_o=s_i_o,
                                                      input_num_steps=steps_methods_considered[m])
                    res_time[m, s] = (time.time() - start)

                # compute error:
                errors[m, s] = (disp_computed - disp_chosen_ground).norm(passe_partout_size=pp, normalized=True)

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
        np.save(fullpath_array_errors_output,       errors)
        np.save(fullpath_array_comp_time_output,    res_time)

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

        # BOXPLOT custom
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
