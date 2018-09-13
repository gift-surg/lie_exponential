import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from controller import methods_t_s, path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables
from visualizer.graphs_and_stats_new import plot_custom_boxplot, plot_custom_cluster

"""
Module for the computation of one 2d SVF generated with matrix of se2_a.
It compares the exponential computation with different methods for a number of
steps defined by the user.

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

    prefix_fn = 'exp_comparing_errors'
    kind   = 'SE2'
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

        pp = 2     # passepartout
        s_i_o = 3  # spline interpolation order

        N = 20

        # Parameters SVF:
        x_1, y_1, z_1 = 20, 20, 5

        if z_1 == 1:
            domain = (x_1, y_1)
            shape = list(domain) + [1, 1, 2]
        else:
            domain = (x_1, y_1, z_1)
            shape = list(domain) + [1, 3]

        interval_theta = (- np.pi / 8, np.pi / 8)
        epsilon = np.pi / 12
        omega = (12, 13, 7, 13)  # where to locate the center of the random rotation

        parameters = [x_1, y_1, z_1,
                      N,                       # number of samples
                      interval_theta[0], interval_theta[1],  # interval of the rotations
                      omega[0], omega[1],      # interval of the center of the rotation x
                      omega[2], omega[3]]      # interval of the center of the rotation y

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

        errors = np.zeros([num_method_considered, N])  # Row: method, col: sampling
        res_time = np.zeros([num_method_considered, N])  # Row: method, col: sampling

        for s in range(N):  # sample

            # generate matrices
            m_0 = se2_g.randomgen_custom_center(interval_theta=interval_theta,
                                                omega=omega,
                                                epsilon_zero_avoidance=epsilon)
            dm_0 = se2_g.log(m_0)

            # Generate svf
            svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))
            disp_0  = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))

            for m in range(num_method_considered):  # method
                if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                    start = time.time()
                    disp_computed = svf_0.exponential_scipy(integrator=names_method_considered[m],
                                                            max_steps=steps_methods_considered[m])
                    res_time[m] = (time.time() - start)

                else:
                    start = time.time()
                    disp_computed = svf_0.exponential(algorithm=names_method_considered[m],
                                                      s_i_o=s_i_o,
                                                      input_num_steps=steps_methods_considered[m])
                    res_time[m, s] = (time.time() - start)

                # compute error:
                errors[m, s] = (disp_computed - disp_0).norm(passe_partout_size=pp, normalized=True)

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
        errors     = np.load(fullpath_array_errors_output)
        res_time   = np.load(fullpath_array_comp_time_output)

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
        # Save table csv
        # np.savetxt(fullpath_csv_table_errors_output, errors, delimiter=" & ")
        # np.savetxt(fullpath_csv_table_comp_time_output, errors, delimiter=" & ")
        # Save image:
        plt.savefig(fullpath_figure_output, format='pdf', dpi=400)
        print 'Figure ' + filename_figure_output + ' saved in the external folder ' + str(fullpath_figure_output)
