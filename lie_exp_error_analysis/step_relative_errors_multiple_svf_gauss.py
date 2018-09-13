import copy
import os
import pickle
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
from visualizer.graphs_and_stats_new import plot_custom_step_error

"""
Study for the estimate of step error for the Scaling and squaring based and Taylor numerical methods.
Multiple Gauss-generated SVF.
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

    # The results, and additional information are loaded in see_error_time_results
    # with a simplified name. They are kept safe from other subsequent tests with the same code.
    save_for_sharing = True

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_step_relative_errors_'
    kind      = 'GAUSS'
    number    = 'multiple'
    file_suffix  = '_' + str(1)

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

    fullpath_array_errors_output = jph(pfo_errors_times_results, fin_array_errors_output + file_suffix + '.npy')
    pfi_array_comp_time_output = jph(pfo_errors_times_results, fin_array_comp_time_output + file_suffix + '.npy')
    fullpath_transformation_parameters = jph(pfo_errors_times_results, fin_transformation_parameters + file_suffix)
    fullpath_field = jph(pfo_errors_times_results, fin_field + file_suffix + '.npy')
    fullpath_numerical_method_table = jph(pfo_errors_times_results, fin_numerical_methods_table + file_suffix)
    fullpath_figure_output  = jph(pfo_notes_figures, fin_figure_output + file_suffix + '.pdf')
    fullpath_csv_table_errors_output = jph(pfo_notes_sharing, fin_csv_table_errors_output + '.csv')
    fullpath_csv_table_comp_time_output = jph(pfo_notes_sharing, fin_csv_table_comp_time_output + '.csv')

    ####################
    ### Computations ###
    ####################

    if compute:  # or compute or load

        s_i_o = 3
        pp = 2

        # Parameters SVF:
        x_1, y_1, z_1 = 50, 50, 50

        if z_1 == 1:
            omega = (x_1, y_1)
        else:
            omega = (x_1, y_1, z_1)

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

            svf_0  = generate_random(omega, parameters=(sigma_init, sigma_gaussian_filter))

            sdisp_step_j_0 = []
            sdisp_step_j_1 = []

            for met in range(num_method_considered):

                if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':
                    disp_0 = lie_exponential_scipy(svf_0, integrator=names_method_considered[met], max_steps=1)

                else:
                    disp_0 = lie_exponential(svf_0, algorithm=names_method_considered[met], s_i_o=s_i_o,
                                             input_num_steps=1)
                sdisp_step_j_0 += [disp_0]

            for stp in range(2, max_steps):
                for met in range(num_method_considered):

                    if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':
                        disp_0 = lie_exponential_scipy(svf_0, integrator=names_method_considered[met], max_steps=stp)

                    else:
                        disp_0 = lie_exponential(svf_0, algorithm=names_method_considered[met], s_i_o=s_i_o,
                                                 input_num_steps=stp)

                    sdisp_step_j_1 += [disp_0]
                    step_errors[met, stp - 1, s] = \
                        vf_norm(sdisp_step_j_1[met] - sdisp_step_j_0[met], passe_partout_size=pp)

                sdisp_step_j_0 = copy.deepcopy(sdisp_step_j_1)
                sdisp_step_j_1 = []

                if verbose:
                    results_by_column = [[met, err] for met, err in zip(names_method_considered,
                                                                        list(step_errors[:, stp - 1, s]))]

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
        np.save(fullpath_field, svf_0)

        with open(fullpath_transformation_parameters, 'wb') as f:
            pickle.dump(parameters, f)

        with open(fullpath_numerical_method_table, 'wb') as f:
            pickle.dump(methods, f)

        if save_for_sharing:

            path_to_sharing_folder = jph(pfo_errors_times_results, 'sharing_folder')
            np.save(jph(path_to_sharing_folder, 'step_relative_errors_gauss'), step_errors)
            np.save(jph(path_to_sharing_folder, 'step_relative_exp_methods_param_gauss'), methods_t_s)

            with open(jph(path_to_sharing_folder,
                                   'step_relative_errors_transformation_parameters_gauss'), 'wb') as f:
                pickle.dump(parameters, f)

            with open(jph(path_to_sharing_folder,
                                   'step_relative_errors_numerical_method_table_gauss'), 'wb') as f:
                pickle.dump(methods, f)

        print
        print '------------------------------------------'
        print 'Data computed and saved in external files!'
        print '------------------------------------------'

    else:

        step_errors = np.load(fullpath_array_errors_output)
        svf_0 = np.load(fullpath_field)

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
