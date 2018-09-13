import copy
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from VECtorsToolkit.tools.fields.generate_vf import generate_random
from VECtorsToolkit.tools.local_operations.lie_exponential import lie_exponential, lie_exponential_scipy
from VECtorsToolkit.tools.fields.queries import vf_norm

from controller import methods_t_s
from path_manager import pfo_notes_figures, pfo_notes_tables, pfo_results
from visualizer.graphs_and_stats_new import plot_custom_bar_chart_with_error

"""
Module for the computation of the error of the exponential map.
Svf involved is one 2d SVF generated with a gaussian filter.
For this kind of svf there is no ground truth available. One of the numerical integrator can be chosen
as ground truth.
"""


if __name__ == "__main__":

    clear_cache()

    ##################
    ### Controller ###
    ##################

    compute       = False
    verbose       = True
    save_external = False
    plot_results  = True

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_errors'
    kind   = 'GAUSS'
    number = 'single'
    file_suffix  = '_' + str(1)

    fin_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    fin_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    fin_csv_table_comp_time_output = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_cp_time'
    fin_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    fin_array_comp_time_output     = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_cp_time'
    fin_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'
    fin_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    fin_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    pfo_errors_times_results = os.path.join(pfo_results, 'errors_times_results')

    os.system('mkdir -p {}'.format(pfo_errors_times_results))
    print("\nPath to results folder {}\n".format(pfo_errors_times_results))

    pfi_array_errors_output = os.path.join(pfo_errors_times_results, fin_array_errors_output + file_suffix + '.npy')
    pfi_array_comp_time_output = os.path.join(pfo_errors_times_results, fin_array_comp_time_output + file_suffix + '.npy')
    pfi_field = os.path.join(pfo_errors_times_results, fin_field + file_suffix + '.npy')
    pfi_transformation_parameters = os.path.join(pfo_errors_times_results, fin_transformation_parameters + file_suffix + '.npy')
    pfi_numerical_method_table = os.path.join(pfo_errors_times_results, fin_numerical_methods_table + file_suffix)

    # path to results external to the project:
    pfi_figure_output  = os.path.join(pfo_notes_figures, fin_figure_output + file_suffix + '.pdf')
    pfi_csv_table_errors_output = os.path.join(pfo_notes_tables, fin_csv_table_errors_output + '.csv')
    pfi_csv_table_comp_time_output = os.path.join(pfo_notes_tables, fin_csv_table_comp_time_output + '.csv')

    ####################
    ### Computations ###
    ####################

    if compute:  # or compute or load

        random_seed = 0

        if random_seed > 0:
            np.random.seed(random_seed)

        s_i_o = 3
        pp = 2

        # Parameters SVF:
        x_1, y_1, z_1 = 60, 60, 10

        if z_1 == 1:
            omega = (x_1, y_1)
        else:
            omega = (x_1, y_1, z_1)

        sigma_init = 5
        sigma_gaussian_filter = 2

        # Numerical method whose result corresponds to the ground truth:
        ground_method = 'rk4'  # in the following table should be false.
        ground_method_steps = 10

        parameters = [x_1, y_1, z_1] + [sigma_init, sigma_gaussian_filter, ground_method, ground_method_steps]

        # import methods from external file aaa_general_controller
        methods = methods_t_s

        indexes_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
        num_method_considered    = len(indexes_methods_considered)

        names_method_considered  = [methods[j][0] for j in indexes_methods_considered]
        steps_methods_considered = [methods[j][2] for j in indexes_methods_considered]

        ###########################
        ### Model: computations ###
        ###########################

        print '---------------------'
        print 'Computations started!'
        print '---------------------'

        errors = np.zeros(num_method_considered)
        res_time = np.zeros(num_method_considered)

        # Generate svf
        svf_0   = generate_random(omega, parameters=(sigma_init, sigma_gaussian_filter))

        # compute the exponential with the selected ground truth method:
        if ground_method == 'vode' or ground_method == 'lsoda':
            disp_silver_ground = lie_exponential_scipy(svf_0, integrator=ground_method, max_steps=ground_method_steps)

        else:
            disp_silver_ground = lie_exponential(svf_0, algorithm=ground_method, s_i_o=s_i_o,
                                                 input_num_steps=ground_method_steps)

        for m in range(num_method_considered):
            if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                start = time.time()
                sdisp_0 = lie_exponential_scipy(svf_0, integrator=names_method_considered[m],
                                                max_steps=steps_methods_considered[m])
                res_time[m] = (time.time() - start)

            else:
                start = time.time()
                sdisp_0 = lie_exponential(svf_0, algorithm=names_method_considered[m], s_i_o=s_i_o,
                                          input_num_steps=steps_methods_considered[m])
                res_time[m] = (time.time() - start)

            # compute error:
            errors[m] = vf_norm(sdisp_0 - disp_silver_ground, passe_partout_size=pp, normalized=True)

            if verbose:
                print '--------------------------------------------------------------------------'
                print 'Computation for the method ' + str(names_method_considered[m]) + ' done.'

        ### Save data to folder ###
        np.save(pfi_array_errors_output,       errors)
        np.save(pfi_array_comp_time_output,    res_time)
        np.save(pfi_field, svf_0)

        with open(pfi_transformation_parameters, 'wb') as f:
            pickle.dump(parameters, f)

        with open(pfi_numerical_method_table, 'wb') as f:
            pickle.dump(methods, f)

        print
        print '------------------------------------------'
        print 'Data computed and saved in external files!'
        print '------------------------------------------'

    else:
        errors       = np.load(pfi_array_errors_output)
        res_time     = np.load(pfi_array_comp_time_output)
        svf_0 = np.load(pfi_field)

        with open(pfi_transformation_parameters, 'rb') as f:
            parameters = pickle.load(f)

        with open(pfi_numerical_method_table, 'rb') as f:
            methods = pickle.load(f)

        print
        print '------------'
        print 'Data loaded!'
        print '------------'

        index_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
        num_method_considered    = len(index_methods_considered)

        names_method_considered       = [methods[j][0] for j in index_methods_considered]
        steps_methods_considered      = [methods[j][2] for j in index_methods_considered]
        color_methods_considered      = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        marker_method_considered      = [methods[j][5] for j in index_methods_considered]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print 'Error-bar and time for one GAUSS generated SVF'
        print '---------------------------------------------'

        print '\nParameters of the transformation se2:'
        print 'domain = ' + str(parameters[:3])
        print 'sigma_init, sigma_gaussian_filter = ' + str(parameters[3:6])
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

    #################
    # Visualization #
    #################

    results_by_column = [[met, err, tim]
                         for met, err, tim in zip(names_method_considered, list(errors), list(res_time))]

    print '\n'
    print 'Results and computational time:'
    print tabulate(results_by_column,
                   headers=['method', 'error', 'comp. time (sec)'])
    print '\n'

    if plot_results:

        plot_custom_bar_chart_with_error(input_data=errors,
                                         input_names=names_method_considered,
                                         fig_tag=11,
                                         titles=('Error exponential map for one GAUSS-generated svf', 'field'),
                                         kind='one_GAUSS',
                                         window_title_input='bar_plot_one_GAUSS',
                                         additional_field=svf_0,
                                         log_scale=True,
                                         input_parameters=parameters,
                                         add_extra_numbers=res_time)
        plt.show()

    ### Save figures and table in external folder ###

    if save_external:

        # Save the table in latex format!
        f = open(pfi_csv_table_errors_output, 'w')
        f.write(tabulate(results_by_column,
                         headers=['method', 'error', 'comp. time (sec)'], tablefmt="latex"))
        f.close()

        # Save image:
        plt.savefig(pfi_figure_output, format='pdf', dpi=400)

        print 'Figure ' + fin_figure_output + ' saved in the external folder ' + str(pfi_figure_output)
