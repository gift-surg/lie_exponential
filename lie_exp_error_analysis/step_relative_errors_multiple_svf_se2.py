import copy
import os
import pickle
from os.path import join as jph

import matplotlib.pyplot as plt
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from VECtorsToolkit.tools.transformations import se2
from VECtorsToolkit.tools.fields.generate_vf import generate_from_matrix
from VECtorsToolkit.tools.local_operations.lie_exponential import lie_exponential_scipy, lie_exponential
from VECtorsToolkit.tools.fields.queries import vf_norm

from controller import methods_t_s
from path_manager import pfo_results, pfo_notes_figures, pfo_notes_sharing
from visualizer.graphs_and_stats_new import plot_custom_step_error

"""
Study for the estimate of step error for the Scaling and squaring based and Taylor numerical methods.
Multiple se2 generated SVFs.
"""

if __name__ == "__main__":

    clear_cache()

    ##################
    #   Controller   #
    ##################

    compute       = True
    verbose       = True
    save_external = True
    plot_results  = True

    #######################
    #   Path management   #
    #######################

    prefix_fn = 'exp_comparing_step_relative_errors_'
    kind   = 'SE2'
    number = 'multiple'
    tag  = '_' + str(1)

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

    pfi_array_errors_output        = jph(pfo_errors_times_results, fin_array_errors_output + tag + '.npy')
    pfi_array_comp_time_output     = jph(pfo_errors_times_results, fin_array_comp_time_output + tag + '.npy')
    pfi_field                      = jph(pfo_errors_times_results, fin_field + tag + '.npy')
    pfi_transformation_parameters  = jph(pfo_errors_times_results, fin_transformation_parameters + tag + '.npy')
    pfi_numerical_method_table     = jph(pfo_errors_times_results, fin_numerical_methods_table + tag)
    pfi_figure_output              = jph(pfo_notes_figures, fin_figure_output + tag + '.pdf')
    pfi_csv_table_errors_output    = jph(pfo_notes_sharing, fin_csv_table_errors_output + '.csv')
    pfi_csv_table_comp_time_output = jph(pfo_notes_sharing, fin_csv_table_comp_time_output + '.csv')

    ####################
    #   Computations   #
    ####################

    svf_0 = None

    if compute:  # or compute or load
        s_i_o = 3
        pp = 2

        # Parameters of the SVF
        N = 20
        x_1, y_1, z_1 = 20, 20, 21

        if z_1 == 1:
            omega = (x_1, y_1)
        else:
            omega = (x_1, y_1, z_1)

        interval_theta = (- np.pi / 8, np.pi / 8)
        epsilon = np.pi / 12
        center = (12, 13, 7, 13)  # where to locate the center of the random rotation

        # maximal number of consecutive steps where to compute the step-relative error
        max_steps = 20

        parameters = [x_1, y_1, z_1,           # Domain
                      N,                       # number of samples
                      interval_theta[0], interval_theta[1],  # interval of the rotations
                      center[0], center[1],      # interval of the center of the rotation x
                      center[2], center[3],       # interval of the center of the rotation y
                      max_steps]

        tag = '_' + str(interval_theta[0]) + '_' + str(interval_theta[1]) + '_' + str(N)

        # import methods from external file aaa_general_controller
        methods = methods_t_s

        index_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
        num_method_considered    = len(index_methods_considered)

        names_method_considered       = [methods[j][0] for j in index_methods_considered]
        color_methods_considered      = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        marker_method_considered      = [methods[j][5] for j in index_methods_considered]

        ###########################
        #   Model: computations   #
        ###########################

        print '---------------------'
        print 'Computations started!'
        print '---------------------'

        # Init results
        step_errors = np.zeros([num_method_considered, max_steps, N])  # Methods x Steps

        for s in range(N):

            m_0 = se2.se2g_randomgen_custom_center(interval_theta=interval_theta, interval_center=center,
                                                   epsilon_zero_avoidance=epsilon)
            dm_0 = se2.se2g_log(m_0)

            svf_0       = generate_from_matrix(omega, dm_0.get_matrix, t=1, structure='algebra')
            disp_ground = generate_from_matrix(omega, m_0.get_matrix, t=1, structure='group')

            # svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))

            sdisp_step_j_0 = []
            sdisp_step_j_1 = []

            for met in range(num_method_considered):
                if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':
                    disp_computed = lie_exponential_scipy(svf_0, integrator=names_method_considered[met], max_steps=1)

                else:
                    disp_computed = lie_exponential(svf_0, algorithm=names_method_considered[met], s_i_o=s_i_o,
                                                    input_num_steps=1)
                sdisp_step_j_0 += [disp_computed]

            for stp in range(2, max_steps):
                for met in range(num_method_considered):

                    if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':
                        disp_computed = lie_exponential_scipy(svf_0, integrator=names_method_considered[met],
                                                              max_steps=stp)

                    else:
                        disp_computed = lie_exponential(svf_0, algorithm=names_method_considered[met], s_i_o=s_i_o,
                                                        input_num_steps=stp)

                    sdisp_step_j_1 += [disp_computed]

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
        np.save(pfi_array_errors_output, step_errors)
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

        step_errors = np.load(pfi_array_errors_output)
        svf_as_array = np.load(pfi_field)

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
        color_methods_considered      = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        marker_method_considered      = [methods[j][5] for j in index_methods_considered]

        max_steps = parameters[10]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print 'Step-wise error for multiple se2 generated SVF'
        print 'step-error is computed as norm(exp(svf,step+1) - exp(svf,step)) '
        print 'Up to step : ' + str(max_steps)

        print '---------------------------------------------'

        print '\nParameters of the multiple transformation se2:'
        print 'number of sample = ' + str(parameters[3])
        print 'domain = ' + str(parameters[:3])
        print 'theta interval = ' + str(parameters[4:6])
        print 'omega (frame where the random center belong) = ' + str(parameters[6:10])

        print '\n'
        print 'Methods and parameters:'
        print tabulate(methods,
                       headers=['name', 'compute (True/False)', 'colour',  'line-style',   'marker'])
        print '\n'

        print 'List of the methods considered:'
        print names_method_considered
        print '---------------------------------------------'

    ############################
    ### Visualization method ###
    ############################

    if plot_results:

        step_errors_mean = np.mean(step_errors, axis=2)
        stdev_errors = np.std(step_errors, axis=2)

        print len(color_methods_considered)
        print color_methods_considered

        plot_custom_step_error(range(1, max_steps - 1),
                               step_errors_mean[:, 1:max_steps - 1],
                               names_method_considered,
                               stdev=None,
                               input_parameters=parameters,
                               fig_tag=2,
                               log_scale=True,
                               kind='multiple_SE2',
                               input_colors=color_methods_considered,
                               input_line_style=line_style_methods_considered,
                               input_marker=marker_method_considered,
                               window_title_input='step errors',
                               titles=('iterations vs. MEANS of the step-errors', ''),
                               additional_field=None,
                               legend_location='upper right')

        plt.show()
