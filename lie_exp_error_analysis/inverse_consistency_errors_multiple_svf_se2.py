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
from VECtorsToolkit.tools.fields.composition import lagrangian_dot_lagrangian


from controller import methods_t_s
from path_manager import pfo_results, pfo_notes_figures, pfo_notes_sharing
from visualizer.graphs_and_stats_new import plot_custom_step_versus_error_multiple


"""
Study for the estimate of inverse consistency error for several numerical methods to compute the exponential of an svf.
Multiple se2 generated SVF.
"""

if __name__ == "__main__":

    clear_cache()

    ##################
    #   Controller   #
    ###################

    compute       = True
    verbose       = True
    save_external = False
    plot_results  = True

    #######################
    #   Path management   #
    #######################

    prefix_fn = 'exp_comparing_inverse_consistency_error'
    kind   = 'SE2'
    number = 'multiple'
    tag  = '_' + str(1)

    fin_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    fin_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    fin_csv_table_comp_time_output = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_cp_time'
    fin_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    fin_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    fin_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'
    fin_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    pfo_errors_times_results = jph(pfo_results, 'errors_times_results')

    os.system('mkdir -p {}'.format(pfo_errors_times_results))
    print("\nPath to results folder {}\n".format(pfo_errors_times_results))
    
    pfi_array_errors_output        = jph(pfo_errors_times_results, fin_array_errors_output + tag + '.npy')
    pfi_transformation_parameters  = jph(pfo_errors_times_results, fin_transformation_parameters + tag)
    pfi_field                      = jph(pfo_errors_times_results, fin_field + tag + '.npy')
    pfi_numerical_method_table     = jph(pfo_errors_times_results, fin_numerical_methods_table + tag)
    pfi_figure_output              = jph(pfo_notes_figures, fin_figure_output + tag + '.pdf')
    pfi_csv_table_errors_output    = jph(pfo_notes_sharing, fin_csv_table_errors_output + '.csv')
    pfi_csv_table_comp_time_output = jph(pfo_notes_sharing, fin_csv_table_comp_time_output + '.csv')

    ####################
    #   Computations   #
    ####################

    if compute:  # or compute or load

        s_i_o = 3
        pp = 2

        N = 20

        x_1, y_1, z_1 = 20, 20, 10

        if z_1 == 1:
            omega = (x_1, y_1)
        else:
            omega = (x_1, y_1, z_1)

        interval_theta = (- np.pi / 8, np.pi / 8)
        epsilon = np.pi / 12
        center = (12, 13, 7, 13)  # where to locate the center of the random rotation

        list_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30, 60]

        parameters  = [x_1, y_1, z_1,          # domain coordinates
                       N,                       # number of samples
                       interval_theta[0], interval_theta[1],  # interval of the rotations
                       center[0], center[1],      # interval of the center of the rotation x
                       center[2], center[3]       # interval of the center of the rotation y
                       ] + list_steps

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

        # initialize results
        errors = np.zeros([num_method_considered, len(list_steps), N])

        for s in range(N):  # sample

            m_0 = se2.se2g_randomgen_custom_center(interval_theta=interval_theta, interval_center=center,
                                                   epsilon_zero_avoidance=epsilon)
            dm_0 = se2.se2g_log(m_0)

            # Generate svf
            svf_0     = generate_from_matrix(omega, dm_0.get_matrix, t=1, structure='algebra')
            svf_0_inv = -1 * svf_0

            # Generate displacement ground truth (for sanity check)
            sdisp_0     = generate_from_matrix(omega, m_0.get_matrix, t=1, structure='group')
            sdisp_0_inv = generate_from_matrix(omega, np.linalg.inv(m_0.get_matrix), t=1, structure='group')

            # # Sanity check: composition of the ground truth, must be very close to the identity field.
            # sdisp_o_sdisp_inv_ground = lagrangian_dot_lagrangian(sdisp_0, sdisp_0_inv, s_i_o=s_i_o)
            # sdisp_inv_o_sdisp_ground = lagrangian_dot_lagrangian(sdisp_0_inv, sdisp_0, s_i_o=s_i_o)
            #
            # np.testing.assert_array_almost_equal(sdisp_o_sdisp_inv_ground, np.zeros_like(sdisp_o_sdisp_inv_ground),
            #                                      decimal=0)
            # np.testing.assert_array_almost_equal(sdisp_inv_o_sdisp_ground, np.zeros_like(sdisp_inv_o_sdisp_ground),
            #                                      decimal=0)

            for step_index, step_num in enumerate(list_steps):

                for met in range(num_method_considered):

                    if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':

                        disp_computed = lie_exponential_scipy(svf_0, integrator=names_method_considered[met],
                                                              max_steps=step_num)

                        disp_computed_inv = lie_exponential_scipy(svf_0_inv, integrator=names_method_considered[met],
                                                                  max_steps=step_num)

                    else:

                        disp_computed = lie_exponential(svf_0, algorithm=names_method_considered[met], s_i_o=s_i_o,
                                                        input_num_steps=step_num)
                        disp_computed_inv = lie_exponential(svf_0_inv, algorithm=names_method_considered[met],
                                                            s_i_o=s_i_o, input_num_steps=step_num)
                    # compute error:
                    sdisp_o_sdisp_inv = lagrangian_dot_lagrangian(disp_computed, disp_computed_inv, s_i_o=s_i_o)
                    sdisp_inv_o_sdisp = lagrangian_dot_lagrangian(disp_computed_inv, disp_computed, s_i_o=s_i_o)

                    errors[met, step_index, s] = 0.5 * (vf_norm(sdisp_o_sdisp_inv, passe_partout_size=pp, normalized=True) +
                                                        vf_norm(sdisp_inv_o_sdisp, passe_partout_size=pp, normalized=True))

                # tabulate the results:
                print 'Step ' + str(step_num) + ' phase  ' + str(step_index + 1) + '/' + str(len(list_steps)) + \
                      ' computed for sample ' + str(s + 1) + '/' + str(N) + '.'
                if verbose:
                    results_by_column = [[met, err] for met, err in zip(names_method_considered,
                                                                        list(errors[:, step_index, s]))]

                    print '---------------------------------------------'
                    print tabulate(results_by_column,
                                   headers=['method', 'inverse consistency error for ' + str(step_num) + ' steps'])
                    print '---------------------------------------------'

        ### Save data to folder ###

        np.save(pfi_array_errors_output, errors)
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

        errors = np.load(pfi_array_errors_output)
        svf_0  = np.load(pfi_field)

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

        list_steps = parameters[10:]
        N = parameters[3]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print '----------------------------------------------------------'
        print 'Inverse consistency error for multiple SE(2) generated SVF'
        print '----------------------------------------------------------'

        print '\nParameters that generate the SVF'
        print 'Svf dimension: ' + str(parameters[:3])
        print 'Number of samples = ' + str(parameters[3])
        print 'theta interval: ' + str(parameters[4:6])
        print 'omega = ' + str(parameters[6:10])
        print 'List of steps considered:'
        print str(parameters[10:])

        print '\n'
        print 'Methods and parameters:'
        print tabulate(methods,
                       headers=['name', 'compute (True/False)', 'colour',  'line-style',   'marker'])
        print '\n'

        print 'List of the methods considered:'
        print names_method_considered
        print 'Steps considered for each method:'
        print list_steps
        print '---------------------------------------------'

    ####################
    #   View results   #
    ####################

    means_errors = np.mean(errors, axis=2)
    means_std = np.std(errors, axis=2)

    if verbose:

        print '---------------------------------'
        print 'Inverse consistency results table of the mean for ' + str(N) + ' samples.'
        print '---------------------------------'

        results_by_column = [[names_method_considered[j]] + list(means_errors[j, :])
                             for j in range(num_method_considered)]
        print tabulate(results_by_column, headers=[''] + list(list_steps))

    if plot_results:

        plot_custom_step_versus_error_multiple(list_steps,
                                               means_errors,  # means
                                               means_std,  # std
                                               names_method_considered,
                                               input_parameters=parameters, fig_tag=2, log_scale=True,
                                               additional_vertical_line=None,
                                               additional_field=svf_0,
                                               kind='multiple_SE2',
                                               titles=('inverse consistency errors vs iterations', 'Fields like:'),
                                               input_marker=marker_method_considered,
                                               input_colors=color_methods_considered,
                                               input_line_style=line_style_methods_considered)

        plt.show()
