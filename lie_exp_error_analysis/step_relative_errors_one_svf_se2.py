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
One se2 generated SVF.
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
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_step_relative_errors_'
    kind   = 'SE2'
    number = 'single'
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

    if compute:  # or compute or load

        s_i_o = 3
        pp = 2

        x_1, y_1, z_1 = 20, 20, 10

        if z_1 == 1:
            omega = (x_1, y_1)
        else:
            omega = (x_1, y_1, z_1)

        x_c   = np.floor(x_1 / 2)
        y_c   = np.floor(y_1 / 2)
        theta = np.pi / 8

        tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
        ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

        max_steps = 20  # maximal number of consecutive steps where to compute the step-relative error

        parameters = [x_1, y_1, z_1] + [theta, tx, ty, max_steps]

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

        # Init results
        svf_as_array = None
        step_errors  = np.zeros([num_method_considered, max_steps])  # Methods x Steps
        # At step_errors[met, step] we save norm(exp(svf,step) - exp(svf,step-1)

        # generate matrices
        m_0 = se2.Se2G(theta, tx, ty)
        dm_0 = se2.se2g_log(m_0)

        # Generate svf
        svf_0 = generate_from_matrix(omega, dm_0.get_matrix, t=1, structure='algebra')

        # init storage of step = step -1 and step = step
        sdisp_step_j_0 = []
        sdisp_step_j_1 = []

        # init the first round step outside the main cycles:
        for met in range(num_method_considered):
            # Compute the displacement with the selected method:
            if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':
                disp_computed = lie_exponential_scipy(svf_0, integrator=names_method_considered[met], max_steps=2)

            else:
                disp_computed = lie_exponential(svf_0, algorithm=names_method_considered[met], s_i_o=s_i_o,
                                                input_num_steps=2)
            sdisp_step_j_0 += [disp_computed]

        # start the main cycle with sdisp_step_j_0 initialized:
        for stp in range(3, max_steps):
            for met in range(num_method_considered):

                # Compute the displacement with the selected method:
                if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':
                    disp_computed = lie_exponential_scipy(svf_0, integrator=names_method_considered[met], max_steps=stp)

                else:
                    disp_computed = lie_exponential(svf_0, algorithm=names_method_considered[met], s_i_o=s_i_o,
                                                    input_num_steps=stp)

                # Store the new displacement in the list_1:
                sdisp_step_j_1 += [disp_computed]

                # Compute the step error and store in the vector step_error
                step_errors[met, stp - 1] = vf_norm(sdisp_step_j_1[met] - sdisp_step_j_0[met], passe_partout_size=pp)

            # copy sdisp_step_j_1 in sdisp_step_j_0 for the next step
            sdisp_step_j_0 = copy.deepcopy(sdisp_step_j_1)
            # erase for safety the list_1
            sdisp_step_j_1 = []

            if verbose:
                # show result at each step.
                results_by_column = [[met, err] for met, err in zip(names_method_considered,
                                                                    list(step_errors[:, stp - 1]))]

                print 'Step ' + str(stp + 1) + '/' + str(max_steps) + '.'
                print '---------------------------------------------'
                print tabulate(results_by_column,
                               headers=['method', 'error'])
                print '---------------------------------------------'

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

        max_steps = parameters[6]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print 'Step-wise error for one se2 generated SVF'
        print 'step-error is computed as norm(exp(svf,step+1) - exp(svf,step)) '
        print 'Up to step : ' + str(max_steps)

        print '---------------------------------------------'

        print '\nParameters of the transformation se2:'
        print 'domain = ' + str(parameters[:3])
        print 'theta, tx, ty = ' + str(parameters[3:6])

        print '\n'
        print 'Methods and parameters from external table:'
        print tabulate(methods,
                       headers=['name', 'compute (True/False)', 'colour',  'line-style',   'marker'])
        print '\n'

        print 'List of the methods considered:'
        print names_method_considered
        print '---------------------------------------------'

    ############################
    ### Visualization method ###
    ############################

    print '\n END'

    if plot_results:

        plot_custom_step_error(range(1, max_steps - 1),
                               step_errors[:, 1:max_steps - 1],
                               names_method_considered,
                               input_parameters=parameters,
                               fig_tag=2,
                               log_scale=True,
                               kind='one_SE2',
                               input_colors=color_methods_considered,
                               input_line_style=line_style_methods_considered,
                               input_marker=marker_method_considered,
                               window_title_input='step errors',
                               titles=('iterations vs.step error', 'Field'),
                               additional_field=svf_as_array,
                               legend_location='upper right')

        plt.show()
