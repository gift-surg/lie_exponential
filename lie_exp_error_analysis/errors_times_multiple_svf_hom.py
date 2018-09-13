import os
import pickle
import time
from os.path import join as jph

import matplotlib.pyplot as plt
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from VECtorsToolkit.tools.transformations.pgl2 import randomgen_homography
from VECtorsToolkit.tools.fields.generate_vf import generate_from_projective_matrix
from VECtorsToolkit.tools.local_operations.lie_exponential import lie_exponential_scipy, lie_exponential
from VECtorsToolkit.tools.fields.queries import vf_norm

from VECtorsToolkit.tools.visualisations.fields.fields_at_the_window import see_2_fields


from controller import methods_t_s
from path_manager import pfo_results, pfo_notes_figures, pfo_notes_tables
from visualizer.graphs_and_stats_new import plot_custom_boxplot

"""
Module for the computation of the error of the exponential map.
Svf involved is one 2d SVF generated with matrix in se2_a.
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

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_errors'
    kind   = 'HOM'
    number = 'multiple'
    file_suffix  = '_' + str(1)  # 1 skew, 2 diag

    fin_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    fin_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    fin_csv_table_comp_time_output = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_cp_time'
    fin_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    fin_array_comp_time_output     = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_cp_time'
    fin_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'
    fin_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    fin_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    pfo_errors_times_results = jph(pfo_results, 'errors_times_results')

    os.system('mkdir -p {}'.format(pfo_errors_times_results))
    print("\nPath to results folder {}\n".format(pfo_errors_times_results))

    pfi_array_errors_output        = jph(pfo_errors_times_results, fin_array_errors_output + file_suffix + '.npy')
    pfi_array_comp_time_output     = jph(pfo_errors_times_results, fin_array_comp_time_output + file_suffix + '.npy')
    pfi_field                      = jph(pfo_errors_times_results, fin_field + file_suffix + '.npy')
    pfi_transformation_parameters  = jph(pfo_errors_times_results, fin_transformation_parameters + file_suffix + '.npy')
    pfi_numerical_method_table     = jph(pfo_errors_times_results, fin_numerical_methods_table + file_suffix)
    pfi_figure_output              = jph(pfo_notes_figures, fin_figure_output + file_suffix + '.pdf')
    pfi_csv_table_errors_output    = jph(pfo_notes_tables, fin_csv_table_errors_output + '.csv')
    pfi_csv_table_comp_time_output = jph(pfo_notes_tables, fin_csv_table_comp_time_output + '.csv')

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
        x_1, y_1, z_1 = 50, 50, 1

        N = 10

        in_psl = False

        if z_1 == 1:
            d = 2
            omega = (x_1, y_1)

            # center of the homography
            x_c = x_1 / 2
            y_c = y_1 / 2
            z_c = 1

            projective_center = [x_c, y_c, z_c]

        else:
            d = 3
            omega = (x_1, y_1, z_1)

            # center of the homography
            x_c = x_1 / 2
            y_c = y_1 / 2
            z_c = z_1 / 2
            w_c = 1

            projective_center = [x_c, y_c, z_c, w_c]

        scale_factor = 1. / (np.max(omega) * 1000)
        hom_attributes = [d, scale_factor, 1, in_psl]

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

        # init structures
        errors = np.zeros([num_method_considered, N])
        res_time = np.zeros([num_method_considered, N])

        for s in range(N):  # sampling

            # Generate SVF and displacement:
            h_a, h_g = randomgen_homography(d=hom_attributes[0], scale_factor=hom_attributes[1],
                                            center=projective_center, sigma=hom_attributes[2],
                                            special=hom_attributes[3], get_as_matrix=True)

            # generate SVF
            svf_0       = generate_from_projective_matrix(omega, h_a, structure='algebra')
            disp_ground = generate_from_projective_matrix(omega, h_g, structure='group')

            see_2_fields(svf_0, disp_ground)

            for m in range(num_method_considered):
                if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                    start = time.time()
                    disp_0 = lie_exponential_scipy(svf_0, integrator=names_method_considered[m],
                                                   max_steps=steps_methods_considered[m])
                    res_time[m, s] = (time.time() - start)

                else:
                    start = time.time()
                    disp_0 = lie_exponential(svf_0, algorithm=names_method_considered[m], s_i_o=s_i_o,
                                             input_num_steps=steps_methods_considered[m])
                    res_time[m, s] = (time.time() - start)

                # compute error:
                errors[m, s] = vf_norm(disp_0 - disp_ground, passe_partout_size=pp, normalized=True)

            print errors

            print '--------------------'
            print 'Sampling ' + str(s + 1) + '/' + str(N) + ' .'
            print '--------------------'

        parameters = [[x_1, y_1, z_1]] + [projective_center] + [hom_attributes] + [N]

        ### Save data to folder ###
        np.save(pfi_array_errors_output,       errors)
        np.save(pfi_array_comp_time_output,    res_time)
        np.save(pfi_field,                     svf_0)

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
        steps_methods_considered      = [methods[j][2] for j in index_methods_considered]
        color_methods_considered      = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        marker_method_considered      = [methods[j][5] for j in index_methods_considered]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print 'Error-bar and time for multiple hom generated SVF'
        print '---------------------------------------------'
        # parameters = [[x_1, y_1, z_1]] + [projective_center] + [hom_attributes] + [N]
        print '\nParameters of the transformation hom:'
        print 'domain = ' + str(parameters[0])
        print 'center = ' + str(parameters[1])
        print 'hom attributes d, scale factor, sigma, special = ' + str(parameters[2])
        print 'number of samples = ' + str(parameters[3])

        print '\n'
        print 'Methods and parameters:'
        print tabulate(methods,
                       headers=['name', 'compute (True/False)', 'num_steps'])
        print '\n'

        print 'You chose to compute ' + str(num_method_considered) + ' methods.'
        print 'List of the methods considered:'
        print names_method_considered
        print 'List of the steps of the methods considered'
        print steps_methods_considered

    #################
    # Visualization #
    #################

    mean_errors   = np.mean(errors, axis=1)
    mean_res_time = np.mean(res_time, axis=1)

    results_by_column = [[met, err, tim]
                         for met, err, tim in zip(names_method_considered, list(mean_errors), list(mean_res_time))]

    print '\n'
    print 'Results and computational time:'
    print tabulate(results_by_column,
                   headers=['method', 'error', 'comp. time (sec)'])
    print '\n END'

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
                            input_titles=('Error exponential map for multiple HOM svf', 'field'),
                            kind='multiple_HOM',
                            window_title_input='bar_plot_multiple_se2',
                            additional_field=None,
                            log_scale=True,
                            input_parameters=None,
                            annotate_mean=True,
                            add_extra_annotation=mean_res_time)
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
