import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from tabulate import tabulate
from sympy.core.cache import clear_cache
import pickle

from utils.projective_algebras import get_random_hom_a_matrices

from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from utils.path_manager import path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables
from visualizer.graphs_and_stats_new import plot_custom_step_versus_error_multiple
from aaa_general_controller import methods_t_s

"""
Study for the estimate of inverse consistency error for several numerical methods to compute the exponential of an svf.
One se2 generated SVF.
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

    prefix_fn = 'exp_comparing_inverse_consistency_error'
    kind   = 'HOM'
    number = 'multiple'
    file_suffix  = '_' + str(1)

    filename_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    filename_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    filename_csv_table_comp_time_output = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_cp_time'
    filename_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    filename_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    filename_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'
    filename_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    path_to_results_folder = os.path.join(path_to_results_folder, 'errors_times_results')
    fullpath_array_errors_output = os.path.join(path_to_results_folder,
                                                filename_array_errors_output + file_suffix + '.npy')
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

        pp = 2     # passepartout
        s_i_o = 3  # spline interpolation order

        N = 10

        x_1, y_1, z_1 = 50, 50, 1

        in_psl = False

        if z_1 == 1:
            d = 2
            domain = (x_1, y_1)
            shape = list(domain) + [1, 1, 2]

            # center of the homography
            x_c = x_1 / 2
            y_c = y_1 / 2
            z_c = 1

            projective_center = [x_c, y_c, z_c]

        else:
            d = 3
            domain = (x_1, y_1, z_1)
            shape = list(domain) + [1, 3]

            # center of the homography
            x_c = x_1 / 2
            y_c = y_1 / 2
            z_c = z_1 / 2
            w_c = 1

            projective_center = [x_c, y_c, z_c, w_c]

        list_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30, 60]

        parameters = [x_1, y_1, z_1] + [N] + list_steps

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

        # init results
        errors = np.zeros([num_method_considered, len(list_steps), N])
        svf_as_array = None

        for s in range(N):  # sample

            # Generate SVF and displacement:
            scale_factor = 1. / (np.max(domain) * 10)
            hom_attributes = [d, scale_factor, 1, in_psl]

            h_a, h_g = get_random_hom_a_matrices(d=hom_attributes[0],
                                                 scale_factor=hom_attributes[1],
                                                 sigma=hom_attributes[2],
                                                 special=hom_attributes[3])

            # Generate SVF
            svf_0 = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)
            svf_0_inv   = -1*svf_0

            for step_index, step_num in enumerate(list_steps):
                for met in range(num_method_considered):
                    if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':
                        disp_computed = svf_0.exponential_scipy(integrator=names_method_considered[met],
                                                                max_steps=step_num)

                        disp_computed_inv = svf_0_inv.exponential_scipy(integrator=names_method_considered[met],
                                                                        max_steps=step_num)
                    else:
                        disp_computed = svf_0.exponential(algorithm=names_method_considered[met],
                                                          s_i_o=s_i_o,
                                                          input_num_steps=step_num)
                        disp_computed_inv = svf_0_inv.exponential(algorithm=names_method_considered[met],
                                                                  s_i_o=s_i_o,
                                                                  input_num_steps=step_num)
                    # compute error:
                    sdisp_o_sdisp_inv = SDISP.composition(disp_computed, disp_computed_inv, s_i_o=s_i_o)
                    sdisp_inv_o_sdisp = SDISP.composition(disp_computed_inv, disp_computed, s_i_o=s_i_o)

                    errors[met, step_index, s] = 0.5 * (sdisp_o_sdisp_inv.norm(passe_partout_size=pp, normalized=True)
                                                 + sdisp_inv_o_sdisp.norm(passe_partout_size=pp, normalized=True))

                # tabulate the result:
                print 'Step ' + str(step_num) + ' phase  ' + str(step_index + 1) + '/' + str(len(list_steps)) + \
                      ' computed for sample ' + str(s + 1) + '/' + str(N) + '.'
                if verbose:
                    results_by_column = [[met, err] for met, err in zip(names_method_considered,
                                                                        list(errors[:, step_index, s]))]

                    print '---------------------------------------------'
                    print tabulate(results_by_column,
                                   headers=['method', 'inverse consistency error for ' + str(step_num) + ' steps'])
                    print '---------------------------------------------'

        svf_as_array = copy.deepcopy(svf_0.field)

        ### Save data to folder ###

        np.save(fullpath_array_errors_output, errors)
        np.save(fullpath_field, svf_as_array)

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

        list_steps = parameters[9:]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print '-----------------------------------------------------'
        print 'Inverse consistency error for one HOM generated SVF'
        print '-----------------------------------------------------'

        print '\nParameters that generate the SVF'
        print 'domain = ' + str(parameters[:3])
        print 'center = ' + str(parameters[3])
        print 'kind = ' + str(parameters[4])
        print 'scale factor = ' + str(parameters[5])
        print 'sigma = ' + str(parameters[6])
        print 'in psl = ' + str(parameters[7])
        print 'Number of samples = ' + str(parameters[8])
        print 'List of steps considered:'
        print str(parameters[9:])

        print '\n'
        print 'Methods and parameters:'
        print tabulate(methods,
                       headers=['name', 'compute (True/False)', 'colour',  'line-style',   'marker'])
        print '\n'
        print 'List of the methods considered:'
        print names_method_considered
        print '---------------------------------------------'

    ####################
    ### View results ###
    ####################

    means_errors = np.mean(errors, axis=2)
    means_std = np.std(errors, axis=2)

    if verbose:

        print '---------------------------------'
        print 'Inverse consistency results table'
        print '---------------------------------'

        results_by_column = [[names_method_considered[j]] + list(means_errors[j, :])
                             for j in range(num_method_considered)]
        print tabulate(results_by_column, headers=[''] + list(list_steps))

        print '\nEND'

    if plot_results:

        plot_custom_step_versus_error_multiple(list_steps,
                                               means_errors,
                                               means_std,
                                               names_method_considered,
                                               input_parameters=parameters, fig_tag=2, log_scale=True,
                                               additional_vertical_line=None,
                                               additional_field=svf_as_array,
                                               kind='multiple_HOM',
                                               input_marker=marker_method_considered,
                                               input_colors=color_methods_considered,
                                               input_line_style=line_style_methods_considered
                                               )

        plt.show()
