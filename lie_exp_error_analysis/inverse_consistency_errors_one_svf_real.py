import copy
import os
import pickle

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from controller import methods_t_s, path_to_results_folder, path_to_exp_notes_figures, \
    path_to_exp_notes_tables
from visualizer.graphs_and_stats_new import plot_custom_step_versus_error_single

"""
Study for the estimate of inverse consistency error for several numerical methods to compute the exponential of an svf.
One Gauss generated SVF.
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

    prefix_fn = 'exp_comparing_inverse_consistency_error'
    kind   = 'REAL'
    number = 'single'
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

        s_i_o = 3
        pp = 30

        ### Select SVF from ADNII data: ###

        # svf data -> id of the loaded element:
        id_element = 3

        list_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30, 60]

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
        errors = np.zeros([num_method_considered, len(list_steps)])

        disp_name_A_C = 'displacement_AD_' + str(id_element) + '_.nii.gz'
        # Load as nib:
        nib_A_C = nib.load(os.path.join(displacements_aei_fp, disp_name_A_C))

        # reduce from 3d to 2d:
        data_A_C = nib_A_C.get_data()
        header_A_C = nib_A_C.header
        affine_A_C = nib_A_C.affine

        array_2d_A_C = data_A_C[pp:-pp, pp:-pp, 100:101, :, 0:2]

        # Create svf over the array:
        svf_0 = SVF.from_array_with_header(array_2d_A_C, header=header_A_C, affine=affine_A_C)

        svf_0_inv   = -1 * svf_0
        svf_as_array = copy.deepcopy(svf_0.field)

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

                errors[met, step_index] = 0.5 * (sdisp_o_sdisp_inv.norm(passe_partout_size=pp, normalized=True)
                                          + sdisp_inv_o_sdisp.norm(passe_partout_size=pp, normalized=True))

            # tabulate the result:
            print 'Step ' + str(step_num) + ' computed: phase  ' + str(step_index + 1) + '/' \
                  + str(len(list_steps)) + '.'
            if verbose:
                results_by_column = [[met, err] for met, err in zip(names_method_considered,
                                                                    list(errors[:, step_index]))]

                print '---------------------------------------------'
                print tabulate(results_by_column,
                               headers=['method', 'inverse consistency error for ' + str(step_num) + ' steps'])
                print '---------------------------------------------'

        parameters = list(svf_as_array.shape[:3]) + [id_element] + list_steps

        ### Save data to folder ###

        np.save(fullpath_array_errors_output, errors)
        np.save(fullpath_field, svf_as_array)
        np.save(fullpath_transformation_parameters, parameters)

        with open(fullpath_transformation_parameters, 'wb') as f:
            pickle.dump(parameters, f)

        with open(fullpath_numerical_method_table, 'wb') as f:
            pickle.dump(methods, f)

        print
        print '------------------------------------------'
        print 'Data computed and saved in external files!'
        print '------------------------------------------'

    else:

        ### Load data to folder ###

        errors       = np.load(fullpath_array_errors_output)
        svf_as_array = np.load(fullpath_field)

        with open(fullpath_transformation_parameters, 'rb') as f:
            parameters = pickle.load(f)

        with open(fullpath_numerical_method_table, 'rb') as f:
            methods = pickle.load(f)

        print
        print '--------------------------------'
        print 'Data loaded from external files!'
        print '--------------------------------'

        index_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
        num_method_considered    = len(index_methods_considered)

        names_method_considered       = [methods[j][0] for j in index_methods_considered]
        color_methods_considered      = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        marker_method_considered      = [methods[j][5] for j in index_methods_considered]

        list_steps = parameters[4:]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print '-----------------------------------------------------'
        print 'Inverse consistency error for one Real generated SVF'
        print '-----------------------------------------------------'

        print '\nParameters of the real SVF'
        print 'Svf dimension: ' + str(parameters[:3])
        print 'id subject: ' + str(parameters[3])
        print 'List of steps considered:'
        print str(parameters[4:])

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

    if verbose:
        print '---------------------------------'
        print 'Inverse consistency results table'
        print '---------------------------------'

        results_by_column = [[names_method_considered[j]] + list(errors[j, :]) for j in range(num_method_considered)]
        print tabulate(results_by_column, headers=[''] + list(list_steps))

    if plot_results:
        plot_custom_step_versus_error_single(list_steps, errors, names_method_considered,
                                             input_parameters=parameters, fig_tag=2, log_scale=True,
                                             additional_vertical_line=None,
                                             additional_field=svf_as_array,
                                             kind='one_REAL',
                                             input_marker=marker_method_considered,
                                             input_colors=color_methods_considered,
                                             input_line_style=line_style_methods_considered
                                             )

        plt.show()
