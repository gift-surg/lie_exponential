import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import os
from sympy.core.cache import clear_cache
import nibabel as nib
import pickle

from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from utils.path_manager import path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables, \
    displacements_folder_path_AD, path_to_sharing_folder

from visualizer.graphs_and_stats_new import plot_custom_time_error_steps

from main_error_analysis.aaa_general_controller import methods_t_s

"""
Module for the computation of the scalar associativity of the exponential map.
Svf involved are more than one 2d SVF generated from ADNI database.
For this kind of svf there is no ground truth available. One of the numerical integrator can be chosen
as ground truth.
"""


if __name__ == "__main__":

    clear_cache()

    ##################
    ### Controller ###
    ##################

    compute = False
    verbose = True
    save_external = False
    plot_results = True

    # The results, and additional information are loaded in see_error_time_results
    # with a simplified name. They are kept safe from other subsequent tests with the same code.
    save_for_sharing = False

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_scalar_associativity'
    kind   = 'REAL'
    number = 'multiple'
    file_suffix  = '_' + str('new')

    filename_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    filename_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    filename_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    filename_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'
    filename_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    filename_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    path_to_results_folder = os.path.join(path_to_results_folder, 'errors_times_results')

    fullpath_array_errors_output = os.path.join(path_to_results_folder,
                                                filename_array_errors_output + file_suffix + '.npy')
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

    ####################
    ### Computations ###
    ####################

    if compute:  # or compute or load

        random_seed = 0

        if random_seed > 0:
            np.random.seed(random_seed)

        s_i_o = 3
        pp = 2

        ### Insert parameters SVF from REAL data: ###

        # svf data - id of the loaded element:
        id_elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        num_subjects = len(id_elements)
        corresponding_str = '_'
        for j in id_elements:
            corresponding_str += str(j) + '_'

        list_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        num_steps = len(list_steps)

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

        errors = np.zeros([num_method_considered, num_steps, num_subjects])

        for subject_num, subject_id in enumerate(id_elements):

            if verbose:
                print '--------------------------------------------------------------------------'
                print 'subject ' + str(subject_id) + ' (' + str(subject_num + 1) + '/' + str(num_subjects) + ').'

            # path flows:
            disp_name_A_C = 'disp_' + str(subject_id) + '_A_C.nii.gz'
            # Load as nib:
            nib_A_C = nib.load(os.path.join(displacements_folder_path_AD, disp_name_A_C))

            # reduce from 3d to 2d:
            data_A_C = nib_A_C.get_data()
            header_A_C = nib_A_C.header
            affine_A_C = nib_A_C.affine

            array_2d_A_C = data_A_C[pp:-pp, pp:-pp, 100:101, :, 0:2]

            # Create svf over the array:
            a = 0.35
            b = 0.35
            c = 0.3

            svf_1 = SVF.from_array_with_header(array_2d_A_C, header=header_A_C, affine=affine_A_C)
            svf_a = a * svf_1
            svf_b = b * svf_1
            svf_c = c * svf_1

            for step_index, step_num in enumerate(list_steps):
                for m in range(num_method_considered):

                    if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                        disp_1 = svf_1.exponential(algorithm=names_method_considered[m],
                                                   s_i_o=s_i_o,
                                                   input_num_steps=list_steps[step_index])
                        disp_a = svf_a.exponential(algorithm=names_method_considered[m],
                                                         s_i_o=s_i_o,
                                                         input_num_steps=list_steps[step_index])
                        disp_b = svf_b.exponential(algorithm=names_method_considered[m],
                                                         s_i_o=s_i_o,
                                                         input_num_steps=list_steps[step_index])
                        disp_c = svf_c.exponential(algorithm=names_method_considered[m],
                                                         s_i_o=s_i_o,
                                                         input_num_steps=list_steps[step_index])

                    else:
                        disp_1 = svf_1.exponential(algorithm=names_method_considered[m],
                                                   s_i_o=s_i_o,
                                                   input_num_steps=list_steps[step_index])
                        disp_a = svf_a.exponential(algorithm=names_method_considered[m],
                                                   s_i_o=s_i_o,
                                                   input_num_steps=list_steps[step_index])
                        disp_b = svf_b.exponential(algorithm=names_method_considered[m],
                                                   s_i_o=s_i_o,
                                                   input_num_steps=list_steps[step_index])
                        disp_c = svf_c.exponential(algorithm=names_method_considered[m],
                                                   s_i_o=s_i_o,
                                                   input_num_steps=list_steps[step_index])

                    disp_abc = SDISP.composition(disp_a, SDISP.composition(disp_b, disp_c, s_i_o=s_i_o), s_i_o=s_i_o)

                    errors[m, step_index, subject_num] = (disp_abc - disp_1).norm(passe_partout_size=pp,
                                                                                  normalized=True)

                    # Tabulate partial results:
                    if verbose:

                        results_errors_by_slice = [[names_method_considered[j]] + list(errors[j, :, subject_num])
                                                   for j in range(num_method_considered)]

                        print 'Subject ' + str(subject_num+1) + '/' + str(num_subjects) + '.'
                        print 'Errors: '
                        print '---------------------------------------------'
                        print tabulate(results_errors_by_slice,
                                       headers=['method'] + [str(j) for j in list_steps])
                        print '---------------------------------------------'

        parameters = list(array_2d_A_C.shape[:3]) + [corresponding_str] + list_steps

        ### Save data to folder ###
        np.save(fullpath_array_errors_output, errors)

        with open(fullpath_transformation_parameters, 'wb') as f:
            pickle.dump(parameters, f)

        with open(fullpath_numerical_method_table, 'wb') as f:
            pickle.dump(methods, f)

        print
        print '------------------------------------------'
        print 'Data computed and saved in external files!'
        print '------------------------------------------'

        if save_for_sharing:

            np.save(os.path.join(path_to_sharing_folder, 'exp_scalar_associativity_errors_real'+file_suffix), errors)

            with open(os.path.join(path_to_sharing_folder, 'exp_scalar_associativity_parameters_real'+file_suffix), 'wb') as f:
                pickle.dump(parameters, f)

            with open(os.path.join(path_to_sharing_folder, 'exp_scalar_associativity_methods_table_real'+file_suffix), 'wb') as f:
                pickle.dump(methods, f)

    else:
        errors = np.load(fullpath_array_errors_output)

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

        list_steps = parameters[4:]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print 'Scalar associativity for multiple REAL SVF'
        print '---------------------------------------------'

        print '\nParameters of the transformation REAL:'
        print 'domain = ' + str(parameters[:3])
        print 'id elements = ' + str(parameters[3])
        print 'steps = ' + str(parameters[4:])

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

    mean_errors   = np.mean(errors, axis=2)

    percentile_25 = np.percentile(errors, 25,  axis=2)
    percentile_75 = np.percentile(errors, 75,  axis=2)

    # Tabulate Errors
    mean_errors_by_column = [[names_method_considered[j]] + list(mean_errors[j, :])
                              for j in range(num_method_considered)]

    print type(list_steps)
    print list_steps
    print '\n'
    print 'Results Errors per steps of the numerical integrators:'
    print tabulate(mean_errors_by_column, headers=[0] + list_steps)  # TODO correct this
    print '\n END'

    # plot results
    if plot_results:

        steps_for_all = np.array(list_steps * num_method_considered).reshape(mean_errors.shape)

        plot_custom_time_error_steps(steps_for_all,
                                     mean_errors,
                                     fig_tag=10,
                                     y_error=[percentile_25, percentile_75],
                                     label_lines=names_method_considered,
                                     additional_field=None,
                                     kind='multiple_HOM',
                                     titles=('Scalar associativity, percentile', 'Field sample'),
                                     x_log_scale=False,
                                     y_log_scale=True,
                                     input_parameters=parameters,
                                     input_marker=markers_methods_considered,
                                     input_colors=colour_methods_considered,
                                     input_line_style=linestyle_methods_considered,
                                     legend_location='upper right')




        plt.show()

    ### Save figures and table in external folder ###

