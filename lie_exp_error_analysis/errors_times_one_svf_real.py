import copy
import os
import pickle
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from controller import methods_t_s, path_to_results_folder, path_to_exp_notes_figures, \
    path_to_exp_notes_tables, displacements_aei_fp
from visualizer.graphs_and_stats_new import plot_custom_bar_chart_with_error

"""
Module for the computation of the error of the exponential map.
Svf involved is one 2d SVF generated from ADNI database.
For this kind of svf there is no ground truth available. One of the numerical integrator can be chosen
as ground truth.
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

    prefix_fn = 'exp_comparing_errors'
    kind   = 'REAL'
    number = 'single'
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

        s_i_o = 3
        pp = 2

        # Numerical method whose result corresponds to the ground truth:
        ground_method = 'rk4'  # in the following table should be false.
        ground_method_steps = 10

        ### Select SVF from ADNII data: ###

        # svf data -> id of the loaded element:
        id_element = 3

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

        # init results data:
        errors = np.zeros(num_method_considered)
        res_time = np.zeros(num_method_considered)

        # path flows:
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

        svf_as_array = copy.deepcopy(svf_0.field)

        parameters = list(svf_as_array.shape[:3]) + [id_element, ground_method, ground_method_steps]

        # compute the exponential with the selected ground truth method:
        if ground_method == 'vode' or ground_method == 'lsoda':
            disp_chosen_ground = svf_0.exponential_scipy(integrator=ground_method,
                                                         max_steps=ground_method_steps)

        else:
            disp_chosen_ground = svf_0.exponential(algorithm=ground_method,
                                                   s_i_o=s_i_o,
                                                   input_num_steps=ground_method_steps)

        for m in range(num_method_considered):
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
                res_time[m] = (time.time() - start)

            # compute error:
            errors[m] = (disp_computed - disp_chosen_ground).norm(passe_partout_size=pp, normalized=True)

            if verbose:
                print '--------------------------------------------------------------------------'
                print 'Computation for the method ' + str(names_method_considered[m]) + ' done.'

        ### Save data to folder ###
        np.save(fullpath_array_errors_output,       errors)
        np.save(fullpath_array_comp_time_output,    res_time)
        np.save(fullpath_field,                     svf_as_array)

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
        res_time     = np.load(fullpath_array_comp_time_output)
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
        steps_methods_considered      = [methods[j][2] for j in index_methods_considered]
        color_methods_considered      = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        marker_method_considered      = [methods[j][5] for j in index_methods_considered]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print 'Error-bar and time for one REAL SVF'
        print '---------------------------------------------'

        print '\nParameters of the transformation se2:'
        print 'domain = ' + str(parameters[:3])
        print 'id element = ' + str(parameters[3])
        print 'dummy ground truth and steps = ' + str(parameters[4:6])

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
                                         titles=('Error exponential map for one ADNI-generated svf', 'field'),
                                         kind='one_REAL',
                                         window_title_input='bar_plot_one_ADNI',
                                         additional_field=svf_as_array,
                                         log_scale=True,
                                         input_parameters=parameters,
                                         add_extra_numbers=res_time)
        plt.show()

    ### Save figures and table in external folder ###

    if save_external:

        # Save the table in latex format!
        f = open(fullpath_csv_table_errors_output, 'w')
        f.write(tabulate(results_by_column,
                         headers=['method', 'error', 'comp. time (sec)'], tablefmt="latex"))
        f.close()

        # Save image:
        plt.savefig(fullpath_figure_output, format='pdf', dpi=400)

        print 'Figure ' + filename_figure_output + ' saved in the external folder ' + str(fullpath_figure_output)
