import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import time
import os
from sympy.core.cache import clear_cache
import nibabel as nib
import pickle

from transformations.s_vf import SVF

from utils.path_manager import path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables
from utils.path_manager import displacements_aei_fp

from visualizer.graphs_and_stats_new import plot_custom_boxplot, plot_custom_cluster
from visualizer.fields_at_the_window import see_field

from aaa_general_controller import methods_t_s

"""
Module for the computation of the error of the exponential map.
Svf involved are more than one 2d SVF generated from ADNI database.
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
    see_image_each_step = False  # slow down the computations... Only for visual assessment!

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_errors'
    kind   = 'REAL'
    number = 'multiple'
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

        ### Insert parameters SVF from ADNII data: ###

        # svf data - id of the loaded element:
        id_elements = [1, 2, 3, 4, 5, 6, 9]

        num_subjects = len(id_elements)
        corresponding_str = '_'
        for j in id_elements:
            corresponding_str += str(j) + '_'

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

        errors = np.zeros([num_method_considered, num_subjects])
        res_time = np.zeros([num_method_considered, num_subjects])

        for subject_num, subject_id in enumerate(id_elements):
            if verbose:
                print '--------------------------------------------------------------------------'
                print 'subject ' + str(subject_id) + ' (' + str(subject_num + 1) + '/' + str(num_subjects) + ').'
            # path flows:
            disp_name_A_C = 'displacement_AD_' + str(subject_id) + '_.nii.gz'
            # Load as nib:
            nib_A_C = nib.load(os.path.join(displacements_aei_fp, disp_name_A_C))

            # reduce from 3d to 2d:
            data_A_C = nib_A_C.get_data()
            header_A_C = nib_A_C.header
            affine_A_C = nib_A_C.affine

            array_2d_A_C = data_A_C[pp:-pp, pp:-pp, 100:101, :, 0:2]

            # Create svf over the array:
            svf_0 = SVF.from_array_with_header(array_2d_A_C, header=header_A_C, affine=affine_A_C)

            if see_image_each_step:
                see_field(svf_0)
                plt.show()

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
                    res_time[m, subject_num] = (time.time() - start)

                else:
                    start = time.time()
                    disp_computed = svf_0.exponential(algorithm=names_method_considered[m],
                                                      s_i_o=s_i_o,
                                                      input_num_steps=steps_methods_considered[m])
                    res_time[m, subject_num] = (time.time() - start)

                # compute error:
                errors[m, subject_num] = (disp_computed - disp_chosen_ground).norm(passe_partout_size=pp,
                                                                                   normalized=True)

                if verbose:
                    print 'Computation for the method ' + str(names_method_considered[m]) + ' done.'

        parameters = list(array_2d_A_C.shape[:3]) + [corresponding_str, ground_method, ground_method_steps]

        ### Save data to folder ###
        np.save(fullpath_array_errors_output,       errors)
        np.save(fullpath_array_comp_time_output,    res_time)

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

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print 'Error-bar and time for multiple REAL SVF'
        print '---------------------------------------------'

        print '\nParameters of the transformation se2:'
        print 'domain = ' + str(parameters[:3])
        print 'id elements = ' + str(parameters[3])
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

    mean_errors   = np.mean(errors, axis=1)
    mean_res_time = np.mean(res_time, axis=1)

    results_by_column = [[met, err, tim]
                         for met, err, tim in zip(names_method_considered, list(mean_errors), list(mean_res_time))]

    print '\n'
    print 'Results and computational time:'
    print tabulate(results_by_column,
                   headers=['method', 'mean error', 'mean comp. time (sec)'])
    print '\n END'

    # plot results
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
                            input_titles=('Error exponential map for multiple ADNI svf', 'field'),
                            kind='multiple_REAL',
                            window_title_input='bar_plot_multiple_se2',
                            additional_field=None,
                            log_scale=False,
                            input_parameters=parameters,
                            annotate_mean=True,
                            add_extra_annotation=mean_res_time)

        plot_custom_cluster(reordered_errors_for_plot, reordered_times_for_plot,
                            fig_tag=42,
                            kind='multiple_REAL',
                            clusters_labels=names_method_considered,
                            clusters_markers=markers_methods_considered,
                            clusters_colors=colour_methods_considered)

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
