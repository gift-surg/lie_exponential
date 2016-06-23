import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from tabulate import tabulate
from sympy.core.cache import clear_cache
import pickle

from transformations.s_vf import SVF
from utils.path_manager import path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables
from visualizer.graphs_and_stats_new import plot_custom_step_error
from aaa_general_controller import methods_t_s


"""
Study for the estimate of step error for the Scaling and squaring-based and Taylor-based numerical methods.
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

    prefix_fn = 'exp_comparing_step_relative_errors_'
    kind   = 'GAUSS'
    number = 'single'
    file_suffix  = '_' + str(1)

    filename_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    filename_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    filename_csv_table_comp_time_output = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_cp_time'

    filename_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    filename_array_comp_time_output     = str(prefix_fn) + '_' + str(number) + str(kind) + '_array_cp_time'

    filename_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    filename_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'

    filename_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    path_to_results_folder = os.path.join(path_to_results_folder, 'errors_times_results')

    fullpath_array_errors_output = os.path.join(path_to_results_folder,
                                                    filename_array_errors_output + file_suffix + '.npy')
    fullpath_array_comp_time_output = os.path.join(path_to_results_folder,
                                                       filename_array_comp_time_output + file_suffix + '.npy')
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
        pp = 2

        x_1, y_1, z_1 = 20, 20, 1

        if z_1 == 1:
            domain = (x_1, y_1)
            shape = list(domain) + [1, 1, 2]
        else:
            domain = (x_1, y_1, z_1)
            shape = list(domain) + [1, 3]

        sigma_init = 4
        sigma_gaussian_filter = 2

        # maximal number of consecutive steps where to compute the step-relative error
        max_steps = 15

        parameters = [x_1, y_1, z_1] + [sigma_init, sigma_gaussian_filter, max_steps]

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
        # At step_errors[met, step] = norm(exp(svf,step) - exp(svf,step-1)

        # Generate svf
        svf_0   = SVF.generate_random_smooth(shape=shape,
                                             sigma=sigma_init,
                                             sigma_gaussian_filter=sigma_gaussian_filter)

        # Store the vector field (for the image)
        svf_as_array = copy.deepcopy(svf_0.field)

        # init storage of step = step -1 and step = step
        sdisp_step_j_0 = []
        sdisp_step_j_1 = []

        # init the first round step outside the main cycles:
        for met in range(num_method_considered):
            # Compute the displacement with the selected method:
            if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':
                disp_computed = svf_0.exponential_scipy(integrator=names_method_considered[met],
                                                        max_steps=2)

            else:
                disp_computed = svf_0.exponential(algorithm=names_method_considered[met],
                                                  s_i_o=s_i_o,
                                                  input_num_steps=2)
            sdisp_step_j_0 += [disp_computed]

        # start the main cycle with sdisp_step_j_0 initialized:
        for stp in range(3, max_steps):
            for met in range(num_method_considered):

                # Compute the displacement with the selected method:
                if names_method_considered[met] == 'vode' or names_method_considered[met] == 'lsoda':
                    disp_computed = svf_0.exponential_scipy(integrator=names_method_considered[met],
                                                            max_steps=stp)

                else:
                    disp_computed = svf_0.exponential(algorithm=names_method_considered[met],
                                                      s_i_o=s_i_o,
                                                      input_num_steps=stp)

                # Store the new displacement in the list_1:
                sdisp_step_j_1 += [disp_computed]

                # Compute the step error and store in the vector step_error
                step_errors[met, stp-1] = (sdisp_step_j_1[met] -   sdisp_step_j_0[met]).norm(passe_partout_size=pp)

            # copy sdisp_step_j_1 in sdisp_step_j_0 for the next step
            sdisp_step_j_0 = copy.deepcopy(sdisp_step_j_1)
            # erase for safety the list_1
            sdisp_step_j_1 = []

            if verbose:
                # show result at each step.
                results_by_column = [[met, err] for met, err in zip(names_method_considered,
                                                                    list(step_errors[:, stp-1]))]

                print 'Step-error for each method computed at ' + str(stp) + 'th. step out of ' + str(parameters[1])
                print '---------------------------------------------'
                print tabulate(results_by_column,
                               headers=['method', 'error'])
                print '---------------------------------------------'

        np.save(fullpath_array_errors_output, step_errors)
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

        step_errors = np.load(fullpath_array_errors_output)
        svf_as_array = np.load(fullpath_field)

        with open(fullpath_transformation_parameters, 'rb') as f:
            parameters = pickle.load(f)

        with open(fullpath_numerical_method_table, 'rb') as f:
            methods = pickle.load(f)

        print
        print '-----------'
        print 'Data loaded'
        print '-----------'

        index_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
        num_method_considered    = len(index_methods_considered)

        names_method_considered       = [methods[j][0] for j in index_methods_considered]
        color_methods_considered      = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        marker_method_considered      = [methods[j][5] for j in index_methods_considered]

        max_steps = parameters[5]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        #parameters = [x_1, y_1, z_1] + [sigma_init, sigma_gaussian_filter, max_steps]

        print 'Step-wise error for one Gauss generated SVF'
        print 'step-error is computed as norm(exp(svf,step+1) - exp(svf,step)) '
        print 'Up to step : ' + str(max_steps)

        print '---------------------------------------------'

        print '\nParameters of the transformation from real data:'
        print 'domain = ' + str(parameters[:3])
        print 'sigma_init, sigma_gaussian_filter = ' + str(parameters[4])

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

        plot_custom_step_error(range(1, max_steps-1),
                               step_errors[:, 1:max_steps-1],
                               names_method_considered,
                               input_parameters=parameters,
                               fig_tag=2,
                               kind='one_GAUSS',
                               log_scale=True,
                               input_colors=color_methods_considered,
                               window_title_input='step errors',
                               titles=('iterations vs.step error', 'Field'),
                               additional_field=svf_as_array,
                               legend_location='upper right',
                               input_line_style=line_style_methods_considered,
                               input_marker=marker_method_considered)

        plt.show()
