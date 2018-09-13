import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from controller import methods_t_s, path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables
from visualizer.graphs_and_stats_new import plot_custom_step_error

"""
Study for the estimate of step error for the Scaling and squaring based and Taylor numerical methods.
One se2 generated SVF.
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

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'exp_comparing_step_relative_errors_'
    kind   = 'HOM'
    number = 'multiple'
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

        random_seed = 0

        if random_seed > 0:
            np.random.seed(random_seed)

        pp = 2     # passepartout
        s_i_o = 3  # spline interpolation order

        # Parameters of the SVFs
        N = 20  # number of samples

        x_1, y_1, z_1 = 51, 51, 1

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

        scale_factor = 1. / (np.max(domain) * 10)
        hom_attributes = [d, scale_factor, 1, in_psl]

        max_steps = 20  # maximal number of consecutive steps where to compute the step-relative error

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
        step_errors  = np.zeros([num_method_considered, max_steps, N])  # Methods x Steps
        # At step_errors[met, step] we save norm(exp(svf,step) - exp(svf,step-1)

        for s in range(N):

            # Generate SVF and displacement
            h_a, h_g = get_random_hom_a_matrices(d=hom_attributes[0],
                                                 scale_factor=hom_attributes[1],
                                                 sigma=hom_attributes[2],
                                                 special=hom_attributes[3])

            if verbose:
                print 'h = '
                print str(h_a)

                print 'H = '
                print str(h_g)

            svf_0 = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)
            disp_0 = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)

            if d == 2:
                svf_as_array = copy.deepcopy(svf_0.field)
            elif d == 3:
                svf_as_array = copy.deepcopy(svf_0.field[:, :, z_c:(z_c + 1), :, :2])

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
                    step_errors[met, stp - 1] = \
                        (sdisp_step_j_1[met] - sdisp_step_j_0[met]).norm(passe_partout_size=pp)

                # copy sdisp_step_j_1 in sdisp_step_j_0 for the next step
                sdisp_step_j_0 = copy.deepcopy(sdisp_step_j_1)
                # erase for safety the list_1
                sdisp_step_j_1 = []

                if verbose:
                    # show result at each step.
                    results_by_column = [[met, err] for met, err in zip(names_method_considered,
                                                                        list(step_errors[:, stp - 1, s]))]

                    print 'Step-error for each method computed at ' + str(stp) + 'th. step. Sample ' + str(s) + '/N'
                    print '---------------------------------------------'
                    print tabulate(results_by_column,
                                   headers=['method', 'error'])
                    print '---------------------------------------------'

        parameters = [x_1, y_1, z_1] + hom_attributes + [N, max_steps]

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
        print '------------'
        print 'Data loaded!'
        print '------------'

        index_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
        num_method_considered    = len(index_methods_considered)

        names_method_considered       = [methods[j][0] for j in index_methods_considered]
        color_methods_considered      = [methods[j][3] for j in index_methods_considered]
        line_style_methods_considered = [methods[j][4] for j in index_methods_considered]
        marker_method_considered      = [methods[j][5] for j in index_methods_considered]

        N = parameters[8]
        max_steps = parameters[9]

    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:

        print 'Step-wise error for multiple HOM generated SVF'
        print 'step-error is computed as norm(exp(svf,step+1) - exp(svf,step)) '
        print 'domain = ' + str(parameters[:3])
        print 'center = ' + str(parameters[3])
        print 'kind = ' + str(parameters[4])
        print 'scale factor = ' + str(parameters[5])
        print 'sigma = ' + str(parameters[6])
        print 'in psl = ' + str(parameters[7])
        print 'number of samples = ' + str(parameters[8])
        print 'max steps = ' + str(parameters[9])

        print '---------------------------------------------'

        print '\nParameters of the transformations hom:'
        print 'domain = ' + str(parameters[:3])

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

    if plot_results:

        step_errors_mean = np.mean(step_errors, axis=2)
        stdev_errors = np.std(step_errors, axis=2)

        plot_custom_step_error(range(1, max_steps - 1),
                               step_errors_mean[:, 1:max_steps - 1],
                               names_method_considered,
                               stdev=stdev_errors[:, 1:max_steps - 1],
                               input_parameters=parameters,
                               fig_tag=2,
                               log_scale=True,
                               kind='multiple_HOM',
                               input_colors=color_methods_considered,
                               input_line_style=line_style_methods_considered,
                               input_marker=marker_method_considered,
                               window_title_input='step errors',
                               titles=('iterations vs.step error', 'Field'),
                               additional_field=svf_as_array,
                               legend_location='upper right')

        plt.show()
