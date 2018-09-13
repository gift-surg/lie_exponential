import copy
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from controller import methods_t_s, pfo_results, pfo_notes_figures, pfo_notes_sharing
from visualizer.graphs_and_stats_new import plot_custom_bar_chart_with_error

"""
Module for the computation of the error of the exponential map.
Svf involved is one 2d SVF generated with matrix in se2_a.
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
    kind   = 'HOM'
    number = 'single'
    file_suffix  = '_' + str(2)  # 1 skew, 2 diag

    filename_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    filename_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    filename_csv_table_comp_time_output = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_cp_time'
    filename_array_errors_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    filename_array_comp_time_output     = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_cp_time'
    filename_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'
    filename_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    filename_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    path_to_results_folder = os.path.join(pfo_results, 'errors_times_results')

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
    fullpath_figure_output  = os.path.join(pfo_notes_figures,
                                           filename_figure_output + file_suffix + '.pdf')
    fullpath_csv_table_errors_output = os.path.join(pfo_notes_sharing,
                                                    filename_csv_table_errors_output + '.csv')
    fullpath_csv_table_comp_time_output = os.path.join(pfo_notes_sharing,
                                                       filename_csv_table_comp_time_output + '.csv')

    ####################
    ### Computations ###
    ####################

    if compute:  # or compute or load

        random_seed = 10

        if random_seed > 0:
            np.random.seed(random_seed)

        s_i_o = 3
        pp = 2

        # Parameters SVF:
        x_1, y_1, z_1 = 30, 30, 1

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
        errors = np.zeros(num_method_considered)
        res_time = np.zeros(num_method_considered)

        # generate matrices homography
        scale_factor = 1. / (np.max(domain) * 10)
        hom_attributes = [d, scale_factor, 1, in_psl]

        h_a, h_g = get_random_hom_a_matrices(d=hom_attributes[0],
                                             scale_factor=hom_attributes[1],
                                             sigma=hom_attributes[2],
                                             special=hom_attributes[3])

        '''
        h_a = np.array([[  9.85981526e-01,   1.78819744e-02,  -3.86350073e-02],
                        [ -2.09596248e-04,   1.00200981e+00,  -1.80021390e-02],
                        [  6.63778964e-03,   2.71371314e-03,   9.06484972e-01]])

        h_g = expm(h_a)
        '''

        # generate SVF
        svf_0 = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)
        disp_0 = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)

        # Store the vector field (for the image)
        svf_as_array = copy.deepcopy(svf_0.field)

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
            errors[m] = (disp_computed - disp_0).norm(passe_partout_size=pp, normalized=True)

        parameters = [x_1, y_1, z_1] + hom_attributes

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

        print 'Error-bar and time for one hom generated SVF'
        print '---------------------------------------------'

        print '\nParameters of the transformation hom:'
        print 'domain = ' + str(parameters[:3])
        print 'center = ' + str(parameters[3])
        print 'kind = ' + str(parameters[4])
        print 'scale factor = ' + str(parameters[5])
        print 'sigma = ' + str(parameters[6])
        print 'in psl = ' + str(parameters[7])

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
    print '\n END'

    if plot_results:

        plot_custom_bar_chart_with_error(input_data=errors,
                                         input_names=names_method_considered,
                                         fig_tag=11,
                                         titles=('Error exponential map for one HOM-generated svf', 'field'),
                                         kind='one_HOM',
                                         window_title_input='bar_plot_one_se2',
                                         additional_field=svf_as_array,
                                         log_scale=False,
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
