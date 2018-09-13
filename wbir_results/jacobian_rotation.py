import os
import numpy as np
from sympy.core.cache import clear_cache
from tabulate import tabulate

from path_manager import pfo_results, pfo_notes_figures, pfo_notes_tables
from controller import methods_t_s

"""
Mean Jacobian determinant of SVFs exponentiated with several methods.
"""


if __name__ == "__main__":

    clear_cache()

    ##################
    ### Controller ###
    ##################

    compute = True
    verbose = True
    save_external = False
    plot_results = False

    # The results, and additional information are loaded in see_error_time_results
    # with a simplified name. They are kept safe from other subsequent tests with the same code.
    save_for_sharing = False

    #######################
    ### Path management ###
    #######################

    prefix_fn = 'zzz_exp_jacobian_det_'
    kind   = 'SE2'
    number = 'multiple'
    file_suffix  = '_' + str(1)

    filename_figure_output              = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_figure'
    filename_csv_table_errors_output    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_csv_errors'
    filename_array_jacobian_output        = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_array_errors'
    filename_field                      = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_field'
    filename_transformation_parameters  = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_parameters'
    filename_numerical_methods_table    = str(prefix_fn) + '_' + str(number) + '_svf_' + str(kind) + '_methods'

    # paths to results in internal to the project
    path_to_results_folder = os.path.join(pfo_results, 'errors_times_results')

    fullpath_array_jacobian_output = os.path.join(path_to_results_folder,
                                                  filename_array_jacobian_output + file_suffix + '.npy')
    fullpath_field = os.path.join(path_to_results_folder,
                                  filename_field + file_suffix + '.npy')
    fullpath_transformation_parameters = os.path.join(path_to_results_folder,
                                                      filename_transformation_parameters + file_suffix + '.npy')
    fullpath_numerical_method_table = os.path.join(path_to_results_folder,
                                                   filename_numerical_methods_table + file_suffix)

    # path to results external to the project:
    fullpath_figure_output  = os.path.join(pfo_notes_figures,
                                           filename_figure_output + file_suffix + '.pdf')
    fullpath_csv_table_errors_output = os.path.join(pfo_notes_tables,
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

        ### Insert parameters SVF from ADNII data: ###

        # svf data - id of the loaded element:
        id_elements = [1]  # , 2, 3, 4, 5, 6, 7, 8, 9, 10]

        N = 10

        x_1, y_1, z_1 = 50, 50, 50

        if z_1 == 1:
            domain = (x_1, y_1)
            shape = list(domain) + [1, 1, 2]
        else:
            domain = (x_1, y_1, z_1)
            shape = list(domain) + [1, 3]

        interval_theta = (- np.pi / 8, np.pi / 8)
        epsilon = np.pi / 12
        omega = (20, 40, 20, 40)  # where to locate the center of the random rotation on the axial plane

        list_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30]
        num_steps = len(list_steps)

        parameters = [x_1, y_1, z_1,           # Domain
                      N,                       # number of samples
                      interval_theta[0], interval_theta[1],  # interval of the rotations
                      omega[0], omega[1],      # interval of the center of the rotation x
                      omega[2], omega[3]] + list_steps  # interval of the center of the rotation y

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

        jacobians = np.zeros([num_method_considered, num_steps, N])

        for subject_num, subject_id in enumerate(id_elements):

            # generate matrices
            m_0 = se2_g.randomgen_custom_center(interval_theta=interval_theta,
                                                omega=omega,
                                                epsilon_zero_avoidance=epsilon)
            dm_0 = se2_g.log(m_0)

            svf_1   = SVF.generate_from_matrix(domain[:2], dm_0.get_matrix, affine=np.eye(4))

            for step_index, step_num in enumerate(list_steps):
                for m in range(num_method_considered):

                    if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                        disp_1 = svf_1.exponential(algorithm=names_method_considered[m],
                                                   s_i_o=s_i_o,
                                                   input_num_steps=list_steps[step_index])
                    else:
                        disp_1 = svf_1.exponential(algorithm=names_method_considered[m],
                                                   s_i_o=s_i_o,
                                                   input_num_steps=list_steps[step_index])

                    jacobians[m, step_index, subject_num] =\
                        np.mean(Field.compute_jacobian_determinant(disp_1, is_displacement=True).field)

                    # Tabulate partial results:
                    if verbose:

                        results_errors_by_slice = [[names_method_considered[j]] + list(jacobians[j, :, subject_num])
                                                   for j in range(num_method_considered)]

                        print 'Subject ' + str(subject_num + 1) + '/' + str(N) + '.'
                        print 'Jacobians: '
                        print '---------------------------------------------'
                        print tabulate(results_errors_by_slice,
                                       headers=['method'] + [str(j) for j in list_steps])
                        print '---------------------------------------------'
