import numpy as np
import matplotlib.pyplot as plt
import copy
from tabulate import tabulate
from sympy.core.cache import clear_cache
import time

from transformations.s_vf import SVF

from visualizer.graphs_and_stats_new import plot_custom_time_error_steps


if __name__ == "__main__":

    clear_cache()

    ## Parameters
    pp = 2
    s_i_o = 3

    # Parameters SVF:
    sigma_init = 4
    sigma_gf_list = [1, 2, 3, 4, 5, 6, 7, 10]

    random_seed = 2

    if random_seed > 0:
        np.random.seed(random_seed)

    x_1, y_1, z_1 = 20, 20, 10

    if z_1 == 1:
        domain = (x_1, y_1)
        shape = list(domain) + [1, 1, 2]
    else:
        domain = (x_1, y_1, z_1)
        shape = list(domain) + [1, 3]

    # Numerical method whose result corresponds to the ground truth:
    ground_method = 'series'  # in the following table should be false.
    ground_method_steps = 10

    # exponential map methods and parameters:
    methods     = ['ss', 'gss_aei', 'euler', 'heun', 'midpoint']
    colors      = ['b', 'b', 'g', 'k', 'r', 'c']
    markers     = ['x', '+', '>', '.', 'x', '.']
    lines_style = ['-', '--', '-', '-', '-', '-']

    steps_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]

    # init empty matrices:
    errors = np.zeros([len(methods), len(steps_list), len(sigma_gf_list)])
    comp_time = np.zeros([len(methods), len(steps_list), len(sigma_gf_list)])
    max_norms = np.zeros([len(sigma_gf_list)])  # max norm each SVF generated per angle
    average_norms = np.zeros([len(sigma_gf_list)])

    ### Computations ###
    for num_sigma_gf, sigma_gf in enumerate(sigma_gf_list):

        # generate SVF and ground truth
        svf_0   = SVF.generate_random_smooth(shape=shape,
                                             sigma=sigma_init,
                                             sigma_gaussian_filter=sigma_gf)

        # compute the dummy ground truth
        disp_chosen_ground = svf_0.exponential(algorithm=ground_method,
                                                   s_i_o=s_i_o,
                                                   input_num_steps=ground_method_steps)

        # Store the field in external folder
        svf_as_array = copy.deepcopy(svf_0.field)

        # compute max of the norm of the SVF generated (norm of the first axial slice is enough even for 3d svf)
        norms_list = [np.linalg.norm(svf_as_array[x, y, 0, 0, :]) for x in range(domain[0])
                                                                                 for y in range(domain[1])]
        max_norms[num_sigma_gf] = np.max(norms_list)
        average_norms[num_sigma_gf] = np.mean(norms_list)

        if 0:  # sanity test to see if the grow angle norm is linear or not
            print ''
            print num_sigma_gf
            print max_norms

        for num_step, step in enumerate(steps_list):

            for num_method, method in enumerate(methods):

                start = time.time()
                # compute the exponential with the appropriate number of steps and method.
                disp_computed = svf_0.exponential(algorithm=method,
                                                          s_i_o=s_i_o,
                                                          input_num_steps=step)
                # compute time
                comp_time[num_method, num_step, num_sigma_gf] = (time.time() - start)

                # compute error:
                errors[num_method, num_step, num_sigma_gf] = \
                    (disp_computed - disp_chosen_ground).norm(passe_partout_size=pp,
                                                              normalized=True)

        # Print table for each denominator
        print '------------------'
        print '------------------'
        print 'Sigma GF ' + str(sigma_gf) + ' .'
        print 'Max norm of the SVF : ' + str(max_norms[num_sigma_gf])
        print '(' + str(num_sigma_gf+1) + '/' +  str(len(sigma_gf_list)) + ')'
        print '------------------'

        results_by_column_error = [[methods[j]] + list(errors[j, :, num_sigma_gf])
                                   for j in range(len(methods))]

        results_by_column_time  = [[methods[j]] + list(comp_time[j, :, num_sigma_gf])
                                   for j in range(len(methods))]

        print 'Results Errors per steps of the numerical integrators:'
        print tabulate(results_by_column_error,
                       headers=[''] + steps_list)

        print '\nResults Computational time per steps of the numerical integrators:'
        print tabulate(results_by_column_time,
                       headers=[''] + steps_list)
        print '\n'

        # Plot one figure per selected angle
        if z_1 == 1:
            parameters = list(domain) + [1]
        else:
            parameters = list(domain)
        parameters += [sigma_init, sigma_gf] + [ground_method, ground_method_steps] + steps_list

        plot_custom_time_error_steps(comp_time[:, :, num_sigma_gf],
                                       errors[:, :, num_sigma_gf],
                                       fig_tag=num_sigma_gf,
                                       label_lines=methods,
                                       additional_field=svf_as_array,
                                       kind='one_GAUSS',
                                       x_log_scale=True,
                                       y_log_scale=True,
                                       input_parameters=parameters,
                                       input_marker=markers,
                                       input_colors=colors,
                                       input_line_style=lines_style,
                                       legend_location='upper right',
                                       additional_data=[max_norms[num_sigma_gf], average_norms[num_sigma_gf]],
                                       window_title_input='errors_sigma_' + str(sigma_gf))

    plt.show()
