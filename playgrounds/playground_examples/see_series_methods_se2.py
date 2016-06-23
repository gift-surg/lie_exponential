import numpy as np
import matplotlib.pyplot as plt
import copy
from tabulate import tabulate
from sympy.core.cache import clear_cache
import time

from transformations.s_vf import SVF
from transformations.s_disp import SDISP
from transformations import se2_g

from visualizer.graphs_and_stats_new import plot_custom_time_error_steps


if __name__ == "__main__":

    clear_cache()

    ## Parameters
    pp = 2
    s_i_o = 3

    # Parameters SVF:
    denominators = [60, 50, 20, 10, 5, 4]  # pi/denominator to increase the size of the deformation

    x_1, y_1, z_1 = 10, 10, 10

    if z_1 == 1:
        domain = (x_1, y_1)
        shape = list(domain) + [1, 1, 2]
    else:
        domain = (x_1, y_1, z_1)
        shape = list(domain) + [1, 3]

    x_c = int(x_1/2)
    y_c = int(y_1/2)

    # exponential map methods and parameters:
    methods     = ['ss', 'gss_aei', 'euler', 'gss_ei', 'rk4', 'series']
    #methods_name = ['a', 'b', 'c', 'd', 'e', 'f']
    colors      = ['b', 'b', 'g', 'k', 'r', 'c']
    markers     = ['x', '+', '>', '.', 'x', '.']
    lines_style = ['-', '--', '-', '-', '-', '-']

    steps_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]

    # init empty matrices:
    errors = np.zeros([len(methods), len(steps_list), len(denominators)])
    comp_time = np.zeros([len(methods), len(steps_list), len(denominators)])

    max_norms = np.zeros([len(denominators)])  # max norm each SVF generated per angle
    average_norms = np.zeros([len(denominators)])

    ### Computations ###
    for num_den, den in enumerate(denominators):

        # generate transformation matrix
        theta = np.pi/den

        tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
        ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

        m_0 = se2_g.se2_g(theta, tx, ty)
        dm_0 = se2_g.log(m_0)

        # generate SVF and ground truth
        svf_0  = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))
        disp_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))

        # Store the field in external folder
        svf_as_array = copy.deepcopy(svf_0.field)

        # compute max of the norm of the SVF generated (norm of the first axial slice is enough even for 3d svf)
        norms_list = [np.linalg.norm(svf_as_array[x, y, 0, 0, :]) for x in range(domain[0])
                                                                                 for y in range(domain[1])]
        max_norms[num_den] = np.max(norms_list)
        average_norms[num_den] = np.mean(norms_list)

        if 0:  # sanity test to see if the grow angle norm is linear or not
            print ''
            print np.pi/den
            print max_norms

        for num_step, step in enumerate(steps_list):

            for num_method, method in enumerate(methods):

                start = time.time()
                # compute the exponential with the appropriate number of steps and method.
                disp_computed = svf_0.exponential(algorithm=method,
                                                          s_i_o=s_i_o,
                                                          input_num_steps=step)
                # compute time
                comp_time[num_method, num_step, num_den] = (time.time() - start)

                # compute error:
                errors[num_method, num_step, num_den] = (disp_computed - disp_0).norm(passe_partout_size=pp,
                                                                                      normalized=True)

        # Print table for each denominator
        print '------------------'
        print '------------------'
        print 'Angle ' + str(theta) + ' = pi/' + str(den), '. tx, ty = ' + str(tx) + ', ' + str(ty)
        print 'Max norm of the SVF : ' + str(max_norms[num_den])
        print '(' + str(num_den+1) + '/' +  str(len(denominators)) + ')'
        print '------------------'

        results_by_column_error = [[methods[j]] + list(errors[j, :, num_den])
                                   for j in range(len(methods))]

        results_by_column_time  = [[methods[j]] + list(comp_time[j, :, num_den])
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
        parameters += [den, tx, ty] + steps_list

        plot_custom_time_error_steps(comp_time[:, :, num_den],
                                       errors[:, :, num_den],
                                       fig_tag=num_den,
                                       label_lines=methods,
                                       additional_field=svf_as_array,
                                       kind='one_SE2',
                                       x_log_scale=True,
                                       y_log_scale=True,
                                       input_parameters=parameters,
                                       input_marker=markers,
                                       input_colors=colors,
                                       input_line_style=lines_style,
                                       legend_location='upper right',
                                       additional_data=[max_norms[num_den], average_norms[num_den]],
                                       window_title_input='errors_den_' + str(den))

    print 'max norms per angle'
    print max_norms

    plt.show()
