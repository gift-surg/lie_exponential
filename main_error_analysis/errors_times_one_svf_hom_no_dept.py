import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import time
import os
import copy
from sympy.core.cache import clear_cache
import pickle


methods_t_s = [['ss',        True,    7,    'b',      '-',     '+'],
               ['gss_aei',   True,    7,   'b',     '--',     '+'],
               ['gss_ei',    True,    7,    'r',      '-',     '.'],
               ['gss_rk4',   True,    7,    'r',      '--',     'x'],
               ['series',    False,   10,   'b',      '-',     '*'],
               ['midpoint',  True,   10,   'c',      '-',      '.'],
               ['euler',     True,    40,    'g',     '-',      '>'],
               ['euler_mod', True,    40,    'm',     '-',      '>'],
               ['euler_aei', True,    40,    'm',     '--',      '>'],
               ['heun',      True,    10,   'k',     '-',      '.'],
               ['heun_mod',  True,    10,   'k',     '--',     '.'],
               ['rk4',       True,    10,   'y',      '--',    'x'],
               ['vode',      False,    7,   'r',    '--',       '.'],
               ['lsoda',     False,    7,  'g',    '--',      '.']]


if __name__ == "__main__":

    clear_cache()


    random_seed = 0

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

    methods = methods_t_s

    indexes_methods_considered = [j for j in range(len(methods)) if methods[j][1] is True]
    num_method_considered = len(indexes_methods_considered)

    names_method_considered = [methods[j][0] for j in indexes_methods_considered]
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
    svf_0 = generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a)
    disp_0 = generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g)

    # Store the vector field (for the image)
    svf_as_array = copy.deepcopy(svf_0.field)

    for m in range(num_method_considered):
        if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
            start = time.time()
            disp_computed = exponential_scipy(integrator=names_method_considered[m],
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

    print 'Error-bar and time for one hom generated SVF'
    print '---------------------------------------------'

    print '\nParameters of the transformation hom:'
    print 'domain = ' + str(parameters[:3])
    print 'center = ' + str(parameters[3])
    print 'kind = ' + str(parameters[4])
    print 'scale factor = ' + str(parameters[5])
    print 'sigma = ' + str(parameters[6])

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