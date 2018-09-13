"""
Module for the computation of 2d SVF generated with matrix of se2_a.
It compares the exponential computation with different strategies.
"""

import numpy as np

import time
import os

from transformations.s_vf import SVF
from transformations.s_disp import SDISP
from transformations.se2_a import se2_g

from z_utils.path_manager import path_to_results_folder

from visualizer.fields_comparisons import see_n_fields_special

import matplotlib.pyplot as plt

from visualizer.graphs_and_stats import custom_boxplot

from z_utils.path_manager import path_to_results_folder


### compute matrix of transformations:

# -----------------
# Control panel:
domain = (16, 16)  # Matrix coordinates: x = -Y, y = X
passe_partout_size = 3
omega = (5, 11, 5, 11)  # where to locate the center of the random rotation

interval_theta = (- np.pi / 4, np.pi / 4)
epsilon = 0.001
interval_tx    = [-8, ]
interval_ty    = [8, ]
spline_interpolation_order = 3

N = 20
verbose = True
all_plot = False
save_external = True
# -----------------

# File where to save the data:
filename_errors = 'err_0_pi4_bis_20'  # results_error_compared_0 vode, isoda, steps=7, results_error_compared_1 steps=6.
filename_times  = 'time_0_pi4_bis_20'  # results_error_compared_2 dopri5 and dop581
fullpath_filename_errors = os.path.join(path_to_results_folder, filename_errors)
fullpath_filename_times  = os.path.join(path_to_results_folder, filename_times)

# output data structure:
res   = np.zeros([N, 12])  # storing results
res_t = np.zeros([N, 10])  # storing time of the computation.

for i in range(N):

    print '--------------------'
    print '--------------------'
    print 'Step: ' + str(i + 1) + '/' + str(N)

    ### Generate random matrix of transformations:

    m_0 = se2_g.randomgen_custom_center(interval_theta=interval_theta,
                                        omega=omega,
                                        epsilon_zero_avoidance=epsilon)
    dm_0 = se2_g.log(m_0)

    ### generate subsequent vector fields

    svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))
    sdisp_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))

    ### compute exponential with different available methods, and compute the time:
    start = time.time()
    sdisp_ss = svf_0.exponential(algorithm='ss', s_i_o=spline_interpolation_order)
    res_t[i, 0] = (time.time() - start)

    start = time.time()
    sdisp_ss_pa = svf_0.exponential(algorithm='gss_aei', s_i_o=spline_interpolation_order)
    res_t[i, 1] = (time.time() - start)

    start = time.time()
    sdisp_euler = svf_0.exponential(algorithm='euler', s_i_o=spline_interpolation_order)
    res_t[i, 2] = (time.time() - start)

    start = time.time()
    sdisp_series = svf_0.exponential(algorithm='series', s_i_o=spline_interpolation_order)
    res_t[i, 3] = (time.time() - start)

    start = time.time()
    sdisp_mid_p = svf_0.exponential(algorithm='midpoint', s_i_o=spline_interpolation_order)
    res_t[i, 4] = (time.time() - start)

    start = time.time()
    sdisp_euler_m = svf_0.exponential(algorithm='euler_mod', s_i_o=spline_interpolation_order)
    res_t[i, 5] = (time.time() - start)

    start = time.time()
    sdisp_rk4 = svf_0.exponential(algorithm='heun', s_i_o=spline_interpolation_order)
    res_t[i, 6] = (time.time() - start)

    start = time.time()
    sdisp_heun = svf_0.exponential(algorithm='heun_mod', s_i_o=spline_interpolation_order)
    res_t[i, 7] = (time.time() - start)

    start = time.time()
    sdisp_heun_m = svf_0.exponential(algorithm='rk4', s_i_o=spline_interpolation_order)
    res_t[i, 8] = (time.time() - start)

    start = time.time()
    sdisp_ss_rk4 = svf_0.exponential(algorithm='gss_rk4', s_i_o=spline_interpolation_order)
    res_t[i, 9] = (time.time() - start)

    '''
    start = time.time()
    sdisp_dodpri5 = svf_0.exponential_scipy(integrator='dopri5',
                                            max_steps=7, passepartout=passe_partout_size)
    res_t[i, 9] = (time.time() - start)
    '''
    ### store result data:
    res[i, 0] = svf_0.norm(passe_partout_size=passe_partout_size)
    res[i, 1] = sdisp_0.norm(passe_partout_size=passe_partout_size)
    res[i, 2] = (sdisp_ss - sdisp_0).norm(passe_partout_size=passe_partout_size)        # |ss - disp|
    res[i, 3] = (sdisp_ss_pa - sdisp_0).norm(passe_partout_size=passe_partout_size)     # |ss_pa - disp|
    res[i, 4] = (sdisp_euler - sdisp_0).norm(passe_partout_size=passe_partout_size)     # |euler - disp|
    res[i, 5] = (sdisp_series - sdisp_0).norm(passe_partout_size=passe_partout_size)     # |ss_pa - disp|
    res[i, 6] = (sdisp_mid_p - sdisp_0).norm(passe_partout_size=passe_partout_size)     # |midpoint - disp|
    res[i, 7] = (sdisp_euler_m - sdisp_0).norm(passe_partout_size=passe_partout_size)   # |euler_mod - disp|
    res[i, 8] = (sdisp_heun - sdisp_0).norm(passe_partout_size=passe_partout_size)      # |heun - disp|
    res[i, 9] = (sdisp_heun_m - sdisp_0).norm(passe_partout_size=passe_partout_size)    # |heun_mod - disp|
    res[i, 10] = (sdisp_rk4 - sdisp_0).norm(passe_partout_size=passe_partout_size)       # |rk4 - disp|
    res[i, 11] = (sdisp_ss_rk4 - sdisp_0).norm(passe_partout_size=passe_partout_size)     # |sdisp_vode - disp|
    #res[i, 11] = (sdisp_dodpri5 - sdisp_0).norm(passe_partout_size=passe_partout_size)  # |sdisp_lsoda - disp|

    if verbose:  # Verbose: print the errors of each methods:
        print '--------------------'
        print 'generated matrix parameters:'
        print 'theta, tx, ty =    ' + str(m_0.get)
        print 'dtheta, dtx, dty = ' + str(dm_0.get)

        print '--------------------'
        print "Norm of the svf:"
        print res[i, 0]

        print '--------------------'
        print "Norm of the displacement field:"
        print res[i, 1]

        print '--------------------'
        print "Norm of the errors:"
        print '|ss - disp|          = ' + str(res[i, 2])
        print '|ss_pa - disp|       = ' + str(res[i, 3])
        print '|euler - disp|       = ' + str(res[i, 4])
        print '|midpoint - disp|    = ' + str(res[i, 5])
        print '|euler_mod - disp|   = ' + str(res[i, 6])
        print '|heun - disp|        = ' + str(res[i, 7])
        print '|heun_mod - disp|    = ' + str(res[i, 8])
        print '|rk4 - disp|         = ' + str(res[i, 9])
        print '|sdisp_ss_rk4 - disp|  = ' + str(res[i, 10])
        print '|sdisp_dopri - disp| = ' + str(res[i, 11])
        print
        print 'computational time of each method : '
        print '> ss        = ' + str(res_t[i, 0])
        print '> ss_pa     = ' + str(res_t[i, 1])
        print '> euler     = ' + str(res_t[i, 2])
        print '> midpoint  = ' + str(res_t[i, 3])
        print '> euler_mod = ' + str(res_t[i, 4])
        print '> heun      = ' + str(res_t[i, 5])
        print '> heun_mod  = ' + str(res_t[i, 6])
        print '> rk4       = ' + str(res_t[i, 7])
        print '> ss_rk4    = ' + str(res_t[i, 8])
        print '> dopri     = ' + str(res_t[i, 9])
        print

    if all_plot:  # show images of each field

        title_input_l = ['Sfv Input',
                         'Ground Output',
                         'Scaling and Squaring',
                         'Affine Scal. and Sq.',
                         'Euler',
                         'Midpoint',
                         'Euler Modif',
                         'Heun',
                         'Heun Modif',
                         'Runge Kutta 4',
                         'dopri5 (scipy)',
                         'dop853 (scipy)']

        list_fields_of_field = [[svf_0], [sdisp_0]]
        list_colors = ['r', 'b']
        fields_list = [svf_0, sdisp_0, sdisp_ss,   sdisp_ss_pa,   sdisp_euler,
                       sdisp_mid_p,   sdisp_euler_m,   sdisp_rk4, sdisp_ss_rk4]
        for third_field in fields_list[2:]:
            list_fields_of_field += [[svf_0, sdisp_0, third_field]]
            list_colors += ['r', 'b', 'm']

        see_n_fields_special(list_fields_of_field, fig_tag=10 + i,
                             colors_input=list_colors,
                             titles_input=title_input_l,
                             zoom_input=[0, 16, 0, 16], sample=(2, 2),
                             window_title_input='matrix, random generated' + str(i),
                             legend_on=False)


# save matrices in external folder:
if save_external:
    np.save(fullpath_filename_errors, res)
    np.save(fullpath_filename_times, res_t)
    if verbose:
        print "Data stored in matrices and saved in datafiles"
        print fullpath_filename_errors
        print fullpath_filename_times
        print


if 1:  # Print boxplot!

    reordered_data_for_boxplot = [list(res[:, 2])] +\
        [list(res[:, 3])] + [list(res[:, 4])] + [list(res[:, 5])] + [list(res[:, 6])] +\
        [list(res[:, 7])] + [list(res[:, 8])] + [list(res[:, 9])] + [list(res[:, 10])] + [list(res[:, 11])]

    title_input_l = ['Sca and Sq',
                     'Poly Sca and Sq',
                     'Euler method',
                     'Midpoint',
                     'Euler mod',
                     'Heun',
                     'Heun mod',
                     'Runge Kutta 4',
                     'ss rk4',
                     'lsoda (scipy)']

    mean_time = np.mean(res_t,axis=0)
    custom_boxplot(reordered_data_for_boxplot, x_labels=title_input_l, fig_tag=2, add_extra_numbers=mean_time)


if 1:  # Print computational time per sample

    x = range(1, N+1)
    title_input_l = ['Sca and Sq',
                     'Poly Sca and Sq',
                     'Euler ',
                     'Midpoint ',
                     'Euler modif ',
                     'Heun',
                     'Heun mod',
                     'Runge Kutta 4',
                     'ss rk4',
                     '0']

    list_colors = ['b', '0.75', 'm', 'r', 'c', 'y', 'k', 'g', 'b', '0.75',]

    fig, ax0 = plt.subplots(ncols=1, nrows=1, figsize=(10, 5.5), dpi=100)
    fig.subplots_adjust(left=0.075, right=0.9, top=0.9, bottom=0.15)
    # Shrink current axis by 20%
    box = ax0.get_position()
    ax0.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    for j in range(len(title_input_l)):
        ax0.plot(x, res_t[:, j], '-o', label=title_input_l[j], color=list_colors[j])

    ax0.set_yscale('log')
    # Put a legend to the right of the current axis
    ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax0.set_xlabel(r'Index of 20 SE(2)-generated SVF, angle in $(- \pi/2, -3\pi/8)\cup (3\pi/8, \pi/2)$', labelpad=20)
    # $(- \pi/8, -0.01)\cup (-0.01, \pi/8)$
    # $(- \pi/4, -\pi/8)\cup (\pi/8, \pi/4)$
    # $(- 3\pi/8, -\pi/4)\cup (\pi/4, 3\pi/8)$
    # $(- \pi/2, -3\pi/8)\cup (3\pi/8, \pi/2)$

    ax0.set_xlim([0, 21])
    ax0.set_ylabel('time (sec.) (log-scale)')

    ax0.set_title('Integrators computational time')

plt.show()