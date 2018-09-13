import os
import numpy as np
import matplotlib.pyplot as plt
from visualizer.graphs_and_stats import custom_boxplot

from z_utils.path_manager import path_to_results_folder

# PLOT data obtained in the main_boxplot_matrix_generated

# Load files from the computations of the previous module:

specified_folder = os.path.join(path_to_results_folder, 'scipy_integrator_results_parameters')

filename_errors = 'err_0_pi4_bis_20'
filename_times  = 'time_0_pi4_bis_20'

path_to_save_images = os.path.join(path_to_results_folder, 'figures')
filename_image_to_save_interval_1 = os.path.join(path_to_save_images, 'boxplot_interval_1.pdf')
filename_image_to_save_interval_2 = os.path.join(path_to_save_images, 'boxplot_interval_2.pdf')
filename_image_to_save_interval_3 = os.path.join(path_to_save_images, 'boxplot_interval_3.pdf')
filename_image_to_save_interval_4 = os.path.join(path_to_save_images, 'boxplot_interval_4.pdf')

'''
Possible filenames:

err_0_pi_8
time_0_pi_8

err_pi_8_pi_4
time_pi_8_pi_4

err_pi_4_3pi_8
time_pi_4_3pi_8


err_3pi_8_pi_2
time_3pi_8_pi_2
'''


fullpath_filename_errors = os.path.join(specified_folder, filename_errors)
fullpath_filename_times  = os.path.join(specified_folder, filename_times)

res = np.load(fullpath_filename_errors + '.npy')

res_t = np.load(fullpath_filename_times + '.npy')

print res.shape
print res_t.shape

print
print 'vode: '
print res[:, 8]
print
print 'lsoda:'
print res[:, 9]
print



N = res.shape[0]


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
                     'vode (scipy)',
                     'lsoda (scipy)']

    mean_time = np.mean(res_t,axis=0)
    custom_boxplot(reordered_data_for_boxplot, x_labels=title_input_l, fig_tag=2, add_extra_numbers=mean_time)
    plt.savefig(filename_image_to_save_interval_1, dpi=400,  format='pdf')


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
                     'vode (scipy)',
                     'lsoda (scipy)']

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

