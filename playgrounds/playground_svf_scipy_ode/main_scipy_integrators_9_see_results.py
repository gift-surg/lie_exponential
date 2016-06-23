import os
import numpy as np
import matplotlib.pyplot as plt

# PLOT data obtained in the main_boxplot_matrix_generated

# Load files from the computations of the previous module:
specified_folder = '/Users/sebastiano/Documents/UCL/z_software/exponential_map/' \
                   'results_folder/scipy_integrator_results_parameters'
filename_errors = 'file_error_1_40_5_steps_all_methods_linear'  # 'file_error_4'
filename_times  = 'file_time_1_40_5_steps_all_methods_linear'  # 'file_time_4'


fullpath_filename_errors = os.path.join(specified_folder, filename_errors)
fullpath_filename_times  = os.path.join(specified_folder, filename_times)

res_error = np.load(fullpath_filename_errors + '.npy')

res_time = np.load(fullpath_filename_times + '.npy')

print res_error.shape
print res_time.shape


print 'steps ' + str(range(1, 40, 5))
print 'vode linear, element 1 of 20'
print res_error[:, 0, 0, 0]
print 'lsoda: linear, element 1 of 20 '
print res_error[:, 1, 0, 0]
print 'dopri5: linear, element 1 of 20'
print res_error[:, 2, 0, 0]
print 'dop853: linear, element 1 of 20'
print res_error[:, 3, 0, 0]
print
'''
print 'vode cubic, element 1 of 20'
print res_error[:, 0, 1, 0]
print 'lsoda: cubic, element 1 of 20 '
print res_error[:, 1, 1, 0]
print 'dopri5: cubic, element 1 of 20'
print res_error[:, 2, 1, 0]
print 'dop853: cubic, element 1 of 20'
print res_error[:, 3, 1, 0]
'''

N = res_error.shape[0]


if 1:  # Print computational time per sample

    x = range(1, N+1)
    title_input_l = ['Sca and Sq',
                     'Poly Sca and Sq',
                     'Euler method',
                     'Midpoint Method',
                     'Euler mod Method',
                     'Runge Kutta 4',
                     'vode (scipy)',
                     'lsoda (scipy)']

    list_colors = ['b', '0.75', 'm', 'r', 'c', 'y', 'k', 'g']

    fig, ax0 = plt.subplots(ncols=1, nrows=1)

    for j in range(len(title_input_l)):
        ax0.plot(x, res_time[:, j], '-o', label=title_input_l[j], color=list_colors[j])

    ax0.set_title('computational time for each SE(2)-generated')

plt.show()

