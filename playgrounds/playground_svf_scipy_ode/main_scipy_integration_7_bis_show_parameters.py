import numpy as np
import os
import matplotlib.pyplot as plt


### Controller ###

verbose = True

# Load files from the computations of the previous module:
specified_folder = '/Users/sebastiano/Documents/UCL/z_software/exponential_map/' \
                   'results_folder/scipy_integrator_results_parameters'
filename_errors = 'file_error_3'
filename_times  = 'file_time_3'


fullpath_filename_errors = os.path.join(specified_folder, filename_errors)
fullpath_filename_times  = os.path.join(specified_folder, filename_times)

main_errors = np.load(fullpath_filename_errors + '.npy')

main_computational_time = np.load(fullpath_filename_times + '.npy')

if verbose:
    print 'Main matrices loaded:'
    print main_errors.shape
    print main_computational_time.shape

integrators_labels    = ['vode', 'lsoda', 'dopri5', 'dop853']

x = range(2, 15, 1)

print len(x)

j = 9

y_vode_bdf_linear_err   = main_errors[:, 0, 0, j]
y_vode__bdf_cubic_err   = main_errors[:, 0, 1, j]

y_vode_adams_linear_err = main_errors[:, 0, 0, j]
y_vode__adams_cubic_err = main_errors[:, 0, 1, j]
'''


y_lsoda_linear_err  = main_errors[:, 1, 0, 0]
y_lsoda_cubic_err   = main_errors[:, 1, 1, 0]
y_dopri5_linear_err = main_errors[:, 2, 0, 0]
y_dopri5_cubic_err  = main_errors[:, 2, 1, 0]
y_dop835_linear_err = main_errors[:, 3, 0, 0]
y_dop835_cubic_err  = main_errors[:, 3, 1, 0]
'''

fig, ax0 = plt.subplots(ncols=1, nrows=1)

ax0.plot(x, y_vode_adams_linear_err, '-o', label='asdf', color='r')
ax0.plot(x, y_vode__adams_cubic_err, '--o', color='r')

ax0.plot(x, y_vode_bdf_linear_err, '-o', label='asdf', color='b')
ax0.plot(x, y_vode__bdf_cubic_err, '--o', color='b')



'''
ax0.plot(x, y_dopri5_linear_err, '-o', label='asdf', color='g')
ax0.plot(x, y_dopri5_cubic_err, '--o', color='g')

ax0.plot(x, y_dop835_linear_err, '-o', label='asdf', color='g')
ax0.plot(x, y_dop835_cubic_err, '--o', color='g')
'''


ax0.set_title('errors for each method')

plt.show()