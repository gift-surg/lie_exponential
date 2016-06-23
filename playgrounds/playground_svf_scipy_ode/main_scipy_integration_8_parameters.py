"""
Try BDF with steps < 7 where the error is still stable.

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from transformations.s_vf import SVF
from transformations.s_disp import SDISP
from transformations.se2_a import se2_g

from visualizer.fields_comparisons import see_n_fields_special


### Controller ###

domain = (14, 14)  # Matrix coordinates: x = -Y, y = X
passepartout = 3
omega = (3, 11, 3, 11)  # where to locate the center of the random rotation

interval_theta = (- np.pi / 4, np.pi / 4)
epsilon = 0.01

N = 20

verbose            = True  # moderate amount of information
verbose_logorrheic = False  # Too many information
plot_em_all = False  # to be True only for emergency debugging!

# Parameter choices for exponential scipy

max_steps      = range(2, 15, 1)
integrators    = ['vode']
methods_vode   = ['bdf', 'adams']  # consider only adams, since they provides almost the same results!
interp_methods = ['linear', 'cubic']
verbose_exp    = False

### Data structure to store data ###

# 4d matrix x:steps, y:integrators or methods, z:interpolation, t:error
main_errors = np.zeros([len(max_steps), len(methods_vode), len(interp_methods), N])
main_computational_time = np.zeros([len(max_steps), len(methods_vode), len(interp_methods), N])

if verbose or verbose_logorrheic:
    print 'Main matrices for storage created:'
    print main_errors.shape
    print main_computational_time.shape

# folder where to save the resulting matrices

specified_folder = '/Users/sebastiano/Documents/UCL/z_software/exponential_map/' \
                   'results_folder/scipy_integrator_results_parameters'
filename_errors = 'file_error_3'
filename_times  = 'file_time_3'
fullpath_filename_errors = os.path.join(specified_folder, filename_errors)
fullpath_filename_times  = os.path.join(specified_folder, filename_times)

if verbose or verbose_logorrheic:
    print 'Main matrices will be saved in the folder '
    print specified_folder
    print 'with file names: ' + str(filename_errors) + ' and ' + str(filename_times)
    print


### Loops: ###

for i in range(N):

#compute random matrices of transformations:
    m_0 = se2_g.randomgen_custom_center(interval_theta=interval_theta,
                                                        omega=omega,
                                                        epsilon_zero_avoidance=epsilon)
    dm_0 = se2_g.log(m_0)

    if verbose_logorrheic:
        print 'Matrices to generate svf and disp ground truth created:'
        print dm_0.get_matrix
        print m_0.get_matrix
        print

    ### generate subsequent vector fields out of the matrices
    svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))
    sdisp_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))

    if verbose_logorrheic:
        print 'Svf and disp ground truth created:'
        print type(svf_0)
        print type(sdisp_0)
        print

    for interp_method_i in range(len(interp_methods)):
        for method in range(len(methods_vode)):
            for max_step_i in range(len(max_steps)):

                if verbose or verbose_logorrheic:
                    print '------- Beginning of a new cycle for matrix ' + str(i+1) + ' out of ' + str(N) + '---------'
                    print 'interp method     : ' + str(interp_methods[interp_method_i])
                    print 'vode method       : ' + str(methods_vode[method])
                    print 'step              : ' + str(max_steps[max_step_i])
                    print '---------------------------------------------------------------'

                ### compute exponential with different available methods: ###

                start = time.time()

                sdisp_scipy =  svf_0.exponential_scipy(integrator='vode',
                                                       method=methods_vode[method],
                                                       max_steps=max_steps[max_step_i],
                                                       interpolation_method=interp_methods[interp_method_i],
                                                       verbose=verbose_exp,
                                                       passepartout=passepartout,
                                                       return_integral_curves=False)

                operation_time = (time.time() - start)
                error = (sdisp_scipy - sdisp_0).norm(passe_partout_size=passepartout)

                if verbose or verbose_logorrheic:
                    print '----------  Error  and Computational Time  ----'
                    print '|vode - disp| = ' + str(error) + ' voxel'
                    print 'Comp Time     = ' + str(operation_time) + ' sec.'
                    print '-----------------------------------------------'
                    print

                # store errors and computational time in appropriate matrices:
                # 4d matrix x:steps, y:integrators and methods, z:interpolation, t:error/computational time
                main_errors[max_step_i, method, interp_method_i, i] = error
                main_computational_time[max_step_i, method, interp_method_i, i] = operation_time

                if plot_em_all:
                    # Keep it false!!! add an if, if there is some data that does not work
                    fields_list_0 = [svf_0, sdisp_0, sdisp_scipy]

                    list_fields_of_field = [[svf_0], [sdisp_0], [svf_0, sdisp_0, sdisp_scipy]]
                    list_colors = ['r', 'b', 'r', 'b', 'm']

                    see_n_fields_special(list_fields_of_field,
                                         fig_tag=i,
                                         row_fig=1, col_fig=3,
                                         colors_input=list_colors,
                                         zoom_input=[0, 20, 0, 20], sample=(1, 1),
                                         window_title_input='matrix, random generated',
                                         legend_on=False)

                    plt.show()

if verbose_logorrheic:
    print 'Cycles end!'

# save matrices in external folder:
np.save(fullpath_filename_errors, main_errors)
np.save(fullpath_filename_times, main_computational_time)
if verbose:
    print "Data stored in matrices and saved in datafiles"
    print fullpath_filename_errors
    print fullpath_filename_times
    print

# compute the means of the error and the standard deviation for each method:


# print some slices of the matrices:


if verbose:
    print "THE END!"



