"""
Study for the estimate of step-error for the Scaling and squaring based and Taylor numerical methods.
Multiple se2 generated SVF.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import nibabel as nib

from transformations.s_vf import SVF
from visualizer.graphs_and_stats import plot_multiple_graphs

from z_utils.path_manager import displacements_folder_path_AD, path_to_results_folder


if __name__ == "__main__":

    #######################
    ### Path management ###
    #######################

    path_to_folder = os.path.join(path_to_results_folder, 'exp_methods_results')

    saved_file_index = '1_k_5_n_70'

    path_to_parameters  = os.path.join(path_to_folder, 'step_error_parameters_multiple_adnii_svf_' +
                                       str(saved_file_index) + '.npy')
    path_to_step_errors = os.path.join(path_to_folder, 'step_error_errors_multiple_adnii_svf_'     +
                                       str(saved_file_index) + '.npy')

    path_to_adnii_folder = copy.deepcopy(displacements_folder_path_AD)

    #####################
    ### Control Panel ###
    #####################

    compute = False  # or compute, or load
    verbose = True

    k = 5  # number of considered SVF max 17
    n = 70  # maximal number of consecutive steps where to compute the truncation error

    shape = (176, 240, 1)

    passepartout_x = 28
    passepartout_y = 60

    passepartout = 5

    ### compute exponential with different available methods:
    spline_interpolation_order = 3

    ### Init data structure to be saved in external files ###

    parameters   = np.array([n, k])
    step_errors  = np.zeros([n, 8, k])  # row: step, col: method, slice: svf

    #############
    ### START ###
    #############

    print 'COMPUTATIONS OF THE TRUNCATION ERROR FOR MULTIPLE SE2-GENERATED SVF.'
    print ''
    print 'INITIAL DATA:'
    print 'n = ' + str(n)
    print 'k = ' + str(k)
    print 'parameters of the transformation n, k = ' + str(parameters)

    if compute:

        print '----------------------'
        print 'Computations started'

        for num_element in range(1, k):

            # path flows:
            disp_name_A_C = 'disp_' + str(num_element) + '_A_C.nii.gz'

            # Load as nib:
            nib_A_C = nib.load(os.path.join(path_to_adnii_folder, disp_name_A_C))

            # reduce from 3d to 2d:
            data_A_C = nib_A_C.get_data()
            array_2d_A_C = data_A_C[passepartout_x:-passepartout_x, passepartout_y:-passepartout_y, 148:149, :, 0:2]

            # initialize as array:
            svf_0 = SVF.from_array(array_2d_A_C)

            print svf_0.shape

            sdisp_ss_j      = svf_0.exponential(algorithm='ss',
                                                s_i_o=spline_interpolation_order, input_num_steps=1)
            sdisp_ss_pa_j   = svf_0.exponential(algorithm='ss_pa',
                                                s_i_o=spline_interpolation_order, input_num_steps=1)
            sdisp_euler_j   = svf_0.exponential(algorithm='euler',
                                                s_i_o=spline_interpolation_order, input_num_steps=1)
            sdisp_mid_p_j   = svf_0.exponential(algorithm='midpoint',
                                                s_i_o=spline_interpolation_order, input_num_steps=1)
            sdisp_euler_m_j = svf_0.exponential(algorithm='euler_mod',
                                                s_i_o=spline_interpolation_order, input_num_steps=1)
            sdisp_heun_j    = svf_0.exponential(algorithm='heun',
                                                s_i_o=spline_interpolation_order, input_num_steps=1)
            sdisp_heun_m_j  = svf_0.exponential(algorithm='heun_mod',
                                                s_i_o=spline_interpolation_order, input_num_steps=1)
            sdisp_rk4_j     = svf_0.exponential(algorithm='rk4',
                                                s_i_o=spline_interpolation_order, input_num_steps=1)

            for step in range(2, n):
                # Store the old svf_0.exp in the temporary list:
                sdisp_step_j_minus_1 = [sdisp_ss_j, sdisp_ss_pa_j, sdisp_euler_j, sdisp_mid_p_j,
                                        sdisp_euler_m_j, sdisp_heun_j, sdisp_heun_m_j, sdisp_rk4_j]

                # Compute the new svf_0.exp at the step j
                if step < 60:
                    sdisp_ss_j      = svf_0.exponential(algorithm='ss',
                                                        s_i_o=spline_interpolation_order, input_num_steps=step)
                    sdisp_ss_pa_j   = svf_0.exponential(algorithm='ss_pa',
                                                        s_i_o=spline_interpolation_order, input_num_steps=step)
                sdisp_euler_j   = svf_0.exponential(algorithm='euler',
                                                    s_i_o=spline_interpolation_order, input_num_steps=step)
                sdisp_mid_p_j   = svf_0.exponential(algorithm='midpoint',
                                                    s_i_o=spline_interpolation_order, input_num_steps=step)
                sdisp_euler_m_j = svf_0.exponential(algorithm='euler_mod',
                                                    s_i_o=spline_interpolation_order, input_num_steps=step)
                sdisp_heun_j    = svf_0.exponential(algorithm='heun',
                                                    s_i_o=spline_interpolation_order, input_num_steps=step)
                sdisp_heun_m_j  = svf_0.exponential(algorithm='heun_mod',
                                                    s_i_o=spline_interpolation_order, input_num_steps=step)
                sdisp_rk4_j     = svf_0.exponential(algorithm='rk4',
                                                    s_i_o=spline_interpolation_order, input_num_steps=step)

                # compute the norm of the differences
                if step < 60:
                    step_errors[step, 0, num_element] = (sdisp_ss_j -
                                                         sdisp_step_j_minus_1[0]).norm(passe_partout_size=passepartout)
                    step_errors[step, 1, num_element] = (sdisp_ss_pa_j -
                                                         sdisp_step_j_minus_1[1]).norm(passe_partout_size=passepartout)
                step_errors[step, 2, num_element] = (sdisp_euler_j -
                                                     sdisp_step_j_minus_1[2]).norm(passe_partout_size=passepartout)
                step_errors[step, 3, num_element] = (sdisp_mid_p_j -
                                                     sdisp_step_j_minus_1[3]).norm(passe_partout_size=passepartout)
                step_errors[step, 4, num_element] = (sdisp_euler_m_j -
                                                     sdisp_step_j_minus_1[4]).norm(passe_partout_size=passepartout)
                step_errors[step, 5, num_element] = (sdisp_heun_j -
                                                     sdisp_step_j_minus_1[5]).norm(passe_partout_size=passepartout)
                step_errors[step, 6, num_element] = (sdisp_heun_m_j -
                                                     sdisp_step_j_minus_1[6]).norm(passe_partout_size=passepartout)
                step_errors[step, 7, num_element] = (sdisp_rk4_j -
                                                     sdisp_step_j_minus_1[7]).norm(passe_partout_size=passepartout)

                # plot results on the screen!
                print 'Element ' + str(num_element) + '. Step ' + str(step) + ' computed for each methods.'
                if verbose:
                    print '------------------------------------------------------------'
                    print 'Element ' + str(num_element) + '. Norm of the errors for number of steps ' + str(step)

                    print 'Truncation_ ' + str(step) + '(ss_j - ss_j_minus_1)               = ' + \
                          str(step_errors[step, 0, num_element])
                    print 'Truncation_ ' + str(step) + '(ss_pa_j - ss_pa_j_minus_1)         = ' + \
                          str(step_errors[step, 1, num_element])
                    print 'Truncation_ ' + str(step) + '(euler_j - euler_j_minus_1)         = ' + \
                          str(step_errors[step, 2, num_element])
                    print 'Truncation_ ' + str(step) + '(midpoint_j - midpoint_j_minus_1)   = ' \
                          + str(step_errors[step, 3, num_element])
                    print 'Truncation_ ' + str(step) + '(euler_mod_j - euler_mod_j_minus_1) = ' + \
                          str(step_errors[step, 4, num_element])
                    print 'Truncation_ ' + str(step) + '(heun_j - heun_j_minus_1)           = ' + \
                          str(step_errors[step, 5, num_element])
                    print 'Truncation_ ' + str(step) + '(heun_mod_j - heun_mod_j_minus_1)   = ' + \
                          str(step_errors[step, 6, num_element])
                    print 'Truncation_ ' + str(step) + '(rk4_j - rk4_j_minus_1)             = ' + \
                          str(step_errors[step, 7, num_element])
                    print '------------------------------------------------------------'

        # Save data on external files:

        np.save(path_to_parameters,  parameters)
        np.save(path_to_step_errors, step_errors)

        print '----------------------'
        print 'Data computed and saved in external files!'
        print '----------------------'

    else:
        # if compute=False, load the data from external files:
        parameters   = np.load(path_to_parameters)
        step_errors  = np.load(path_to_step_errors)

        print '----------------------'
        print 'Data loaded!'
        print '----------------------'

    ############################
    ### Visualization method ###
    ############################

    n = step_errors.shape[0]

    print step_errors.shape

    # Plot all:
    methods_names_list = ['ss', 'ss_pa', 'euler', 'midpoint', 'euler_mod', 'heun', 'heun_mod', 'rk4']
    methods_color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.6'] * 2

    plot_multiple_graphs(range(0, n), step_errors[0:, :, :], methods_titles=methods_names_list,
                         fig_tag=2, num_col=4, num_row=2,
                         log_scale=True, input_colors_per_methods=methods_color_list,
                         window_title_input='step errors for 16 svf generated with adnii and Niftyreg')

    plt.show()

    ### save figure in external folder ###
