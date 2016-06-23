import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

from transformations.s_vf import SVF
from transformations.s_disp import SDISP
import transformations.se2_g as se2_g

from utils.aux_functions import get_in_out_liers

from visualizer.fields_at_the_window import see_2_fields, see_field
from visualizer.graphs_and_stats import plot_splitted_bar_chart
from utils.path_manager import path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables


"""
OLD module for the inverse consistency.

Plain module aimed to the visualization and saving of the inverse consistency data.
One SVF, se2,

Output:
    Figure with barchart inverse consistency results.
    Figure name:
    exp_inverse_consistency_barchart_1SE2genSVF_pi4_ALLmethods.pdf

    CSV table with the numerical inverse consistency errors for each method.
    Table name:
    exp_inverse_consistency_table_1SE2genSVF_pi4_ALLmethods.csv

    numpy array with the errors
    npy data structure name
    exp_inverse_consistency_array_1SE2genSVF_ALLmethods_pi.npy

"""


if __name__ == "__main__":

    #######################
    ### Path management ###
    #######################

    filename_figure_output    = 'exp_inverse_consistency_barchart_1SE2genSVF_ALLmethods'  # pdf
    filename_csv_table_output = 'exp_inverse_consistency_table_1SE2genSVF_ALLmethods'  # csv
    filename_np_array_output  = 'exp_inverse_consistency_array_1SE2genSVF_ALLmethods'  # npy

    saved_file_suffix = '_pi8'

    path_to_results_folder = os.path.join(path_to_results_folder, 'exponential_error_results')

    fullpath_image_to_external = os.path.join(path_to_exp_notes_figures,
                                              filename_figure_output + saved_file_suffix + '.pdf')
    fullpath_to_table_external = os.path.join(path_to_exp_notes_tables,
                                              filename_csv_table_output + saved_file_suffix + '.csv')
    fullpath_to_array_in_results_folder = os.path.join(path_to_results_folder,
                                                       filename_np_array_output + saved_file_suffix + '.npy')

    ##################
    ### Controller ###
    ##################

    x_1, y_1 = 25, 25
    domain = (x_1, y_1)
    shape = list(domain) + [1, 1, 2]

    # Passepartout keep greater than 1
    pp = 8

    spline_interpolation_order = 2

    compute = True

    save_data_in_external_folder = True

    visualize_extra = False
    show_bar_chart_splitted = False
    show_bar_chart_log_scale = True

    compute_vode_method = False

    verbose = True

    # transformations data:

    x_c = int(x_1/2)
    y_c = int(y_1/2)
    theta = np.pi/8

    ###########################
    ### Model: computations ###
    ###########################

    # To avoid warning in the editor
    sdisp_vode, sdisp_vode_inv, sdisp_o_sdisp_inv_vode, sdisp_inv_o_sdisp_vode = [None, ] * 4

    if compute:

        tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c + 0.5
        ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c + 0.5

        m_0 = se2_g.se2_g(theta, tx, ty)
        dm_0 = se2_g.log(m_0)

        # Generate svf
        svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))
        svf_0_inv   = -1*svf_0

        # Generate displacement ground truth
        sdisp_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))
        sdisp_0_inv = SDISP.generate_from_matrix(domain, np.linalg.inv(m_0.get_matrix) - np.eye(3), affine=np.eye(4))

        identity = SDISP.generate_id(shape=shape)

        # Exponentiate svf:
        sdisp_ss      = svf_0.exponential(algorithm='ss',        s_i_o=spline_interpolation_order)
        sdisp_ss_pa   = svf_0.exponential(algorithm='gss_aei',     s_i_o=spline_interpolation_order,
                                          input_num_steps=1)  # In the linear case one step is enough.
        sdisp_mid_p   = svf_0.exponential(algorithm='midpoint',  s_i_o=spline_interpolation_order)
        sdisp_euler   = svf_0.exponential(algorithm='euler',     s_i_o=spline_interpolation_order)
        sdisp_euler_m = svf_0.exponential(algorithm='euler_mod', s_i_o=spline_interpolation_order)
        sdisp_heun    = svf_0.exponential(algorithm='heun',      s_i_o=spline_interpolation_order)
        sdisp_heun_m  = svf_0.exponential(algorithm='heun_mod',  s_i_o=spline_interpolation_order)
        sdisp_rk4     = svf_0.exponential(algorithm='rk4',       s_i_o=spline_interpolation_order)
        if compute_vode_method:
            sdisp_vode = svf_0.exponential_scipy(passepartout=pp-3, verbose=verbose)

        # Exponentiate svf inverse:
        sdisp_ss_inv      = svf_0_inv.exponential(algorithm='ss',        s_i_o=spline_interpolation_order)
        sdisp_ss_pa_inv   = svf_0_inv.exponential(algorithm='gss_aei',     s_i_o=spline_interpolation_order,
                                                  input_num_steps=1)  # In the linear case one step is enough.
        sdisp_mid_p_inv   = svf_0_inv.exponential(algorithm='midpoint',  s_i_o=spline_interpolation_order)
        sdisp_euler_inv   = svf_0_inv.exponential(algorithm='euler',     s_i_o=spline_interpolation_order)
        sdisp_euler_m_inv = svf_0_inv.exponential(algorithm='euler_mod', s_i_o=spline_interpolation_order)
        sdisp_heun_inv    = svf_0_inv.exponential(algorithm='heun',      s_i_o=spline_interpolation_order)
        sdisp_heun_m_inv  = svf_0_inv.exponential(algorithm='heun_mod',  s_i_o=spline_interpolation_order)
        sdisp_rk4_inv     = svf_0_inv.exponential(algorithm='rk4',       s_i_o=spline_interpolation_order)
        if compute_vode_method:
            sdisp_vode_inv    = svf_0_inv.exponential_scipy(passepartout=pp-3, verbose=verbose)

        ### Compute the composition of displacements with different methods ###
        # Ideally they should be the identity #

        # For sanity check composition of the ground truth. Must be very close to the identity field.
        sdisp_o_sdisp_inv_ground = SDISP.composition(sdisp_0, sdisp_0_inv, s_i_o=spline_interpolation_order)
        sdisp_inv_o_sdisp_ground = SDISP.composition(sdisp_0_inv, sdisp_0, s_i_o=spline_interpolation_order)

        # scaling and squaring
        sdisp_o_sdisp_inv_ss = SDISP.composition(sdisp_ss, sdisp_ss_inv, s_i_o=spline_interpolation_order)
        sdisp_inv_o_sdisp_ss = SDISP.composition(sdisp_ss_inv, sdisp_ss, s_i_o=spline_interpolation_order)

        # polyaffine scaling and squaring
        sdisp_o_sdisp_inv_ss_pa = SDISP.composition(sdisp_ss_pa, sdisp_ss_pa_inv, s_i_o=spline_interpolation_order)
        sdisp_inv_o_sdisp_ss_pa = SDISP.composition(sdisp_ss_pa_inv, sdisp_ss_pa, s_i_o=spline_interpolation_order)

        # midpoint
        sdisp_o_sdisp_inv_mid_p = SDISP.composition(sdisp_mid_p, sdisp_mid_p_inv, s_i_o=spline_interpolation_order)
        sdisp_inv_o_sdisp_mid_p = SDISP.composition(sdisp_mid_p_inv, sdisp_mid_p, s_i_o=spline_interpolation_order)

        # euler
        sdisp_o_sdisp_inv_euler = SDISP.composition(sdisp_euler, sdisp_euler_inv, s_i_o=spline_interpolation_order)
        sdisp_inv_o_sdisp_euler = SDISP.composition(sdisp_euler_inv, sdisp_euler, s_i_o=spline_interpolation_order)

        # euler modified
        sdisp_o_sdisp_inv_eul_m = SDISP.composition(sdisp_euler_m, sdisp_euler_m_inv, s_i_o=spline_interpolation_order)
        sdisp_inv_o_sdisp_eul_m = SDISP.composition(sdisp_euler_m_inv, sdisp_euler_m, s_i_o=spline_interpolation_order)

        # heun
        sdisp_o_sdisp_inv_heun = SDISP.composition(sdisp_heun, sdisp_heun_inv, s_i_o=spline_interpolation_order)
        sdisp_inv_o_sdisp_heun = SDISP.composition(sdisp_heun_inv, sdisp_heun, s_i_o=spline_interpolation_order)

        # heun modified
        sdisp_o_sdisp_inv_heun_m = SDISP.composition(sdisp_heun_m, sdisp_heun_m_inv, s_i_o=spline_interpolation_order)
        sdisp_inv_o_sdisp_heun_m = SDISP.composition(sdisp_heun_m_inv, sdisp_heun_m, s_i_o=spline_interpolation_order)

        # rk4
        sdisp_o_sdisp_inv_rk4 = SDISP.composition(sdisp_rk4, sdisp_rk4_inv, s_i_o=spline_interpolation_order)
        sdisp_inv_o_sdisp_rk4 = SDISP.composition(sdisp_rk4_inv, sdisp_rk4, s_i_o=spline_interpolation_order)

        if compute_vode_method:
            # vode
            sdisp_o_sdisp_inv_vode = SDISP.composition(sdisp_vode, sdisp_vode_inv, s_i_o=spline_interpolation_order)
            sdisp_inv_o_sdisp_vode = SDISP.composition(sdisp_vode_inv, sdisp_vode, s_i_o=spline_interpolation_order)

        if visualize_extra:
            see_2_fields(svf_0, svf_0_inv, input_color='r')
            see_2_fields(sdisp_ss_pa, sdisp_ss_pa_inv, input_color='b')

            see_field(sdisp_0, fig_tag=30, input_color='g')
            see_field(sdisp_0_inv, fig_tag=30, input_color='g')
            see_field(sdisp_o_sdisp_inv_ground, fig_tag=30, input_color='m')

            if compute_vode_method:
                see_field(sdisp_vode, fig_tag=30, input_color='b')
                see_field(sdisp_vode_inv, fig_tag=30, input_color='b')
                see_field(sdisp_o_sdisp_inv_vode, fig_tag=30, input_color='k')

            plt.show()

        ### Compute the norm of each composition ###
        # data structure
        # cols: norm inversion with methods: 0) ground truth 1) scal_sq 2) Poly scal sq 3) midp 4) euler
        # 5) euler mod 6) heun 7) heun mod 8) vode
        # rows: 0) norm(disp o disp_inv, method cols) 1) norm(disp o disp_inv, method cols)

        data_structure = np.zeros([2, 9])
        if compute_vode_method:
            data_structure = np.zeros([2, 10])

        sdisp_o_sdisp_inv_data = [sdisp_o_sdisp_inv_ground, sdisp_o_sdisp_inv_ss, sdisp_o_sdisp_inv_ss_pa,
                                  sdisp_o_sdisp_inv_mid_p, sdisp_o_sdisp_inv_euler, sdisp_o_sdisp_inv_eul_m,
                                  sdisp_o_sdisp_inv_heun, sdisp_o_sdisp_inv_heun_m, sdisp_o_sdisp_inv_rk4]

        sdisp_inv_o_sdisp_data = [sdisp_inv_o_sdisp_ground, sdisp_inv_o_sdisp_ss, sdisp_inv_o_sdisp_ss_pa,
                                  sdisp_inv_o_sdisp_mid_p, sdisp_inv_o_sdisp_euler, sdisp_inv_o_sdisp_eul_m,
                                  sdisp_inv_o_sdisp_heun, sdisp_inv_o_sdisp_heun_m, sdisp_inv_o_sdisp_rk4]

        if compute_vode_method:
            sdisp_o_sdisp_inv_data += [sdisp_o_sdisp_inv_vode]
            sdisp_inv_o_sdisp_data += [sdisp_inv_o_sdisp_vode]

        for col, (obj_0, obj_1) in enumerate(zip(sdisp_o_sdisp_inv_data, sdisp_inv_o_sdisp_data)):

            data_structure[0, col] = obj_0.norm(normalized=True, passe_partout_size=pp)
            data_structure[1, col] = obj_1.norm(normalized=True, passe_partout_size=pp)

        ### Save data to external folder ###
        np.save(fullpath_to_array_in_results_folder, data_structure)

    else:
        ### Load data to external folder ###
        data_structure = np.load(fullpath_to_array_in_results_folder)
        print 'Data loaded from the folder ' + fullpath_to_array_in_results_folder
        print data_structure

    #################
    # Visualization #
    #################

    # define labels for graphs:
    title_input_l = ['ground', 'scal sq', 'Poly scal sq', 'midp',
                         'euler', 'euler mod', 'heun', 'heun mod', 'RK4']

    if compute_vode_method:
        title_input_l += ['vode']

    if verbose:

        print '\n \n'
        print 'results data: '
        print tabulate(data_structure, headers=title_input_l)
        print '\n \n'

    if show_bar_chart_splitted:

        out_liers_values, in_liers_values = get_in_out_liers(data_structure[0, :], coeff=0.01, return_values=True)
        epsilon, delta = 0.00005, 0.008

        interval_lower_y = [.000,  np.max(in_liers_values) + epsilon]
        interval_upper_y = [np.min(out_liers_values) - delta, np.max(out_liers_values) + 2 * delta]

        plot_splitted_bar_chart(data_structure, title_input=title_input_l,
                                y_intervals=(interval_lower_y, interval_upper_y))

    if show_bar_chart_log_scale:

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=120)

        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)

        ax.grid(True)

        index = np.arange(len(data_structure[1]))
        bar_width = 0.35

        r_1 = plt.bar(index, list(data_structure[0, :]), bar_width,
                         color='b',
                         label=r'$\varphi \circ \varphi^{-1}$', log=True)

        r_2 = plt.bar(index + bar_width, list(data_structure[1, :]), bar_width,
                         color='r',
                         label=r'$\varphi^{-1} \circ \varphi$', log=True)

        plt.xlabel('methods')
        plt.ylabel('error')
        plt.title(r'Inverse consistency errors: $| \varphi \circ \varphi^{-1} - I |$')
        plt.xticks(index + bar_width, title_input_l)
        ax.legend(loc=1)

    plt.show()

