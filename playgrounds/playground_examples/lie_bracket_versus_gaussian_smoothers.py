
"""
Aimed to see how the exp works with different available algorithms.

Old to be refactored!
"""
import os
import numpy as np

from transformations.s_vf import SVF
from matplotlib import pyplot as plt


from utils.path_manager import path_to_results_folder

print 'Hello! It may take some minutes...'


file_name_image = 'SVF_image_scale_bracket_versus_gaussian'
fullpath_image  = os.path.join(path_to_results_folder, file_name_image)

file_name_data_0  = 'matrix_norms_0'
file_name_data_1  = 'matrix_norms_1'
file_name_data_bk = 'matrix_norms_lie_bracket'

fullpath_data_0  = os.path.join(path_to_results_folder, file_name_data_0)
fullpath_data_1  = os.path.join(path_to_results_folder, file_name_data_1)
fullpath_data_bk = os.path.join(path_to_results_folder, file_name_data_bk)

# if true save a second copy of relevant figures in specified path
save_data_also_in_specified_folder = False
specified_folder = ''
fullpath_image_specified_folder = os.path.join(specified_folder, file_name_image)

# fonts
font_on_fig   = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 10}
font_on_fig_w = {'family': 'serif', 'color': 'white', 'weight': 'normal', 'size': 10}

'''
---
relation between sigma_gf, sigma_init and norm of the exponentiated SVF

Each node of the graph is the mean of number_of_samples_per_step samples with
sigma of the gaussian smoother equals to i in range(5)
and
sigma of the initial SVF equals to j in range np.linspace(0.0, 1.0, num=11)
'''

number_of_samples_per_step = 1

steps_sigma_gf = np.linspace(1.0, 3.0, 7)
sigma_init = 2

lie_bk = SVF.lie_bracket


svf_test_0 = SVF.generate_random_smooth((150, 150, 1, 1, 2), sigma=1, sigma_gaussian_filter=5)
svf_test_1 = SVF.generate_random_smooth((150, 150, 1, 1, 2), sigma=1, sigma_gaussian_filter=5)
lie_bracket_0_1 = lie_bk(svf_test_0, svf_test_1)

# initial test
'''
print 'Norm 0'
print np.linalg.norm(svf_test_0.field.data)
print 'Norm 1'
print np.linalg.norm(svf_test_1.field.data)
print 'Lie bracket type'
print type(lie_bracket_0_1)
print 'Lie bracket norm'
print np.linalg.norm(lie_bracket_0_1.field.data)
'''


def half_lie_bracket_2(left, right):
    """
    Compute the Lie bracket of two velocity fields.

    Parameters:
    -----------
    :param left: Left velocity field 2 dimensionals
    :param right: Right velocity field.
    truncated second order Lie bracket  = Jac(left)*Jac(left)*right - Jac(left)*Jac(right)*left
    :return Return the resulting velocity field
    """

    left_jac = left.compute_jacobian()
    right_jac = right.compute_jacobian()

    result = SVF.generate_id_from_obj(left.field)

    result.field.data[..., 0] = \
        (left_jac.data[..., 0] * right.field.data[..., 0] +
         left_jac.data[..., 1] * right.field.data[..., 1]) -\
        (right_jac.data[..., 0] * left.field.data[..., 0] +
         right_jac.data[..., 1] * left.field.data[..., 1])

    result.field.data[..., 1] = \
        (left_jac.data[..., 2] * right.field.data[..., 0] +
         left_jac.data[..., 3] * right.field.data[..., 1]) - \
        (right_jac.data[..., 2] * left.field.data[..., 0] +
         right_jac.data[..., 3] * left.field.data[..., 1])


compute = False


if compute:

    ans_matrix_lie_bk = np.zeros([7, 7, 5])
    # first slice: the mean of the sample 0
    # second slice: the means fo the sample 1
    # third slice the respective lie bracket of the samples

    for i in range(7):
        for j in range(7):

            sample_0 = [SVF.generate_random_smooth((150, 150, 1, 1, 2),
                                                          sigma=sigma_init,
                                                          sigma_gaussian_filter=steps_sigma_gf[i])
                          for _ in range(number_of_samples_per_step)]

            sample_1 = [SVF.generate_random_smooth((150, 150, 1, 1, 2),
                                                          sigma=sigma_init,
                                                          sigma_gaussian_filter=steps_sigma_gf[j])
                          for _ in range(number_of_samples_per_step)]

            sample_lie_bracket_1   = [lie_bk(m0, m1).field for m0, m1 in zip(sample_0, sample_1)]
            sample_lie_bracket_3_2 = [lie_bk(m0, lie_bk(m0, m1)).field for m0, m1 in zip(sample_0, sample_1)]
            sample_lie_bracket_2   = [(lie_bk(m0, lie_bk(m0, m1)) + lie_bk(m1, lie_bk(m1, m0))).field for m0, m1 in zip(sample_0, sample_1)]

            ans_matrix_lie_bk[i, j, 0] = np.mean([np.linalg.norm(m0.field.data) for m0 in sample_0])
            ans_matrix_lie_bk[i, j, 1] = np.mean([np.linalg.norm(m1.field.data) for m1 in sample_1])
            ans_matrix_lie_bk[i, j, 2] = np.mean(np.linalg.norm(sample_lie_bracket_1))
            ans_matrix_lie_bk[i, j, 3] = np.mean(np.linalg.norm(sample_lie_bracket_3_2))
            ans_matrix_lie_bk[i, j, 4] = np.mean(np.linalg.norm(sample_lie_bracket_2))

    np.save(fullpath_data_bk, ans_matrix_lie_bk)

else:
    ans_matrix_lie_bk   = np.load(fullpath_data_bk + '.npy')


if 1:
    # figure 1
    min_val_0 = np.min(ans_matrix_lie_bk[:, :, 2:4])
    max_val_0 = np.max(ans_matrix_lie_bk[:, :, 2:4])
    min_val_1 = np.min(ans_matrix_lie_bk[:, :, 2:4])
    max_val_1 = np.max(ans_matrix_lie_bk[:, :, 2:4])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5), dpi=90)
    fig.canvas.set_window_title('SVF lie bracket versus gaussian filter')

    plt.subplots_adjust(left=0.07, bottom=0.15)

    plt.sca(axes[0])
    plt.xticks(range(0, 7), np.round(steps_sigma_gf, decimals=2))
    plt.yticks(range(0, 7), np.round(steps_sigma_gf, decimals=2))

    plt.sca(axes[1])
    plt.xticks(range(0, 7), np.round(steps_sigma_gf, decimals=2))
    plt.yticks(range(0, 7), np.round(steps_sigma_gf, decimals=2))

    #
    ax_0 = axes.flat[0]
    img_0 = ax_0.imshow(ans_matrix_lie_bk[:, :, 2], vmin=min_val_0, vmax=max_val_0, cmap=plt.get_cmap('YlOrRd'), interpolation='nearest', origin='lower') #YlOrRd

    ax_0.set_xlabel('Value of $\sigma_{gf}$, $\mathbf{u}_0$')
    ax_0.set_ylabel('Value of $\sigma_{gf}$, $\mathbf{u}_1$')
    ax_0.set_title('Norm of [$\mathbf{u}_0$, $\mathbf{u}_1$]', y=1.01)
    ax_0.xaxis.labelpad = 7
    ax_0.yaxis.labelpad = 1

    ax_0.text(-0.4, -0.05,  str(np.round(ans_matrix_lie_bk[0, 0, 2], decimals=3)), fontdict=font_on_fig_w)
    ax_0.text(-0.4, 5.95,   str(np.round(ans_matrix_lie_bk[0, 6, 2], decimals=3)), fontdict=font_on_fig)
    ax_0.text(5.55, 5.95,   str(np.round(ans_matrix_lie_bk[6, 6, 2], decimals=3)), fontdict=font_on_fig)
    ax_0.text(5.55, -0.05,  str(np.round(ans_matrix_lie_bk[6, 0, 2], decimals=3)), fontdict=font_on_fig)

    ax_0.text(0.55, 0.95,   str(np.round(ans_matrix_lie_bk[1, 1, 2], decimals=3)), fontdict=font_on_fig)
    ax_0.text(1.55, 1.95,   str(np.round(ans_matrix_lie_bk[2, 2, 2], decimals=3)), fontdict=font_on_fig)
    ax_0.text(2.55, 2.95,   str(np.round(ans_matrix_lie_bk[3, 3, 2], decimals=3)), fontdict=font_on_fig)
    ax_0.text(3.55, 3.95,   str(np.round(ans_matrix_lie_bk[4, 4, 2], decimals=3)), fontdict=font_on_fig)
    ax_0.text(4.55, 4.95,   str(np.round(ans_matrix_lie_bk[5, 5, 2], decimals=3)), fontdict=font_on_fig)
    #
    ax_1 = axes.flat[1]
    img_1 = ax_1.imshow(ans_matrix_lie_bk[:, :, 3], vmin=min_val_1, vmax=max_val_1, cmap=plt.get_cmap('YlOrRd'), interpolation='nearest', origin='lower')

    ax_1.set_xlabel('Value of $\sigma_{gf}$, $\mathbf{u}_0$')
    ax_1.set_ylabel('Value of $\sigma_{gf}$, $\mathbf{u}_1$')
    ax_1.set_title('Norm of [$\mathbf{u}_0$,[$\mathbf{u}_0$, $\mathbf{u}_1$]]', y=1.01)
    ax_1.xaxis.labelpad = 7
    ax_1.yaxis.labelpad = 1

    ax_1.text(-0.4, -0.05,  str(np.round(ans_matrix_lie_bk[0, 0, 3], decimals=3)), fontdict=font_on_fig_w)
    ax_1.text(-0.4, 5.95,   str(np.round(ans_matrix_lie_bk[0, 6, 3], decimals=3)), fontdict=font_on_fig)
    ax_1.text(5.55, 5.95,   str(np.round(ans_matrix_lie_bk[6, 6, 3], decimals=3)), fontdict=font_on_fig)
    ax_1.text(5.55, -0.05,  str(np.round(ans_matrix_lie_bk[6, 0, 3], decimals=3)), fontdict=font_on_fig)

    ax_1.text(0.55, 0.95,   str(np.round(ans_matrix_lie_bk[1, 1, 3], decimals=3)), fontdict=font_on_fig)
    ax_1.text(1.55, 1.95,   str(np.round(ans_matrix_lie_bk[2, 2, 3], decimals=3)), fontdict=font_on_fig)
    ax_1.text(2.55, 2.95,   str(np.round(ans_matrix_lie_bk[3, 3, 3], decimals=3)), fontdict=font_on_fig)
    ax_1.text(3.55, 3.95,   str(np.round(ans_matrix_lie_bk[4, 4, 3], decimals=3)), fontdict=font_on_fig)
    ax_1.text(4.55, 4.95,   str(np.round(ans_matrix_lie_bk[5, 5, 3], decimals=3)), fontdict=font_on_fig)

    position1 = fig.add_axes([0.93, 0.15, 0.02, 0.75])
    plt.colorbar(img_1, cax=position1)

    # save the figure
    plt.savefig(fullpath_image)
    print 'Figure '+ file_name_image + ' saved in the project folder ' + str(fullpath_image)

    if save_data_also_in_specified_folder:
        plt.savefig(fullpath_image_specified_folder, dpi=400)
        print 'Figure '+ file_name_image + ' saved in the external folder ' + str(fullpath_image_specified_folder)

    plt.show()


if 0:
    # figure 1
    min_val = np.min(ans_matrix_lie_bk)
    max_val = np.max(ans_matrix_lie_bk)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.4), dpi=80)
    plt.subplots_adjust(left=0.07, bottom=0.15)

    plt.sca(axes[0])
    plt.xticks(range(0, 7), range(1, 8))
    plt.yticks(range(0, 7), np.round(steps_sigma_gf, decimals=2))

    plt.sca(axes[1])
    plt.xticks(range(0, 7), np.round(steps_sigma_gf, decimals=2))
    plt.yticks(range(0, 7), range(1, 8))

    plt.sca(axes[2])
    plt.xticks(range(0, 7), np.round(steps_sigma_gf, decimals=2))
    plt.yticks(range(0, 7), np.round(steps_sigma_gf, decimals=2))

    #
    ax_0 = axes.flat[0]
    img_0 = ax_0.imshow(ans_matrix_lie_bk[:, :, 0], vmin=min_val, vmax=max_val, cmap=plt.get_cmap('YlOrRd'), interpolation='nearest', origin='lower') #YlOrRd

    ax_0.set_xlabel('Id sample')
    ax_0.set_ylabel('Value of $\sigma_{gf}$')
    ax_0.set_title('Norm of SVF0')
    ax_0.xaxis.labelpad = 2
    ax_0.yaxis.labelpad = 2
    #
    ax_1 = axes.flat[1]
    img_1 = ax_1.imshow(ans_matrix_lie_bk[:, :, 1], vmin=min_val, vmax=max_val, cmap=plt.get_cmap('YlOrRd'), interpolation='nearest', origin='lower')

    ax_1.set_xlabel('Value of $\sigma_{gf}$')
    ax_1.set_ylabel('Id sample')
    ax_1.set_title('Norm of SVF1')
    ax_1.xaxis.labelpad = 9
    ax_1.yaxis.labelpad = 7
    #
    ax_2 = axes.flat[2]
    img_2 = ax_2.imshow(ans_matrix_lie_bk[:, :, 2], vmin=min_val, vmax=max_val, cmap=plt.get_cmap('YlOrRd'), interpolation='nearest', origin='lower')

    ax_2.set_xlabel('Value of $\sigma_{gf}$ SVF0')
    ax_2.set_ylabel('Value of $\sigma_{gf}$ SVF1')
    ax_2.set_title('Norm of [SVF0, SVF1]')
    ax_2.xaxis.labelpad = 9
    ax_2.yaxis.labelpad = -3

    position = fig.add_axes([0.93, 0.15, 0.02, 0.75])
    plt.colorbar(img_1, cax=position)

    # save the figure
    plt.savefig(fullpath_image)
    print 'Figure '+ file_name_image + ' saved in the project folder ' + str(fullpath_image)

    if save_data_also_in_specified_folder:
        plt.savefig(fullpath_image_specified_folder, dpi=400)
        print 'Figure '+ file_name_image + ' saved in the external folder ' + str(fullpath_image_specified_folder)

    plt.show()