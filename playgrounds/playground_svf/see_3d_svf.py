import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import os
import nibabel as nib

from transformations.s_vf import SVF
from transformations import se2_g
from visualizer.fields_at_the_window import see_field

from utils.path_manager import path_to_results_folder, path_to_exp_notes_figures, path_to_exp_notes_tables, \
    displacements_folder_path_AD

"""
Simple and as direct as possible module aimed to the investigation and the visualization
of the 3d generated vector fields.

1) generated from SE2.
2) generated from Gauss.
3) generated from real data.

"""

if __name__ == "__main__":

    from_se2   = True
    from_Gauss = False
    from_real  = False

    if from_se2:

        x_1, y_1, z_1 = 8, 8, 8

        domain = (x_1, y_1, z_1)

        x_c = np.floor(x_1/2)
        y_c = np.floor(y_1/2)
        theta = np.pi/50

        tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
        ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

        # generate matrices
        m_0 = se2_g.se2_g(theta, tx, ty)
        dm_0 = se2_g.log(m_0)

        m_0 = se2_g.se2_g(0, 1, 0)
        dm_0 = se2_g.log(m_0)

        # Generate 3d svf
        svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))

        # see svf 3d
        field_from_svf = svf_0.field

        fig = plt.figure(1)

        ax_1 = fig.gca(projection='3d')

        ax_1.set_title('quiver')
        xx, yy, zz = np.meshgrid(np.arange(field_from_svf.shape[0]),
                                 np.arange(field_from_svf.shape[1]),
                                 np.arange(field_from_svf.shape[2]))
        ax_1.quiver(yy,
                    xx,
                    zz,
                    field_from_svf[:, :, :, 0, 0],
                    field_from_svf[:, :, :, 0, 1],
                    field_from_svf[:, :, :, 0, 2], color='r')

        print field_from_svf[1, 1, 1, 0, :]
        print field_from_svf[2, 2, 2, 0, :]
        print field_from_svf[1, 1, 3, 0, :]

        plt.show()

    if from_Gauss:

        shape = (8, 8, 8, 1, 3)

        sigma_init = 4
        sigma_gaussian_filter = 1

        svf_im0   = SVF.generate_random_smooth(shape=shape,
                                                 sigma=sigma_init,
                                                 sigma_gaussian_filter=sigma_gaussian_filter)

        # see svf 3d
        field_from_svf = svf_im0.field

        fig = plt.figure(2)

        ax_1 = fig.gca(projection='3d')

        ax_1.set_title('quiver')
        xx, yy, zz = np.meshgrid(np.arange(field_from_svf.shape[0]),
                                 np.arange(field_from_svf.shape[1]),
                                 np.arange(field_from_svf.shape[2]))
        ax_1.quiver(yy,
                    xx,
                    zz,
                    field_from_svf[:, :, :, 0, 0],
                    field_from_svf[:, :, :, 0, 1],
                    field_from_svf[:, :, :, 0, 2], color='r')

        print ''
        print 'STATS: '
        print 'min svf ' + str(np.min(field_from_svf))
        print 'max svf ' + str(np.max(field_from_svf))
        print 'median svf ' + str(np.median(field_from_svf))
        print 'Norm: ' + str(svf_im0.norm(normalized=True))
        print ''
        print 'Some points: '
        print field_from_svf[1, 1, 1, 0, :]
        print field_from_svf[2, 2, 2, 0, :]
        print field_from_svf[1, 1, 3, 0, :]

        sdisp_im1 = svf_im0.exponential(algorithm='ss')

        print 'Some points of the exponential of the svf: '
        print sdisp_im1.field[1, 1, 1, 0, :]
        print sdisp_im1.field[2, 2, 2, 0, :]
        print sdisp_im1.field[1, 1, 3, 0, :]

        plt.show()

    if from_real:

        id_element = 1
        # path flows:
        disp_name_A_C = 'disp_' + str(id_element) + '_A_C.nii.gz'
        # Load as nib:
        nib_A_C = nib.load(os.path.join(displacements_folder_path_AD, disp_name_A_C))

        # reduce from 3d to 2d:
        data_A_C = nib_A_C.get_data()
        header_A_C = nib_A_C.header
        affine_A_C = nib_A_C.affine

        array_2d_A_C = data_A_C[50:60, 50:60, 50:60, :, :3]

        # Create svf over the array:
        svf_0 = SVF.from_array_with_header(array_2d_A_C, header=header_A_C, affine=affine_A_C)

        field_from_svf = svf_0.field

        fig = plt.figure(2)

        ax_1 = fig.gca(projection='3d')

        ax_1.set_title('quiver')
        xx, yy, zz = np.meshgrid(np.arange(field_from_svf.shape[0]),
                                 np.arange(field_from_svf.shape[1]),
                                 np.arange(field_from_svf.shape[2]))
        ax_1.quiver(yy,
                    xx,
                    zz,
                    field_from_svf[:, :, :, 0, 0],
                    field_from_svf[:, :, :, 0, 1],
                    field_from_svf[:, :, :, 0, 2], color='r')

    plt.show()
