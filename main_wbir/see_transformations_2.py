import copy

import numpy as np
from matplotlib import pyplot as plt
import os
import nibabel as nib

from transformations import se2_g
from transformations.s_vf import SVF
from utils.projective_algebras import get_random_hom_a_matrices
from utils.path_manager import displacements_folder_path_AD

from utils.aux_functions import grid_generator
from utils.helper import generate_position_from_displacement
from utils.resampler import NearestNeighbourResampler
from utils.image import Image
from visualizer.fields_at_the_window import triptych_quiver_quiver_quiver, triptych_quiver_quiver_image, \
    triptych_image_quiver_image, quadrivium_quiver

from utils.path_manager_2 import flows_aei_fp, displacements_aei_fp

"""
Module created to show images of the first four data-sets.

"""

if __name__ == "__main__":

    random_seed = 4

    if random_seed > 0:
        np.random.seed(random_seed)

    #### SE2 generated image ####

    x_1, y_1 = 20, 20
    x_c = np.floor(x_1/2)+3
    y_c = np.floor(y_1/2)-2

    theta = np.pi/12

    tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
    ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

    m_0 = se2_g.se2_g(theta, tx, ty)
    dm_0 = se2_g.log(m_0)

    svf_se2   = SVF.generate_from_matrix([x_1, y_1], dm_0.get_matrix, affine=np.eye(4))

    #### HOM generated image ####

    x_1, y_1 = 20, 20
    x_c = np.floor(x_1/2)
    y_c = np.floor(y_1/2)

    projective_center = [x_1, y_1, 1]

    scale_factor = 1./(np.max([x_1, y_1])*5)
    hom_attributes = [scale_factor, 1, True]
    h_a, h_g = get_random_hom_a_matrices(d=2,
                                          scale_factor=hom_attributes[0],
                                          sigma=hom_attributes[1],
                                          special=hom_attributes[2])

    svf_hom = SVF.generate_from_projective_matrix_algebra(input_vol_ext=[x_1, y_1], input_h=h_a)

    #### Gauss generated image ####

    shape_gauss = [30, 30] + [1, 1, 2]

    sigma_init = 5
    sigma_gaussian_filter = 2

    svf_gauss = SVF.generate_random_smooth(shape=shape_gauss,
                                               sigma=sigma_init,
                                               sigma_gaussian_filter=sigma_gaussian_filter)

    #### ADNI generated image ####

    # svf data -> id of the loaded element:
    id_element = 1

    # Load as nib:
    #nib_A_C = nib.load(os.path.join(displacements_folder_path_AD,  'disp_' + str(id_element) + '_A_C.nii.gz'))

    nib_A_C = nib.load(os.path.join(displacements_aei_fp, 'displacement_AD_0_.nii.gz'))

    # reduce from 3d to 2d:
    data_A_C = nib_A_C.get_data()
    header_A_C = nib_A_C.header
    affine_A_C = nib_A_C.affine

    print data_A_C.shape

    #array_2d_A_C = data_A_C[:, 32:-32, 100:101, :, 0:2]
    array_2d_A_C = data_A_C[35:-35, 62:63, 35:-35, :, 0:2]

    print 'spam'
    print array_2d_A_C.shape

    # Create svf over the array:
    svf_adni = SVF.from_array_with_header(array_2d_A_C, header=header_A_C, affine=affine_A_C)

    print svf_adni.shape

    #### ADNI deformed grid ####

    # Convert to position from displacement:
    svf_adni_def = generate_position_from_displacement(svf_adni)

    # generate source and target
    source_grid_im = Image.from_array(grid_generator(x_size=svf_adni.shape[0], y_size=svf_adni.shape[2],
                                                     x_step=44, y_step=44))

    target_grid_im = Image.from_array(np.zeros_like(source_grid_im.field))

    print source_grid_im.shape
    print svf_adni.shape
    print target_grid_im.shape

    csr = NearestNeighbourResampler()
    csr.resample(source_grid_im, svf_adni, target_grid_im)

    ### Plot the triptychs ###

    triptych_quiver_quiver_quiver(svf_se2.field, svf_gauss.field, svf_adni.field, fig_tag=1)

    #triptych_quiver_quiver_image(svf_se2.field, svf_gauss.field, target_grid_im.field, fig_tag=2)

    #triptych_image_quiver_image(source_grid_im.field, svf_adni.field, target_grid_im.field, fig_tag=3)

    ### plot the four images: ###

    print
    print svf_se2.field.shape, svf_hom.field.shape, svf_gauss.field.shape, svf_adni.field.shape

    quadrivium_quiver(svf_se2.field, svf_hom.field, svf_gauss.field, svf_adni.field, fig_tag=10)

    plt.show()
