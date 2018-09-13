import copy

import numpy as np
from matplotlib import pyplot as plt

from transformations.s_vf import SVF
from transformations import se2_g

from z_utils.image import Image
from z_utils.aux_functions import grid_generator
from z_utils.helper import generate_position_from_displacement
from z_utils.resampler import NearestNeighbourResampler
from z_utils.path_manager import path_to_tmp_folder

from visualizer.fields_at_the_window import triptych_image_quiver_image

"""
Module aimed to the creation of deformed grid according to a given SVF.

"""


if __name__ == "__main__":

    #### PARAMETERS ####

    x_1, y_1 = 204, 204
    domain = (x_1, y_1)
    shape = list(domain) + [1, 1, 2]

    random_image = True  # random or se2

    # generate the grid and copy as target
    grid_array = grid_generator(x_size=x_1,
                                y_size=y_1,
                                x_step=20,
                                y_step=20,
                                line_thickness=1)
    source_grid_im = Image.from_array(grid_array)

    zeros_array = np.zeros_like(grid_array)
    target_grid_im = Image.from_array(zeros_array)

    #### Generate the random SVF: ####

    if random_image:

        sigma_init = 6
        sigma_gaussian_filter = 2

        svf_im0 = SVF.generate_random_smooth(shape=shape,
                                             sigma=sigma_init,
                                             sigma_gaussian_filter=sigma_gaussian_filter)

    else:

        x_c = np.floor(x_1/2)
        y_c = np.floor(y_1/2)
        theta = np.pi/50

        tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
        ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

        m_0 = se2_g.se2_g(theta, tx, ty)
        dm_0 = se2_g.log(m_0)

        svf_im0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))

    if 0:
        print svf_im0

        ### Verbose!!
        print svf_im0.shape
        print source_grid_im.shape
        print target_grid_im.shape
        print type(svf_im0)
        print type(source_grid_im)
        print type(target_grid_im)

        print 'min:    ' + str(np.min(svf_im0.field))
        print 'max:    ' + str(np.max(svf_im0.field))
        print 'median: ' + str(np.median(svf_im0.field))
        print 'Norm:   ' + str(svf_im0.norm(normalized=True))

        print type(source_grid_im.nib_image)

        print 'spam0'
        print source_grid_im.nib_image.header

        print 'spam1'

        print target_grid_im.nib_image.header

        print source_grid_im.nib_image.affine
        print target_grid_im.nib_image.affine

    # Save
    if 0:
        svf_im0.save(path_to_tmp_folder)
        source_grid_im.save(path_to_tmp_folder)
        target_grid_im.save(path_to_tmp_folder)

    ### RESAMPLING!!!
    # Convert to position from displacement:
    svf_im0_def = generate_position_from_displacement(svf_im0)

    csr = NearestNeighbourResampler()
    #csr.order = 5
    csr.resample(source_grid_im, svf_im0_def, target_grid_im)

    ### Visualize:
    triptych_image_quiver_image(source_grid_im.field,
                                svf_im0.field,
                                target_grid_im.field,
                                interval_svf=1)

    #print target_grid_im.field[2, 40]

    plt.show()
