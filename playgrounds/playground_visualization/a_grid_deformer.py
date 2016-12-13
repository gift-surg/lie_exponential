"""
Module to create grids as 2d numpy arrays made of 0 and 1 whose 1 are structured as grid.
"""

from os.path import join as jph

import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
import copy

from utils.image import Image
from transformations.s_vf import SVF
from utils.aux_functions import grid_generator
from utils.helper import generate_position_from_displacement
from utils.resampler import NearestNeighbourResampler
from visualizer.fields_at_the_window import see_field
from visualizer.fields_at_the_window import triptych_image_quiver_image


# Import SVF:

path_to_folder = '/Users/sebastiano/Documents/UCL/z_data/z_for_presentations/single_sub'
path_t0_nifti = jph(path_to_folder, 'time0.nii')
path_t1_nifti = jph(path_to_folder, 'time1.nii')

path_to_svf = jph(path_to_folder, 'disp.nii.gz')

path_to_tmp = '/Users/sebastiano/Desktop/tmp'

### import source image:

source = nib.load(path_t0_nifti)

data_source = source.get_data()
header_source = source.header
affine_source = source.affine

print data_source.shape

array_source = np.rot90(data_source[:, 62:63, :].reshape(256, 256), 0)

source_time0_im = Image.from_array(array_source)

### Import flow

flow = nib.load(path_to_svf)

data_flow = flow.get_data()
header_flow = flow.header
affine_flow = flow.affine

print data_flow.shape

array_flow = np.zeros([256, 256, 1, 1, 2])

for x in range(256):
    for y in range(256):
        array_flow[x, y, 0, 0, 0] = data_flow[x, 62:63, y, :, 0]
        array_flow[x, y, 0, 0, 1] = data_flow[x, 62:63, y, :, 2]

svf_1 = SVF.from_array(array_flow)

print 'SVF shape : ' + str(svf_1.shape)

### generate the grid and copy as target

grid_array = grid_generator(x_size=256,
                            y_size=256,
                            x_step=20,
                            y_step=20,
                            line_thickness=1)

source_grid_im = Image.from_array(grid_array)

zeros_array = np.zeros_like(grid_array)
target_grid_im = Image.from_array(zeros_array)

target_time0_im = Image.from_array(zeros_array)

### RESAMPLING!!!
csr = NearestNeighbourResampler()


# print resampling steps:

coeff_steps = np.linspace(0, 0.4, 10)

for s in range(len(coeff_steps)):

    # import and modify the grid:
    svf_reduced = copy.deepcopy(coeff_steps[s] * svf_1)

    exp_svf = svf_reduced.exponential()

    def_reduced = generate_position_from_displacement(exp_svf)

    print source_grid_im.shape, def_reduced.shape, target_grid_im.shape
    print source_time0_im.shape, def_reduced.shape, target_time0_im.shape

    # resample of the grid
    csr.resample(source_grid_im, def_reduced, target_grid_im)

    # create the image
    fig = plt.figure(s, figsize=(7, 7), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)
    ax  = fig.add_subplot(111)
    # add grid
    ax.imshow(target_grid_im.field, cmap='Greys', interpolation='none')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    plt.savefig(jph(path_to_tmp, 'grid_' + str(s) + '.png'))

    #plt.show()

    # resample of the image
    csr.resample(source_time0_im, def_reduced, target_time0_im)

    fig2 = plt.figure(s+10, figsize=(7, 7), dpi=100)
    fig2.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)
    ax  = fig2.add_subplot(111)
    # add grid
    ax.imshow(target_time0_im.field, cmap='Greys', interpolation='none')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    #plt.savefig(jph(path_to_tmp, 'time_' + str(s) + '.png'))

    # add the field
    #see_field(svf_reduced, fig_tag=s+10)

    #plt.show()

    # add the field
    #see_field(svf_reduced, fig_tag=s+10)

    #plt.show()


'''
see_field(svf_1, fig_tag=100)
see_field(0.5 * svf_1, fig_tag=101)
see_field(0.01 * svf_1, fig_tag=102)
'''

# import initial image


