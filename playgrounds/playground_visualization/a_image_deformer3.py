"""
Module to create grids as 2d numpy arrays made of 0 and 1 whose 1 are structured as grid.
"""

from os.path import join as jph
import os
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

path_to_folder = '/Users/sebastiano/a_data/z_for_presentations/2_rabbits'
path_t0_nifti = jph(path_to_folder, '1305_segment.nii.gz')
path_t1_nifti = jph(path_to_folder, '1702_T1.nii.gz')

path_to_svf = jph(path_to_folder, 'vel_cpp_disp.nii.gz')

path_output_img = jph(path_to_folder, 'output_img_segm')

os.system('mkdir -p {0}'.format(path_output_img))

### import source image:

source = nib.load(path_t0_nifti)

data_source = source.get_data()
header_source = source.header
affine_source = source.affine

print data_source.shape

array_source = np.rot90(data_source[:, 193:194, :].reshape(320, 320),0)

source_time0_im = Image.from_array(array_source)

### Import flow

flow = nib.load(path_to_svf)

data_flow = 10*flow.get_data()[:]

#data_flow = data_flow.field - Image.displacement_from_deformation(data_flow).field
header_flow = flow.header
affine_flow = flow.affine

print data_flow.shape

array_flow = data_flow[:, 193:194, :, :, 0:3:2].reshape(320, 320, 1, 1, 2)

svf_1 = SVF.from_array(array_flow)

print 'SVF shape : ' + str(svf_1.shape)

### generate the grid and copy as target

grid_array = grid_generator(x_size=320,
                            y_size=320,
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

coeff_steps = np.linspace(0, 1, 20)
#
# for s in range(len(coeff_steps)):
#
#     # import and modify the grid:
#     svf_reduced = copy.deepcopy(coeff_steps[s] * svf_1)
#     svf_def_reduced = generate_position_from_displacement(svf_reduced)
#
#     print source_grid_im.shape, svf_def_reduced.shape, target_grid_im.shape
#     print source_time0_im.shape, svf_def_reduced.shape, target_time0_im.shape
#
#     # resample of the image
#     csr.resample(source_time0_im, svf_def_reduced, target_time0_im)
#
#     fig = plt.figure(s+10, figsize=(7, 7), dpi=100)
#     fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)
#     ax  = fig.add_subplot(111)
#     # add grid
#     ax.imshow(np.rot90(target_time0_im.field), cmap='gray_r')
#     plt.setp(ax.get_xticklabels(), visible=False)
#     plt.setp(ax.get_yticklabels(), visible=False)
#     plt.savefig(jph(path_output_img, 'fig_{}.png'.format(s)))
#
#     # add the field
#     #see_field(svf_reduced, fig_tag=s+10)
#
#     plt.show(block=True)


grid = grid_generator(x_size=320,
                   y_size=320,
                   x_step=20,
                   y_step=20,
                   line_thickness=1)

source_time0_im.field = source_time0_im.field + grid * np.mean(source_time0_im.field)

for s in range(len(coeff_steps)):

    # import and modify the grid:
    svf_reduced = copy.deepcopy(coeff_steps[s] * svf_1)
    svf_def_reduced = generate_position_from_displacement(svf_reduced)

    print source_grid_im.shape, svf_def_reduced.shape, target_grid_im.shape
    print source_time0_im.shape, svf_def_reduced.shape, target_time0_im.shape

    # resample of the image
    csr.resample(source_time0_im, svf_def_reduced, target_time0_im)

    fig = plt.figure(s+10, figsize=(7, 7), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)
    ax  = fig.add_subplot(111)
    # add grid


    ax.imshow(np.rot90(target_time0_im.field), cmap='gray_r')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.savefig(jph(path_output_img, 'fig_grid_{}.png'.format(s)))

    plt.show(block=True)

print('Producing animation.gif using ImageMagick...')
os.system("convert -delay 1 -dispose Background +page " + str(path_output_img)
      + "/*.png -loop 0 " + str(path_output_img) + "/animation.gif")
