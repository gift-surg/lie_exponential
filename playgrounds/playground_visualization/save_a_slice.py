"""
Module to create grids as 2d numpy arrays made of 0 and 1 whose 1 are structured as grid.
"""

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from os.path import join as jph

from transformations.s_vf import SVF
from utils.image import Image

from utils.aux_functions import grid_generator

from visualizer.fields_at_the_window import see_field

import scipy.misc


# Paths:

path_to_folder = '/Users/sebastiano/Documents/UCL/z_data/z_for_presentations/single_sub'
path_t0_nifti = jph(path_to_folder, 'time0.nii')
path_t1_nifti = jph(path_to_folder, 'time1.nii')

path_slice_t0_nifti = jph(path_to_folder, 'time0_slice.nii.gz')
path_slice_t1_nifti = jph(path_to_folder, 'time1_slice.nii.gz')

path_slice_t0_jpg = jph(path_to_folder, 'time0_slice.jpg')
path_slice_t1_jpg = jph(path_to_folder, 'time1_slice.jpg')


# Import image t0 and select slice:
nib_t0 = nib.load(path_t0_nifti)

data_t0 = nib_t0.get_data()
header_t0 = nib_t0.header
affine_t0 = nib_t0.affine

im0 = data_t0[:, 72:73, :]
im_nifty0 = Image.from_array(im0)
im0 = np.rot90(im0.reshape(256, 256))

# Import image t1 and select slice:
nib_t1 = nib.load(path_t1_nifti)

data_t1 = nib_t1.get_data()
header_t1 = nib_t1.header
affine_t1 = nib_t1.affine

im1 = data_t1[:, 72:73, :]
im_nifty1 = Image.from_array(im1)
im1 = np.rot90(im1.reshape(256, 256))

# save the slice as jpg:
scipy.misc.imsave(path_slice_t0_jpg, im0)
scipy.misc.imsave(path_slice_t1_jpg, im1)

# save the slice as nifti:
im_nifty0.save(path_slice_t0_nifti)
im_nifty1.save(path_slice_t1_nifti)

# visualise
fig = plt.figure(1, figsize=(7, 7), dpi=100)
fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)
ax  = fig.add_subplot(111)
ax.imshow(im0, cmap='Greys',  interpolation='nearest')
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)


fig = plt.figure(2, figsize=(7, 7), dpi=100)
fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)
ax  = fig.add_subplot(111)
ax.imshow(im1, cmap='Greys',  interpolation='nearest')
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)

plt.show()
