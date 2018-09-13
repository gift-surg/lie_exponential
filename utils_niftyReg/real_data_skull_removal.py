"""
Aimed to load images and masks and obtain the segmented region.
"""

import os
import numpy as np
import nibabel as nib


from path_manager import root_data_MIRIAD

original_images_fp = os.path.join(root_data_MIRIAD, 'a__original_images')
skull_less_fp      = os.path.join(root_data_MIRIAD, 'a_original_images_sl')  # here images and matrices


# Baseline

path_template_0           = os.path.join(original_images_fp, 'template_0.nii')
path_template_mask_0      = os.path.join(original_images_fp, 'template_0_brain.nii')
path_template_segmented_0 = os.path.join(skull_less_fp, 'AAA_template_sl.nii')

im1 = Image(nib.load(path_template_0))

mask_1 = Image(nib.load(path_template_mask_0))

# load a second image to be modified, so that header is the same.
segmented_im1 = Image(nib.load(path_template_0))

segmented_im1.field = np.multiply(im1.field, mask_1.field)
segmented_im1.update_field()

print segmented_im1.nib_image.header

segmented_im1.save(path_template_segmented_0)

loaded = Image(nib.load(path_template_segmented_0))

print
print loaded.nib_image.header

path_segmented = os.path.join(skull_less_fp, 'A_erase_me.nii')

loaded2 = Image(nib.load(path_template_segmented_0))


print
print loaded2.nib_image.header


if 0:
    for j in range(10):

        # Baseline

        path_im_j   = os.path.join(original_images_fp, 'AD_' + str(j) + '_baseline.nii')
        path_mask_j = os.path.join(original_images_fp, 'AD_' + str(j) + '_baseline_brain.nii')
        path_segmented_j = os.path.join(skull_less_fp, 'AD_' + str(j) + '_baseline_sl.nii')

        im1 = Image(nib.load(path_im_j))

        mask_1 = Image(nib.load(path_mask_j))

        # load a second image to be modified, so that header is the same.
        segmented_im1 = Image(nib.load(path_im_j))

        segmented_im1.field = np.multiply(im1.field, mask_1.field)
        segmented_im1.update_field()

        segmented_im1.save(path_segmented_j)

        # Followup

        path_im_j   = os.path.join(original_images_fp, 'AD_' + str(j) + '_followup.nii')
        path_mask_j = os.path.join(original_images_fp, 'AD_' + str(j) + '_followup_brain.nii')
        path_segmented_j = os.path.join(skull_less_fp, 'AD_' + str(j) + '_followup_sl.nii')

        im1 = Image(nib.load(path_im_j))

        mask_1 = Image(nib.load(path_mask_j))

        # load a second image to be modified, so that header is the same.
        segmented_im1 = Image(nib.load(path_im_j))

        segmented_im1.field = np.multiply(im1.field, mask_1.field)
        segmented_im1.update_field()

        segmented_im1.save(path_segmented_j)

