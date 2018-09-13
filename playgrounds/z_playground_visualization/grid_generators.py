"""
Module to create grids as 2d numpy arrays made of 0 and 1 whose 1 are structured as grid.
"""

import matplotlib.pyplot as plt
import nibabel as nib
from os.path import join as jph

from transformations.s_vf import SVF

from z_utils.aux_functions import grid_generator
from z_utils.path_manager import original_common_space_fp

from visualizer.fields_at_the_window import see_field

# Create grid:

gr = grid_generator(x_size=301,
                    y_size=301,
                    x_step=30,
                    y_step=30,
                    line_thickness=1)

fig = plt.figure(1, figsize=(7, 7), dpi=100)
fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)
ax  = fig.add_subplot(111)
ax.imshow(gr, cmap='Greys',  interpolation='nearest')
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)


# Import SVF

disp_name_A_C = 'disp_' + str(1) + '_A_C.nii.gz'

nib_A_C = nib.load(jph(original_common_space_fp, disp_name_A_C))

data_A_C = nib_A_C.get_data()
header_A_C = nib_A_C.header
affine_A_C = nib_A_C.affine

array_2d_A_C = data_A_C[:, 32:-32, 50:51, :, 0:2]

svf_1 = SVF.from_array_with_header(array_2d_A_C, header=header_A_C, affine=affine_A_C)

print 'SVF shape : ' + str(svf_1.shape)

see_field(svf_1)

# import initial image







plt.show()
