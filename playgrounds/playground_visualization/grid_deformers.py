"""
Module to create grids as 2d numpy arrays made of 0 and 1 whose 1 are structured as grid.
"""

from os.path import join as jph

import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np

from utils.image import Image
from transformations.s_vf import SVF
from utils.aux_functions import grid_generator
from utils.helper import generate_position_from_displacement
from utils.resampler import NearestNeighbourResampler
from visualizer.fields_at_the_window import see_field
from visualizer.fields_at_the_window import triptych_image_quiver_image



# Create grid:

gr = grid_generator(x_size=301,
                    y_size=301,
                    x_step=30,
                    y_step=30,
                    line_thickness=1)

fig = plt.figure(2, figsize=(7, 7), dpi=100)
fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)
ax  = fig.add_subplot(111)
ax.imshow(gr, cmap='Greys',  interpolation='nearest')
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)



# Import SVF:

path_to_folder = '/Users/sebastiano/Documents/UCL/z_data/z_for_presentations/single_sub'
path_t0_nifti = jph(path_to_folder, 'time0.nii')
path_t1_nifti = jph(path_to_folder, 'time1.nii')

path_to_svf = jph(path_to_folder, 'disp_vel.nii.gz')


flow = nib.load(path_to_svf)

data_flow = flow.get_data()
header_flow = flow.header
affine_flow = flow.affine

print data_flow.shape


array_flow = data_flow[:, 76:77, :, :, 0:3:2].reshape(256,256,1,1,2)

svf_1 = SVF.from_array(array_flow)

print 'SVF shape : ' + str(svf_1.shape)


# generate the grid and copy as target
grid_array = grid_generator(x_size=256,
                            y_size=256,
                            x_step=20,
                            y_step=20,
                            line_thickness=1)

source_grid_im = Image.from_array(grid_array)

zeros_array = np.zeros_like(grid_array)
target_grid_im = Image.from_array(zeros_array)

 ### RESAMPLING!!!
# Convert to position from displacement:
svf_im0_def = generate_position_from_displacement(svf_1)

csr = NearestNeighbourResampler()
#csr.order = 5
csr.resample(source_grid_im, svf_im0_def, target_grid_im)

### Visualize:
triptych_image_quiver_image(source_grid_im.field,
                            svf_1.field,
                            target_grid_im.field,
                            interval_svf=1)

#print target_grid_im.field[2, 40]




see_field(svf_1)

# import initial image







plt.show()
