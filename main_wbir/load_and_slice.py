import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
import copy
import nibabel as nib

from transformations.s_vf import SVF
from transformations.s_disp import SDISP


from nibabel.testing import data_path

from utils.path_manager import path_to_results_folder, \
    path_to_exp_notes_figures, path_to_exp_notes_tables, displacements_folder_path_AD
from utils.path_manager_2 import displacements_aei_fp

from visualizer.graphs_and_stats import plot_steps_vs_multiple_lines

"""
Simple module to play around and visualize some image, svf and displacement
obtained from Adnii images and

"""

if __name__ == "__main__":

    # Load SVF from Adnii data:
    num_element = 3

    # path flows:  displacement_AD_0_.nii
    disp_name_A_C = 'displacement_AD_' + str(3) + '_.nii.gz'
    # Load as nib:
    nib_A_C = nib.load(os.path.join(displacements_aei_fp, disp_name_A_C))

    affine_A_C =  nib_A_C.affine
    header_A_C = nib_A_C.header

    data_A_C = nib_A_C.get_data()

    print affine_A_C
    print header_A_C
    print data_A_C.shape

    pp = 30
    array_2d_A_C = data_A_C[pp:-pp, pp:-pp, 148:149, :, 0:2]

    # Create svf over the array:
    svf_0 = SVF.from_array_with_header(array_2d_A_C, header=header_A_C, affine=affine_A_C)

