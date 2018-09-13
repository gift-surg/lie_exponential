"""
Path manager of the whole code.
Set the root_data_storage variable to your specific folder where you want to save the output and where to retrieve the
MRI data for real data expeirments. If you have no real SVF you can create it with the synthetic data-set BrainWeb.
See README for further information on this part.
---
Nomenclature:
root : main folder upon which all the other folders depends.
pfo : path to folder
pfi : path to file
"""

import os
from os.path import join as jph


root_dir = os.path.abspath(os.path.dirname(__file__))
root_data_storage = jph('/Volumes', 'SmartWare', 'lie_exponential')

if not os.path.exists(root_data_storage):
    print('Connect to data storage or set path in {}'.format(__file__))


""" Paths to OUTPUT folders """


pfo_results       = jph(root_data_storage, 'elaborations')
pfo_notes_figures = jph(root_data_storage, 'elaborations', 'figures')
pfo_notes_tables  = jph(root_data_storage, 'elaborations', 'tables')
pfo_notes_sharing = jph(root_data_storage, 'elaborations', 'sharing')


""" Segmentation propagation paths """


seg_prop_result_folder = jph(root_data_storage, 'elaborations', 'seg_prop_results')

pfi_roi_indexes           = os.path.join(seg_prop_result_folder, 'roi_indexes.npy')

pfi_means_dice_no_aei     = os.path.join(seg_prop_result_folder, 'means_dice_no_aei.npy')
pfi_means_dice_aei        = os.path.join(seg_prop_result_folder, 'means_dice_aei.npy')
pfi_labelled_dice_no_aei  = os.path.join(seg_prop_result_folder, 'labelled_dice_no_aei.npy')
pfi_labelled_dice_aei     = os.path.join(seg_prop_result_folder, 'labelled_dice_aei.npy')

pfi_means_times_no_aei    = os.path.join(seg_prop_result_folder, 'means_times_no_aei.npy')
pfi_means_times_aei       = os.path.join(seg_prop_result_folder, 'means_times_aei.npy')
pfi_labelled_times_no_aei = os.path.join(seg_prop_result_folder, 'labelled_times_no_aei.npy')
pfi_labelled_times_aei    = os.path.join(seg_prop_result_folder, 'labelled_times_aei.npy')


""" Paths to INPUT MIRIAD dataset """


root_data_MIRIAD = os.path.join(root_data_storage, 'ipmi_dataset')

if not os.path.exists(root_data_MIRIAD):
    print('MIRIAD dataset not present, correct the variable root_data_MIRIAD or use alternative datasets as brainweb')

# specified folders with original data:
original_images_fp               = os.path.join(root_data_MIRIAD, 'a__original_images')
original_image_skull_less_fp     = os.path.join(root_data_MIRIAD, 'a_original_images_sl')
original_common_space_fp         = os.path.join(root_data_MIRIAD, 'a_original_images_sl_common_space')

# For original version of niftyReg:
non_rigid_alignment_o_fp           = os.path.join(root_data_MIRIAD, 'non_rigid_alignment')
flows_o_fp                         = os.path.join(root_data_MIRIAD, 'flows')
deformations_o_fp                  = os.path.join(root_data_MIRIAD, 'deformations')
displacements_o_fp                 = os.path.join(root_data_MIRIAD, 'displacements')
grids_o_fp                         = os.path.join(root_data_MIRIAD, 'grids')

# For modified version of niftyReg with Approximated Exponential Integrators:
non_rigid_alignment_aei_fp           = os.path.join(root_data_MIRIAD, 'non_rigid_alignment_aei')
flows_aei_fp                         = os.path.join(root_data_MIRIAD, 'flows_aei')
deformations_aei_fp                  = os.path.join(root_data_MIRIAD, 'deformations_aei')
displacements_aei_fp                 = os.path.join(root_data_MIRIAD, 'displacements_aei')
grids_aei_fp                         = os.path.join(root_data_MIRIAD, 'grids_aei')


""" paths to input ADNI dataset """


# Path to patient data
root_data_AD = os.path.join('/Users/sebastiano/Documents/UCL/z_data/ADNI_MRes', 'data_svf_AD/')
root_data_CTL = os.path.join('/Users/sebastiano/Documents/UCL/z_data/ADNI_MRes', 'data_svf_CTL/')

# specified folders after elaborations
original_images_folder_path_AD                = os.path.join(root_data_AD, 'original_images')
rigid_aligned_images_folder_path_AD           = os.path.join(root_data_AD, 'rigid_aligned_images')
rigid_matrix_transform_folder_path_AD         = os.path.join(root_data_AD, 'rigid_matrix_transform')
non_rigid_aligned_images_folder_path_AD       = os.path.join(root_data_AD, 'non_rigid_aligned_images')
non_rigid_control_point_grid_folder_path_AD   = os.path.join(root_data_AD, 'non_rigid_control_point_grid')
flows_folder_path_AD                          = os.path.join(root_data_AD, 'flows')
deformations_folder_path_AD                   = os.path.join(root_data_AD, 'deformations')
displacements_folder_path_AD                  = os.path.join(root_data_AD, 'displacements')
grids_folder_path_AD                          = os.path.join(root_data_AD, 'grids')

original_images_folder_path_CTL                = os.path.join(root_data_CTL, 'original_images')
rigid_aligned_images_folder_path_CTL           = os.path.join(root_data_CTL, 'rigid_aligned_images')
rigid_matrix_transform_folder_path_CTL         = os.path.join(root_data_CTL, 'rigid_matrix_transform')
non_rigid_aligned_images_folder_path_CTL       = os.path.join(root_data_CTL, 'non_rigid_aligned_images')
non_rigid_control_point_grid_folder_path_CTL   = os.path.join(root_data_CTL, 'non_rigid_control_point_grid')
flows_folder_path_CTL                          = os.path.join(root_data_CTL, 'flows')
deformations_folder_path_CTL                   = os.path.join(root_data_CTL, 'deformations')
displacements_folder_path_CTL                  = os.path.join(root_data_CTL, 'displacements')
grids_folder_path_CTL                          = os.path.join(root_data_CTL, 'grids')


""" paths to INPUT NEUROMORPHOMETRICS data set"""


maxit = True  # 0: original data from NiftyReg

root_data_NMM = os.path.join(root_data_storage, 'neuromorphometrics')

pfo_txt_original_data_no_aei = jph(root_data_NMM, 'Data/f3d2_no_aei')

if maxit:
    pfo_NMM_elaborations = jph(root_data_NMM, 'log_elaboration_maxit')
    path_to_copy_elaborated_data_step_1_no_aei = jph(pfo_NMM_elaborations, 'step_1', 'no_aei')
    path_to_copy_elaborated_data_step_1_aei = jph(pfo_NMM_elaborations, 'step_1', 'aei')
    path_to_txt_original_data_aei = jph(root_data_NMM, 'Data_maxit', 'f3d2_aei')

    path_to_copy_for_numpy_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py_maxit/no_aei'
    path_to_copy_for_numpy_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py_maxit/aei'
    pfo_txt_original_data_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data_maxit/f3d2_no_aei'
    path_to_txt_original_data_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data_maxit/f3d2_aei'


else:
    pfo_NMM_elaborations = jph(root_data_NMM, 'log_elaboration')
    path_to_copy_elaborated_data_step_1_no_aei = jph(pfo_NMM_elaborations, 'step_1', 'no_aei')
    path_to_copy_elaborated_data_step_1_aei = jph(pfo_NMM_elaborations, 'step_1', 'aei')
    path_to_txt_original_data_aei = jph(root_data_NMM, 'Data', 'f3d2_aei')

    path_to_copy_for_numpy_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py/no_aei'
    path_to_copy_for_numpy_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py/no_aei'
    path_to_txt_original_data_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data/f3d2_aei'
    path_to_copy_for_numpy_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py/aei'


'''

path_to_project = os.path.dirname(os.path.dirname(__file__))

path_to_results_folder = os.path.join(path_to_project, 'results_folder')
path_to_sharing_folder = os.path.join(path_to_results_folder, 'sharing_folder')

path_to_tmp_folder = '/Users/sebastiano/Desktop/test_image'




# folders for latex documents where to save the produced images:
path_to_exp_notes_figures = '/Users/sebastiano/Documents/UCL/notes/Notes_on_exp/figures'
path_to_exp_notes_tables  = '/Users/sebastiano/Documents/UCL/notes/Notes_on_exp/tables'

# path to external table
fullpath_tables_external_folder = '/Users/sebastiano/Documents/UCL/ResearchProjects/paper_drafts/WBIR_16/tables'

""" Nifty-Reg Path to reg-apps """



""" Segmentation propagation paths """

seg_prop_result_folder = '/Users/sebastiano/Documents/UCL/z_software/exponential_map/results_folder/seg_prop_results'

full_path_roi_indexes             = os.path.join(seg_prop_result_folder, 'roi_indexes.npy')

full_path_means_dice_no_aei       = os.path.join(seg_prop_result_folder, 'means_dice_no_aei.npy')
full_path_means_dice_aei          = os.path.join(seg_prop_result_folder, 'means_dice_aei.npy')
full_path_labelled_dice_no_aei    = os.path.join(seg_prop_result_folder, 'labelled_dice_no_aei.npy')
full_path_labelled_dice_aei       = os.path.join(seg_prop_result_folder, 'labelled_dice_aei.npy')

full_path_means_times_no_aei       = os.path.join(seg_prop_result_folder, 'means_times_no_aei.npy')
full_path_means_times_aei          = os.path.join(seg_prop_result_folder, 'means_times_aei.npy')
full_path_labelled_times_no_aei    = os.path.join(seg_prop_result_folder, 'labelled_times_no_aei.npy')
full_path_labelled_times_aei       = os.path.join(seg_prop_result_folder, 'labelled_times_aei.npy')


def save_my_data(path, filename, tag, names, values):
    """
    save_my_file(path, filename, tag, names, values)
    :param path:
    :param filename: no extension
    :param tag:
    :param names:  [name1, name1, name3, ...]
    :param values: [value1 value2, value3 ,...]
        NO DICT because we want to read data in the same order we have inserted!
    :return:
    """

    # under construction, save the data with the same name of the variable.
    # This should work as a path manager, may save both images and files in appropriate formats!!

    filename += tag + '.dat'
    fullpath = os.path.join(path, filename)

    if len(names) == len(values):

        if os.path.exists(fullpath):
            f = open(fullpath, "w")
        else:
            f = open(fullpath, "w")
            #f = open(fullpath, 'w+')

        for j in range(len(names)):
            f.write("%s = %s\n" % (names[j], repr(values[j])))
            f.write("\n")

        f.close()

        return 'data saved in ' + filename

    else:

        raise TypeError("Warning data_saver: wrong input data type: names, values. "
                        " must have the same length ")


'''

