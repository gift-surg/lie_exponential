import os


path_to_project = os.path.dirname(os.path.dirname(__file__))

path_to_results_folder = os.path.join(path_to_project, 'results_folder')
path_to_sharing_folder = os.path.join(path_to_results_folder, 'sharing_folder')

path_to_tmp_folder = '/Users/sebastiano/Desktop/test_image'

""" ADNI dataset (not used for WBIR paper)"""

# Path to patient data: Chose here if you want to deal with CTL or AD
root_data_AD = os.path.join('/Users/sebastiano/Documents/UCL/z_data/ADNI_MRes', 'data_svf_AD/')
root_data_CTL = os.path.join('/Users/sebastiano/Documents/UCL/z_data/ADNI_MRes', 'data_svf_CTL/')

# specified folders
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


# folders for latex documents where to save the produced images:
path_to_exp_notes_figures = '/Users/sebastiano/Documents/UCL/notes/Notes_on_exp/figures'
path_to_exp_notes_tables  = '/Users/sebastiano/Documents/UCL/notes/Notes_on_exp/tables'

# path to external table
fullpath_tables_external_folder = '/Users/sebastiano/Documents/UCL/ResearchProjects/paper_drafts/WBIR_16/tables'

""" Nifty-Reg Path to reg-apps """

niftyReg_path = '/Users/sebastiano/Workspace/niftiREG/build/reg-apps/'


""" Paths to MIRIAD dataset """

root_data = os.path.join('/Users/sebastiano/Documents/UCL/z_data/', 'ipmi_dataset/')

# specified folders with original data:
original_images_fp               = os.path.join(root_data, 'a__original_images')
original_image_skull_less_fp     = os.path.join(root_data, 'a_original_images_sl')
original_common_space_fp         = os.path.join(root_data, 'a_original_images_sl_common_space')


# For original version of niftyReg:
non_rigid_alignment_o_fp           = os.path.join(root_data, 'non_rigid_alignment')

flows_o_fp                         = os.path.join(root_data, 'flows')
deformations_o_fp                  = os.path.join(root_data, 'deformations')
displacements_o_fp                 = os.path.join(root_data, 'displacements')
grids_o_fp                         = os.path.join(root_data, 'grids')


# For modified version of niftyReg with Approximated Exponential Integrators:
non_rigid_alignment_aei_fp           = os.path.join(root_data, 'non_rigid_alignment_aei')

flows_aei_fp                         = os.path.join(root_data, 'flows_aei')
deformations_aei_fp                  = os.path.join(root_data, 'deformations_aei')
displacements_aei_fp                 = os.path.join(root_data, 'displacements_aei')
grids_aei_fp                         = os.path.join(root_data, 'grids_aei')


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
