import os

### Segmentation propagation path manager: ###

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
