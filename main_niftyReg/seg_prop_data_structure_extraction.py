import numpy as np
import os

from utils.path_manager import full_path_roi_indexes, full_path_means_dice_no_aei, full_path_means_dice_aei, \
    full_path_labelled_dice_no_aei, full_path_labelled_dice_aei
# full_path_means_times_no_aei, full_path_means_times_aei,
#  full_path_labelled_times_no_aei, full_path_labelled_times_aei


"""
Phase 3: from .txt files data are extracted ad numpy arrays are obtained, validated and stored in external folder.
Input: txt file with cluster output written in appropriate ways
Output: numpy array written in specific data structures

roi_indexes = indexes of the regions where the label is not nan
"""

### paths where the data are loaded


max_iteration = True

if max_iteration:
    path_to_copy_for_numpy_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py_maxit/no_aei'
    path_to_copy_for_numpy_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py_maxit/aei'
else:
    path_to_copy_for_numpy_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py/no_aei'
    path_to_copy_for_numpy_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py/aei'

### Not all the regions are classified as nan in all the images:
# Searching for the intersection of all of the index that are not nan:
# Mean must be computed again,

num_scans = 35
roi_indexes_found = set(range(208))

list_index_ref = range(num_scans)


for ref_index in list_index_ref:

    list_index_flo = list_index_ref[:]
    del list_index_flo[ref_index]

    for flo_index in list_index_flo:

        folder_name_txt = str(ref_index) + '_f3d2'
        file_name_txt = 'dice2_ref' + str(ref_index) + '_flo' + str(flo_index) + '.txt'

        file_path_copy_for_numpy_no_aei = os.path.join(os.path.join(path_to_copy_for_numpy_no_aei, folder_name_txt),
                                                       file_name_txt)
        file_path_copy_for_numpy_aei    = os.path.join(os.path.join(path_to_copy_for_numpy_aei, folder_name_txt),
                                                       file_name_txt)

        m_pilot = np.loadtxt(file_path_copy_for_numpy_no_aei, delimiter=',')
        not_nan_roi_indexes = [i for i in range(m_pilot.shape[0] - 1) if not np.isnan(m_pilot[i, 1])]

        ## intersect with the previous one

        roi_indexes_found = roi_indexes_found & set(not_nan_roi_indexes)


# Label to be ignored given by Jorge:
ignore_labels = range(1, 4) + range(5, 11) + range(12, 23) + range(24, 30) + range(33, 35) + range(42, 44) + \
                range(53, 55) + range(63, 69) + range(70, 71) + range(74, 75) + range(80, 100) + \
                range(110, 112) + range(126, 128) + range(130, 132) + range(158, 160) + range(188, 190)

roi_indexes_given = set(range(208)) -  set(ignore_labels)


print len(roi_indexes_given)
print len(list(roi_indexes_found))

print roi_indexes_given - roi_indexes_found
print roi_indexes_found - roi_indexes_given


# Indexes that we really have to keep (the one given by Jorge where not all):
roi_indexes = np.sort(list(roi_indexes_found & roi_indexes_given))

### initialize data structure ###

num_labels = len(roi_indexes)

#means_dice_no_aei     = np.zeros([num_scans, num_scans])
#means_dice_aei        = np.zeros([num_scans, num_scans])
labelled_dice_no_aei  = np.zeros([num_scans, num_scans, num_labels])
labelled_dice_aei     = np.zeros([num_scans, num_scans, num_labels])

means_times_no_aei    = np.zeros([num_scans, num_scans])
means_times_aei       = np.zeros([num_scans, num_scans])
labelled_times_no_aei = np.zeros([num_scans, num_scans, num_labels])
labelled_times_aei    = np.zeros([num_scans, num_scans, num_labels])


#### Elaborate data ###

list_index_ref = range(num_scans)

for ref_index in list_index_ref:

    list_index_flo = list_index_ref[:]
    del list_index_flo[ref_index]

    for flo_index in list_index_flo:

        folder_name_txt = str(ref_index) + '_f3d2'
        file_name_txt = 'dice2_ref' + str(ref_index) + '_flo' + str(flo_index) + '.txt'

        file_path_copy_for_numpy_no_eai = os.path.join(os.path.join(path_to_copy_for_numpy_no_aei, folder_name_txt),
                                                       file_name_txt)
        file_path_copy_for_numpy_eai = os.path.join(os.path.join(path_to_copy_for_numpy_aei, folder_name_txt),
                                                    file_name_txt)

        m_no_aei = np.loadtxt(file_path_copy_for_numpy_no_eai, delimiter=',')
        m_aei    = np.loadtxt(file_path_copy_for_numpy_eai, delimiter=',')

        ### DICES ###
        # store data per labelled region
        for id_roi_index, id_roi in enumerate(roi_indexes):
            #print id_roi_index, id_roi
            labelled_dice_no_aei[ref_index, flo_index, id_roi_index] = m_no_aei[id_roi, 1]
            labelled_dice_aei[ref_index, flo_index, id_roi_index] = m_aei[id_roi, 1]

        ### COMPUTATIONAL TIME ###

        ### ZZZ

### Recompute the means:
means_dice_no_aei = np.mean(labelled_dice_no_aei, axis=2)
means_dice_aei    = np.mean(labelled_dice_aei, axis=2)


# Save externally:
np.save(full_path_roi_indexes, roi_indexes)

np.save(full_path_means_dice_no_aei, means_dice_no_aei)
np.save(full_path_means_dice_aei, means_dice_aei)
np.save(full_path_labelled_dice_no_aei, labelled_dice_no_aei)
np.save(full_path_labelled_dice_aei, labelled_dice_aei)
