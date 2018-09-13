import numpy as np
import os

from path_manager import path_to_copy_for_numpy_no_aei, path_to_txt_original_data_no_aei, \
    path_to_txt_original_data_aei, path_to_copy_for_numpy_aei


"""
Phase 1: data are restructured in a suitable form for numpy from the labels obtained from the computation on
the cluster.
Input: output data from the cluster
Output: txt files ready to be converted in numpy with np.loadtxt.
"""


num_scans = 35
list_index_ref = range(num_scans)

for ref_index in list_index_ref:

    list_index_flo = list_index_ref[:]
    del list_index_flo[ref_index]

    for flo_index in list_index_flo:

        #### Elaborate data for no aei ####

        folder_name_txt = str(ref_index) + '_f3d2'
        file_name_txt = 'dice2_ref' + str(ref_index) + '_flo' + str(flo_index) + '.txt'

        file_path_original_txt = os.path.join(os.path.join(path_to_txt_original_data_no_aei, folder_name_txt), file_name_txt)
        file_path_copy_for_numpy = os.path.join(os.path.join(path_to_copy_for_numpy_no_aei, folder_name_txt), file_name_txt)

        os.system(' mkdir -p ' + os.path.join(path_to_copy_for_numpy_no_aei, folder_name_txt))

        with open(file_path_original_txt, 'r') as file_txt_original:
            data = file_txt_original.read()

            data = data.replace('Mean Dice', '-1')
            data = data.replace('Label[', '')
            data = data.replace(']', '')
            data = data.replace(' = ', ', ')

            file_txt_original.close()

        with open(file_path_copy_for_numpy, 'w') as file_txt_copy:

            file_txt_copy.write(data)
            file_txt_copy.close()

        #### Elaborate data for aei ####

        file_path_original_txt = os.path.join(os.path.join(path_to_txt_original_data_aei, folder_name_txt), file_name_txt)
        file_path_copy_for_numpy = os.path.join(os.path.join(path_to_copy_for_numpy_aei, folder_name_txt), file_name_txt)

        os.system(' mkdir -p ' + os.path.join(path_to_copy_for_numpy_aei, folder_name_txt))

        with open(file_path_original_txt, 'r') as file_txt_original:
            data = file_txt_original.read()

            data = data.replace('Mean Dice', '-1')
            data = data.replace('Label[', '')
            data = data.replace(']', '')
            data = data.replace(' = ', ', ')

            file_txt_original.close()

        with open(file_path_copy_for_numpy, 'w') as file_txt_copy:

            file_txt_copy.write(data)
            file_txt_copy.close()

# aa = np.loadtxt(file_path_py, delimiter=',')
