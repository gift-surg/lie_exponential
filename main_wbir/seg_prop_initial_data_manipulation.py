import numpy as np
import os


"""
Phase 1: data are restructured in a suitable form for numpy from the labels obtained from the computation on
the cluster.
Input: output data from the cluster
Output: txt files ready to be converted in numpy with np.loadtxt.
"""


max_iteration = True  # data from second nifty_reg_folder obtained with maxit

if max_iteration:

    path_to_txt_original_data_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data_maxit/f3d2_no_aei'
    path_to_copy_for_numpy_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py_maxit/no_aei'

    path_to_txt_original_data_aei      = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data_maxit/f3d2_aei'
    path_to_copy_for_numpy_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py_maxit/aei'

else:

    path_to_txt_original_data_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data/f3d2_no_aei'
    path_to_copy_for_numpy_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py/no_aei'

    path_to_txt_original_data_aei      = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data/f3d2_aei'
    path_to_copy_for_numpy_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/dices_for_py/aei'


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
