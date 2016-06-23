import numpy as np
import os


num_data_set = 1  # 0: original data from NiftyReg

if num_data_set == 0:

    path_root = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/log_elaboration/'

elif num_data_set == 1:

    path_root = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/log_elaboration_maxit/'

else:
    pass

path_to_txt_elaborated_no_aei = os.path.join(path_root, 'step_1/no_aei')
path_to_npy_array_no_aei      = os.path.join(path_root, 'step_2/no_aei')

path_to_txt_elaborated_aei = os.path.join(path_root, 'step_1/aei')
path_to_npy_array_aei      = os.path.join(path_root, 'step_2/aei')

num_scans = 35

list_index_ref = range(num_scans)

for ref_index in list_index_ref:

    list_index_flo = list_index_ref[(ref_index+1):]

    for flo_index in list_index_flo:

        # Init the paths:

        folder_name_log = str(ref_index) + '_f3d2'

        file_name_log_numpy = 'log_ref' + str(ref_index) + '_flo' + str(flo_index) + '_for_numpy.txt'
        file_name_log_numpy_aei = 'log_ref' + str(ref_index) + '_flo' + str(flo_index) + '_aei_for_numpy.txt'

        file_path_elaborated_data = os.path.join(path_to_txt_elaborated_no_aei, folder_name_log, file_name_log_numpy)
        file_path_elaborated_data_aei = os.path.join(path_to_txt_elaborated_aei, folder_name_log, file_name_log_numpy_aei)

        tmp_no_aei = np.loadtxt(file_path_elaborated_data, delimiter=',')
        tmp_aei    = np.loadtxt(file_path_elaborated_data_aei, delimiter=',')

        m_no_aei = np.zeros([2, 3], dtype=float)
        m_aei    = np.zeros([2, 3], dtype=float)

        # store in m the relevant data:
        pyramid_level = 0
        num_rows = tmp_no_aei.shape[0]
        for i in range(1, num_rows):
            if tmp_no_aei[i, 0] == 0:
                m_no_aei[0, pyramid_level] = tmp_no_aei[i-1, 0]  # set step number at first or second pyramid level.
                m_no_aei[1, pyramid_level] = tmp_no_aei[i-1, 1]  # set cost function at first or second pyramid level.
                pyramid_level += 1
                if pyramid_level > 2:
                    assert ArithmeticError

        m_no_aei[0, pyramid_level] = tmp_no_aei[num_rows-1, 0]  # set step number at first at third pyramid level.
        m_no_aei[1, pyramid_level] = tmp_no_aei[num_rows-1, 1]  # set cost function at first at third pyramid level.

        # store in m_aei the relevant data:
        pyramid_level = 0
        num_rows = tmp_aei.shape[0]
        for i in range(1, num_rows):
            if tmp_aei[i, 0] == 0:
                m_aei[0, pyramid_level] = tmp_aei[i-1, 0]  # set step number at first or second pyramid level.
                m_aei[1, pyramid_level] = tmp_aei[i-1, 1]  # set cost function at first or second pyramid level.
                pyramid_level += 1
                if pyramid_level > 2:
                    assert ArithmeticError

        m_aei[0, pyramid_level] = tmp_aei[num_rows-1, 0]  # set step number at first at third pyramid level.
        m_aei[1, pyramid_level] = tmp_aei[num_rows-1, 1]  # set cost function at first at third pyramid level.

        # Save
        name_output_no_aei = 'step_cost_ref' + str(ref_index) + '_flo' + str(flo_index) + '.npy'
        name_output_aei    = 'step_cost_ref' + str(ref_index) + '_flo' + str(flo_index) + '_aei.npy'

        np.save(os.path.join(path_to_npy_array_no_aei, name_output_no_aei), m_no_aei)
        np.save(os.path.join(path_to_npy_array_aei, name_output_aei), m_aei)

print "step 2 data saved!"
