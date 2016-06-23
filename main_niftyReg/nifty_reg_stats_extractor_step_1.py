import os

"""
Elaboration logs from NiftyReg.
"""

num_data_set = 1  # 0: original data from NiftyReg

if num_data_set == 0:

    path_to_txt_original_data_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data/f3d2_no_aei'
    path_to_copy_elaborated_data_step_1_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/log_elaboration/step_1/no_aei'

    path_to_txt_original_data_aei      = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data/f3d2_aei'
    path_to_copy_elaborated_data_step_1_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/log_elaboration/step_1/aei'

elif num_data_set == 1:

    path_to_txt_original_data_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data_maxit/f3d2_no_aei'
    path_to_copy_elaborated_data_step_1_no_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/log_elaboration_maxit/step_1/no_aei'

    path_to_txt_original_data_aei      = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data_maxit/f3d2_aei'
    path_to_copy_elaborated_data_step_1_aei = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/log_elaboration_maxit/step_1/aei'

else:
    pass


num_scans = 35
list_index_ref = range(num_scans)

for ref_index in list_index_ref:

    list_index_flo = list_index_ref[(ref_index+1):]

    for flo_index in list_index_flo:

        #### Elaborate data for no aei ####
        # extract the log from niftyReg:

        folder_name_log = str(ref_index) + '_f3d2'
        file_name_log_txt = 'log_ref' + str(ref_index) + '_flo' + str(flo_index) + '.txt'

        folder_original_txt = os.path.join(path_to_txt_original_data_no_aei, folder_name_log)
        file_path_elaborated_data_txt = os.path.join(path_to_copy_elaborated_data_step_1_no_aei, folder_name_log, file_name_log_txt)

        os.system(' mkdir -p ' + os.path.join(path_to_copy_elaborated_data_step_1_no_aei, folder_name_log))

        log_index = flo_index - ref_index

        if num_data_set == 0:

            partial_name_log = "reg_f3d2_" + str(ref_index) + ".o17*." + str(log_index)

        if num_data_set == 1:

            partial_name_log = "reg_f3d2_" + str(ref_index) + ".o19*." + str(log_index)

        partial_path_log = os.path.join(folder_original_txt, partial_name_log)

        msg = " sed -l 's/'Initial'/'[0]'/g' " + partial_path_log + " | grep 'objective function' > " + file_path_elaborated_data_txt

        print msg

        os.system(msg)

        file_name_log_numpy = 'log_ref' + str(ref_index) + '_flo' + str(flo_index) + '_for_numpy.txt'
        file_path_elaborated_data_numpy = os.path.join(path_to_copy_elaborated_data_step_1_no_aei, folder_name_log, file_name_log_numpy)

        with open(file_path_elaborated_data_txt, 'r') as file_log_elaborated:
            data = file_log_elaborated.read()

            data = data.replace(' - (wJAC)0', ', 0 ')
            data = data.replace('[NiftyReg F3D2] [', '')
            data = data.replace('] Current objective function:', ', ')
            data = data.replace('] objective function:', ', ')
            data = data.replace('= (wSIM)', ', ')
            data = data.replace(' - (wBE)', ', ')
            data = data.replace(' - (wLE)', ', ')
            data = data.replace('[+', ', ')
            data = data.replace('mm]', '')

            file_log_elaborated.close()

        with open(file_path_elaborated_data_numpy, 'w') as file_txt_for_numpy:

            file_txt_for_numpy.write(data)
            file_txt_for_numpy.close()

        #### Elaborate data for aei ####
        # extract the log from niftyReg:

        folder_name_log = str(ref_index) + '_f3d2'
        file_name_log_txt = 'log_ref' + str(ref_index) + '_flo' + str(flo_index) + '.txt'

        folder_original_txt = os.path.join(path_to_txt_original_data_aei, folder_name_log)
        file_path_elaborated_data_txt = os.path.join(path_to_copy_elaborated_data_step_1_aei, folder_name_log, file_name_log_txt)

        os.system(' mkdir -p ' + os.path.join(path_to_copy_elaborated_data_step_1_aei, folder_name_log))

        log_index = flo_index - ref_index

        if num_data_set == 0:

            partial_name_log = "reg_f3d2_aei_" + str(ref_index) + ".o17*." + str(log_index)

        if num_data_set == 1:

            partial_name_log = "reg_f3d2_aei_" + str(ref_index) + ".o19*." + str(log_index)

        partial_path_log = os.path.join(folder_original_txt, partial_name_log)

        msg = " sed -l 's/'Initial'/'[0]'/g' " + partial_path_log + " | grep 'objective function' > " + file_path_elaborated_data_txt

        print msg

        os.system(msg)

        file_name_log_numpy = 'log_ref' + str(ref_index) + '_flo' + str(flo_index) + '_aei_for_numpy.txt'
        file_path_elaborated_data_numpy = os.path.join(path_to_copy_elaborated_data_step_1_aei, folder_name_log, file_name_log_numpy)

        with open(file_path_elaborated_data_txt, 'r') as file_log_elaborated:
            data = file_log_elaborated.read()

            data = data.replace(' - (wJAC)0', ', 0 ')
            data = data.replace('[NiftyReg F3D2] [', '')
            data = data.replace('] Current objective function:', ', ')
            data = data.replace('] objective function:', ', ')
            data = data.replace('= (wSIM)', ', ')
            data = data.replace(' - (wBE)', ', ')
            data = data.replace(' - (wLE)', ', ')
            data = data.replace('[+', ', ')
            data = data.replace('mm]', '')

            file_log_elaborated.close()

        with open(file_path_elaborated_data_numpy, 'w') as file_txt_for_numpy:

            file_txt_for_numpy.write(data)
            file_txt_for_numpy.close()

print "step 1 data saved!"
