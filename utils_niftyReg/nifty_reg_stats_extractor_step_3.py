import numpy as np
import os


from path_manager import path_root

path_to_npy_array_no_aei = os.path.join(path_root, 'step_2/no_aei')
path_to_npy_array_aei    = os.path.join(path_root, 'step_2/aei')

path_to_npy_final_arrays = os.path.join(path_root, 'step_3')

num_scans = 35

num_registrations = 35 * 34 / 2

# init final data structures:
global_no_aei = np.zeros([2, 3, num_registrations])
global_aei    = np.zeros([2, 3, num_registrations])

list_index_ref = range(num_scans)

index_registration = 0

for ref_index in list_index_ref:

    list_index_flo = list_index_ref[(ref_index+1):]

    for flo_index in list_index_flo:

        # Load matrices:

        name_output_no_aei = 'step_cost_ref' + str(ref_index) + '_flo' + str(flo_index) + '.npy'
        name_output_aei    = 'step_cost_ref' + str(ref_index) + '_flo' + str(flo_index) + '_aei.npy'

        m_no_aei = np.load(os.path.join(path_to_npy_array_no_aei, name_output_no_aei))
        m_aei    = np.load(os.path.join(path_to_npy_array_aei, name_output_aei))

        # Add values to the new matrices
        global_no_aei[..., index_registration] = m_no_aei[...]
        global_aei[..., index_registration] = m_aei[...]

        index_registration += 1

print
print index_registration

# save the created matrices:
np.save(os.path.join(path_to_npy_final_arrays, 'steps_and_costs.npy'), global_no_aei)
np.save(os.path.join(path_to_npy_final_arrays, 'steps_and_costs_aei.npy'), global_aei)

print "step 3 data saved!"
