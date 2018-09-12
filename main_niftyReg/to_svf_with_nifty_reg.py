import os
from utils.path_manager import niftyReg_path

"""
Notes on how to OBTAIN SVFs from NiftyReg

SVF are obtained from NiftyReg as follows reg_f3d with command -vel returns the corresponding cpp grid as the
control point grid we are interested in.
The dense vector field that corresponds to the given gpp grid is then provided with -flow and it
is obtained in 'deformation coordinates' (Eulerian coordinate system).
To have it in displacement coordinate system (Lagrangian coordinate system) for our elaboration we need to
subtract the identity with python (not with - disp in niftyReg, otherwise it will be exponentiated again).
"""


def get_flow_field(path_tool,
                   path_input_fixed,
                   path_input_control_point_grid,
                   path_output_flow):

    path_reg_tool = os.path.join(path_tool, 'reg_transform')
    command = path_reg_tool + \
              ' -ref ' + path_input_fixed + \
              ' -flow ' + path_input_control_point_grid + \
              ' ' + path_output_flow

    output_msg = 'reg_transform ' + \
                 ' -ref ' + os.path.basename(path_input_fixed) + \
                 ' -flow ' + os.path.basename(path_input_control_point_grid) + \
                 ' ' + os.path.basename(path_output_flow)

    return command, output_msg


ref_image = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data_maxit/neuromorphometrics_data/cropped/cropped_1000_3.nii.gz '
cpp_path = '/Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/Data/f3d2_aei/0_f3d2/cpp2_ref0_flo1.nii.gz'
flow_output = 'Users/sebastiano/Documents/UCL/z_data/neuromorphometrics/svf/flows/svf_def_ref0_flo1.nii.gz'


cmd, msg = get_flow_field(niftyReg_path, ref_image, cpp_path, flow_output)

print msg
print os.system(cmd)
