import os

"""
Notes on how to OBTAIN SVFs from NiftyReg

SVF are obtained from NiftyReg as follows reg_f3d with command -vel returns the corresponding cpp grid as the
control point grid we are interested in.
The dense vector field that corresponds to the given gpp grid is then provided with -flow and it
is obtained in 'deformation coordinates' (Eulerian coordinate system).
To have it in displacement coordinate system (Lagrangian coordinate system) for our elaboration we need to
subtract the identity with python (not with - disp in niftyReg, otherwise it will be exponentiated again).
"""


def get_flow_field(path_input_fixed,
                   path_input_control_point_grid,
                   path_output_flow):

    command = 'reg_transform ' + \
              ' -ref ' + path_input_fixed + \
              ' -flow ' + path_input_control_point_grid + \
              ' ' + path_output_flow

    output_msg = 'reg_transform ' + \
                 ' -ref ' + os.path.basename(path_input_fixed) + \
                 ' -flow ' + os.path.basename(path_input_control_point_grid) + \
                 ' ' + os.path.basename(path_output_flow)

    return command, output_msg
