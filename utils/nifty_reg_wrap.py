"""
Path based methods to use Nifty_reg with os.system(command)
"""
import os


def rigid_registration(path_tool,
                       path_input_fixed,
                       path_input_moving,
                       path_output_img,
                       path_output_matrix_transform,
                       open_mp_threads=4,
                       speed=False):

    path_reg_tool = os.path.join(path_tool, 'reg_aladin')
    command = path_reg_tool + \
              ' -ref ' + path_input_fixed + \
              ' -flo ' + path_input_moving + \
              ' -res ' + path_output_img + \
              ' -aff ' + path_output_matrix_transform + \
              ' -omp ' + str(open_mp_threads)

    output_msg = 'reg_aladin ' + \
                 ' -ref ' + os.path.basename(path_input_fixed) + \
                 ' -flo ' + os.path.basename(path_input_moving) + \
                 ' -res ' + os.path.basename(path_output_img) + \
                 ' -aff ' + os.path.basename(path_output_matrix_transform) + \
                 ' -omp ' + str(open_mp_threads)
    if speed:
        command += ' -lp 1 -speeeeed'
        output_msg += ' -lp 1 -speeeeed'

    return command, output_msg


def non_rigid_registration(path_tool,
                           path_input_fixed,
                           path_input_moving,
                           path_output_img,
                           path_output_control_point_grid,
                           open_mp_threads=4,
                           speed=False):

    path_reg_tool = os.path.join(path_tool, 'reg_f3d')
    command = path_reg_tool + \
              ' -ref ' + path_input_fixed + \
              ' -flo ' + path_input_moving + \
              ' -cpp ' + path_output_control_point_grid + \
              ' -res ' + path_output_img +  \
              ' -vel ' + \
              ' -omp ' + str(open_mp_threads)

    # cpp is a nii image
    output_msg = 'reg_f3d ' + \
                 ' -ref ' + os.path.basename(path_input_fixed) + \
                 ' -flo ' + os.path.basename(path_input_moving) + \
                 ' -cpp ' + os.path.basename(path_output_control_point_grid) + \
                 ' -res ' + os.path.basename(path_output_img) + \
                 ' -vel ' + \
                 ' -omp ' + str(open_mp_threads)

    if speed:
        command += ' -lp 3 -maxit 10'
        output_msg += ' -lp 3 -maxit 10'

    return command, output_msg


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


def get_deformation_field(path_tool,
                          path_input_fixed,
                          path_input_control_point_grid,
                          path_output_def):

    path_reg_tool = os.path.join(path_tool, 'reg_transform')
    command = path_reg_tool + \
              ' -ref ' + path_input_fixed + \
              ' -def ' + path_input_control_point_grid + \
              ' ' + path_output_def

    output_msg = 'reg_transform ' + \
                 ' -ref ' + os.path.basename(path_input_fixed) + \
                 ' -def ' + os.path.basename(path_input_control_point_grid) + \
                 ' ' + os.path.basename(path_output_def)

    return command, output_msg


def get_displacement_field(path_tool,
                           path_input_fixed,
                           path_input_control_point_grid,
                           path_output_disp):

    path_reg_tool = os.path.join(path_tool, 'reg_transform')
    command = path_reg_tool + \
              ' -ref ' + path_input_fixed + \
              ' -disp ' + path_input_control_point_grid + \
              ' ' + path_output_disp

    output_msg = 'reg_transform ' + \
                 ' -ref ' + os.path.basename(path_input_fixed) + \
                 ' -disp ' + os.path.basename(path_input_control_point_grid) + \
                 ' ' + os.path.basename(path_output_disp)

    return command, output_msg


def get_non_rigid_grid(path_tool,
                       path_input_fixed,
                       path_input_moving,
                       path_input_transformation,
                       path_output_grid):

    path_reg_tool = os.path.join(path_tool, 'reg_resample')
    command = path_reg_tool + \
              ' -ref ' + path_input_fixed + \
              ' -flo ' + path_input_moving + \
              ' -trans ' + path_input_transformation + \
              ' -blank ' + path_output_grid

    output_msg = 'reg_sample ' + \
                 ' -ref ' + os.path.basename(path_input_fixed) + \
                 ' -flo ' + os.path.basename(path_input_moving) + \
                 ' -trans ' + os.path.basename(path_input_transformation) + \
                 ' -blank ' + os.path.basename(path_output_grid)

    return command, output_msg
