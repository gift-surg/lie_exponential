"""
Aimed to embed the niftiReg methods for all the images image.
Set the path to the build reg-apps of niftyReg

514 >> /Users/sebastiano/Workspace/niftiREG/build/reg-apps/reg_aladin -ref ../AD_0_baseline.nii -flo ../CTL_0_baseline.nii -aff matrix.txt -res affine_refAD0_floCTL0.nii.gz -lp 1 -speeeeed
515 >> /Users/sebastiano/Workspace/niftiREG/build/reg-apps/reg_f3d -ref ../AD_0_baseline.nii -flo affine_refAD0_floCTL0.nii.gz -cpp cpp_refAD0_floCTL0.nii.gz -res res_refAD0_floCTL0.nii.gz -vel -lp 1 -maxit 10
516 >> /Users/sebastiano/Workspace/niftiREG/build/reg-apps/reg_transform -ref ../AD_0_baseline.nii -flow cpp_refAD0_floCTL0.nii.gz flow_refAD0_floCTL0.nii.gz

"""

import os

from path_manager import original_image_skull_less_fp, original_common_space_fp, \
    non_rigid_alignment_aei_fp, flows_aei_fp, deformations_aei_fp, displacements_aei_fp, grids_aei_fp, \
    non_rigid_alignment_o_fp, flows_o_fp, deformations_o_fp, displacements_o_fp, grids_o_fp
from z_utils.nifti_warp import nifti

######### Controller ############

align_in_common_space_process = False
speed_rigid                   = False

non_rigid_registration_process = True
speed_non_rigid                = False

get_flow_fields   = True
get_displacements = True
get_deformations  = True
get_grids         = True


original_nifty_reg = True


######### Set path to store the output according to NiftyReg version ############
# Set folder results for original NiftyReg or the EAI version

if original_nifty_reg:  # folder path for the data when niftyReg is NOT modified with EAI

    non_rigid_alignment_fp = non_rigid_alignment_o_fp
    flows_fp               = flows_o_fp
    deformations_fp        = deformations_o_fp
    displacements_fp       = displacements_o_fp
    grids_fp               = grids_o_fp

else:  # set folder paths for the data when niftyReg is modified with EAI

    non_rigid_alignment_fp = non_rigid_alignment_aei_fp
    flows_fp               = flows_aei_fp
    deformations_fp        = deformations_aei_fp
    displacements_fp       = displacements_aei_fp
    grids_fp               = grids_aei_fp


############################ COMPUTATIONS  #############################

if align_in_common_space_process:

    # align all AD with the template:
    template = os.path.join(original_image_skull_less_fp, 'AAA_template_sl.nii')

    for kind in ['baseline', 'followup']:

        for subj in range(10):

            image_1  = os.path.join(original_image_skull_less_fp, 'AD_' + str(subj) + '_' + kind + '_sl.nii')

            output_1      = os.path.join(original_common_space_fp, 'AD_' + str(subj) + '_' + kind + '_sl.nii')
            output_matrix = os.path.join(original_common_space_fp,
                                         'z_matrix_transform_AD_' + str(subj) + '_' + kind + '_in_template_sl.txt')

            cmd, msg = nifty.rigid_registration(niftyReg_path,
                                                template,
                                                image_1,
                                                output_1,
                                                output_matrix,
                                                open_mp_threads=4,
                                                speed=speed_rigid)

            print
            print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '
            print 'subject: ' + str(subj) + ' ' + str(kind) + ' rigid alignment to the template, 10 subjects'
            print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '
            print
            print msg
            print

            print os.system(cmd)

            print '\n\n'


if non_rigid_registration_process:

    for subj in range(10):

        baseline = os.path.join(original_common_space_fp, 'AD_' + str(subj) + '_baseline_sl.nii')
        followup = os.path.join(original_common_space_fp, 'AD_' + str(subj) + '_followup_sl.nii')

        output_im  = os.path.join(non_rigid_alignment_fp,
                                  'AD_' + str(subj) + '_non_rigid_aligned_baseline_on_followup.nii.gz')
        output_cpp = os.path.join(non_rigid_alignment_fp,
                                  'cpp_AD_' + str(subj) + '_baseline_on_followup.nii.gz')

        cmd, msg = nifty.non_rigid_registration(niftyReg_path,
                                                baseline,  # fixed
                                                followup,  # moving
                                                output_im,  # output image
                                                output_cpp,  # output cpp
                                                open_mp_threads=4,
                                                speed=False)
        print
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '
        print 'subject: ' + str(subj) + ' non rigid alignment to the template, 10 subjects'
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '
        print
        print msg
        print

        print os.system(cmd)

        print '\n\n'


if get_flow_fields:

    for subj in range(10):

        fixed = os.path.join(non_rigid_alignment_fp,
                             'AD_' + str(subj) + '_non_rigid_aligned_baseline_on_followup.nii.gz')

        cpp = os.path.join(non_rigid_alignment_fp,
                           'cpp_AD_' + str(subj) + '_baseline_on_followup.nii.gz')

        output = os.path.join(flows_fp,
                              'flow_AD_' + str(subj) + '_.nii.gz')

        cmd, msg = nifty.get_flow_field(niftyReg_path,
                                        fixed,  # input reference
                                        cpp,    # input cpp
                                        output)  # output

        print
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print 'subject: ' + str(subj) + ' flow creations, 10 subjects'
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print
        print msg
        print

        print os.system(cmd)

        print '\n\n'


if get_displacements:

    for subj in range(10):

        fixed = os.path.join(non_rigid_alignment_fp,
                             'AD_' + str(subj) + '_non_rigid_aligned_baseline_on_followup.nii.gz')

        cpp = os.path.join(non_rigid_alignment_fp,
                           'cpp_AD_' + str(subj) + '_baseline_on_followup.nii.gz')

        output = os.path.join(displacements_fp,
                              'displacement_AD_' + str(subj) + '_.nii.gz')

        cmd, msg = nifty.get_displacement_field(niftyReg_path,
                                                fixed,
                                                cpp,
                                                output)

        print
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print 'subject: ' + str(subj) + ' displacement creations, 10 subjects'
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print
        print msg
        print

        print os.system(cmd)

        print '\n\n'

if get_deformations:

    for subj in range(10):

        fixed = os.path.join(non_rigid_alignment_fp,
                             'AD_' + str(subj) + '_non_rigid_aligned_baseline_on_followup.nii.gz')

        cpp = os.path.join(non_rigid_alignment_fp,
                           'cpp_AD_' + str(subj) + '_baseline_on_followup.nii.gz')

        output = os.path.join(deformations_fp,
                              'deformation_AD_' + str(subj) + '_.nii.gz')

        cmd, msg = nifty.get_deformation_field(niftyReg_path,
                                               fixed,
                                               cpp,
                                               output)

        print
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print 'subject: ' + str(subj) + ' deformation creations, 10 subjects'
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print
        print msg
        print

        print os.system(cmd)

        print '\n\n'


if get_grids:
    pass
