import numpy as np
import copy
from scipy.misc import factorial as fact
from scipy.linalg import expm
import matplotlib.pyplot as plt

from scipy.integrate import ode

from utils.image import Image
from utils.fields import Field

from transformations.s_disp import SDISP
from transformations.s_vf import SVF

from visualizer.fields_at_the_window import see_field
from visualizer.fields_comparisons import see_2_fields_separate_and_overlay, \
    see_overlay_of_n_fields, see_n_fields_special


def exponential_beta(svf, algorithm='ss', s_i_o=3, input_num_steps=None, input_num_steps_rk4=7):

    v = copy.deepcopy(SVF.from_field(svf, header=svf.nib_image.get_header()))
    phi = copy.deepcopy(SDISP.generate_zero(shape=svf.shape, header=svf.nib_image.get_header()))

    # compose works only with images, and children!
    if algorithm == 'ss':

        norm = np.linalg.norm(svf.field, axis=svf.field.ndim - 1)
        max_norm = np.max(norm[:])

        if max_norm < 0:
            raise ValueError('Maximum norm is invalid.')
        if max_norm == 0:
            return phi

        if input_num_steps is None:
            # automatic computation of the optimal number of steps:
            pix_dims = np.asarray(svf.zooms)
            min_size = np.min(pix_dims[pix_dims > 0])
            num_steps = max(0, np.ceil(np.log2(max_norm / (min_size / 2))).astype('int')) + 2  # adaptative method.
        else:
            num_steps = input_num_steps

        # (1)
        init = 1 << num_steps  # equivalent to 1 * pow(2, num_steps)
        phi.field = svf.field / init

        # (2)
        for _ in range(0, num_steps):
            phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

        return phi

    if algorithm == 'ss_part_1':

        norm = np.linalg.norm(svf.field, axis=svf.field.ndim - 1)
        max_norm = np.max(norm[:])

        if max_norm < 0:
            raise ValueError('Maximum norm is invalid.')
        if max_norm == 0:
            return phi

        if input_num_steps is None:
            # automatic computation of the optimal number of steps:
            pix_dims = np.asarray(svf.zooms)
            min_size = np.min(pix_dims[pix_dims > 0])
            num_steps = max(0, np.ceil(np.log2(max_norm / (min_size / 2))).astype('int')) + 2  # adaptative method.
        else:
            num_steps = input_num_steps

        # (1)
        init = 1 << num_steps  # equivalent to 1 * pow(2, num_steps)
        phi.field = svf.field / init

        return phi

    if algorithm == 'ss_part_1_and_rk4':

        norm = np.linalg.norm(svf.field, axis=svf.field.ndim - 1)
        max_norm = np.max(norm[:])

        if max_norm < 0:
            raise ValueError('Maximum norm is invalid.')
        if max_norm == 0:
            return phi

        if input_num_steps is None:
            # automatic computation of the optimal number of steps:
            pix_dims = np.asarray(svf.zooms)
            min_size = np.min(pix_dims[pix_dims > 0])
            num_steps = max(0, np.ceil(np.log2(max_norm / (min_size / 2))).astype('int')) + 2  # adaptative method.
        else:
            num_steps = input_num_steps

        # (1)
        init = 1 << num_steps  # equivalent to 1 * pow(2, num_steps)
        v.field /= init

        # rk steps:

        h = 1.0 / input_num_steps_rk4

        for i in range(input_num_steps_rk4):

            phi_def = SDISP.deformation_from_displacement(phi)

            r_1 = SDISP.from_array(phi.field)
            r_2 = SDISP.from_array(phi.field)
            r_3 = SDISP.from_array(phi.field)
            r_4 = SDISP.from_array(phi.field)

            psi_1 = SDISP.from_array(phi.field)
            psi_2 = SDISP.from_array(phi.field)
            psi_3 = SDISP.from_array(phi.field)

            r_1.field   = h * SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field

            psi_1.field = phi.field + .5 * r_1.field
            psi_1_def  = SDISP.deformation_from_displacement(psi_1)
            r_2.field   = h  * SDISP.compose_with_deformation_field(v, psi_1_def, s_i_o=s_i_o).field

            psi_2.field = phi.field + .5 * r_2.field
            psi_2_def  = SDISP.deformation_from_displacement(psi_2)
            r_3.field   = h  * SDISP.compose_with_deformation_field(v, psi_2_def, s_i_o=s_i_o).field

            psi_3.field = phi.field + r_3.field
            psi_3_def  = SDISP.deformation_from_displacement(psi_3)
            r_4.field = h  * SDISP.compose_with_deformation_field(v, psi_3_def, s_i_o=s_i_o).field

            phi.field += (1. / 6) * (r_1.field + 2 * r_2.field + 2 * r_3.field + r_4.field)

        phi_partial = copy.deepcopy(phi)

        # (2)
        for _ in range(num_steps):
            phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

        return phi_partial, phi


random_seed = 0
if random_seed > 0:
        np.random.seed(random_seed)

shape = (20, 20, 20, 1, 3)
svf_0   = SVF.generate_random_smooth(shape=shape, sigma_gaussian_filter=2, sigma=5)
sdisp_0 = svf_0.exponential(algorithm='rk4', input_num_steps=10)

spline_interpolation_order = 3

sdisp_beta_v = exponential_beta(svf_0, algorithm='ss_part_1', input_num_steps=2)
sdisp_beta_phi_partial, sdisp_beta_phi = exponential_beta(svf_0, algorithm='ss_part_1_and_rk4', input_num_steps=2)

# See how far the sdisp_beta_phi, is away from the ground truth!
print (sdisp_beta_phi-sdisp_0).norm(passe_partout_size=5)

if 1:

    see_overlay_of_n_fields([svf_0, sdisp_0, sdisp_beta_phi, sdisp_beta_phi_partial],
                            fig_tag=1,
                            title_input='beta field',
                            input_color=['r', 'b', 'm', 'g', '0.5', 'm', 'r', 'b', 'm'])

    plt.show()
