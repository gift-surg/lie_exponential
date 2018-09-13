import numpy as np
import copy
from scipy.misc import factorial as fact
from scipy.linalg import expm
import matplotlib.pyplot as plt

from scipy.integrate import ode

from z_utils.image import Image
from z_utils.fields import Field

from transformations.s_disp import SDISP
from transformations.s_vf import SVF

from visualizer.fields_at_the_window import see_field
from visualizer.fields_comparisons import see_2_fields_separate_and_overlay, \
    see_overlay_of_n_fields, see_n_fields_special


def exponential_beta(svf, algorithm='ss', s_i_o=3, input_num_steps=None):

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

    elif algorithm == 'ss_part_1':

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

    elif algorithm == 'ss_part_1_and_affine':

        # automatic computation of the optimal number of steps:
        if input_num_steps is None:

            norm = np.linalg.norm(svf.field, axis=svf.field.ndim - 1)
            max_norm = np.max(norm[:])

            if max_norm < 0:
                raise ValueError('Maximum norm is invalid.')
            if max_norm == 0:
                return phi
            pix_dims = np.asarray(svf.zooms)
            min_size = np.min(pix_dims[pix_dims > 0])
            num_steps = max(0, np.ceil(np.log2(max_norm / (min_size / 2))).astype('int')) + 2
        else:
            num_steps = input_num_steps

        init = 1 << num_steps
        phi.field = svf.field / init

        # (1)
        jv = SDISP.compute_jacobian(phi)

        if svf.dim == 2:

            for x in range(phi.shape[0]):
                for y in range(phi.shape[1]):

                    A = jv.field[x, y, 0, 0, :].reshape([2, 2])
                    tr = phi.field[x, y, 0, 0, 0:2]
                    A_tr = A.dot(tr)
                    phi.field[x, y, 0, 0, :] = tr + 0.5 * A_tr  # + 1/6. * A.dot(A_v)

        # (2)
        #for _ in range(num_steps):
        #    phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

    else:
        print 'error!'

    return phi


random_seed = 50
if random_seed > 0:
        np.random.seed(random_seed)

shape = (20, 20, 1, 1, 2)
svf_0   = SVF.generate_random_smooth(shape=shape, sigma_gaussian_filter=1.2, sigma=4)
sdisp_0 = svf_0.exponential(algorithm='rk4', input_num_steps=10)

spline_interpolation_order = 3

sdisp_beta_v = exponential_beta(svf_0, algorithm='ss_part_1', input_num_steps=2)
sdisp_beta_phi = exponential_beta(svf_0, algorithm='ss_part_1_and_affine', input_num_steps=2)

if 1:

    see_overlay_of_n_fields([svf_0, sdisp_0, sdisp_beta_v, sdisp_beta_phi],
                            fig_tag=1,
                            title_input='beta field',
                            input_color=['r', 'b', 'k', 'g', 'b', 'm', 'r', 'b', 'm'])

    plt.show()
