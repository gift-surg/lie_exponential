import numpy as np

import transformations.se2_a as se2_a
import transformations.se2_g as se2_g
from utils.aux_functions import bch_right_jacobian

from utils.resampler import DisplacementFieldComposer
import copy
import transformations.s_vf as SVF

"""
NOTE
r and dr are BOTH in the lie algebra (dr stands for delta r, the little one).
Here we use the close form for each formula! (to be compared with bch_M)
passing through the matrix form just for the parallel transport case.

Input are elements of the Lie algebra, output is in the same
Lie algebra.

Only parallel transport refers to matrices for computations.
(waiting for close form)
"""


"""
Log-composition for se2_a
"""


def log_composition_se2_a(r, dr, kind='ground'):

    if kind == 'ground':
        """
        bch_ground(r, dr)
        bch close formula of r and dr
        """
        if not (isinstance(r, se2_a.se2_a) and isinstance(dr, se2_a.se2_a)):
            raise TypeError('warning: wrong input data format, 2 se2_a elements expected.')

        ans = se2_g.log(se2_a.exp(r)*se2_a.exp(dr))

    elif kind == 'bch0':
        """
        bch_0(r, dr)

        bch zero 'order' approximation
        """
        if not (isinstance(r, se2_a.se2_a) and isinstance(dr, se2_a.se2_a)):
            raise TypeError('warning: wrong input data format, 2 se2_a elements expected.')

        ans = r + dr

    elif kind == 'bch1':
        """
        bch_1(r, dr)

        bch first 'order' approximation
        """
        if not (isinstance(r, se2_a.se2_a) and isinstance(dr, se2_a.se2_a)):
            raise TypeError('warning: wrong input data format, 2 se2_a elements expected.')

        ans = r + dr + se2_a.scalarpr(0.5, se2_a.lie_bracket(r, dr))

    elif kind == 'bch2':
        """
        bch_2(r, dr)

        bch first 'order' approximation
        """
        if not (isinstance(r, se2_a.se2_a) and isinstance(dr, se2_a.se2_a)):
            raise TypeError('warning: wrong input data format, 2 se2_a elements expected.')

        ans = r + dr + se2_a.scalarpr(0.5, se2_a.lie_bracket(r, dr)) + \
            se2_a.scalarpr(0.08333333333333333, se2_a.lie_bracket(r, se2_a.lie_bracket(r, dr))
                              + se2_a.lie_bracket(dr, se2_a.lie_bracket(dr, r)))

    elif kind == 'taylor':
        """
        """
        if not (isinstance(r, se2_a.se2_a) and isinstance(dr, se2_a.se2_a)):
            raise TypeError('warning: wrong input data format, 2 se2_a elements expected.')

        v = np.array(r.get)
        dv = np.array(dr.get)
        J = bch_right_jacobian(v)

        vect_ans = v + J.dot(dv)

        ans = se2_a.se2_a(vect_ans[0], vect_ans[1], vect_ans[2])

    elif kind == 'pt':
        """
        bch_pt(r,dr)

        bch using parallel transport. Elements are considered as matrices to respect the original formula
        No closed formula yet... Passing through matrices
            bch_pt = V + dV_parallel = V + \exp(V/2)*\exp(dV)*\exp(-V/2) - Id

        computations are done as matrices, but the result is back an element in se2
        """
        if not (isinstance(r, se2_a.se2_a) and isinstance(dr, se2_a.se2_a)):
            raise TypeError('warning: wrong input data format, 2 se2_a elements expected.')

        #Id = np.identity(3)
        null_elem = se2_a.se2_a(0, 0, 0)

        #m = r.get_matrix
        exp_half_r = se2_a.exp(se2_a.scalarpr(0.5, r))
        exp_dr = se2_a.exp(dr)
        exp_half_neg_r = se2_a.exp(se2_a.scalarpr(0.5, null_elem - r))

        dr_pt = (exp_half_r*exp_dr*exp_half_neg_r).get_matrix - np.identity(3)

        ans = se2_a.matrix2se2_a(r.get_matrix + dr_pt, eat_em_all=True)

    else:
        raise TypeError('warning: wrong input data, composition must be one of bch0, bch1, bch2, taylor, pt, sym.')

    return ans


"""
Alternative for the parallel transport

if not (isinstance(r, se2_a.se2_a) and isinstance(dr, se2_a.se2_a)):
    raise TypeError('warning: wrong input data format, 2 se2_a elements expected.')

Id = np.identity(3)
null_elem = se2_a.se2_a(0, 0, 0)

exp_half_r = se2_a.exp(se2_a.scalarpr(0.5, r))
exp_dr = se2_a.exp(dr)
exp_half_neg_r = se2_a.exp(se2_a.scalarpr(0.5, r)) # null_elem - r

matrix_dr_pt = (exp_half_r*exp_dr*exp_half_neg_r).get_matrix - Id

return bch_1(r, se2_a.matrix2se2_a(matrix_dr_pt,eat_em_all=True))
"""


def log_composition_matrix_se2_a(r, dr, kind='ground'):
    """
    :param r:
    :param dr:
    :param composition:
    :return:

    Here only with matrices!
    """
    r_m = r.get_matrix
    dr_m = dr.get_matrix

    if kind == 'ground':
        # compute ground truth close formulas
        ans = se2_g.log_for_matrices(se2_a.exp_for_matrices(r_m)*se2_a.exp_for_matrices(dr_m))
    elif kind == 'bch0':
        ans = r_m + dr_m
    elif kind == 'bch1':
        ans = r_m + dr_m + 0.5 * se2_a.bracket_for_matrices(r, dr)
    elif kind == 'bch2':
        ans = r_m + dr_m + 0.5 * se2_a.bracket_for_matrices(r, dr) + \
            (1/12.0)*(se2_a.lie_multi_bracket_for_matrices([r, r, dr]) +
            se2_a.lie_multi_bracket_for_matrices([dr, dr, r]))
    elif kind == 'taylor':
        ans = []
    elif kind == 'pt':
        id = np.identity(3)

        exp_half_r = 0.5 * se2_a.exp_for_matrices(r)
        exp_dr = se2_a.exp_for_matrices(dr)
        exp_half_neg_r = -0.5 * se2_a.exp_for_matrices(r)

        ans = exp_half_r*exp_dr*exp_half_neg_r - id  # projection operator needed

    else:
        raise TypeError('warning: wrong input data, composition must be one of bch0, bch1, bch2, taylor, pt, sym.')

    return ans


"""
Log-composition for svf
Elements are defined in the Lie group as exponential of some vector.
This is not a proper log-composition, since to have a ground truth we remain in the lie group.

def_  elements in the Lie group (transformations)
svf_  elements in the Lie algebra (vector fields)

The results is the Lie group, where we perform an improper norm (norm that do not relies on the Lie group structure but
only on the fact that the object we are dealing with are matrices, and so have many norms).
"""


def log_composition_svf(svf_im0, svf_im1, kind='ground'):
    """
    From two stationary velocity fields in the tangent space he returns their composition in the Lie group.
    No log is computed on the resulting def (deformation field)
    :param svf_im0:
    :param svf_im1:
    :param kind: 'ground'
        ground truth for the composition, it returns the value \exp(svf_im0)\circ \exp(svf_im1)

    kind: 'bch0'
        composition using the bch 0: it returns
        \exp(svf_im0)\circ \exp(svf_im1) \simeq \exp(svf_im0) + \exp(svf_im1)
    kind: 'bch1'
        composition using the bch 1: it returns
        \exp(svf_im0)\circ \exp(svf_im1) \simeq \exp(svf_im0) + \exp(svf_im1) + .5 [\exp(svf_im0), \exp(svf_im1)]
    kind: 'pt'
        composition using parallel transport: it returns
        \exp(svf_im0)\circ \exp(svf_im1) \simeq ...
    kind: 'pt_warp'
        composition using parallel transport warped: it returns
        \exp(svf_im0)\circ \exp(svf_im1) \simeq ...

    :return: exp(svf_im0)oexp(svf_im1). It is in the Lie group and not in the Lie algebra,
    since the computation of the error is in the Lie group.
    """
    if kind == 'ground':
        def_im0 = svf_im0.exponential()
        def_im1 = svf_im1.exponential()
        dfc = DisplacementFieldComposer()
        def_result = dfc.compose(def_im0, def_im1)

    elif kind == 'bch0':
        svf_bch0 = copy.deepcopy(svf_im0)
        svf_bch0.field.data += svf_im1.field.data
        def_result = svf_bch0.exponential()

    elif kind == 'bch1':
        vel_bch1 = copy.deepcopy(svf_im0)
        vel_bch1.field.data += svf_im1.field.data
        vel_bch1.field.data += 0.5*SVF.SVF.lie_bracket(svf_im0, svf_im1).field.data
        def_result = vel_bch1.exponential()

    elif kind == 'bch1.5':
        vel_bch2 = copy.deepcopy(svf_im0)
        vel_bch2.field.data += svf_im1.field.data
        vel_bch2.field.data += 0.5*SVF.SVF.lie_bracket(svf_im0, svf_im1).field.data
        vel_bch2.field.data += (1/12.0)*SVF.SVF.lie_bracket(svf_im0, SVF.SVF.lie_bracket(svf_im0, svf_im1)).field.data

        def_result = vel_bch2.exponential()

    elif kind == 'bch2':
        vel_bch2 = copy.deepcopy(svf_im0)
        vel_bch2.field.data += svf_im1.field.data
        vel_bch2.field.data += 0.5*SVF.SVF.lie_bracket(svf_im0, svf_im1).field.data
        vel_bch2.field.data += (1/12.0)*(SVF.SVF.lie_bracket(svf_im0, SVF.SVF.lie_bracket(svf_im0, svf_im1)).field.data
                                         +
                                         SVF.SVF.lie_bracket(svf_im1, SVF.SVF.lie_bracket(svf_im1, svf_im0)).field.data
                                        )
        def_result = vel_bch2.exponential()

    elif kind == 'pt':
        dfc = DisplacementFieldComposer()
        tmp_vel = copy.deepcopy(svf_im0)
        tmp_vel.field.data /= 2

        tmp_def_a = tmp_vel.exponential()
        tmp_vel.field.data = -tmp_vel.field.data
        tmp_def_b = tmp_vel.exponential()
        tmp_def_c = dfc.compose(dfc.compose(tmp_def_a, svf_im1.exponential()), tmp_def_b)

        vel_pt = copy.deepcopy(svf_im0)
        vel_pt.field.data += tmp_def_c.data
        def_result = vel_pt.exponential()

    elif kind == 'pt_alternate':
        dfc = DisplacementFieldComposer()
        tmp_vel = copy.deepcopy(svf_im0)
        tmp_vel.field.data /= 2

        tmp_def_a = tmp_vel.exponential()
        tmp_vel.field.data = -tmp_vel.field.data
        tmp_def_b = tmp_vel.exponentiatial()
        tmp_def_c = dfc.compose(dfc.compose(tmp_def_a, svf_im1.exponential()), tmp_def_b)

        vel_pt = copy.deepcopy(svf_im0)
        vel_pt.field.data += tmp_def_c.data
        def_result = vel_pt.exponential()

    else:
        raise TypeError('warning: wrong input data for the kind of composition.')

    return def_result


