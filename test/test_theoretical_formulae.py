from random import uniform
import numpy as np

from transformations.s_disp import SDISP
from transformations.s_vf import SVF
import transformations.se2_g as se2_g
import transformations.se2_a as se2_a
from numpy.testing import assert_array_almost_equal


### confirm that exp and log are ones other inverse for matrices ###


def test_theory_inverse_exp_log():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    a = se2_g.se2_g(any_angle_1, any_tx_1, any_ty_1)
    ans = se2_a.exp(se2_g.log(a))
    assert_array_almost_equal(a.get, ans.get)


def test_theory_inverse_log_exp():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    a = se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1)
    ans = se2_g.log(se2_a.exp(a))
    assert_array_almost_equal(a.get, ans.get)


def test_theory_inverse_log_exp_input_not_in_quotient():
    any_angle_1 = 2 * np.pi + uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    a = se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1)
    ans = se2_g.log(se2_a.exp(a))
    assert_array_almost_equal(a.get, ans.get)


### Deformation and displacement Methods


def test_deformation_and_displacement_passages_from_matrices():
    # passing from deformation to displacement and vice versa using a matrix as
    # ground truth.
    domain = (16, 16)
    theta, tx, ty = np.pi / 6, 2.5, -2
    m_0 = se2_g.se2_g(theta, tx, ty)
    dm_0 = se2_g.log(m_0)

    print dm_0.get_matrix
    print m_0.get_matrix

    ### generate subsequent vector fields
    svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))

    # This provides the displacement since I am subtracting the id.
    disp_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))

    # This provides the deformation since I am not subtracting the id
    def_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix, affine=np.eye(4))

    # I generate the deformation from the displacement using deformation_from_displacement
    def_generated_0 = SDISP.deformation_from_displacement(disp_0)

    # I generate the displacement from the deformation using displacement_from_deformation
    disp_generated_0 = SDISP.displacement_from_deformation(def_0)

    assert_array_almost_equal(def_0.field, def_generated_0.field)
    assert_array_almost_equal(disp_0.field, disp_generated_0.field)


test_deformation_and_displacement_passages_from_matrices()
test_theory_inverse_exp_log()