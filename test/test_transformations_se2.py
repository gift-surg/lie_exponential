import numpy as np
from random import uniform
import scipy.linalg as lin
from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_almost_equal, assert_array_equal

from utils.aux_functions import mod_pipi

import transformations.se2_a as se2_a
import transformations.se2_g as se2_g

'''
Test for the module translation
'''

meaningful_decimals = 6

'''
Test for the module se2_a, Lie algebra of rotations and translations
quotient over the equivalence relation given by exp
'''

''' test class object se2_a. '''


def test_init_se2_a_random_input():
    any_angle_in_pi_pi = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx = uniform(-10, 10)
    any_ty = uniform(-10, 10)
    assert_array_equal(se2_a.se2_a(any_angle_in_pi_pi, any_tx, any_ty).get, [any_angle_in_pi_pi, any_tx, any_ty])


def test_se2_a_get_matrix():
    any_angle_in_pi_pi = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx = uniform(-10, 10)
    any_ty = uniform(-10, 10)
    m = se2_a.se2_a(any_angle_in_pi_pi, any_tx, any_ty).get_matrix
    assert m[0, 0] == m[1, 1] == 0 \
        and -m[0, 1] == m[1, 0] == any_angle_in_pi_pi \
        and m[2, 2] == 0 and m[0, 2] == any_tx and m[1, 2] == any_ty


def test_se2_a_get_matrix_det():
    any_angle_in_pi_pi = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx = uniform(-10, 10)
    any_ty = uniform(-10, 10)
    m = se2_a.se2_a(any_angle_in_pi_pi, any_tx, any_ty).get_matrix
    given_output = np.linalg.det(m)
    expected_output = 0.0
    global meaningful_decimals
    assert round(abs(given_output-expected_output), meaningful_decimals) == 0.0


def test_se2_a_quotient_projection_greater_than_pi():
    theta_in = np.pi + 0.3
    tx_in = 3
    ty_in = 4
    fact = mod_pipi(theta_in)/theta_in
    theta_out = -np.pi + 0.3
    tx_out = tx_in * fact
    ty_out = ty_in * fact
    assert_array_equal(se2_a.se2_a(theta_in, tx_in, ty_in).get, se2_a.se2_a(theta_out, tx_out, ty_out).get)


def test_se2_a_quotient_peojection_smaller_than_pi():
    theta_in = np.pi - 0.3
    tx_in = 3
    ty_in = 4
    assert_array_equal(se2_a.se2_a(theta_in, tx_in, ty_in).get, [theta_in, tx_in, ty_in])


def test_se2_a_add_0_translation():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx_1 = 0
    ty_1 = 0
    any_angle_2 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx_2 = 0
    ty_2 = 0
    expected_output_angle = mod_pipi(any_angle_1 + any_angle_2)
    expected_output_tx = 0
    expected_output_ty = 0
    expected_output = [expected_output_angle, expected_output_tx, expected_output_ty]
    given_output = (se2_a.se2_a(any_angle_1, tx_1, ty_1) + se2_a.se2_a(any_angle_2, tx_2, ty_2)).get
    assert_array_equal(given_output, expected_output)


def test_se2_a_add_0_rotation():
    angle_1 = 0
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    angle_2 = 0
    any_tx_2 = uniform(-10, 10)
    any_ty_2 = uniform(-10, 10)
    expected_output_angle = 0
    expected_output_tx = any_tx_1 + any_tx_2
    expected_output_ty = any_ty_1 + any_ty_2
    expected_output = [expected_output_angle, expected_output_tx, expected_output_ty]
    given_output = (se2_a.se2_a(angle_1, any_tx_1, any_ty_1) + se2_a.se2_a(angle_2, any_tx_2, any_ty_2)).get
    assert_array_equal(given_output, expected_output)


def test_se2_a_add_random_input():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    any_angle_2 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_2 = uniform(-10, 10)
    any_ty_2 = uniform(-10, 10)

    elem1 = se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1)
    elem2 = se2_a.se2_a(any_angle_2, any_tx_2, any_ty_2)

    # this sum is must be done mod the relation given by exp

    sum_angle = any_angle_1 + any_angle_2
    sum_tx = any_tx_1 + any_tx_2
    sum_ty = any_ty_1 + any_ty_2

    sum_angle_mod = mod_pipi(sum_angle)
    if not sum_angle == sum_angle_mod:
        modfact = sum_angle_mod / sum_angle
        sum_tx = modfact*sum_tx
        sum_ty = modfact*sum_ty

    elem_sum = se2_a.se2_a(sum_angle_mod, sum_tx, sum_ty)

    print elem1.get
    print elem2.get
    print elem_sum.get

    assert_array_equal((elem1 + elem2).get, elem_sum.get)


def test_se2_a_comparing_sum_restricted_form_and_matrix_form():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    any_angle_2 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_2 = uniform(-10, 10)
    any_ty_2 = uniform(-10, 10)
    elem1 = se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1)
    elem2 = se2_a.se2_a(any_angle_2, any_tx_2, any_ty_2)

    print elem1.get
    print elem2.get

    matrix_output_of_sums = (elem1 + elem2).get_matrix
    # the matrix here must be constructed carefully. It is not the sum of the matrix
    # but some operations (_mod) must be done on its elements to have the quotient.
    matrix_sum_output = \
        se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1).get_matrix+se2_a.se2_a(any_angle_2, any_tx_2, any_ty_2).get_matrix

    theta_mod = mod_pipi(matrix_sum_output[1, 0])
    if not matrix_sum_output[1, 0] == theta_mod:
        modfact = theta_mod / matrix_sum_output[1, 0]
        matrix_sum_output[0, 2] = matrix_sum_output[0, 2] * modfact
        matrix_sum_output[1, 2] = matrix_sum_output[1, 2] * modfact
        matrix_sum_output[0, 1] = - theta_mod
        matrix_sum_output[1, 0] = theta_mod

    assert_array_almost_equal(matrix_output_of_sums, matrix_sum_output)


# test inv
def test_se2_a_add_element_to_null_equals_element():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    null_element = se2_a.se2_a(0, 0, 0)
    element1 = se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1)
    assert_array_equal((element1 + null_element).get, element1.get)


def test_se2_a_element_plus_opposite_equals_null_element():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    element1 = se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1)
    element1_opp = se2_a.se2_a(-any_angle_1, -any_tx_1, -any_ty_1)

    assert_array_equal((element1_opp + element1).get, [0, 0, 0])


def test_se2_a_opposite_random_element_verification_with_matrix():
    angle = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-10, 10)
    ty = uniform(-10, 10)
    element = se2_a.se2_a(angle, tx, ty)
    element_m = element.get_matrix
    element_opp_m = se2_a.se2_a(-angle, -tx, -ty).get_matrix
    assert_array_almost_equal(element_m + element_opp_m, np.array([0]*9).reshape(3, 3))


# test norm
def test_se2_a_norm_standard_fixed_input_0():
    angle, tx,  ty = 0, 0, 0
    element = se2_a.se2_a(angle, tx, ty)
    assert_equal(element.norm('lamb', 1), 0)


def test_se2_a_norm_translation_fixed_input_0():
    angle, tx,  ty = 0, 0, 0
    element = se2_a.se2_a(angle, tx, ty)
    assert_equal(element.norm('translation'), 0)


def test_se2_a_norm_fro_fixed_input_0():
    angle, tx,  ty = 0, 0, 0
    element = se2_a.se2_a(angle, tx, ty)
    assert_equal(element.norm('fro'), 0.0)


def test_se2_a_norm_standard_fixed_input_non0():
    angle, tx, ty = se2_a.se2_a(4, 5, 6).quot.get  # input data has to be considered in the quotient space
    lamb = 1
    element = se2_a.se2_a(angle, tx, ty)
    assert_equal(element.norm('lamb', lamb), np.sqrt(mod_pipi(angle)**2 + lamb*(tx**2 + ty**2)))


def test_se2_a_norm_translation_fixed_input_non0():
    angle, tx, ty = se2_a.se2_a(4, 5, 6).quot.get  # input data has to be considered in the quotient space
    element = se2_a.se2_a(angle, tx, ty)
    factmod = mod_pipi(angle)/angle
    assert_equal(element.norm('translation'), factmod*np.sqrt(tx**2 + ty**2))


def test_se2_a_norm_fro_fixed_input_non0():
    angle, tx, ty = se2_a.se2_a(4, 5, 6).quot.get  # input data has to be considered in the quotient space
    element = se2_a.se2_a(angle, tx, ty)
    factmod = mod_pipi(angle)/angle
    assert_equal(element.norm('fro'), np.sqrt(2*angle**2 + factmod**2 * (tx**2 + ty**2)))


def test_se2_a_norm_standard_random_input():
    angle = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-10, 10)
    ty = uniform(-10, 10)
    lamb = 1.4
    element = se2_a.se2_a(angle, tx, ty)
    assert_equal(element.norm('lamb', lamb), np.sqrt(angle**2 + lamb*(tx**2 + ty**2)))


def test_se2_a_norm_translation_random_input():
    angle = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-10, 10)
    ty = uniform(-10, 10)
    element = se2_a.se2_a(angle, tx, ty)
    assert_equal(element.norm('translation'), np.sqrt(tx**2 + ty**2))


def test_se2_a_norm_fro_random_input():
    angle = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-10, 10)
    ty = uniform(-10, 10)
    element = se2_a.se2_a(angle, tx, ty)
    assert_equal(element.norm('fro'), np.sqrt(2*angle**2 + tx**2 + ty**2))


def test_se2_a_norm_spam_type_input():
    angle, tx,  ty = 4, 5, 6
    element = se2_a.se2_a(angle, tx, ty)
    assert_equal(element.norm('spam'), -1)


# norm fro and norm translation are not invariant for angle in the quotient space:
# (certainly not if the angle is greater than pi)
def test_se2_a_norm_translation_and_fro_not_invariance_for_angle_greater_than_pi():
    angle_1, tx,  ty = 2 * np.pi + 0.5, 5, 6
    angle_2 = angle_1 + uniform(-np.pi, np.pi)
    element_1 = se2_a.se2_a(angle_1, tx, ty)
    element_2 = se2_a.se2_a(angle_2, tx, ty)
    if element_1.norm('translation') == element_2.norm('translation') \
            or element_1.norm('fro') == element_2.norm('fro'):
        assert False
    else:
        assert True


''' test is_a_matrix_in_se2_a '''


def test_se2_a_is_a_matrix_in_se2_a_bad_shape_input():
    tt = np.array([1, 2, 3])
    assert not se2_a.is_a_matrix_in_se2_a(tt)


def test_se2_a_is_a_matrix_in_se2_a_bad_type_input():
    tt = '42'
    assert not se2_a.is_a_matrix_in_se2_a(tt)


def test_se2_a_is_a_matrix_in_se2_a_non_skew_input():
    tt = np.array([[1, 2, 0], [2, 1, 0], [0, 0, 1]])
    assert not se2_a.is_a_matrix_in_se2_a(tt)


def test_se2_a_is_a_matrix_in_se2_a_last_row_wrong():
    tt = np.array([[1, 2, 0], [-2, 1, 0], [0, 4, 1]])
    assert not se2_a.is_a_matrix_in_se2_a(tt)


def test_se2_a_is_a_matrix_in_se2_a_good_input():
    tt = np.array([[0, -2, 4], [2, 0, 3], [0, 0, 0]])
    assert se2_a.is_a_matrix_in_se2_a(tt)


''' Test Lie bracket se2_a quotient'''


def test_se2_a_lie_bracket_insane_input():
    element1 = '42'
    element2 = [1, 2]
    with assert_raises(Exception):
        se2_a.lie_bracket(element1, element2)


def test_se2_a_lie_bracket_both_zero_rotation_input():
    angle_1 = 0
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    angle_2 = 0
    any_tx_2 = uniform(-10, 10)
    any_ty_2 = uniform(-10, 10)

    elem1 = se2_a.se2_a(angle_1, any_tx_1, any_ty_1)
    elem2 = se2_a.se2_a(angle_2, any_tx_2, any_ty_2)
    exp_ans = [0, 0.0, 0.0]

    assert_array_almost_equal(se2_a.lie_bracket(elem1, elem2).get, exp_ans)


def test_se2_a_lie_bracket_both_zero_translation_input():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx_1 = 0
    ty_1 = 0
    any_angle_2 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx_2 = 0
    ty_2 = 0

    elem1 = se2_a.se2_a(any_angle_1, tx_1, ty_1)
    elem2 = se2_a.se2_a(any_angle_2, tx_2, ty_2)
    exp_ans = [0, 0.0, 0.0]

    assert_array_almost_equal(se2_a.lie_bracket(elem1, elem2).get, exp_ans)


def test_se2_a_lie_bracket_one_element_out_quotient():
    any_angle_1 = np.pi + uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    any_angle_2 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_2 = uniform(-10, 10)
    any_ty_2 = uniform(-10, 10)
    # the first input manually reduced to quotient space
    mod_fact = mod_pipi(any_angle_1)/any_angle_1
    any_angle_1 = mod_pipi(any_angle_1)
    any_tx_1 *= mod_fact
    any_ty_1 *= mod_fact
    #
    elem1 = se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1)
    elem2 = se2_a.se2_a(any_angle_2, any_tx_2, any_ty_2)
    exp_ans = [0, any_angle_2 * any_ty_1 - any_angle_1 * any_ty_2, any_angle_1 * any_tx_2 - any_angle_2 * any_tx_1]

    assert_array_almost_equal(se2_a.lie_bracket(elem1, elem2).get, exp_ans)


def test_se2_a_lie_bracket_random_input():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    any_angle_2 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_2 = uniform(-10, 10)
    any_ty_2 = uniform(-10, 10)

    elem1 = se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1)
    elem2 = se2_a.se2_a(any_angle_2, any_tx_2, any_ty_2)
    exp_ans = [0, any_angle_2 * any_ty_1 - any_angle_1 * any_ty_2, any_angle_1 * any_tx_2 - any_angle_2 * any_tx_1]

    assert_array_almost_equal(se2_a.lie_bracket(elem1, elem2).get, exp_ans)


''' Test Lie multi bracket se2_a quotient'''


def test_se2_a_lie_multi_bracket():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    any_angle_2 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_2 = uniform(-10, 10)
    any_ty_2 = uniform(-10, 10)
    any_angle_3 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_3 = uniform(-10, 10)
    any_ty_3 = uniform(-10, 10)
    any_angle_4 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_4 = uniform(-10, 10)
    any_ty_4 = uniform(-10, 10)

    elem1 = se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1)
    elem2 = se2_a.se2_a(any_angle_2, any_tx_2, any_ty_2)
    elem3 = se2_a.se2_a(any_angle_3, any_tx_3, any_ty_3)
    elem4 = se2_a.se2_a(any_angle_4, any_tx_4, any_ty_4)

    l = [elem1, elem2, elem3, elem4]
    ans_provided = se2_a.lie_multi_bracket(l).get
    ans_expected = se2_a.lie_bracket(elem1, se2_a.lie_bracket(elem2, se2_a.lie_bracket(elem3, elem4))).get

    assert_array_almost_equal(ans_provided, ans_expected)


''' test rmul as scalarpr '''


def test_se2_a_rmul_():
    angle = 0.98
    tx = 5
    ty = 7
    scalar = 0.5
    ans_expected = [scalar*angle, scalar*tx, scalar*ty]
    scalar_prod = scalar*se2_a.se2_a(angle, tx, ty)
    ans_provided = [scalar_prod.rotation_angle, scalar_prod.tx, scalar_prod.ty]

    assert_array_almost_equal(ans_provided, ans_expected)


''' Test scalarpr se2_a '''


def test_se2_a_scalarpr_smaller_than_pi():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    scalar = 0.3
    if any_angle_1 != 0:
        scalar = uniform(0, np.pi/any_angle_1)
    elem = se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1)
    expected_ans = se2_a.se2_a(scalar*any_angle_1, scalar*any_tx_1, scalar*any_ty_1)
    scalarpr_ans = se2_a.scalarpr(scalar, elem)
    assert_array_almost_equal(expected_ans.get, scalarpr_ans.get)


def test_se2_a_scalarpr_greater_than_pi():
    any_angle_1 = 2*np.pi + uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    scalar = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    # obtain quotient and its list
    elem_quot = se2_a.se2_a(any_angle_1, any_tx_1, any_ty_1)
    elem_quot_get = elem_quot.get

    # (scalar * elem).quot :
    expected_ans = se2_a.se2_a(scalar*elem_quot_get[0], scalar*elem_quot_get[1], scalar*elem_quot_get[2])
    # scalar * (elem.quot) :
    scalarpr_ans = se2_a.scalarpr(scalar, elem_quot)

    assert_array_almost_equal(expected_ans.get, scalarpr_ans.get)

''' test matrix2se2_a '''


def test_se2_a_matrix2se2_a_insane_input_eat_em_all_false():
    tt = np.array(range(9)).reshape(3, 3)
    with assert_raises(Exception):
        se2_a.matrix2se2_a(tt)


def test_se2_a_matrix2se2_a_sane_input():
    theta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-5, 5)
    ty = uniform(-5, 5)
    m = np.array([[0, -theta, tx], [theta, 0, ty], [0, 0, 0]])
    ans = se2_a.matrix2se2_a(m)
    exp_ans = se2_a.se2_a(theta, tx, ty)
    assert_array_almost_equal(ans.get, exp_ans.get)


''' test list2se2_a'''


def test_se2_a_list2se2_a_insane_input():
    tt = [1, 2, 3, 4]
    with assert_raises(TypeError):
        se2_a.list2se2_a(tt)


def test_se2_a_list2se2_a_good_input():
    tt = [mod_pipi(3.2), 1, 2]
    ans = se2_a.list2se2_a(tt)
    assert_array_almost_equal(ans.get, tt)


''' test exp se2_a '''


def test_se2_a_exp_pade_approx_comparison():
    theta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-5, 5)
    ty = uniform(-5, 5)
    element = se2_a.se2_a(theta, tx, ty)
    ans_exp = se2_a.exp(element).get_matrix
    ans_pade = lin.expm(element.get_matrix)
    assert_array_almost_equal(ans_exp, ans_pade)


def test_se2_a_exp_0_angle():
    theta = 0
    tx = uniform(-5, 5)
    ty = uniform(-5, 5)
    element = se2_a.se2_a(theta, tx, ty)

    tx1 = tx
    ty1 = ty

    ans_exp = se2_a.exp(element)
    if ans_exp.rotation_angle == 0 and ans_exp.tx == tx1 and ans_exp.ty == ty1:
        assert True
    else:
        assert False


''' test exp_for_matrices se2_a '''


def test_se2_g_log_for_matrices_pade_approx_comparison():
    theta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-5, 5)
    ty = uniform(-5, 5)
    element_m = se2_a.se2_a(theta, tx, ty).get_matrix
    ans_exp_m = se2_a.exp_for_matrices(element_m)
    ans_pade = np.around(lin.expm(element_m), 10).real
    print ans_exp_m
    print ans_pade
    assert_array_almost_equal(ans_exp_m, ans_pade)


'''
Test for the module se2_g, Lie group of rotations and translations
'''

''' test class object se2_g '''


def test_init_se2_g_random_input():
    any_angle_in_pi_pi = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx = uniform(-10, 10)
    any_ty = uniform(-10, 10)
    assert_array_equal(se2_g.se2_g(any_angle_in_pi_pi, any_tx, any_ty).get, [any_angle_in_pi_pi, any_tx, any_ty])


def test_se2_g_get_matrix():
    any_angle_in_pi_pi = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx = uniform(-10, 10)
    any_ty = uniform(-10, 10)
    m = se2_g.se2_g(any_angle_in_pi_pi, any_tx, any_ty).get_matrix
    assert m[0, 0] == m[1, 1] == np.cos(any_angle_in_pi_pi) \
        and -m[0, 1] == m[1, 0] == np.sin(any_angle_in_pi_pi) \
        and m[2, 2] == 1 and m[0, 2] == any_tx and m[1, 2] == any_ty


def test_se2_g_get_matrix_det():
    any_angle_in_pi_pi = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx = uniform(-10, 10)
    any_ty = uniform(-10, 10)
    m = se2_g.se2_g(any_angle_in_pi_pi, any_tx, any_ty).get_matrix
    given_output = np.linalg.det(m)
    expected_output = 1.0
    global meaningful_decimals
    assert round(abs(given_output-expected_output), meaningful_decimals) == 0.0


def test_se2_g_mul_0_translation():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx_1 = 0
    ty_1 = 0
    any_angle_2 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx_2 = 0
    ty_2 = 0
    expected_output_angle = mod_pipi(any_angle_1 + any_angle_2)
    expected_output_tx = 0
    expected_output_ty = 0
    expected_output = [expected_output_angle, expected_output_tx, expected_output_ty]
    given_output = (se2_g.se2_g(any_angle_1, tx_1, ty_1)*se2_g.se2_g(any_angle_2, tx_2, ty_2)).get
    assert_array_equal(given_output, expected_output)


def test_se2_g_mul_0_rotation():
    angle_1 = 0
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    angle_2 = 0
    any_tx_2 = uniform(-10, 10)
    any_ty_2 = uniform(-10, 10)
    expected_output_angle = 0
    expected_output_tx = any_tx_1 + any_tx_2
    expected_output_ty = any_ty_1 + any_ty_2
    expected_output = [expected_output_angle, expected_output_tx, expected_output_ty]
    given_output = (se2_g.se2_g(angle_1, any_tx_1, any_ty_1)*se2_g.se2_g(angle_2, any_tx_2, any_ty_2)).get
    assert_array_equal(given_output, expected_output)


def test_se2_g_mul_random_input():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    any_angle_2 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_2 = uniform(-10, 10)
    any_ty_2 = uniform(-10, 10)
    expected_output_angle = mod_pipi(any_angle_1 + any_angle_2)
    expected_output_tx = any_tx_1 + any_tx_2 * np.cos(any_angle_1) - any_ty_2 * np.sin(any_angle_1)
    expected_output_ty = any_ty_1 + any_tx_2 * np.sin(any_angle_1) + any_ty_2 * np.cos(any_angle_1)
    expected_output = [expected_output_angle, expected_output_tx, expected_output_ty]
    given_output = (se2_g.se2_g(any_angle_1, any_tx_1, any_ty_1)*se2_g.se2_g(any_angle_2, any_tx_2, any_ty_2)).get
    assert_array_equal(given_output, expected_output)


def test_se2_g_comparing_product_restricted_form_and_matrix_form():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    any_angle_2 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_2 = uniform(-10, 10)
    any_ty_2 = uniform(-10, 10)
    get_matrix_output = \
        (se2_g.se2_g(any_angle_1, any_tx_1, any_ty_1)*se2_g.se2_g(any_angle_2, any_tx_2, any_ty_2)).get_matrix
    matrix_product_output = \
        se2_g.se2_g(any_angle_1, any_tx_1, any_ty_1).get_matrix.dot(
            se2_g.se2_g(any_angle_2, any_tx_2, any_ty_2).get_matrix)
    assert_array_almost_equal(get_matrix_output, matrix_product_output)


# test inv
def test_se2_g_inv_null_element():
    angle = 0
    tx = 0
    ty = 0

    element = se2_g.se2_g(angle, tx, ty)
    inverse_element = element.inv()
    expected_inverse_element = se2_g.se2_g(-angle, -tx, -ty)

    assert_array_almost_equal(inverse_element.get, expected_inverse_element.get)


def test_se2_g_inv_random_element():
    angle = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-10, 10)
    ty = uniform(-10, 10)
    element = se2_g.se2_g(angle, tx, ty)
    inverse_element = element.inv().get_matrix
    expected = np.linalg.inv(element.get_matrix)
    assert_array_almost_equal(inverse_element, expected)


def test_se2_g_inv_in_matrix_product():
    angle = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-10, 10)
    ty = uniform(-10, 10)
    element = se2_g.se2_g(angle, tx, ty)
    matr = element.get_matrix
    matri_inv = element.inv().get_matrix
    assert_array_almost_equal(matr.dot(matri_inv), np.identity(3))


# test norm
def test_se2_g_norm_standard_fixed_input_0():
    angle, tx,  ty = 0, 0, 0
    element = se2_g.se2_g(angle, tx, ty)
    assert_equal(element.norm('standard', 1), 0)


def test_se2_g_norm_translation_fixed_input_0():
    angle, tx,  ty = 0, 0, 0
    element = se2_g.se2_g(angle, tx, ty)
    assert_equal(element.norm('translation'), 0)


def test_se2_g_norm_fro_fixed_input_0():
    angle, tx,  ty = 0, 0, 0
    element = se2_g.se2_g(angle, tx, ty)
    print element.get
    assert_equal(element.norm('fro'), np.sqrt(3))


def test_se2_g_norm_standard_fixed_input_non0():
    angle, tx,  ty = 4, 5, 6
    lamb = 1
    element = se2_g.se2_g(angle, tx, ty)
    assert_equal(element.norm('standard', lamb), np.sqrt(mod_pipi(angle)**2 + lamb*(tx**2 + ty**2)))


def test_se2_g_norm_translation_fixed_input_non0():
    angle, tx,  ty = 4, 5, 6
    element = se2_g.se2_g(angle, tx, ty)
    assert_equal(element.norm('translation'), np.sqrt(61.0))


def test_se2_g_norm_fro_fixed_input_non0():
    angle, tx,  ty = 4, 5, 6
    element = se2_g.se2_g(angle, tx, ty)
    assert_equal(element.norm('fro'), np.sqrt(3 + 61.0))


def test_se2_g_norm_standard_random_input():
    angle = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-10, 10)
    ty = uniform(-10, 10)
    lamb = 1.4
    element = se2_g.se2_g(angle, tx, ty)
    assert_equal(element.norm('standard', lamb), np.sqrt(angle**2 + lamb*(tx**2 + ty**2)))


def test_se2_g_norm_translation_random_input():
    angle = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-10, 10)
    ty = uniform(-10, 10)
    element = se2_g.se2_g(angle, tx, ty)
    assert_equal(element.norm('translation'), np.sqrt(tx**2 + ty**2))


def test_se2_g_norm_fro_random_input():
    angle = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-10, 10)
    ty = uniform(-10, 10)
    element = se2_g.se2_g(angle, tx, ty)
    assert_equal(element.norm('fro'), np.sqrt(3 + tx**2 + ty**2))


def test_se2_g_norm_fro_random_input_comparing_matrix_numpy_norm():
    # comparing frobenius with frobenius norm matrix (random matrix
    angle = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = uniform(-10, 10)
    ty = uniform(-10, 10)
    element = se2_g.se2_g(angle, tx, ty)
    output_norm = element.norm('fro')
    linalg_norm = np.linalg.norm(element.get_matrix, 'fro')
    global meaningful_decimals
    assert round(abs(output_norm-linalg_norm), meaningful_decimals) == 0.0


def test_se2_g_norm_spam_input():
    angle, tx,  ty = 4, 5, 6
    element = se2_g.se2_g(angle, tx, ty)
    assert_equal(element.norm('spam'), -1)


def test_se2_g_norm_translation_invariance_for_angle():
    angle_1, tx,  ty = 4, 5, 6
    angle_2 = angle_1 + uniform(-np.pi, np.pi)
    element_1 = se2_g.se2_g(angle_1, tx, ty)
    element_2 = se2_g.se2_g(angle_2, tx, ty)
    assert_equal(element_1.norm('translation'), element_2.norm('translation'))


def test_se2_g_norm_fro_invariance_for_angle():
    angle_1, tx,  ty = 4, 5, 6
    angle_2 = angle_1 + uniform(-np.pi, np.pi)
    element_1 = se2_g.se2_g(angle_1, tx, ty)
    element_2 = se2_g.se2_g(angle_2, tx, ty)
    assert_equal(element_1.norm('fro'), element_2.norm('fro'))


''' test randomgen standard se2_g '''


def test_se2_g_randomgen_standard_insane_interval():
    interval = (1, 2, 3)
    with assert_raises(Exception):
        se2_g.randomgen(interval)


def test_se2_g_randomgen_standard_negative_interval():
    interval = (1, -2)
    with assert_raises(Exception):
        se2_g.randomgen(interval)


def test_se2_g_randomgen_standard_swapped_interval():
    interval = (3, 1)
    with assert_raises(Exception):
        se2_g.randomgen(interval)


def test_se2_g_randomgen_standard_a_negative_interval():
    interval = (-3, 1)
    with assert_raises(Exception):
        se2_g.randomgen(interval)


def test_se2_g_randomgen_standard_b_negative_interval():
    interval = (3, -1)
    with assert_raises(Exception):
        se2_g.randomgen(interval)


def test_se2_g_randomgen_standard_lamb_negative():
    interval = (1, 2)
    with assert_raises(Exception):
        se2_g.randomgen(interval, lamb=-1)


def test_se2_g_randomgen_standard_lamb_0_interval_insane():
    interval = (7, 8)
    with assert_raises(Exception):
        se2_g.randomgen(interval, lamb=0)


def test_se2_g_randomgen_standard_lamb_0_no_restriction_interval_dx():
    a = 1
    b = 15
    lamb = 0
    interval = (a, b)
    given_output = se2_g.randomgen(interval, lamb=lamb)
    assert a <= given_output.norm('standard', lamb=lamb) <= np.pi


def test_se2_g_randomgen_standard_lamb_0_no_restriction_interval_sx():
    a = 0
    b = 2
    lamb = 0
    interval = (a, b)
    given_output = se2_g.randomgen(interval, lamb=lamb)
    assert a <= given_output.norm('standard', lamb=lamb) <= b


def test_se2_g_randomgen_standard_lamb_0_restricted_interval_1_2():
    a = 1
    b = 2
    lamb = 0
    interval = (a, b)
    given_output = se2_g.randomgen(interval, lamb=lamb)
    assert a <= given_output.norm('standard', lamb=lamb) <= b


def test_se2_g_randomgen_standard_lamb_positive_insane_interval():
    interval = (1, 2, 3)
    with assert_raises(Exception):
        se2_g.randomgen(interval)


def test_se2_g_randomgen_standard_lamb_positive_empty_interval():
    interval = ()
    element = se2_g.randomgen(interval)
    assert 0 <= element.norm('standard') <= 10


def test_se2_g_randomgen_standard_lamb_positive_strict_interval():
    a = 1.5
    b = 2
    lamb = 1
    interval = (a, b)
    given_output = se2_g.randomgen(interval, lamb=lamb)
    assert a <= given_output.norm('standard', lamb=lamb) <= b


''' test randomgen translation se2_g '''


def test_se2_g_randomgen_translation_insane_interval():
    interval = (1, 2, 3)
    with assert_raises(Exception):
        se2_g.randomgen_translation(interval)


def test_se2_g_randomgen_translation_negative_interval():
    interval = (-1, -2)
    with assert_raises(Exception):
        se2_g.randomgen_translation(interval)


def test_se2_g_randomgen_translation_swapped_interval():
    interval = (3, 1)
    with assert_raises(Exception):
        se2_g.randomgen_translation(interval)


def test_se2_g_randomgen_translation_a_negative_interval():
    interval = (-3, 1)
    with assert_raises(Exception):
        se2_g.randomgen_translation(interval)


def test_se2_g_randomgen_translation_b_negative_interval():
    interval = (3, -1)
    with assert_raises(Exception):
        se2_g.randomgen_translation(interval)


def test_se2_g_randomgen_translation_empty_interval():
    interval = ()
    given_output = se2_g.randomgen_translation(interval)
    assert 0 <= given_output.norm('translation') <= 10


def test_se2_g_randomgen_translation_bigger_interval():
    a = 0
    b = 15
    interval = (a, b)
    given_output = se2_g.randomgen_translation(interval)
    assert a <= given_output.norm('translation') <= b


def test_se2_g_randomgen_translation_restricted_interval():
    a = 2.3
    b = 2.5
    interval = (a, b)
    given_output = se2_g.randomgen_translation(interval)
    assert a <= given_output.norm('translation') <= b


''' test randomgen fro se2_g '''


def test_se2_g_randomgen_fro_insane_interval():
    interval = (1, 2, 3)
    with assert_raises(Exception):
        se2_g.randomgen_fro(interval)


def test_se2_g_randomgen_fro_negative_interval():
    interval = (-1, -2)
    with assert_raises(Exception):
        se2_g.randomgen_fro(interval)


def test_se2_g_randomgen_fro_swapped_interval():
    interval = (3, 1)
    with assert_raises(Exception):
        se2_g.randomgen_fro(interval)


def test_se2_g_randomgen_fro_a_negative_interval():
    interval = (-3, 1)
    with assert_raises(Exception):
        se2_g.randomgen_fro(interval)


def test_se2_g_randomgen_fro_b_negative_interval():
    interval = (3, -1)
    with assert_raises(Exception):
        se2_g.randomgen_fro(interval)


def test_se2_g_randomgen_fro_a_less_sqrt3():
    interval = (1.71, 5)
    with assert_raises(Exception):
        se2_g.randomgen_fro(interval)


def test_se2_g_randomgen_fro_empty_interval():
    interval = ()
    given_output = se2_g.randomgen_fro(interval)
    assert np.sqrt(3) <= given_output.norm('fro') <= 10


def test_se2_g_randomgen_fro_bigger_interval():
    a = np.sqrt(3)
    b = 15
    interval = (a, b)
    given_output = se2_g.randomgen_fro(interval)
    assert a <= given_output.norm('fro') <= b


def test_se2_g_randomgen_fro_restricted_interval():
    a = 2.3
    b = 2.5
    interval = (a, b)
    given_output = se2_g.randomgen_fro(interval)
    print str(given_output.get)
    assert a <= given_output.norm('fro') <= b


#verify this also for each kind of norm in the random generator!
''' test is_a_matrix_in_se2_g '''


def test_se2_g_is_a_matrix_in_se2_g_bad_shape_input():
    tt = np.array([1, 2, 3])
    assert not se2_g.is_a_matrix_in_se2_g(tt)


def test_se2_g_is_a_matrix_in_se2_g_bad_type_input():
    tt = '42'
    assert not se2_g.is_a_matrix_in_se2_g(tt)


def test_se2_g_is_a_matrix_in_se2_g_non_skew_input():
    tt = np.array([[1, 2, 0], [2, 1, 0], [0, 0, 1]])
    assert not se2_g.is_a_matrix_in_se2_g(tt)


def test_se2_g_is_a_matrix_in_se2_g_last_row_wrong():
    tt = np.array([[1, 2, 0], [-2, 1, 0], [0, 4, 1]])
    assert not se2_g.is_a_matrix_in_se2_g(tt)


def test_se2_g_is_a_matrix_in_se2_g_good_input():
    tt = np.array([[1, 2, 0], [-2, 1, 0], [0, 0, 1]])
    assert se2_g.is_a_matrix_in_se2_g(tt)


''' test matrix2se2_g '''


def test_se2_gmatrix2se2_g_insane_input_eat_em_all_false():
    tt = np.array(range(9)).reshape(3, 3)
    with assert_raises(Exception):
        se2_g.matrix2se2_g(tt)


def test_se2_g_matrix2se2_g_sane_input():
    theta = uniform(-np.pi, np.pi)
    tx = uniform(-5, 5)
    ty = uniform(-5, 5)
    m = np.array([[np.cos(theta), -np.sin(theta), tx], [np.sin(theta), np.cos(theta), ty], [0, 0, 1]])
    ans = se2_g.matrix2se2_g(m)
    exp_ans = se2_g.se2_g(theta, tx, ty)
    assert_array_almost_equal(ans.get, exp_ans.get)


''' test list2se2_g'''


def test_se2_g_list2se2_g_insane_input():
    tt = [1, 2, 3, 4]
    with assert_raises(TypeError):
        se2_g.list2se2_g(tt)


def test_se2_g_list2se2_g_good_input():
    tt = [mod_pipi(3.2), 1, 2]
    ans = se2_g.list2se2_g(tt)
    assert_array_almost_equal(ans.get, tt)


''' test log se2_g '''


def test_se2_g_log_pade_approx_comparison():
    theta = uniform(-np.pi, np.pi)
    tx = uniform(-5, 5)
    ty = uniform(-5, 5)
    element = se2_g.se2_g(theta, tx, ty)
    ans_log = se2_g.log(element).get_matrix
    ans_pade = lin.logm(element.get_matrix)
    assert_array_almost_equal(ans_log, ans_pade)


def test_se2_g_log_0_angle():
    theta = 0
    tx = uniform(-5, 5)
    ty = uniform(-5, 5)
    element = se2_g.se2_g(theta, tx, ty)
    ans_log = se2_g.log(element)
    if ans_log.rotation_angle == 0 and ans_log.tx == tx and ans_log.ty == ty:
        assert True
    else:
        assert False


''' test log_for_matrices '''


def test_se2_g_log_for_matrices_pade_approx_comparison_1():
    theta = uniform(-np.pi, np.pi)
    tx = uniform(-5, 5)
    ty = uniform(-5, 5)
    element_m = se2_g.se2_g(theta, tx, ty).get_matrix
    ans_log_m = se2_g.log_for_matrices(element_m)
    ans_pade = np.around(lin.logm(element_m), 10).real
    assert_array_almost_equal(ans_log_m, ans_pade)

