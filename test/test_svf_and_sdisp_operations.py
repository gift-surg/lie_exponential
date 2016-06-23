import numpy as np
import copy

from numpy.testing import assert_array_equal

from transformations.s_vf import SVF
from transformations.s_disp import SDISP


def test_sum_of_two_svf(verbose=True):

    def function_0(t, x):
        t = float(t); x = [float(y) for y in x]
        return np.array([x[1], -1 * x[0]])

    def function_1(t, x):
        t = float(t); x = [float(y) for y in x]
        return np.array([0.5 * x[0] + 0.6 * x[1], 0.8 * x[1]])

    def function_sum(t, x):
        t = float(t); x = [float(y) for y in x]
        return function_0(t, x) + function_1(t, x)

    def function_difference(t, x):
        t = float(t); x = [float(y) for y in x]
        return function_0(t, x) - function_1(t, x)

    def function_0_times_alpha(t, x):
        alpha = 2.5
        t = float(t); x = [float(y) for y in x]
        return alpha * function_0(t, x)

    def function_1_times_minus_one(t, x):
        t = float(t); x = [float(y) for y in x]
        return - 1 * function_1(t, x)

    field_0     = SVF.generate_zero(shape=(20, 20, 1, 1, 2))
    field_1     = SVF.generate_zero(shape=(20, 20, 1, 1, 2))
    field_sum   = SVF.generate_zero(shape=(20, 20, 1, 1, 2))
    field_diff  = SVF.generate_zero(shape=(20, 20, 1, 1, 2))
    field_scal  = SVF.generate_zero(shape=(20, 20, 1, 1, 2))
    field_2_inv = SVF.generate_zero(shape=(20, 20, 1, 1, 2))

    assert field_0.__class__.__name__ == 'SVF'
    assert field_1.__class__.__name__ == 'SVF'

    for i in range(0, 20):
        for j in range(0, 20):
            field_0.field[i, j, 0, 0, :]     = function_0(1, [i, j])
            field_1.field[i, j, 0, 0, :]     = function_1(1, [i, j])
            field_sum.field[i, j, 0, 0, :]   = function_sum(1, [i, j])
            field_diff.field[i, j, 0, 0, :]  = function_difference(1, [i, j])
            field_scal.field[i, j, 0, 0, :] = function_0_times_alpha(1, [i, j])
            field_2_inv.field[i, j, 0, 0, :] = function_1_times_minus_one(1, [i, j])

    x_0, y_0 = 3, 1
    # some assertions to verify the sum is really the sum we wanted in the ground truth
    assert_array_equal(field_sum.field[x_0, y_0, 0, 0, :], field_0.field[x_0, y_0, 0, 0, :] + field_1.field[x_0, y_0, 0, 0, :])
    assert_array_equal(field_diff.field[x_0, y_0, 0, 0, :], field_0.field[x_0, y_0, 0, 0, :] - field_1.field[x_0, y_0, 0, 0, :])
    assert_array_equal(field_scal.field[x_0, y_0, 0, 0, :], 2.5 * field_0.field[x_0, y_0, 0, 0, :])
    assert_array_equal(field_2_inv.field[x_0, y_0, 0, 0, :], -1 * field_1.field[x_0, y_0, 0, 0, :])

    if verbose:
        print 'f1(' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_0.field[x_0, y_0, 0, 0, :]
        print ''
        print 'f2(' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_1.field[x_0, y_0, 0, 0, :]
        print ''
        print '(f1 + f2) (' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_sum.field[x_0, y_0, 0, 0, :]
        print ''
        print '(f1 - f2) (' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_diff.field[x_0, y_0, 0, 0, :]
        print ''
        print '(2.5 * f1) (' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_scal.field[x_0, y_0, 0, 0, :]
        print ''
        print '(-f1) (' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_2_inv.field[x_0, y_0, 0, 0, :]

    # compare the ground truth with the computed sum and scalar product!
    field_sum_directly_computed   = field_0 + field_1
    field_diff_directly_computed  = field_0 - field_1
    field_scal_directly_computed  = 2.5 * field_0 
    field_2_inv_directly_computed = -1 * field_1
    
    # verify the types remains the proper one:
    assert field_sum_directly_computed.__class__.__name__ == 'SVF'
    assert field_diff_directly_computed.__class__.__name__ == 'SVF'
    assert field_scal_directly_computed.__class__.__name__ == 'SVF'
    assert field_2_inv_directly_computed.__class__.__name__ == 'SVF'

    # verify the operations are not destructive
    assert field_0.__class__.__name__ == 'SVF'
    assert field_1.__class__.__name__ == 'SVF'
    assert_array_equal(field_0.field[x_0, y_0, 0, 0, :], function_0(1, [x_0, y_0]))
    assert_array_equal(field_1.field[x_0, y_0, 0, 0, :], function_1(1, [x_0, y_0]))

    # verify the sum directly computed is the sum of the field build from the functions
    assert_array_equal(field_sum_directly_computed.field, field_sum.field)
    assert_array_equal(field_diff_directly_computed.field, field_diff.field)
    assert_array_equal(field_scal_directly_computed.field, field_scal.field)
    assert_array_equal(field_2_inv_directly_computed.field, field_2_inv.field)


def test_sum_of_two_sdisp(verbose=True):

    # NOTE: a displacement can still have the operation of scalar field
    # since it is a vector field!!! Displacement is a vector field. Period.

    def function_0(t, x):
        t = float(t); x = [float(y) for y in x]
        return np.array([x[1], -1 * x[0]])

    def function_1(t, x):
        t = float(t); x = [float(y) for y in x]
        return np.array([0.5 * x[0] + 0.6 * x[1], 0.8 * x[1]])

    def function_sum(t, x):
        t = float(t); x = [float(y) for y in x]
        return function_0(t, x) + function_1(t, x)

    def function_difference(t, x):
        t = float(t); x = [float(y) for y in x]
        return function_0(t, x) - function_1(t, x)

    def function_0_times_alpha(t, x):
        alpha = 2.5
        t = float(t); x = [float(y) for y in x]
        return alpha * function_0(t, x)

    def function_1_times_minus_one(t, x):
        t = float(t); x = [float(y) for y in x]
        return - 1 * function_1(t, x)

    field_0     = SDISP.generate_zero(shape=(20, 20, 1, 1, 2))
    field_1     = SDISP.generate_zero(shape=(20, 20, 1, 1, 2))
    field_sum   = SDISP.generate_zero(shape=(20, 20, 1, 1, 2))
    field_diff  = SDISP.generate_zero(shape=(20, 20, 1, 1, 2))
    field_scal  = SDISP.generate_zero(shape=(20, 20, 1, 1, 2))
    field_2_inv = SDISP.generate_zero(shape=(20, 20, 1, 1, 2))

    assert field_0.__class__.__name__ == 'SDISP'
    assert field_1.__class__.__name__ == 'SDISP'

    for i in range(0, 20):
        for j in range(0, 20):
            field_0.field[i, j, 0, 0, :]     = function_0(1, [i, j])
            field_1.field[i, j, 0, 0, :]     = function_1(1, [i, j])
            field_sum.field[i, j, 0, 0, :]   = function_sum(1, [i, j])
            field_diff.field[i, j, 0, 0, :]  = function_difference(1, [i, j])
            field_scal.field[i, j, 0, 0, :] = function_0_times_alpha(1, [i, j])
            field_2_inv.field[i, j, 0, 0, :] = function_1_times_minus_one(1, [i, j])

    x_0, y_0 = 3, 1
    # some assertions to verify the sum is really the sum we wanted in the ground truth
    assert_array_equal(field_sum.field[x_0, y_0, 0, 0, :], field_0.field[x_0, y_0, 0, 0, :] + field_1.field[x_0, y_0, 0, 0, :])
    assert_array_equal(field_diff.field[x_0, y_0, 0, 0, :], field_0.field[x_0, y_0, 0, 0, :] - field_1.field[x_0, y_0, 0, 0, :])
    assert_array_equal(field_scal.field[x_0, y_0, 0, 0, :], 2.5 * field_0.field[x_0, y_0, 0, 0, :])
    assert_array_equal(field_2_inv.field[x_0, y_0, 0, 0, :], -1 * field_1.field[x_0, y_0, 0, 0, :])

    if verbose:
        print 'f1(' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_0.field[x_0, y_0, 0, 0, :]
        print ''
        print 'f2(' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_1.field[x_0, y_0, 0, 0, :]
        print ''
        print '(f1 + f2) (' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_sum.field[x_0, y_0, 0, 0, :]
        print ''
        print '(f1 - f2) (' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_diff.field[x_0, y_0, 0, 0, :]
        print ''
        print '(2.5 * f1) (' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_scal.field[x_0, y_0, 0, 0, :]
        print ''
        print '(-f1) (' + str(x_0) + ' , ' + str(y_0) + ')'
        print field_2_inv.field[x_0, y_0, 0, 0, :]

    # compare the ground truth with the computed sum and scalar product!
    field_sum_directly_computed   = field_0 + field_1
    field_diff_directly_computed  = field_0 - field_1
    field_scal_directly_computed  = 2.5 * field_0
    field_2_inv_directly_computed = -1 * field_1

    # verify the types remains the proper one:
    assert field_sum_directly_computed.__class__.__name__ == 'SDISP'
    assert field_diff_directly_computed.__class__.__name__ == 'SDISP'
    assert field_scal_directly_computed.__class__.__name__ == 'SDISP'
    assert field_2_inv_directly_computed.__class__.__name__ == 'SDISP'

    # verify the operations are not destructive
    assert field_0.__class__.__name__ == 'SDISP'
    assert field_1.__class__.__name__ == 'SDISP'
    assert_array_equal(field_0.field[x_0, y_0, 0, 0, :], function_0(1, [x_0, y_0]))
    assert_array_equal(field_1.field[x_0, y_0, 0, 0, :], function_1(1, [x_0, y_0]))

    # verify the sum directly computed is the sum of the field build from the functions
    assert_array_equal(field_sum_directly_computed.field, field_sum.field)
    assert_array_equal(field_diff_directly_computed.field, field_diff.field)
    assert_array_equal(field_scal_directly_computed.field, field_scal.field)
    assert_array_equal(field_2_inv_directly_computed.field, field_2_inv.field)


def test_non_destructivity_of_the_operations():

    shape = (20, 20, 1, 1, 2)
    sigma_init = 4
    sigma_gaussian_filter = 2

    svf_0   = SVF.generate_random_smooth(shape=shape,
                                         sigma=sigma_init,
                                         sigma_gaussian_filter=sigma_gaussian_filter)

    svf_1   = SVF.generate_random_smooth(shape=shape,
                                         sigma=sigma_init,
                                         sigma_gaussian_filter=sigma_gaussian_filter)

    svf_0_back_up = copy.deepcopy(svf_0)
    svf_1_back_up = copy.deepcopy(svf_1)

    assert_array_equal(svf_0.field, svf_0_back_up.field)
    assert_array_equal(svf_1.field, svf_1_back_up.field)
