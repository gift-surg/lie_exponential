import numpy as np
from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_almost_equal
from scipy.linalg import expm

from utils.projective_algebras import ProjectiveAlgebra, ProjectiveGroup


### TESTS projective general linear algebra and group ###


def test_init_pgl_a_fake_input():
    a = np.array([1, 2, 3])
    with assert_raises(IOError):
        ProjectiveAlgebra(d=3, m=a)


def test_init_pgl_a_good_input():
    a = np.array(range(9)).reshape([3, 3])
    dd = 2
    m1 = ProjectiveAlgebra(d=dd, m=a)
    assert_array_equal(m1.matrix, a)
    assert_equals(dd, m1.dim)
    assert_array_equal(m1.shape, [3, 3])


def test_randomgen_pgl_a():
    m2 = ProjectiveAlgebra.randomgen()
    m4 = ProjectiveAlgebra.randomgen(d=4)
    assert isinstance(m2, ProjectiveAlgebra)
    assert isinstance(m4, ProjectiveAlgebra)
    assert_array_equal(m2.shape, [3, 3])
    assert_array_equal(m4.shape, [5, 5])


def test_exponentiate_pgl_a_1():
    m4_a = ProjectiveAlgebra.randomgen(d=4)
    m4_m = m4_a.matrix

    exp_of_m4_a = m4_a.exponentiate()
    exp_of_m4_m = expm(m4_m)

    # check class
    assert isinstance(exp_of_m4_a, ProjectiveGroup)

    # check values
    assert_array_equal(exp_of_m4_a.matrix, exp_of_m4_m)


def test_exponentiate_pgl_a_2():
    m6_a = ProjectiveAlgebra.randomgen(d=6)
    m6_m = m6_a.matrix

    exp_of_m6_a = m6_a.exponentiate()
    exp_of_m6_m = expm(m6_m)

    # check class and dim
    assert isinstance(exp_of_m6_a, ProjectiveGroup)
    assert exp_of_m6_a.dim == 6
    assert_array_equal(exp_of_m6_a.shape, [7, 7])

    # check values of matrix
    assert_array_equal(exp_of_m6_a.matrix, exp_of_m6_m)


def test_ode_solution_pgl_a_1():
    m_a = ProjectiveAlgebra.randomgen()
    m_m = m_a.matrix

    exp_of_m_m = expm(m_m)
    init_cond = np.array([2, 3.5, 1])

    s = exp_of_m_m.dot(init_cond)

    assert_array_equal(s, m_a.ode_solution(init_cond=init_cond, affine_coordinates=False))


def test_ode_solution_pgl_a_2():
    m_a = ProjectiveAlgebra.randomgen(d=3)
    m_m = m_a.matrix

    exp_of_m_m = expm(m_m)
    init_cond = np.array([2, 3.5, 1, 1])

    s = exp_of_m_m.dot(init_cond)

    assert_array_equal(s[0:3]/s[3], m_a.ode_solution(init_cond=init_cond, affine_coordinates=True))


def test_generated_psl_a():
    a = np.array(range(9)).reshape([3, 3])
    dd = 2
    m1 = ProjectiveAlgebra(d=dd, m=a, special=False)

    assert_array_equal(m1.matrix, a)
    assert_equals(dd, m1.dim)
    assert_array_equal(m1.shape, [3, 3])
    with assert_raises(IOError):
        # assert is not in the special linear algebra
        ProjectiveAlgebra(d=3, m=a, special=True)


def test_generated_psl_a_1():
    #
    a = np.array(range(9)).reshape([3, 3])
    a[2, 2] = -4
    dd = 2
    m1 = ProjectiveAlgebra(d=dd, m=a, special=True)

    # special linear algebra element must have trace = 0.
    assert_equals(np.trace(m1.matrix), 0)
    # special linear group element should have det = 1.
    assert_almost_equal(np.linalg.det(m1.exponentiate().matrix), 1)


def test_randomgen_psl_a_2():
    dd = 2
    m1 = ProjectiveAlgebra.randomgen(d=dd, special=True)

    # special linear algebra element must have trace = 0.
    assert_almost_equal(np.trace(m1.matrix), 0)
    # special linear group element should have det = 1.
    assert_almost_equal(np.linalg.det(m1.exponentiate().matrix), 1)
