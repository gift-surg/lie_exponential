"""
Test module for the class field, basics.
"""
from utils.fields import Field
import numpy as np
import matplotlib.pyplot as plt
import time

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_almost_equal

from visualizer.fields_at_the_window import see_2_fields
from sympy.core.cache import clear_cache


### TESTS ###


# set to true if you want to see the figures to compare the fields!
open_f = True
seconds_fig = 1

### Initializations test: ###

def test_initialization_field():
    array = np.ones([5, 5, 1, 1, 2])
    field_1 = Field(array)
    assert_array_equal(field_1.field, array)


def test_initialisation_stationary_2d_scalar_field():
    array = np.ones([5, 6])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 1)
    assert_equals(field_1.vol_ext, (5, 6))
    assert_equals(field_1.shape, (5, 6))
    assert_equals(field_1.dim, 2)


def test_initialisation_stationary_3d_scalar_field():
    array = np.ones([5, 6, 7])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 1)
    assert_equals(field_1.vol_ext, (5, 6, 7))
    assert_equals(field_1.shape, (5, 6, 7))
    assert_equals(field_1.dim, 3)


def test_initialisation_time_varying_2d_scalar_field():
    array = np.ones([5, 6, 1, 8])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 8)
    assert_equals(field_1.vol_ext, (5, 6))
    assert_equals(field_1.shape, (5, 6, 1, 8))
    assert_equals(field_1.dim, 2)


def test_initialisation_time_varying_3d_scalar_field():
    array = np.ones([5, 6, 7, 8])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 8)
    assert_equals(field_1.vol_ext, (5, 6, 7))
    assert_equals(field_1.shape, (5, 6, 7, 8))
    assert_equals(field_1.dim, 3)


def test_initialisation_stationary_2d_vector_field():
    array = np.ones([5, 6, 1, 1, 2])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 1)
    assert_equals(field_1.vol_ext, (5, 6))
    assert_equals(field_1.shape, (5, 6, 1, 1, 2))
    assert_equals(field_1.dim, 2)


def test_initialisation_stationary_3d_vector_field():
    array = np.ones([5, 6, 7, 1, 3])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 1)
    assert_equals(field_1.vol_ext, (5, 6, 7))
    assert_equals(field_1.shape, (5, 6, 7, 1, 3))
    assert_equals(field_1.dim, 3)


def test_initialisation_time_varying_2d_vector_field():
    array = np.ones([5, 6, 1, 9, 2])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 9)
    assert_equals(field_1.vol_ext, (5, 6))
    assert_equals(field_1.shape, (5, 6, 1, 9, 2))
    assert_equals(field_1.dim, 2)


def test_initialisation_time_varying_3d_vector_field():
    array = np.ones([5, 6, 7, 9, 3])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 9)
    assert_equals(field_1.vol_ext, (5, 6, 7))
    assert_equals(field_1.shape, (5, 6, 7, 9, 3))
    assert_equals(field_1.dim, 3)


def test_initialisation_stationary_2d_to_nd_vector_field_1():
    array = np.ones([5, 6, 1, 1, 3])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 1)
    assert_equals(field_1.vol_ext, (5, 6))
    assert_equals(field_1.shape, (5, 6, 1, 1, 3))
    assert_equals(field_1.dim, 2)


def test_initialisation_time_varying_2d_to_nd_vector_field_1():
    array = np.ones([5, 6, 1, 9, 3])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 9)
    assert_equals(field_1.vol_ext, (5, 6))
    assert_equals(field_1.shape, (5, 6, 1, 9, 3))
    assert_equals(field_1.dim, 2)


def test_initialisation_stationary_3d_to_nd_vector_field_1():
    array = np.ones([5, 6, 5, 1, 4])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 1)
    assert_equals(field_1.vol_ext, (5, 6, 5))
    assert_equals(field_1.shape, (5, 6, 5, 1, 4))
    assert_equals(field_1.dim, 3)


def test_initialisation_time_varying_3d_to_nd_vector_field_1():
    array = np.ones([5, 6, 5, 9, 4])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 9)
    assert_equals(field_1.vol_ext, (5, 6, 5))
    assert_equals(field_1.shape, (5, 6, 5, 9, 4))
    assert_equals(field_1.dim, 3)


def test_initialisation_stationary_2d_to_nd_vector_field_2():
    array = np.ones([12, 13, 1, 1, 6])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 1)
    assert_equals(field_1.vol_ext, (12, 13))
    assert_equals(field_1.shape, (12, 13, 1, 1, 6))
    assert_equals(field_1.dim, 2)


def test_initialisation_time_varying_2d_to_nd_vector_field_2():
    array = np.ones([12, 13, 1, 9, 6])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 9)
    assert_equals(field_1.vol_ext, (12, 13))
    assert_equals(field_1.shape, (12, 13, 1, 9, 6))
    assert_equals(field_1.dim, 2)


def test_initialisation_stationary_3d_to_nd_vector_field_2():
    array = np.ones([5, 6, 5, 1, 7])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 1)
    assert_equals(field_1.vol_ext, (5, 6, 5))
    assert_equals(field_1.shape, (5, 6, 5, 1, 7))
    assert_equals(field_1.dim, 3)


def test_initialisation_time_varying_3d_to_nd_vector_field_2():
    array = np.ones([5, 6, 5, 9, 8])
    field_1 = Field(array)

    assert_array_equal(field_1.field, array)
    assert_equals(field_1.time_points, 9)
    assert_equals(field_1.vol_ext, (5, 6, 5))
    assert_equals(field_1.shape, (5, 6, 5, 9, 8))
    assert_equals(field_1.dim, 3)


### Vector field operation tests ###


def test_sum_images():
    field_0 = np.random.randn(5, 5, 5, 1, 3)
    field_1 = np.random.randn(5, 5, 5, 1, 3)
    f_0 = Field(field_0)
    f_1 = Field(field_1)
    f_sum = f_0 + f_1
    assert_array_equal(f_0.field, field_0)  # Check is not destructive
    assert_array_equal(f_1.field, field_1)
    assert_array_equal(f_sum.field, field_0 + field_1)


def test_sub_images():
    field_0 = np.random.randn(5, 5, 5, 1, 3)
    field_1 = np.random.randn(5, 5, 5, 1, 3)
    f_0 = Field(field_0)
    f_1 = Field(field_1)
    f_sub = f_0 - f_1
    assert_array_equal(f_0.field, field_0)  # Check is not destructive
    assert_array_equal(f_1.field, field_1)
    assert_array_equal(f_sub.field, field_0 - field_1)


def test_scalar_multiplication_field():
    field_0 = np.random.randn(5, 5, 5, 1, 3)
    alpha = np.pi
    f_0 = Field(field_0)
    f_ans = alpha * f_0
    assert_array_equal(alpha, np.pi)  # Check the operation is not destructive
    assert_array_equal(f_0.field, field_0)
    assert_array_equal(f_ans.field, alpha * field_0)


### Affine homogeneous methods test: ###


def test_to_homogeneous_method():
    array = np.zeros([10, 10, 10, 1, 3])
    for i in range(10):
        for j in range(10):
            for k in range(10):
                array[i, j, k, 0, :] = [i + 1, j + 1, k + 1]

    flag = True
    field_1 = Field(array)
    assert_equals(field_1.homogeneous, False)

    field_1.to_homogeneous()
    assert_equals(field_1.homogeneous, True)
    assert_array_equal(field_1.shape, [10, 10, 10, 1, 4])

    for i in range(10):
        for j in range(10):
            for k in range(10):
                if not np.array_equal(field_1.field[i, j, k, 0, :], [i + 1, j + 1, k + 1, 1]):
                    flag = False

    assert_equals(flag, True)


test_to_homogeneous_method()

def test_to_homogeneous_and_going_back_method():
    array = np.zeros([5, 5, 5, 1, 3])
    for i in range(5):
        for j in range(5):
            for k in range(5):
                array[i, j, k, 0, :] = [i * j + 1, k + 1, j + 1]

    flag = True
    field_1 = Field(array)
    assert_equals(field_1.homogeneous, False)

    field_1.to_homogeneous()
    assert_equals(field_1.homogeneous, True)
    assert_array_equal(field_1.shape, (5, 5, 5, 1, 4))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                if not np.array_equal(field_1.field[i, j, k, 0, :], [i * j + 1, k + 1, j + 1, 1]):
                    flag = False

    assert_equals(flag, True)

    field_1.to_affine()
    assert_equals(field_1.homogeneous, False)
    assert_array_equal(field_1.shape, [5, 5, 5, 1, 3])

    for i in range(3):
        for j in range(3):
            for k in range(3):
                if not np.array_equal(field_1.field[i, j, k, 0, :], [i * j + 1, k + 1, j + 1]):
                    flag = False

    assert_equals(flag, True)


def test_to_homogeneous_fake_input():
    spam_array = np.zeros([5, 5, 5, 1])
    spam_field = Field(spam_array)
    with assert_raises(TypeError):
        spam_field.to_homogeneous()


def test_to_affine_fake_input():
    spam_array = np.zeros([5, 5, 5, 1])
    spam_field = Field(spam_array, homogeneous=True)
    with assert_raises(TypeError):
        spam_field.to_affine()


### Test Inter-class methods ###


### Test Jacobian-methods for fields are in the module test_field_jacobian_computation ###


### Test norm methods ###


def test_norm_of_a_field_2d():
    field_0 = np.ones([5, 5, 1, 1, 2])
    f = Field(field_0)
    assert_equals(f.norm(normalized=False, passe_partout_size=0), np.sqrt(2 * 5 * 5))


def test_norm_of_a_field_3d():
    field_0 = np.ones([5, 5, 5, 1, 3])
    f = Field(field_0)
    assert_equals(f.norm(normalized=False, passe_partout_size=0), np.sqrt(3 * 5 * 5 * 5))


def test_norm_of_a_field_2d_passe_partout():
    field_0 = np.ones([9, 9, 1, 1, 2])
    f = Field(field_0)
    assert_equals(f.norm(normalized=False, passe_partout_size=3), np.sqrt(2 * (9 - 3 * 2) * (9 - 3 * 2)))


def test_norm_of_a_field_3d_passe_partout():
    field_0 = np.ones([9, 9, 7, 1, 3])
    f = Field(field_0)
    assert_equals(f.norm(normalized=False, passe_partout_size=2), np.sqrt(3 * 5 * 5 * 3))


def test_norm_of_a_field_3d_normalized():
    field_0 = np.ones([9, 9, 7, 1, 3])
    f = Field(field_0)
    assert_equals(f.norm(normalized=True, passe_partout_size=0), np.sqrt(3 * 9 * 9 * 7) / np.sqrt(9 * 9 * 7))


def test_norm_of_a_field_3d_passe_partout_normalized():
    field_0 = np.ones([9, 9, 7, 1, 3])
    f = Field(field_0)
    assert_equals(f.norm(normalized=True, passe_partout_size=2), np.sqrt(3 * 5 * 5 * 3) / np.sqrt(5 * 5 * 3))


def test_norm_of_an_images_2d():
    # Using sum of the first n squared number formula.
    field_0 = np.zeros([6, 6])
    for j in range(6):
        field_0[j, :] = [float(j)] * 6
    f = Field(field_0)
    assert_equals(f.norm(normalized=False, passe_partout_size=0), np.sqrt(5 * 6 * 11))


def test_norm_of_an_images_2d_normalized(verbose=False):
    field_0 = np.zeros([12, 12])
    for j in range(12):
        field_0[j, :] = [float(j)] * 12
    if verbose:
        print field_0
    f = Field(field_0)
    assert_almost_equals(f.norm(normalized=True, passe_partout_size=0), np.sqrt(2 * 11 * 12 * 23) / np.sqrt(12 * 12))


def test_norm_of_an_images_2d_normalized_passe_partout(verbose=False):

    def sum_squared_integer_between_m_and_n(m, n):
        if m < n:
            return (1. / 6) * (n * (n + 1) * (2 * n + 1) - (m - 1) * m * (2 * m - 1))

    field_0 = np.zeros([12, 12])
    for j in range(12):
        field_0[j, :] = [float(j)] * 12
    if verbose:
        print field_0
    f = Field(field_0)
    if verbose:
        print f.norm(normalized=True, passe_partout_size=3)
        print np.sqrt(6 * sum_squared_integer_between_m_and_n(3, 8)) / np.sqrt(6 * 6)

    assert_almost_equals(f.norm(normalized=True, passe_partout_size=3),
                         np.sqrt(6 * sum_squared_integer_between_m_and_n(3, 8)) / np.sqrt(6 * 6))


def test_norm_of_an_images_3d_normalized_random():
    field_0 = np.random.randn(10, 10, 10, 1, 3)
    k = np.random.randint(1, 4, 1)[0]
    field_0_reduced = field_0[k:-k, k:-k, k:-k, :, :]
    manual_norm = np.sqrt(sum([a ** 2 for a in field_0_reduced.ravel()])) / np.sqrt((10 - 2 * k) ** 3)
    f = Field(field_0)
    assert_almost_equals(f.norm(normalized=True, passe_partout_size=k), manual_norm)


def test_norm_of_difference_of_images():
    field_a = np.random.randn(5, 5, 5, 1, 3)
    field_b = np.random.randn(5, 5, 5, 1, 3)
    field_sub = field_a - field_b
    f_a = Field(field_a)
    f_b = Field(field_b)

    manual_norm = np.sqrt(sum([a ** 2 for a in field_sub.ravel()])) / np.sqrt(5 ** 3)
    assert_almost_equals(Field.norm_of_difference_of_fields(f_a, f_b, passe_partout_size=0, normalized=True),
                         manual_norm)


### Test generator methods ###


def test_zero_field_basic():
    clear_cache()
    shape = (128, 128, 128, 5)
    f = Field.generate_zero(shape=shape)
    assert_equals(f.__class__.__name__, 'Field')
    assert_equals(f.shape, (128, 128, 128, 5))
    assert_equals(f.dim, 3)
    assert_equals(f.time_points, 5)


def test_random_smooth_field_basic():
    shape = (128, 128, 128, 1)
    f_rand = Field.generate_random_smooth(shape=shape, mean=0, sigma=10, sigma_gaussian_filter=0)
    assert_equals(f_rand.__class__.__name__, 'Field')
    assert_equals(f_rand.shape, (128, 128, 128, 1))
    assert_equals(f_rand.dim, 3)
    assert_equals(f_rand.time_points, 1)
    assert_almost_equal(np.mean(f_rand.field), 0.0, decimal=1)
    assert_almost_equal(np.std(f_rand.field), 10, decimal=1)


def test_random_smooth_field_filtered():
    sigma = 5
    shape = (128, 128, 128, 1)
    f_rand = \
        Field.generate_random_smooth(shape=shape, mean=0, sigma=10, sigma_gaussian_filter=2)
    assert_equals(f_rand.__class__.__name__, 'Field')
    assert_equals(f_rand.shape, (128, 128, 128, 1))
    assert_equals(f_rand.dim, 3)
    assert_equals(f_rand.time_points, 1)
    assert_almost_equal(np.mean(f_rand.field), 0.0, decimal=1)
    assert np.std(f_rand.field) < sigma  # get reduced after gaussian filter.


def test_generate_id_field_2d_1_time_points():
    shape = (10, 10, 1, 1, 2)
    id_field = Field.generate_id(shape=shape)
    assert isinstance(id_field.field, np.ndarray)
    assert_equals(id_field.shape, (10, 10, 1, 1, 2))
    for i in range(10):
        assert_array_equal(id_field.field[i, ..., 0].reshape([1, 10]), [[float(i)] * 10])
        assert_array_equal(id_field.field[:, i, ..., 1].reshape([1, 10]), [[float(i)] * 10])


def test_generate_id_field_2d_5_time_points():
    shape = (12, 12, 1, 5, 2)
    id_field = Field.generate_id(shape=shape)
    assert isinstance(id_field.field, np.ndarray)
    assert_equals(id_field.shape, (12, 12, 1, 5, 2))
    for i in range(10):
        assert_array_equal(id_field.field[i, ..., 0].reshape([5, 12]), np.array([float(i)] * 12 * 5).reshape([5, 12]))
        assert_array_equal(id_field.field[:, i, ..., 1].reshape([5, 12]),
                           np.array([float(i)] * 12 * 5).reshape([5, 12]))


def test_generate_id_field_3d_1_time_points():
    shape = (16, 16, 16, 1, 3)
    id_field = Field.generate_id(shape=shape)
    assert isinstance(id_field.field, np.ndarray)
    assert_equals(id_field.shape, (16, 16, 16, 1, 3))

    slice_m = np.array([[range(16)] * 16]).reshape([16, 16])

    for i in range(10):
        assert_array_equal(id_field.field[i, ..., 0].reshape([16, 16]), np.array([i] * 16 * 16).reshape(16, 16))
        assert_array_equal(id_field.field[:, i, ..., 1].reshape([16, 16]), np.array([i] * 16 * 16).reshape(16, 16))
        assert_array_equal(id_field.field[:, :, i, ..., 2].reshape([16, 16]), np.array([i] * 16 * 16).reshape(16, 16))

        assert_array_equal(id_field.field[:, i, ..., 2].reshape([16, 16]), slice_m)


def test_generate_id_field_3d_5_time_points():
    shape = (16, 16, 16, 5, 3)
    id_field = Field.generate_id(shape=shape)
    assert isinstance(id_field.field, np.ndarray)
    assert_equals(id_field.shape, (16, 16, 16, 5, 3))

    slice_m = np.array([[range(16)] * 16]).reshape([16, 16])

    for i in range(10):
        t = np.random.choice(range(5))
        assert_array_equal(id_field.field[i, ..., t, 0].reshape([16, 16]), np.array([i] * 16 * 16).reshape(16, 16))
        assert_array_equal(id_field.field[:, i, :, t, 1].reshape([16, 16]), np.array([i] * 16 * 16).reshape(16, 16))
        assert_array_equal(id_field.field[:, :, i, t, 2].reshape([16, 16]), np.array([i] * 16 * 16).reshape(16, 16))


def test_generate_id_field_3d_wrong_input_shape():
    shape = (16, 16, 16, 1)
    with assert_raises(IOError):
        Field.generate_id(shape=shape)


def test_generate_id_field_3d_wrong_input_data():
    shape = (16, 16, 16, 1, 2)
    with assert_raises(IOError):
        Field.generate_id(shape=shape)


def test_generate_id_field_from_field_1():
    shape = (16, 16, 16, 5, 3)
    field_1 = Field.generate_zero(shape=shape)
    field_1.field[1, 1, 1, 0, :] = [42, 42, 42]

    field_2 = Field.generate_id_from_obj(field_1)

    # verify attributes of the new element
    assert_array_equal(field_2.shape, (16, 16, 16, 5, 3))
    # verify it is not destructive:
    # modified original field:
    assert_array_equal(field_1.field[1, 1, 1, 0, :], [42, 42, 42])
    # new id field
    assert_array_equal(field_2.field[1, 1, 1, 0, :], [1, 1, 1])


def test_generate_id_field_from_field_spam_input():
    array = np.array([1, 2, 3, 4])
    with assert_raises(IOError):
        Field.generate_id(array)


def test_generate_from_matrix_rotation_and_translation():
    theta = np.pi / 6
    tx, ty = 1.2, 1.8
    m = np.array([[np.cos(theta), -np.sin(theta), tx],
                  [np.sin(theta), np.cos(theta), ty],
                  [0,             0,             1]])

    shape = [5, 5, 1, 1, 2]
    id_field = Field.generate_id(shape=shape)
    id_field.to_homogeneous()

    array_homogeneous = np.zeros([5, 5, 1, 1, 3])
    for i in range(5):
        for j in range(5):
            array_homogeneous[i, j, 0, 0, :] = m.dot(id_field.field[i, j, 0, 0, :])

    gen_field = Field.generate_from_matrix(input_vol_ext=shape[:2], input_matrix=m)

    if open_f:
        see_2_fields(gen_field, Field.from_array(array_homogeneous[..., 0:2]),
                     title_input_0='generated', title_input_1='computed')

        plt.ion()
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

    assert_array_equal(gen_field.field, array_homogeneous[..., 0:2])


def test_generate_from_matrix_rotation_and_translation_scaled():
    theta = np.pi / 3
    tx, ty = 2, -2
    m = np.array([[0.5 * np.cos(theta), -0.5 * np.sin(theta), tx],
                  [0.5 * np.sin(theta), 0.5 * np.cos(theta),  ty],
                  [0,                   0,                    1]])

    shape = [15, 15, 1, 1, 2]
    id_field = Field.generate_id(shape=shape)
    id_field.to_homogeneous()

    array_homogeneous = np.zeros([15, 15, 1, 1, 3])
    for i in range(15):
        for j in range(15):
            array_homogeneous[i, j, 0, 0, :] = m.dot(id_field.field[i, j, 0, 0, :])

    gen_field = Field.generate_from_matrix(input_vol_ext=shape[:2], input_matrix=m)

    if open_f:
        see_2_fields(gen_field, Field.from_array(array_homogeneous[..., 0:2]),
                     title_input_0='generated', title_input_1='computed')

        plt.ion()
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

    assert_array_equal(gen_field.field, array_homogeneous[..., 0:2])
