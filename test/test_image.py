"""
The Class Image's methods are divided in 5 blocks:
0) Initialization
1) vector space operations
2) Image manager methods  (tested throughout the other tests)
3) Normed space methods
4) Jacobian computation methods (tested in test_image_jacobian_computation.py)
5) Generator Methods - methods that initializes the vector fields.
"""

import numpy as np
from sympy.core.cache import clear_cache
import nibabel as nib

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_almost_equal

from utils.image import Image
from utils.fields import Field

clear_cache()


### 0 - Initialization methods tests ###

def test_empty_image_field():
    field = np.ones([5, 5, 1, 1, 2])
    affine = np.eye(4)
    nib_im = nib.Nifti1Image(field, affine)
    im = Image(nib_im)
    print im.nib_image.get_header()
    assert_array_equal(im.field, field)


def test_empty_image_dim2():
    field = np.ones([5, 5, 1, 1, 2])
    affine = np.array([[3, 0, 0, 2], [0, 1, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]])
    nib_im = nib.Nifti1Image(field, affine)
    im = Image(nib_im)
    assert_array_equal(im.field, field)
    assert im.dim == 2


def test_empty_image_dim3():
    field = np.ones([5, 5, 5, 1, 2])
    affine = np.array([[3, 0, 0, 2], [0, 1, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]])
    nib_im = nib.Nifti1Image(field, affine)
    im = Image(nib_im)
    assert_array_equal(im.field, field)
    assert im.dim == 3


def test_empty_image_affine_voxel_2_mm():
    field = np.ones([5, 5, 1, 1, 2])
    affine = np.array([[3, 0, 0, 2], [0, 1, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]])
    nib_im = nib.Nifti1Image(field, affine)
    im = Image(nib_im)
    assert_array_equal(im.voxel_2_mm, affine)


def test_empty_image_affine_mm_2_voxel():
    field = np.ones([5, 5, 1, 1, 2])
    affine = np.array([[3, 0, 0, 2], [0, 1, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]])
    inv_affine = np.linalg.inv(affine)
    nib_im = nib.Nifti1Image(field, affine)
    im = Image(nib_im)
    assert_array_equal(im.mm_2_voxel, inv_affine)


### 1 - Vector space operations tests: ###


def test_sum_images():
    array_0 = np.random.randn(5, 5, 5, 1, 3)
    array_1 = np.random.randn(5, 5, 5, 1, 3)
    field_0 = Field(array_0)
    field_1 = Field(array_1)

    m_0 = Image.from_field(field_0)
    m_1 = Image.from_field(field_1)
    m_sum = m_0 + m_1
    assert_array_equal(m_0.field, array_0)  # Check is not destructive
    assert_array_equal(m_1.field, array_1)
    assert_array_equal(m_sum.field, array_0 + array_1)


def test_sum_images_attributes():
    array_0 = np.random.randn(5, 5, 5, 1, 3)
    array_1 = np.random.randn(5, 5, 5, 1, 3)
    field_0 = Field(array_0)
    field_1 = Field(array_1)

    m_0 = Image.from_field(field_0)
    m_1 = Image.from_field(field_1)
    m_sum = m_0 + m_1
    assert_array_equal(m_sum.voxel_2_mm, m_0.voxel_2_mm)
    assert_equals(m_sum.nib_image.get_header(), m_0.nib_image.get_header())


def test_sum_images_fake_input_type():
    array_0 = np.random.randn(5, 5, 5, 1, 3)
    field_0 = Field(array_0)
    m_0 = Image.from_field(field_0)
    m_1 = 'Spam!'
    with assert_raises(TypeError):
        m_0.__add__(m_1)


def test_sum_images_fake_shape():
    array_0 = np.random.randn(5, 5, 5, 1, 3)
    array_1 = np.random.randn(5, 6, 5, 1, 3)
    field_0 = Field(array_0)
    field_1 = Field(array_1)
    m_0 = Image.from_field(field_0)
    m_1 = Image.from_field(field_1)
    with assert_raises(TypeError):
        m_0.__add__(m_1)


def test_sum_images_fake_affine_transform():
    array_0 = np.random.randn(5, 5, 5, 1, 3)
    array_1 = np.random.randn(5, 5, 5, 1, 3)
    field_0 = Field(array_0)
    field_1 = Field(array_1)
    aff_0 = np.eye(4)
    aff_1 = np.array([[0,  2,  0,  4],
                     [-2,  0,  0,  3],
                     [0,  0,  4, -2],
                     [0,  0,  0,  1]])
    m_0 = Image.from_field(field_0, affine=aff_0)
    m_1 = Image.from_field(field_1, affine=aff_1)
    with assert_raises(TypeError):
        m_0.__add__(m_1)


def test_sub_images():
    array_0 = np.random.randn(5, 5, 5, 1, 3)
    array_1 = np.random.randn(5, 5, 5, 1, 3)
    field_0 = Field(array_0)
    field_1 = Field(array_1)
    m_0 = Image.from_field(field_0)
    m_1 = Image.from_field(field_1)
    m_sub = m_0 - m_1
    assert_array_equal(m_0.field, array_0)  # Check is not destructive
    assert_array_equal(m_1.field, array_1)
    assert_array_equal(m_sub.field, array_0 - array_1)


def test_sub_images_attributes():
    array_0 = np.random.randn(5, 5, 5, 1, 3)
    array_1 = np.random.randn(5, 5, 5, 1, 3)
    field_0 = Field(array_0)
    field_1 = Field(array_1)
    m_0 = Image.from_field(field_0)
    m_1 = Image.from_field(field_1)
    m_sub = m_0 - m_1
    assert_array_equal(m_sub.voxel_2_mm, m_0.voxel_2_mm)
    assert_equals(m_sub.nib_image.get_header(), m_0.nib_image.get_header())


def test_sub_images_fake_shape():
    array_0 = np.random.randn(5, 5, 5, 1, 3)
    array_1 = np.random.randn(5, 6, 5, 1, 3)
    field_0 = Field(array_0)
    field_1 = Field(array_1)
    m_0 = Image.from_field(field_0)
    m_1 = Image.from_field(field_1)
    with assert_raises(TypeError):
        m_0.__sub__(m_1)


def test_sub_images_fake_affine_transform():
    array_0 = np.random.randn(5, 5, 5, 1, 3)
    array_1 = np.random.randn(5, 5, 5, 1, 3)
    field_0 = Field(array_0)
    field_1 = Field(array_1)
    aff_0 = np.eye(4)
    aff_1 = np.array([[0,  2,  0,  4],
                     [-2,  0,  0,  3],
                     [0,  0,  4, -2],
                     [0,  0,  0,  1]])
    m_0 = Image.from_field(field_0, affine=aff_0)
    m_1 = Image.from_field(field_1, affine=aff_1)
    with assert_raises(TypeError):
        m_0.__sub__(m_1)


def test_sub_to_zero_images():
    array_z = np.zeros([5, 5, 5, 1, 3])
    array_a = np.random.randn(5, 5, 5, 1, 3)
    field_z = Field(array_z)
    field_a = Field(array_a)
    m_z = Image.from_field(field_z)
    m_a = Image.from_field(field_a)
    assert_array_equal(m_z.field, (m_a - m_a).field)


def test_scalar_multiplication_images():
    array_0 = np.random.randn(5, 5, 5, 1, 3)
    field_0 = Field(array_0)
    alpha = np.pi
    m_0 = Image.from_field(field_0)
    m_ans = Image.from_field(alpha * field_0)

    m_mul = alpha * m_0

    assert isinstance(m_mul, m_0.__class__)
    assert_array_equal(alpha, np.pi)  # Check the operation is not destructive
    assert_array_equal(m_0.field, array_0)
    assert_array_equal(m_ans.field, m_mul.field)


### 3 - Normed space methods tests: ###
# Assuming linalg.norm is tested enough, we move to the test results only for the
# Frobenius norm.

def test_norm_of_a_field_2d():
    array_0 = np.ones([5, 5, 1, 1, 2])
    field_0 = Field(array_0)
    im = Image.from_field(field_0)
    assert_equals(im.norm(normalized=False, passe_partout_size=0),
                  np.sqrt(2 * 5 * 5))


def test_norm_of_a_field_3d():
    array_0 = np.ones([5, 5, 5, 1, 3])
    field_0 = Field(array_0)
    im = Image.from_field(field_0)
    assert_equals(im.norm(normalized=False, passe_partout_size=0),
                  np.sqrt(3 * 5 * 5 * 5))


def test_norm_of_a_field_2d_passe_partout():
    array_0 = np.ones([9, 9, 1, 1, 2])
    field_0 = Field(array_0)
    im = Image.from_field(field_0)
    assert_equals(im.norm(normalized=False, passe_partout_size=3),
                  np.sqrt(2 * (9 - 3 * 2) * (9 - 3 * 2)))


def test_norm_of_a_field_3d_passe_partout():
    array_0 = np.ones([9, 9, 7, 1, 3])
    field_0 = Field(array_0)
    im = Image.from_field(field_0)
    assert_equals(im.norm(normalized=False, passe_partout_size=2),
                  np.sqrt(3 * 5 * 5 * 3))


def test_norm_of_a_field_3d_normalized():
    array_0 = np.ones([9, 9, 7, 1, 3])
    field_0 = Field(array_0)
    im = Image.from_field(field_0)
    assert_equals(im.norm(normalized=True, passe_partout_size=0),
                  np.sqrt(3 * 9 * 9 * 7) / np.sqrt(9 * 9 * 7))


def test_norm_of_a_field_3d_passe_partout_normalized():
    array_0 = np.ones([9, 9, 7, 1, 3])
    field_0 = Field(array_0)
    im = Image.from_field(field_0)
    assert_equals(im.norm(normalized=True, passe_partout_size=2),
                  np.sqrt(3 * 5 * 5 * 3) / np.sqrt(5 * 5 * 3))


def test_norm_of_an_images_2d():
    # Using sum of the first n squared number formula.
    field_0 = Field(np.zeros([6, 6]))
    for j in range(6):
        field_0.field[j, :] = [float(j)] * 6
    im = Image.from_field(field_0)
    assert_equals(im.norm(normalized=False, passe_partout_size=0), np.sqrt(5 * 6 * 11))


def test_norm_of_an_images_2d_normalized(verbose=False):
    field_0 = Field(np.zeros([12, 12]))
    for j in range(12):
        field_0.field[j, :] = [float(j)] * 12
    if verbose:
        print field_0.field
    im = Image.from_field(field_0)
    assert_almost_equals(im.norm(normalized=True, passe_partout_size=0),
                         np.sqrt(2 * 11 * 12 * 23) / np.sqrt(12 * 12))


def test_norm_of_an_images_2d_normalized_passe_partout(verbose=True):

    def sum_squared_integer_between_m_and_n(m, n):
        if m < n:
            return (1. / 6) * (n * (n + 1) * (2 * n + 1) - (m - 1) * m * (2 * m - 1))

    array_0 = np.zeros([12, 12])
    for j in range(12):
        array_0[j, :] = [float(j)] * 12
    if verbose:
        print array_0

    field_0 = Field(array_0)
    im = Image.from_field(field_0)
    if verbose:
        print im.norm(normalized=True, passe_partout_size=3)
        print np.sqrt(6 * sum_squared_integer_between_m_and_n(3, 8)) / np.sqrt(6 * 6)

    assert_almost_equals(im.norm(normalized=True, passe_partout_size=3),
                         np.sqrt(6 * sum_squared_integer_between_m_and_n(3, 8)) / np.sqrt(6 * 6))


test_norm_of_an_images_2d_normalized_passe_partout()



def test_norm_of_an_images_3d_normalized_ranodm():
    array_0 = np.random.randn(10, 10, 10, 1, 3)
    k = np.random.randint(1, 4, 1)[0]  # random passepartout size.
    field_0_reduced = array_0[k:-k, k:-k, k:-k, :, :]
    manual_norm = np.sqrt(sum([a ** 2 for a in field_0_reduced.ravel()])) / np.sqrt((10 - 2 * k) ** 3)
    im = Image.from_field(Field(array_0))
    assert_almost_equals(im.norm(normalized=True, passe_partout_size=k), manual_norm)


def test_norm_of_difference_of_images():
    array_a = np.random.randn(5, 5, 5, 1, 3)
    array_b = np.random.randn(5, 5, 5, 1, 3)
    array_sub = array_a - array_b
    m_a = Image.from_field(Field(array_a))
    m_b = Image.from_field(Field(array_b))
    manual_norm = np.sqrt(sum([a ** 2 for a in array_sub.ravel()])) / np.sqrt(5 ** 3)
    assert_almost_equals(Image.norm_of_difference_of_images(m_a, m_b, passe_partout_size=0, normalized=True),
                         manual_norm)


### Test Image generator methods ###


def test_image_constructor_empty_image_3d():
    shape_input = [30, 30, 30, 5, 3]
    im_0 = Image.generate_zero(shape_input, affine=np.eye(4))
    assert_equals(im_0.__class__.__name__, 'Image')
    assert_array_equal(im_0.field, np.zeros(shape_input))
    assert_equals(im_0.time_points, 5)
    assert_array_equal(im_0.voxel_2_mm, np.eye(4))
    assert_array_equal(im_0.mm_2_voxel, np.eye(4))
    assert_array_equal(im_0.vol_ext, (30, 30, 30))
    assert_array_equal(im_0.zooms, tuple([1.0] * 5))
    assert_array_equal(im_0.shape, shape_input)
    assert_equals(im_0.dim, 3)
    assert_equals(im_0.is_matrix_data, False)


def test_image_constructor_empty_image_2d():
    shape_input = [30, 30, 1, 1, 2]
    im_0 = Image.generate_zero(shape_input, affine=np.eye(4))
    assert_equals(im_0.__class__.__name__, 'Image')
    assert_array_equal(im_0.field, np.zeros(shape_input))
    assert_equals(im_0.time_points, 1)
    assert_array_equal(im_0.voxel_2_mm, np.eye(4))
    assert_array_equal(im_0.mm_2_voxel, np.eye(4))
    assert_array_equal(im_0.vol_ext, (30, 30))
    assert_array_equal(im_0.zooms, tuple([1.0] * 5))
    assert_array_equal(im_0.shape, shape_input)
    assert_equals(im_0.dim, 2)
    assert_equals(im_0.is_matrix_data, False)


def test_image_constructor_empty_image_1d():
    shape_input = [1, 1, 1]
    im_0 = Image.generate_zero(shape_input, affine=np.eye(4))
    assert_equals(im_0.__class__.__name__, 'Image')
    assert_array_equal(im_0.field, np.zeros(shape_input))
    assert_equals(im_0.time_points, 0)
    assert_array_equal(im_0.vol_ext, (1, 1, 1))
    assert_array_equal(im_0.zooms, tuple([1.0] * 3))
    assert_equals(im_0.dim, 0)


def test_image_constructor_empty_image_non_id_affine():
    shape_input = [30, 30, 30, 1, 3]
    aff = np.array([[0,  2,  0,  4],
                    [-2,  0,  0,  3],
                    [0,  0,  4, -2],
                    [0,  0,  0,  1]])
    inv_aff = np.array([[0., -0.5,  0.,  1.5],
                        [0.5,  0.,  0., -2.],
                        [0.,  0.,  0.25,  0.5],
                        [0.,  0., 0., 1.]])

    im_0 = Image.generate_zero(shape_input, affine=aff)
    assert_array_equal(im_0.voxel_2_mm, aff)
    assert_array_equal(im_0.mm_2_voxel, inv_aff)


def test_image_constructor_empty_image_fake_affine_type():
    shape_input = [30, 30, 30, 1, 3]
    aff = 'Spam!'
    with assert_raises(ValueError):
        Image.generate_zero(shape_input, affine=aff)


def test_image_constructor_empty_image_fake_affine_shape():
    shape_input = [30, 30, 30, 1, 3]
    aff = np.eye(5)
    with assert_raises(ValueError):
        Image.generate_zero(shape_input, affine=aff)


def test_load_matrix_data_attributes_for_empty_image():
    shape_input = [30, 30, 30, 1, 3]
    im_0 = Image.generate_zero(shape_input)
    assert_equals(im_0.num_matrix_rows, 0)
    assert_equals(im_0.num_matrix_rows, 0)
    assert_equals(im_0.is_matrix_data, False)
    # Nibabel programmers set to the string 'none' the not initialized intent (!!)
    assert_equals(im_0.nib_image.get_header().get_intent()[0], 'none')
    assert_equals(im_0.nib_image.get_header().get_intent()[1], ())
    assert_equals(im_0.nib_image.get_header().get_intent()[2], '')


def test_generate_id_field_2d_1_time_points():
    shape = (10, 10, 1, 1, 2)
    id_im = Image.generate_id(shape=shape)
    assert isinstance(id_im.field, np.ndarray)
    assert isinstance(id_im, Image)
    assert_equals(id_im.shape, (10, 10, 1, 1, 2))
    for i in range(10):
        assert_array_equal(id_im.field[i, ..., 0].reshape(1, 10), [[float(i)] * 10])
        assert_array_equal(id_im.field[:, i, ..., 1].reshape(1, 10), [[float(i)] * 10])


def test_generate_id_field_2d_5_time_points():
    shape = (12, 12, 1, 5, 2)
    id_field = Field.generate_id(shape=shape)
    isinstance(id_field, np.ndarray)
    assert_equals(id_field.shape, (12, 12, 1, 5, 2))
    for i in range(10):
        assert_array_equal(id_field.field[i, ..., 0].reshape(5, 12), np.array([float(i)] * 12 * 5).reshape(5, 12))
        assert_array_equal(id_field.field[:, i, ..., 1].reshape(5, 12), np.array([float(i)] * 12 * 5).reshape(5, 12))


def test_random_smooth_image_filtered():
    sigma = 5
    shape = (128, 128, 128, 1, 3)
    im_rand = Image.generate_random_smooth(shape, sigma_gaussian_filter=2, sigma=sigma)
    assert_equals(im_rand.__class__.__name__, 'Image')
    assert_equals(im_rand.shape, (128, 128, 128, 1, 3))
    assert_equals(im_rand.dim, 3)
    assert_almost_equal(np.mean(im_rand.field), 0, decimal=2)
    # we expect std to get reduced after gaussian filter.
    # ... not the best test in the history...
    assert np.std(im_rand.field) < sigma


def test_generate_id_field_2d_1_time_points_1():
    id_field = Image.generate_id(shape=(10, 10, 1, 1, 2))
    isinstance(id_field, np.ndarray)
    assert_equals(id_field.shape, (10, 10, 1, 1, 2))
    for i in range(10):
        assert_array_equal(id_field.field[i, ..., 0].reshape(1, 10), [[float(i)] * 10])
        assert_array_equal(id_field.field[:, i, ..., 1].reshape(1, 10), [[float(i)] * 10])


def test_generate_id_field_3d_wrong_input_shape():
    shape = (16, 16, 16, 1)
    with assert_raises(IOError):
        Image.generate_id(shape)


def test_generate_id_field_3d_wrong_input_data():
    shape = (16, 16, 16, 1, 2)
    with assert_raises(IOError):
        Image.generate_id(shape)


