"""
The Class SVF methods are divided in 5 blocks:
0) Initialization
1) vector space operations
2) Image manager methods  (tested throughout the other tests)
3) Normed space methods
4) Jacobian computation methods (tested in test_image_jacobian_computation.py)
"""

import numpy as np
from sympy.core.cache import clear_cache

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

from transformations.s_vf import SVF
from utils.fields import Field
from utils.image import Image


clear_cache()


### test SVF manager methods ###


def test_init_svf():
    field_0 = Field(np.zeros([5, 5, 5, 1, 3]))
    im = Image.from_field(field_0)
    nib_image = im.nib_image
    svf_0 = SVF(nib_image)
    assert isinstance(svf_0, SVF)
    assert_array_equal(svf_0.shape, field_0.shape)


def test_svf_from_field():
    field_0 = Field(np.zeros([5, 5, 5, 1, 3]))
    svf_0 = SVF.from_field(field_0)
    assert isinstance(svf_0, SVF)
    assert_array_equal(svf_0.shape, field_0.shape)


def test_sfv_from_nifti_image_0():
    field_0 = Field(np.zeros([5, 5, 1, 1, 2]))
    im = Image.from_field(field_0)
    nib_image = im.nib_image
    svf_0 = SVF.from_nifti_image(nib_image)
    assert isinstance(svf_0, SVF)
    assert_array_equal(svf_0.shape, field_0.shape)


def test_svf_from_nifti_image_1():
    field_0 = Field(np.zeros([5, 5, 5, 1, 3]))
    im = Image.from_field(field_0)
    nib_image = im.nib_image
    svf_0 = SVF.from_nifti_image(nib_image)
    assert isinstance(svf_0, SVF)
    assert_array_equal(svf_0.shape, field_0.shape)


def test_svf_from_image_0():
    field_0 = Field(np.zeros([5, 5, 1, 1, 2]))
    im = Image.from_field(field_0)
    svf_0 = SVF.from_image(im)
    assert isinstance(svf_0, SVF)
    assert_array_equal(svf_0.shape, field_0.shape)


def test_svf_from_image_1():
    field_0 = Field(np.zeros([5, 5, 5, 1, 3]))
    im = Image.from_field(field_0)
    svf_0 = SVF.from_image(im)
    assert isinstance(svf_0, SVF)
    assert_array_equal(svf_0.shape, field_0.shape)


### Test SVF operations ###


def test_jacobian_product():
    pass


def test_lie_bracket():
    pass


### Test SVF generators ###


def test_svf_generate_zero():
    shape_input = [30, 30, 30, 1, 3]
    svf_0 = SVF.generate_zero(shape_input, affine=np.eye(4))
    assert_equals(svf_0.__class__.__name__, 'SVF')
    assert_array_equal(svf_0.field, np.zeros(shape_input))
    assert_equals(svf_0.time_points, 1)
    assert_array_equal(svf_0.voxel_2_mm, np.eye(4))
    assert_array_equal(svf_0.mm_2_voxel, np.eye(4))
    assert_array_equal(svf_0.vol_ext, (30, 30, 30))
    assert_array_equal(svf_0.zooms, tuple([1.0] * 5))
    assert_array_equal(svf_0.shape, shape_input)
    assert_equals(svf_0.dim, 3)
    assert_equals(svf_0.is_matrix_data, False)


def test_svf_generate_id():
    shape = (10, 10, 1, 1, 2)
    id_svf = SVF.generate_id(shape=shape)
    assert_equals(id_svf.__class__.__name__, 'SVF')
    assert_equals(id_svf.shape, (10, 10, 1, 1, 2))
    for i in range(10):
        assert_array_equal(id_svf.field[i, ..., 0].reshape(1, 10), [[float(i)] * 10])
        assert_array_equal(id_svf.field[:, i, ..., 1].reshape(1, 10), [[float(i)] * 10])
