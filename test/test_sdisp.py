
"""
The Class SDISP methods are divided in 5 blocks:
0) Initialization
1) vector space operations
2) Image manager methods  (tested throughout the other tests)
3) Normed space methods
4) Jacobian computation methods (tested in test_image_jacobian_computation.py)
"""

import numpy as np
from sympy.core.cache import clear_cache
import nibabel as nib
import os
import warnings

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

from transformations.s_disp import SDISP
from utils.fields import Field
from utils.image import Image

clear_cache()


### test SDISP manager methods ###


def test_init_sdisp():
    field_0 = Field(np.zeros([5, 5, 5, 1, 3]))
    im = Image.from_field(field_0)
    nib_image = im.nib_image
    sdisp_0 = SDISP(nib_image)
    assert isinstance(sdisp_0, SDISP)
    assert_array_equal(sdisp_0.shape, field_0.shape)


def test_sdisp_from_field():
    field_0 = Field(np.zeros([5, 5, 5, 1, 3]))
    sdisp_0 = SDISP.from_field(field_0)
    assert isinstance(sdisp_0, SDISP)
    assert_array_equal(sdisp_0.shape, field_0.shape)


def test_sfv_from_nifti_image_0():
    field_0 = Field(np.zeros([5, 5, 1, 1, 2]))
    im = Image.from_field(field_0)
    nib_image = im.nib_image
    sdisp_0 = SDISP.from_nifti_image(nib_image)
    assert isinstance(sdisp_0, SDISP)
    assert_array_equal(sdisp_0.shape, field_0.shape)


def test_sdisp_from_nifti_image_1():
    field_0 = Field(np.zeros([5, 5, 5, 1, 3]))
    im = Image.from_field(field_0)
    nib_image = im.nib_image
    sdisp_0 = SDISP.from_nifti_image(nib_image)
    assert isinstance(sdisp_0, SDISP)
    assert_array_equal(sdisp_0.shape, field_0.shape)


def test_sdisp_from_image_0():
    field_0 = Field(np.zeros([5, 5, 1, 1, 2]))
    im = Image.from_field(field_0)
    sdisp_0 = SDISP.from_image(im)
    assert isinstance(sdisp_0, SDISP)
    assert_array_equal(sdisp_0.shape, field_0.shape)


def test_sdisp_from_image_1():
    field_0 = Field(np.zeros([5, 5, 5, 1, 3]))
    im = Image.from_field(field_0)
    sdisp_0 = SDISP.from_image(im)
    assert isinstance(sdisp_0, SDISP)
    assert_array_equal(sdisp_0.shape, field_0.shape)


### Test SDISP operations ###

# Inhibition tests:


def test_inhibition_add():
    field_0 = Field(np.zeros([5, 5, 5, 1, 3]))
    im_0 = Image.from_field(field_0)
    sdisp_0 = SDISP.from_image(im_0)
    field_1 = Field(np.zeros([5, 5, 5, 1, 3]))
    im_1 = Image.from_field(field_1)
    sdisp_1 = SDISP.from_image(im_1)

    with warnings.catch_warnings(record=True):
        sdisp_0.__add__(sdisp_1)
        # print 2+2  # TODO: correct this!


test_inhibition_add()

### Test SDISP generators ###


def test_sdisp_generate_zero():
    shape_input = [30, 30, 30, 1, 3]
    sdisp_0 = SDISP.generate_zero(shape_input, affine=np.eye(4))
    assert_equals(sdisp_0.__class__.__name__, 'SDISP')
    assert_array_equal(sdisp_0.field, np.zeros(shape_input))
    assert_equals(sdisp_0.time_points, 1)
    assert_array_equal(sdisp_0.voxel_2_mm, np.eye(4))
    assert_array_equal(sdisp_0.mm_2_voxel, np.eye(4))
    assert_array_equal(sdisp_0.vol_ext, (30, 30, 30))
    assert_array_equal(sdisp_0.zooms, tuple([1.0] * 5))
    assert_array_equal(sdisp_0.shape, shape_input)
    assert_equals(sdisp_0.dim, 3)
    assert_equals(sdisp_0.is_matrix_data, False)


def test_sdisp_generate_id():
    shape = (10, 10, 1, 1, 2)
    id_sdisp = SDISP.generate_id(shape=shape)
    assert_equals(id_sdisp.__class__.__name__, 'SDISP')
    assert_equals(id_sdisp.shape, (10, 10, 1, 1, 2))
    for i in range(10):
        assert_array_equal(id_sdisp.field[i, ..., 0].reshape(1, 10), [[float(i)] * 10])
        assert_array_equal(id_sdisp.field[:, i, ..., 1].reshape(1, 10), [[float(i)] * 10])

