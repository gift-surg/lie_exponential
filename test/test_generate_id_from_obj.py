import numpy as np

from utils.fields import Field
from utils.image import Image
from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_almost_equal

from visualizer.fields_at_the_window import see_2_fields
from sympy.core.cache import clear_cache


### TESTS generate_id_from_obj. Crossing-classes method ###


# set to true if you want to see the figures to compare the fields!
open_f = True
seconds_fig = 1


# Field #


def test_generate_id_from_object_field():
    shape = (10, 10, 1, 1, 2)

    # create elements:
    f        = Field.generate_random_smooth(shape=shape)
    id_field = Field.generate_id(shape=shape)
    f_gen    = Field.generate_id_from_obj(f)

    # test class
    assert_equals(f.__class__.__name__,        'Field')
    assert_equals(id_field.__class__.__name__, 'Field')
    assert_equals(f_gen.__class__.__name__,    'Field')

    # test field instance and shape
    assert isinstance(f_gen.field, np.ndarray)
    assert_equals(id_field.shape, (10, 10, 1, 1, 2))

    # test if the id is really an id:
    assert_array_equal(id_field.field, f_gen.field)

    # test by slices:
    for i in range(10):
        assert_array_equal(id_field.field[i, ..., 0].reshape(1, 10), [[float(i)] * 10])
        assert_array_equal(id_field.field[:, i, ..., 1].reshape(1, 10), [[float(i)] * 10])

    # verify this is not destructive!
    assert not np.array_equal(f.field[2, 2, 0, 0, :], f_gen.field[2, 2, 0, 0, :])


# Image #


def test_generate_id_from_object_image():
    shape = (10, 10, 1, 1, 2)

    # create elements:
    im     = Image.generate_random_smooth(shape=shape)
    im_id  = Image.generate_id(shape=shape)
    im_gen = Image.generate_id_from_obj(im)

    # test class
    assert_equals(im.__class__.__name__,     'Image')
    assert_equals(im_id.__class__.__name__,  'Image')
    assert_equals(im_gen.__class__.__name__, 'Image')

    # test field instance and shape
    assert isinstance(im_gen.field, np.ndarray)
    assert_equals(im_id.shape, (10, 10, 1, 1, 2))

    # test if the id is really an id:
    assert_array_equal(im_id.field, im_gen.field)

    # test by slices:
    for i in range(10):
        assert_array_equal(im_id.field[i, ..., 0].reshape(1, 10), [[float(i)] * 10])
        assert_array_equal(im_id.field[:, i, ..., 1].reshape(1, 10), [[float(i)] * 10])

    # verify this is not destructive!
    assert not np.array_equal(im.field[2, 2, 0, 0, :], im_gen.field[2, 2, 0, 0, :])


# SVF #


def test_generate_id_from_object_svf():
    shape = (10, 10, 1, 1, 2)

    # create elements:
    svf     = SVF.generate_random_smooth(shape=shape)
    svf_id  = SVF.generate_id(shape=shape)
    svf_gen = SVF.generate_id_from_obj(svf)

    # test class
    assert_equals(svf.__class__.__name__,     'SVF')
    assert_equals(svf_id.__class__.__name__,  'SVF')
    assert_equals(svf_gen.__class__.__name__, 'SVF')

    # test field instance and shape
    assert isinstance(svf_gen.field, np.ndarray)
    assert_equals(svf_id.shape, (10, 10, 1, 1, 2))

    # test if the id is really an id:
    assert_array_equal(svf_id.field, svf_gen.field)

    # test by slices:
    for i in range(10):
        assert_array_equal(svf_id.field[i, ..., 0].reshape(1, 10), [[float(i)] * 10])
        assert_array_equal(svf_id.field[:, i, ..., 1].reshape(1, 10), [[float(i)] * 10])

    # verify this is not destructive!
    assert not np.array_equal(svf.field[2, 2, 0, 0, :], svf_gen.field[2, 2, 0, 0, :])


# SDISP #


def test_generate_id_from_object_sdisp():
    shape = (10, 10, 1, 1, 2)

    # create elements:
    sdisp     = SDISP.generate_random_smooth(shape=shape)
    sdisp_id  = SDISP.generate_id(shape=shape)
    sdisp_gen = SDISP.generate_id_from_obj(sdisp)

    # test class
    assert_equals(sdisp.__class__.__name__,     'SDISP')
    assert_equals(sdisp_id.__class__.__name__,  'SDISP')
    assert_equals(sdisp_gen.__class__.__name__, 'SDISP')

    # test field instance and shape
    assert isinstance(sdisp_gen.field, np.ndarray)
    assert_equals(sdisp_id.shape, (10, 10, 1, 1, 2))

    # test if the id is really an id:
    assert_array_equal(sdisp_id.field, sdisp_gen.field)

    # test by slices:
    for i in range(10):
        assert_array_equal(sdisp_id.field[i, ..., 0].reshape(1, 10), [[float(i)] * 10])
        assert_array_equal(sdisp_id.field[:, i, ..., 1].reshape(1, 10), [[float(i)] * 10])

    # verify this is not destructive!
    assert not np.array_equal(sdisp.field[2, 2, 0, 0, :], sdisp_gen.field[2, 2, 0, 0, :])


# TEST exceptions: generate from object different from the class that calls the method


def test_fake_generate_id_from_object_field_image():
    """
    Call generate_id_from_obj from SVF for class sdisp, and vice versa
    """
    shape = (10, 10, 1, 1, 2)
    f  = Field.generate_random_smooth(shape=shape)
    im = Image.generate_random_smooth(shape=shape)

    with assert_raises(TypeError) as error:
        Field.generate_id_from_obj(im)
    ex = error.exception
    print ex

    with assert_raises(TypeError) as error:
        Image.generate_id_from_obj(f)
    ex = error.exception
    print ex


def test_fake_generate_id_from_object_image_svf():
    shape = (10, 10, 1, 1, 2)
    im = Image.generate_random_smooth(shape=shape)
    svf  = SVF.generate_random_smooth(shape=shape)
    with assert_raises(TypeError) as error:
        Image.generate_id_from_obj(svf)
    ex = error.exception
    print ex

    with assert_raises(TypeError) as error:
        SVF.generate_id_from_obj(im)
    ex = error.exception
    print ex


def test_fake_generate_id_from_object_svf_sdisp():
    shape = (10, 10, 1, 1, 2)

    svf  = SVF.generate_random_smooth(shape=shape)
    sdisp = SDISP.generate_random_smooth(shape=shape)
    with assert_raises(TypeError) as error:
        SVF.generate_id_from_obj(sdisp)
    ex = error.exception
    print ex

    with assert_raises(TypeError) as error:
        SDISP.generate_id_from_obj(svf)
    ex = error.exception
    print ex


def test_fake_generate_id_from_object_sdisp_field():
    shape = (10, 10, 1, 1, 2)


    sdisp = SDISP.generate_random_smooth(shape=shape)
    f  = Field.generate_random_smooth(shape=shape)
    with assert_raises(TypeError) as error:
        SDISP.generate_id_from_obj(f)
    ex = error.exception
    print ex

    with assert_raises(TypeError) as error:
        Field.generate_id_from_obj(sdisp)
    ex = error.exception
    print ex


'''
test_generate_id_from_object_field()
test_generate_id_from_object_image()
test_generate_id_from_object_svf()
test_generate_id_from_object_sdisp()

test_fake_generate_id_from_object_field_image()
test_fake_generate_id_from_object_image_svf()
test_fake_generate_id_from_object_svf_sdisp()
test_fake_generate_id_from_object_sdisp_field()
'''