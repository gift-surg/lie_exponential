import numpy as np
import matplotlib.pyplot as plt
import time

from numpy.testing import assert_array_equal, assert_array_almost_equal
from sympy.core.cache import clear_cache

from visualizer.fields_at_the_window import see_field, see_2_fields, see_jacobian_of_a_field_2d, \
    see_2_jacobian_of_2_fields_2d, see_field_subregion

from utils.fields import Field
from utils.image import Image
from transformations.s_vf import SVF
from transformations.s_disp import SDISP


def test_jacobian_determinant_of_a_translation():
    # here we want to test the visualizers of the following elements
    def function_1(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[1], -1 * x[0]

    # Domain:
    x_dom, y_dom = 20, 20

    field_0 = Field.generate_zero(shape=(x_dom, y_dom, 1, 1, 2))
    jac_0_expected       = Field.generate_zero(shape=(x_dom, y_dom, 1, 1, 4))
    det_jac_0_expected   = Field.generate_zero(shape=(x_dom, y_dom, 1, 1))

    for i in range(0, x_dom):
        for j in range(0, y_dom):
            field_0.field[i, j, 0, 0, :] = function_1(1, [i, j])
            jac_0_expected.field[i, j, 0, 0, :] = [0., 1., -1., 0.]
            det_jac_0_expected.field[i, j, 0, 0] = 1.

    jac_0_computed     = Field.compute_jacobian(field_0)
    det_jac_0_computed = Field.compute_jacobian_determinant(field_0)

    if 0:
        print jac_0_computed.field.shape
        print det_jac_0_computed.field.shape
        print jac_0_computed.field[2, 2, 0, 0, :]
        print det_jac_0_computed.field[2, 2, 0, 0]

    assert_array_equal(jac_0_computed.field, jac_0_expected.field)
    assert_array_equal(det_jac_0_computed.field, det_jac_0_expected.field)


def test_jacobian_determinant_of_any_function():
    # here we want to test the visualizers of the following elements
    def function_1(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[0]*x[1], 3 * x[0]**2 + x[1]

    def jac_map_1(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[1], 2*x[0], 6*x[0], 1

    def det_jac_map_1(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[1] + 12*x[0]**2

    # Domain:
    x_dom, y_dom = 20, 20

    field_0 = Field.generate_zero(shape=(x_dom, y_dom, 1, 1, 2))
    jac_0_expected       = Field.generate_zero(shape=(x_dom, y_dom, 1, 1, 4))
    det_jac_0_expected   = Field.generate_zero(shape=(x_dom, y_dom, 1, 1))

    for i in range(0, x_dom):
        for j in range(0, y_dom):
            field_0.field[i, j, 0, 0, :] = function_1(1, [i, j])
            jac_0_expected.field[i, j, 0, 0, :] = jac_map_1(1, [i, j])
            det_jac_0_expected.field[i, j, 0, 0] = det_jac_map_1(1, [i, j])

    jac_0_computed     = Field.compute_jacobian(field_0)
    det_jac_0_computed = Field.compute_jacobian_determinant(field_0)

    if 1:
        print jac_0_computed.field.shape
        print det_jac_0_computed.field.shape
        print jac_0_computed.field[2, 2, 0, 0, :]
        print det_jac_0_computed.field[2, 2, 0, 0]

    pp = 2
    assert_array_equal(jac_0_computed.field[pp:-pp, pp:-pp, ...], jac_0_expected.field[pp:-pp, pp:-pp, ...])
    #assert_array_equal(det_jac_0_computed.field[pp:-pp, pp:-pp, ...], det_jac_0_expected.field[pp:-pp, pp:-pp, ...])


test_jacobian_determinant_of_a_translation()
test_jacobian_determinant_of_any_function()
