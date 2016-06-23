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


# set to true if you want to see the figures to compare the fields!
open_f = True
seconds_fig = 2


### Jacobian tests for the class Field ###

def test_visualizers_toy_examples(open_fig=open_f):

    clear_cache()

    # here we want to test the visualizers of the following elements
    def function_1(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[1], -1 * x[0]

    def function_2(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5 * x[0] + 0.6 * x[1], 0.8 * x[1]

    def jacobian_1(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.0, 1.0, -1.0, 0.0

    def jacobian_2(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5, 0.6, 0.0, 0.8

    field_0 = Field.generate_zero(shape=(20, 20, 1, 1, 2))
    jac_0   = Field.generate_zero(shape=(20, 20, 1, 1, 4))
    field_1 = Field.generate_zero(shape=(20, 20, 1, 1, 2))
    jac_1   = Field.generate_zero(shape=(20, 20, 1, 1, 4))

    assert field_0.__class__.__name__ == 'Field'
    assert jac_0.__class__.__name__ == 'Field'

    assert field_1.__class__.__name__ == 'Field'
    assert jac_1.__class__.__name__ == 'Field'

    for i in range(0, 20):
        for j in range(0, 20):
            field_0.field[i, j, 0, 0, :] = function_1(1, [i, j])
            jac_0.field[i, j, 0, 0, :] = jacobian_1(1, [i, j])
            field_1.field[i, j, 0, 0, :] = function_2(1, [i, j])
            jac_1.field[i, j, 0, 0, :] = jacobian_2(1, [i, j])

    if open_fig:

        see_field(field_0, fig_tag=0)
        plt.ion()
        plt.pause(0.0001)  # ion requires a short pause to work properly!
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

        see_2_fields(field_0, field_1, fig_tag=1)
        plt.ion()
        plt.pause(0.0001)
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

        see_field_subregion(field_0, fig_tag=1, subregion=([5, 10], [5, 10]))
        plt.ion()
        plt.pause(0.0001)
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

        see_jacobian_of_a_field_2d(jac_0, fig_tag=2)
        plt.ion()
        plt.pause(0.0001)
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

        see_2_jacobian_of_2_fields_2d(jac_0, jac_1, fig_tag=3,
                                             title_input_0='ground truth 1',
                                             title_input_1='Approx 1')
        plt.ion()
        plt.pause(0.0001)
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

    # verify visualization methods are not destructive for i in range(0, 20):
    for j in range(0, 20):
        for i in range(0, 20):
            assert_array_equal(function_1(1, [i, j]), field_0.field[i, j, 0, 0, :])
            assert_array_equal(function_2(1, [i, j]), field_1.field[i, j, 0, 0, :])
            assert_array_equal(jacobian_1(1, [i, j]), jac_0.field[i, j, 0, 0, :])
            assert_array_equal(jacobian_2(1, [i, j]), jac_1.field[i, j, 0, 0, :])

test_visualizers_toy_examples()


def test_jacobian_toy_field_1(open_fig=open_f):

    clear_cache()

    def function_field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[1], -1 * x[0]

    def function_jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.0, 1.0, -1.0, 0.0

    field_0      = Field.generate_zero(shape=(20, 20, 1, 1, 2))
    jac_f_ground = Field.generate_zero(shape=(20, 20, 1, 1, 4))

    assert field_0.__class__.__name__ == 'Field'
    assert jac_f_ground.__class__.__name__ == 'Field'

    for i in range(0, 20):
        for j in range(0, 20):
            field_0.field[i, j, 0, 0, :] = function_field_f(1, [i, j])
            jac_f_ground.field[i, j, 0, 0, :] = function_jacobian_f(1, [i, j])

    jac_f_numeric = Field.compute_jacobian(field_0)

    if open_fig:

        see_2_jacobian_of_2_fields_2d(jac_f_ground, jac_f_numeric,
                                      title_input_0='ground truth 1',
                                      title_input_1='Approx 1')

        plt.ion()
        plt.pause(0.0001)
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

    square_size = range(0, 20)
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, 0, 0, :],
                              jac_f_numeric.field[square_size,square_size, 0, 0, :])
    # verify jacobian is not destructive:
    for i in range(0, 20):
        for j in range(0, 20):
            assert_array_equal(function_field_f(1, [i, j]), field_0.field[i, j, 0, 0, :])


def test_jacobian_toy_field_2(open_fig=open_f):

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5 * x[0] + 0.6 * x[1], 0.8 * x[1]

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5, 0.6, 0.0, 0.8

    field_0      = Field.generate_zero(shape=(20, 20, 1, 1, 2))
    jac_f_ground = Field.generate_zero(shape=(20, 20, 1, 1, 4))

    assert field_0.__class__.__name__ == 'Field'
    assert jac_f_ground.__class__.__name__ == 'Field'

    for i in range(0, 20):
        for j in range(0, 20):
            field_0.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_f_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_F_numeric = Field.compute_jacobian(field_0)

    if open_fig:
        see_2_jacobian_of_2_fields_2d(jac_f_ground, jac_F_numeric,
                                      title_input_0='ground truth', title_input_1='Approx')

        plt.ion()
        plt.pause(0.0001)
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

    square_size = range(0, 20)
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, 0,0, :], jac_F_numeric.field[square_size,square_size,0,0,:])
    # verify jacobian is not destructive:
    for i in range(0, 20):
        for j in range(0, 20):
            assert_array_equal(field_f(1, [i, j]), field_0.field[i, j, 0, 0, :])


def test_jacobian_toy_field_3(open_fig=open_f):

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0] ** 2 + 2 * x[0] + x[1], 3.0 * x[0] + 2.0

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2.0 * x[0] + 2.0, 1.0, 3.0, 0.0

    field_0      = Field.generate_zero(shape=(20, 20, 1, 1, 2))
    jac_f_ground = Field.generate_zero(shape=(20, 20, 1, 1, 4))

    assert field_0.__class__.__name__ == 'Field'
    assert jac_f_ground.__class__.__name__ == 'Field'

    for i in range(0, 20):
        for j in range(0, 20):
            field_0.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_f_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_f_numeric = Field.compute_jacobian(field_0)

    if open_fig:
        see_2_jacobian_of_2_fields_2d(jac_f_ground, jac_f_numeric,
                                      title_input_0='ground truth', title_input_1='approx')

        plt.ion()
        plt.pause(0.0001)
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

    square_size = range(1, 19)  # errors are accumulated on the boundaries.
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, 0, 0, :],
                              jac_f_numeric.field[square_size,square_size, 0, 0, :],
                              decimal=6)


def test_jacobian_toy_field_4(open_fig=open_f):

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5 * x[0] * x[1], 0.5 * (x[0] ** 2) * x[1]

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5 * x[1], 0.5 * x[0],\
               2 * 0.5 * x[0] * x[1], 0.5 * (x[0] ** 2)

    field_0      = Field.generate_zero(shape=(20, 20, 1, 1, 2))
    jac_f_ground = Field.generate_zero(shape=(20, 20, 1, 1, 4))

    assert field_0.__class__.__name__ == 'Field'
    assert jac_f_ground.__class__.__name__ == 'Field'

    for i in range(0, 20):
        for j in range(0, 20):
            field_0.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_f_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_f_numeric = Field.compute_jacobian(field_0)

    if open_fig:
        see_2_jacobian_of_2_fields_2d(jac_f_ground, jac_f_numeric,
                     title_input_0='ground', title_input_1='approx')

        plt.ion()
        plt.pause(0.0001)
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

    square_size = range(1, 19)  # the behaviour on the boundary is tested later.
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, 0, 0, :],
                              jac_f_numeric.field[square_size,square_size, 0, 0, :],
                              decimal=6)


def test_jacobian_toy_field_5(open_fig=open_f):

    clear_cache()
    alpha = 0.05

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return np.sin(alpha * x[0]), np.cos(alpha * x[1])

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return alpha * np.cos(alpha * x[0]), 0.,\
               0.0, -alpha * np.sin(alpha * x[1])

    field_0      = Field.generate_zero(shape=(20, 20, 1, 1, 2))
    jac_f_ground = Field.generate_zero(shape=(20, 20, 1, 1, 4))

    assert field_0.__class__.__name__ == 'Field'
    assert jac_f_ground.__class__.__name__ == 'Field'

    for i in range(0, 20):
        for j in range(0, 20):
            field_0.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_f_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_f_numeric = Field.compute_jacobian(field_0)

    if open_fig:
        see_2_jacobian_of_2_fields_2d(jac_f_ground, jac_f_numeric,
                     title_input_0='ground truth', title_input_1='Approx', scale_0=0.1, scale_1=0.1)

        plt.ion()
        plt.pause(0.0001)
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

    square_size = range(1, 19)
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, 0, 0, :],
                              jac_f_numeric.field[square_size, square_size, 0, 0, :],
                              decimal=4)


### NO more visualizations from here to the end! ###

### Jacobian tests for the class IMAGE ###

def test_jacobian_toy_image_1():

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0] ** 2 + 2 * x[0] + x[1], 3.0 * x[0] + 2.0

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2.0 * x[0] + 2.0, 1.0, 3.0, 0.0

    image_0       = Image.generate_zero(shape=(20, 20, 1, 1, 2))
    jac_im_ground = Image.generate_zero(shape=(20, 20, 1, 1, 4))

    assert image_0.__class__.__name__ == 'Image'
    assert jac_im_ground.__class__.__name__ == 'Image'

    for i in range(0, 20):
        for j in range(0, 20):
            image_0.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_im_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_im_numeric = Field.compute_jacobian(image_0)

    assert jac_im_numeric.__class__.__name__ == 'Field'

    square_size = range(1, 19)  # errors are accumulated on the boundaries.
    assert_array_almost_equal(jac_im_ground.field[square_size, square_size, 0, 0, :],
                              jac_im_numeric.field[square_size, square_size, 0, 0, :],
                              decimal=6)
    # verify jacobian is not destructive:
    for i in range(0, 20):
        for j in range(0, 20):
            assert_array_equal(field_f(1, [i, j]), image_0.field[i, j, 0, 0, :])


### Jacobian tests for the class SVF ###

def test_jacobian_toy_svf_1():

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0] ** 2 + 2 * x[0] + x[1], 3.0 * x[0] + 2.0

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2.0 * x[0] + 2.0, 1.0, 3.0, 0.0

    svf_0          = SVF.generate_zero(shape=(20, 20, 1, 1, 2))
    jac_svf_ground = SVF.generate_zero(shape=(20, 20, 1, 1, 4))

    assert svf_0.__class__.__name__ == 'SVF'
    assert jac_svf_ground.__class__.__name__ == 'SVF'

    for i in range(0, 20):
        for j in range(0, 20):
            svf_0.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_svf_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_im_numeric = SVF.compute_jacobian(svf_0)

    assert jac_im_numeric.__class__.__name__ == 'SVF'

    square_size = range(1, 19)  # errors are accumulated on the boundaries.
    assert_array_almost_equal(jac_svf_ground.field[square_size, square_size, 0, 0, :],
                              jac_im_numeric.field[square_size, square_size, 0, 0, :],
                              decimal=6)
    # verify jacobian is not destructive:
    for i in range(0, 20):
        for j in range(0, 20):
            assert_array_equal(field_f(1, [i, j]), svf_0.field[i, j, 0, 0, :])


### Jacobian tests for the class SDISP ###

def test_jacobian_toy_sdisp_1():

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0] ** 2 + 2 * x[0] + x[1], 3.0 * x[0] + 2.0

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2.0 * x[0] + 2.0, 1.0, 3.0, 0.0

    sdisp_0          = SDISP.generate_zero(shape=(20, 20, 1, 1, 2))
    jac_sdisp_ground = SDISP.generate_zero(shape=(20, 20, 1, 1, 4))

    assert sdisp_0.__class__.__name__ == 'SDISP'
    assert jac_sdisp_ground.__class__.__name__ == 'SDISP'

    for i in range(0, 20):
        for j in range(0, 20):
            sdisp_0.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_sdisp_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_im_numeric = SDISP.compute_jacobian(sdisp_0)

    assert jac_im_numeric.__class__.__name__ == 'SDISP'

    square_size = range(1, 19)  # errors are accumulated on the boundaries.
    assert_array_almost_equal(jac_sdisp_ground.field[square_size, square_size, 0, 0, :],
                              jac_im_numeric.field[square_size, square_size, 0, 0, :],
                              decimal=6)
    # verify jacobian is not destructive:
    for i in range(0, 20):
        for j in range(0, 20):
            assert_array_equal(field_f(1, [i, j]), sdisp_0.field[i, j, 0, 0, :])


### Jacobian tests for 3d objects in various contexts ###


def test_jacobian_toy_field_3d_1():

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0] - 0.5 * x[1], x[1] - 0.5 * x[2], x[2] - 0.5 * x[0]

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 1.0, -0.5, 0.0, \
               0.0, 1.0, -0.5, \
               -0.5, 0.0, 1.0

    shape   = (20, 20, 20, 1, 3)
    shape_j = (20, 20, 20, 1, 9)

    svf_0          = SVF.generate_zero(shape=shape)
    jac_svf_ground = SVF.generate_zero(shape=shape_j)

    assert svf_0.__class__.__name__ == 'SVF'
    assert jac_svf_ground.__class__.__name__ == 'SVF'

    for i in range(0, 20):
        for j in range(0, 20):
            for z in range(0, 20):
                svf_0.field[i, j, z, 0, :] = field_f(1, [i, j, z])
                jac_svf_ground.field[i, j, z, 0, :] = jacobian_f(1, [i, j, z])

    jac_f_numeric = SVF.compute_jacobian(svf_0)

    assert jac_f_numeric.__class__.__name__ == 'SVF'

    square_size = range(1, 19)
    assert_array_almost_equal(jac_svf_ground.field[square_size, square_size, square_size, 0, :],
                              jac_f_numeric.field[square_size,square_size, square_size, 0, :],
                              decimal=9)

    # verify jacobian is not destructive:
    for i in range(0, 20):
        for j in range(0, 20):
            for k in range(0, 20):
                assert_array_equal(field_f(1, [i, j, k]), svf_0.field[i, j, k, 0, :])


def test_jacobian_toy_field_3d_2():

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.2 * (x[0] ** 2) * x[1], 0.3 * x[1] * x[2], 0.8 * x[0] * x[1] * x[2]

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.4 * x[0] * x[1], 0.2 * x[0] ** 2,   0.0,       \
               0.0,               0.3 * x[2],        0.3 * x[1],\
               0.8 * x[1] * x[2], 0.8 * x[0] * x[2], 0.8 * x[0] * x[1]

    shape   = (20, 20, 20, 1, 3)
    shape_j = (20, 20, 20, 1, 9)

    svf_f          = SVF.generate_zero(shape=shape)
    jac_f_ground = SVF.generate_zero(shape=shape_j)

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_f_ground.__class__.__name__ == 'SVF'

    for i in range(20):
        for j in range(20):
            for k in range(20):
                #print i, j, k

                svf_f.field[i, j, k, 0, :] = field_f(1, [i, j, k])
                jac_f_ground.field[i, j, k, 0, :] = jacobian_f(1, [i, j, k])

                #print field_f(1, [i, j, k]), svf_f.field[i, j, k, 0, :]

    jac_f_numeric = SVF.compute_jacobian(svf_f)

    square_size = range(1, 19)
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, square_size, 0, :],
                              jac_f_numeric.field[square_size,square_size, square_size, 0, :],
                              decimal=9)

    # verify jacobian is not destructive:
    for i in range(20):
        for j in range(20):
            for k in range(20):
                assert_array_equal(field_f(1, [i, j, k]), svf_f.field[i, j, k, 0, :])


test_visualizers_toy_examples()
test_jacobian_toy_field_1()
test_jacobian_toy_field_2()
test_jacobian_toy_field_3()
test_jacobian_toy_field_4()
test_jacobian_toy_field_5()
test_jacobian_toy_image_1()
test_jacobian_toy_svf_1()
test_jacobian_toy_sdisp_1()
test_jacobian_toy_field_3d_1()
test_jacobian_toy_field_3d_2()


# TODO: test jacobian determinant method! Refactor the following tests!

'''
### Jacobian determinant tests ###


def test_jacobian_determinant_toy_field_2d_1():
    # build a toy field: with 1 time points
    array = np.zeros([2, 2, 1, 2, 4])

    array[0, 0, 0, 0, :] = [1., 2., 3., 6.]
    array[1, 0, 0, 0, :] = [1., 0., 0., 1.]
    array[0, 1, 0, 0, :] = [0., 1., 1., 0.]
    array[1, 1, 0, 0, :] = [2., 1., 6., 2.]

    array[0, 0, 0, 1, :] = [2., 1., 6., 2.]
    array[1, 0, 0, 1, :] = [0., 1., 1., 0.]
    array[0, 1, 0, 1, :] = [1., 0., 0., 1.]
    array[1, 1, 0, 1, :] = [2., 1., 6., 3.]

    array_ground_truth_det = np.zeros([2, 2, 1, 2])

    array_ground_truth_det[0, 0, 0, 0] = 0.
    array_ground_truth_det[1, 0, 0, 0] = 1.
    array_ground_truth_det[0, 1, 0, 0] = -1.
    array_ground_truth_det[1, 1, 0, 0] = -2.

    array_ground_truth_det[0, 0, 0, 1] = -2.
    array_ground_truth_det[1, 0, 0, 1] = -1.
    array_ground_truth_det[0, 1, 0, 1] = 1.
    array_ground_truth_det[1, 1, 0, 1] = 0.

    m = array.reshape([2, 2, 1, 2, 2, 2])

    # verify ground truth is the same
    assert_array_equal(array_ground_truth_det, np.linalg.det(m))

    # create image from field
    im = Image.from_array(array)
    print array.shape
    print type(im)
    print im.shape

    # compute the jacobian
    im_det = im.compute_jacobian_determinant()

    # verify type
    assert im_det.__class__.__name__ == 'Image'

    # verify intent header
    hdr = im_det.get_header()
    intent = hdr.get_intent()
    assert_equals(intent[2], 'Jacobian Det')

    # verify the shape is the same
    assert_equals(array_ground_truth_det.shape, im_det.shape)

    # verify im_det corresponds to the ground truth
    assert_array_almost_equal(array_ground_truth_det, im_det.field)


test_jacobian_determinant_toy_field_2d_1()


def test_jacobian_determinant_toy_field_3d_0():
    # build a toy field: with 2 time points
    field = np.zeros([2, 2, 2, 1, 9])

    field[0, 0, 0, 0, :] = [1., 0., 0.,
                            0., 2., 3.,
                            0., 1., 4.]
    field[1, 0, 0, 0, :] = [3., 0., 1.,
                            0., 1., 0.,
                            2., 0., -1.]
    field[0, 1, 0, 0, :] = [2., 5., 0.,
                            1., 3., 0.,
                            0., 0., 1.]
    field[0, 0, 1, 0, :] = [1., 0., 0.,
                            0., 2., 3.,
                            0., -1., -2.]
    field[0, 1, 1, 0, :] = [1., 3., 1.,
                            0., 1., 4.,
                            1., 3., 2.]
    field[1, 0, 1, 0, :] = [5., 2., 1.,
                            1., 0., 1.,
                            3., 2., -6.]
    field[1, 1, 0, 0, :] = [4., 5., 1.,
                            2., 1., 1.,
                            3., 2., 0.]
    field[1, 1, 1, 0, :] = [0., 0., 1.,
                            2., 0., 0.,
                            0., 1., 0.]

    field_ground_truth_det = np.zeros([2, 2, 2, 1])

    field_ground_truth_det[0, 0, 0, 0] = 5.
    field_ground_truth_det[1, 0, 0, 0] = -5.
    field_ground_truth_det[0, 1, 0, 0] = 1.
    field_ground_truth_det[0, 0, 1, 0] = -1.
    field_ground_truth_det[0, 1, 1, 0] = 1.
    field_ground_truth_det[1, 0, 1, 0] = 10.
    field_ground_truth_det[1, 1, 0, 0] = 8.
    field_ground_truth_det[1, 1, 1, 0] = 2.

    m = field.reshape([2, 2, 2, 1, 3, 3])
    det_m = np.linalg.det(m)

    # verify ground truth is well computed
    assert_array_almost_equal(field_ground_truth_det, det_m)

    # create image from field
    im = image.Image.from_field(field)

    # compute the jacobian
    im_det = im.compute_jacobian_determinant(jacobian_image=image.Image.from_field(field))

    # verify type
    assert im_det.__class__.__name__ == 'Image'

    # verify intent header
    hdr = im_det.get_header()
    intent = hdr.get_intent()
    assert_equals(intent[2], 'Jacobian Det')

    # verify the shape is the same
    assert_equals(field_ground_truth_det.shape, im_det.shape)

    # verify im_det corresponds to the ground truth
    assert_array_almost_equal(field_ground_truth_det, im_det.field)


def test_jacobian_determinant_toy_field_3d_1():
    # build a toy field: with 2 time points
    field = np.zeros([2, 2, 2, 2, 9])

    field[0, 0, 0, 0, :] = [1., 0., 0.,
                            0., 2., 3.,
                            0., 1., 4.]
    field[1, 0, 0, 0, :] = [3., 0., 1.,
                            0., 1., 0.,
                            2., 0., -1.]
    field[0, 1, 0, 0, :] = [2., 5., 0.,
                            1., 3., 0.,
                            0., 0., 1.]
    field[0, 0, 1, 0, :] = [1., 0., 0.,
                            0., 2., 3.,
                            0., -1., -2.]
    field[0, 1, 1, 0, :] = [1., 3., 1.,
                            0., 1., 4.,
                            1., 3., 2.]
    field[1, 0, 1, 0, :] = [5., 2., 1.,
                            1., 0., 1.,
                            3., 2., -6.]
    field[1, 1, 0, 0, :] = [4., 5., 1.,
                            2., 1., 1.,
                            3., 2., 0.]
    field[1, 1, 1, 0, :] = [0., 0., 1.,
                            2., 0., 0.,
                            0., 1., 0.]

    field[0, 0, 0, 1, :] = [2., 0., 0.,
                            0., 0., 1.,
                            0., 3., 0.]
    field[1, 0, 0, 1, :] = [0., 1., 0.,
                            2., 0., 1.,
                            5., 0., 1.]
    field[0, 1, 0, 1, :] = [1., 0., 0.,
                            0., 0., 1.,
                            0., 0., 0.]
    field[0, 0, 1, 1, :] = [2., 3., 1.,
                            1., 0., 0.,
                            2., 1., 1.]
    field[0, 1, 1, 1, :] = [0., 1., 3.,
                            9., 7., 0.,
                            2., 1., 0.]
    field[1, 0, 1, 1, :] = [3., 2., 0.,
                            2., 0., 1.,
                            1., 4., 0.]
    field[1, 1, 0, 1, :] = [0., 1., 0.,
                            3., 0., 4.,
                            2., 0., 1.]
    field[1, 1, 1, 1, :] = [0., 0., 1.,
                            1., 0., 0.,
                            0., 1., 0.]

    field_ground_truth_det = np.zeros([2, 2, 2, 2])

    field_ground_truth_det[0, 0, 0, 0] = 5.
    field_ground_truth_det[1, 0, 0, 0] = -5.
    field_ground_truth_det[0, 1, 0, 0] = 1.
    field_ground_truth_det[0, 0, 1, 0] = -1.
    field_ground_truth_det[0, 1, 1, 0] = 1.
    field_ground_truth_det[1, 0, 1, 0] = 10.
    field_ground_truth_det[1, 1, 0, 0] = 8.
    field_ground_truth_det[1, 1, 1, 0] = 2.

    field_ground_truth_det[0, 0, 0, 1] = -6.
    field_ground_truth_det[1, 0, 0, 1] = 3.
    field_ground_truth_det[0, 1, 0, 1] = 0.
    field_ground_truth_det[0, 0, 1, 1] = -2.
    field_ground_truth_det[0, 1, 1, 1] = -15.
    field_ground_truth_det[1, 0, 1, 1] = -10.
    field_ground_truth_det[1, 1, 0, 1] = 5.
    field_ground_truth_det[1, 1, 1, 1] = 1.


    m = field.reshape([2, 2, 2, 2, 3, 3])
    det_m = np.linalg.det(m)

    # verify ground truth is well computed
    assert_array_almost_equal(field_ground_truth_det, det_m, decimal=4)

    # create image from field
    im = image.Image.from_field(field)

    # compute the jacobian
    im_det = im.compute_jacobian_determinant(jacobian_image=image.Image.from_field(field))

    # verify type
    assert im_det.__class__.__name__ == 'Image'

    # verify intent header
    hdr = im_det.get_header()
    intent = hdr.get_intent()
    assert_equals(intent[2], 'Jacobian Det')

    # verify the shape is the same
    assert_equals(field_ground_truth_det.shape, im_det.shape)

    # verify im_det corresponds to the ground truth
    assert_array_almost_equal(field_ground_truth_det, im_det.field)


def test_jacobian_determinant_toy_field_3d_close_1():

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0] - 0.5 * x[1], x[1] - 0.5 * x[2], x[2] - 0.5 * x[0]

    def jacobian_det_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 1 - 0.5 ** 3

    svf_f            = svf.generate_id_svf(shape=(20, 20, 20, 0, 3))
    jac_det_f_ground = image.generate_empty_image((20, 20, 20, 1))

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_det_f_ground.__class__.__name__ == 'Image'

    for i in range(0, 20):
        for j in range(0, 20):
            for z in range(0, 20):
                svf_f.field[i, j, z, 0, :] = field_f(1, [i, j, z])
                jac_det_f_ground.field[i, j, z, 0] = jacobian_det_f(1, [i, j, z])

    jac_det_f_numeric = svf_f.compute_jacobian_determinant()

    square_size = range(0, 20)
    assert_array_almost_equal(jac_det_f_ground.field[square_size, square_size, square_size, 0],
                              jac_det_f_numeric.field[square_size,square_size, square_size, 0],
                              decimal=9)


def test_jacobian_determinant_toy_field_2d_close():

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0] - 0.5 * x[1], x[1] - 0.5 * x[2], x[2] - 0.5 * x[0]

    def jacobian_det_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 1 - 0.5 ** 3

    svf_f            = svf.generate_id_svf(shape=(20, 20, 20, 0, 3))
    jac_det_f_ground = image.generate_empty_image((20, 20, 20, 1))

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_det_f_ground.__class__.__name__ == 'Image'

    for i in range(0, 20):
        for j in range(0, 20):
            for z in range(0, 20):
                svf_f.field[i, j, z, 0, :] = field_f(1, [i, j, z])
                jac_det_f_ground.field[i, j, z, 0] = jacobian_det_f(1, [i, j, z])

    jac_det_f_numeric = svf_f.compute_jacobian_determinant()

    square_size = range(0, 20)
    assert_array_almost_equal(jac_det_f_ground.field[square_size, square_size, square_size, 0],
                              jac_det_f_numeric.field[square_size,square_size, square_size, 0],
                              decimal=9)


def test_jacobian_determinant_toy_field_3d_close_2():

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 3 * x[0] * x[2], 2 * x[0] * x[1], 1./3 * x[0] * x[1] + x[2]

    def jacobian_det_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 6.0 * x[0] * x[2]

    svf_f            = svf.generate_id_svf(shape=(20, 20, 20, 0, 3))
    jac_det_f_ground = image.generate_empty_image((20, 20, 20, 1))

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_det_f_ground.__class__.__name__ == 'Image'

    for i in range(0, 20):
        for j in range(0, 20):
            for z in range(0, 20):
                svf_f.field[i, j, z, 0, :] = field_f(1, [i, j, z])
                jac_det_f_ground.field[i, j, z, 0] = jacobian_det_f(1, [i, j, z])

    jac_det_f_numeric = svf_f.compute_jacobian_determinant()

    square_size = range(0, 20)
    assert_array_almost_equal(jac_det_f_ground.field[square_size, square_size, square_size, 0],
                              jac_det_f_numeric.field[square_size,square_size, square_size, 0],
                              decimal=9)


def test_jacobian_determinant_toy_field_3d_close_3():

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2 * x[0] * x[1], 3 * x[2] + x[1], 0.5 * x[2] ** 2 + 2 * x[2]

    def jacobian_det_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2 * x[1] * (x[2] + 2)

    svf_f            = svf.generate_id_svf(shape=(20, 20, 20, 0, 3))
    jac_det_f_ground = image.generate_empty_image((20, 20, 20, 1))

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_det_f_ground.__class__.__name__ == 'Image'

    print svf_f.field.shape


    for i in range(0, 20):
        for j in range(0, 20):
            for k in range(0, 20):
                svf_f.field[i, j, k, 0, :] = field_f(1, [i, j, k])
                jac_det_f_ground.field[i, j, k, 0] = jacobian_det_f(1, [i, j, k])

    jac_det_f_numeric = svf_f.compute_jacobian_determinant()

    square_size = range(1, 19)
    assert_array_almost_equal(jac_det_f_ground.field[square_size, square_size, square_size, 0],
                              jac_det_f_numeric.field[square_size,square_size, square_size, 0],
                              decimal=6)


# TODO: jacobian for multiple time points!


def test_jacobian_determinant_toy_field_2d_rand():
    field = np.random.randn(30, 30, 1, 1, 4)

    manual_det = np.linalg.det(field.reshape([30, 30, 1, 1, 2, 2]))

    # create image from field
    im = image.Image.from_field(field)

    # compute the jacobian
    im_det = im.compute_jacobian_determinant(jacobian_image=image.Image.from_field(field))

    # verify type
    assert im_det.__class__.__name__ == 'Image'

    # verify intent header
    hdr = im_det.get_header()
    intent = hdr.get_intent()
    assert_equals(intent[2], 'Jacobian Det')

    # verify the shape is the same
    assert_equals(manual_det.shape, im_det.shape)

    # verify im_det corresponds to the ground truth
    assert_array_almost_equal(manual_det, im_det.field)


def test_jacobian_determinant_toy_field_3d_time_rand():

    field = np.random.randn(30, 30, 30, 5, 9)

    manual_det = np.linalg.det(field.reshape([30, 30, 30, 5, 3, 3]))

    # create image from field
    im = image.Image.from_field(field)

    # compute the jacobian
    im_det = im.compute_jacobian_determinant(jacobian_image=image.Image.from_field(field))

    # verify type
    assert im_det.__class__.__name__ == 'Image'

    # verify intent header
    hdr = im_det.get_header()
    intent = hdr.get_intent()
    assert_equals(intent[2], 'Jacobian Det')

    # verify the shape is the same
    assert_equals(manual_det.shape, im_det.shape)

    # verify im_det corresponds to the ground truth
    assert_array_almost_equal(manual_det, im_det.field)


# TODO: Jacobian for images with multiple time points.
# TODO: see jacobian is not destuctive
# ### Jacobian test on the boundary domain ###
'''


