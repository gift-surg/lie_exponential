import numpy as np
import matplotlib.pyplot as plt
import time

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_array_almost_equal

from transformations.s_vf import SVF
from utils.image import Image
from visualizer.fields_at_the_window import see_field, see_2_fields
from sympy.core.cache import clear_cache


### Jacobian tests for the class Image 2d ###


def test_jacobian_toy_field_1():
    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[1], -1 * x[0]

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.0, 1.0, -1.0, 0.0

    svf_f        = SVF.generate_id(shape=(20, 20, 1, 1, 2))
    jac_f_ground = Image.initialise_jacobian(svf_f)

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_f_ground.__class__.__name__ == 'Image'

    for i in range(0, 20):
        for j in range(0, 20):
            svf_f.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_f_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_f_numeric = SVF.compute_jacobian(svf_f)

    square_size = range(0, 20)
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, 0, 0, :], 
                              jac_f_numeric.field[square_size, square_size, 0, 0, :])


def test_jacobian_toy_field_2():
    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5 * x[0] + 0.6 * x[1], 0.8 * x[1]

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5, 0.6, 0.0, 0.8

    svf_f        = SVF.generate_id(shape=(20, 20, 1, 1, 2))
    jac_f_ground = Image.initialise_jacobian(svf_f)

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_f_ground.__class__.__name__ == 'Image'

    for i in range(0, 20):
        for j in range(0, 20):
            svf_f.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_f_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_f_numeric = SVF.compute_jacobian(svf_f)

    square_size = range(0, 20)
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, 0, 0, :],
                              jac_f_numeric.field[square_size, square_size, 0, 0, :])


def test_jacobian_toy_field_3():
    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0] ** 2 + 2 * x[0] + x[1], 3.0 * x[0] + 2.0

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2.0 * x[0] + 2.0, 1.0, 3.0, 0.0

    svf_f        = SVF.generate_id(shape=(30, 30, 1, 1, 2))
    jac_f_ground = Image.initialise_jacobian(svf_f)

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_f_ground.__class__.__name__ == 'Image'

    for i in range(0, 30):
        for j in range(0, 30):
            svf_f.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_f_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_f_numeric = SVF.compute_jacobian(svf_f)

    pp = 2
    assert_array_almost_equal(jac_f_ground.field[pp:-pp, pp:-pp, 0, 0, :],
                              jac_f_numeric.field[pp:-pp, pp:-pp, 0, 0, :])


test_jacobian_toy_field_1()
test_jacobian_toy_field_2()
test_jacobian_toy_field_3()


### "Jacobian product" u*v = Jv.u for SVF tests ###


def test_jacobian_product_toy_example_2d_1():

    def field_vector_u(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0]**2 + 2*x[0] + x[1], 3 * x[0] + 2 * x[1]

    def field_vector_v(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[1] - x[0], 2 * x[0]

    def ground_jac_product_u_v(t, x):
        t = float(t); x = [float(y) for y in x]
        return - x[0]**2 + x[0] + x[1], 2 * x[0]**2 + 4 * x[0] + 2 * x[1]

    def ground_jac_product_v_u(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[0]*x[1] - 2*x[0]**2 + 2*x[1], 3 * x[1] + x[0]

    # init:
    u   = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))
    v   = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))

    ground_u_v = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))
    ground_v_u = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))

    # construction:
    for i in range(0, 20):
        for j in range(0, 20):
            u.field[i, j, 0, 0, :]   = field_vector_u(1, [i, j])
            v.field[i, j, 0, 0, :]   = field_vector_v(1, [i, j])
            ground_u_v.field[i, j, 0, 0, :] = ground_jac_product_u_v(1, [i, j])
            ground_v_u.field[i, j, 0, 0, :] = ground_jac_product_v_u(1, [i, j])

    jac_prod_u_v = SVF.jacobian_product(u, v)
    jac_prod_v_u = SVF.jacobian_product(v, u)

    assert isinstance(jac_prod_u_v, SVF)
    pp = 1
    assert_array_almost_equal(jac_prod_u_v.field[pp:-pp, pp:-pp, 0, 0, :], ground_u_v.field[pp:-pp, pp:-pp, 0, 0, :])
    assert_array_almost_equal(jac_prod_v_u.field[pp:-pp, pp:-pp, 0, 0, :], ground_v_u.field[pp:-pp, pp:-pp, 0, 0, :])


def test_jacobian_product_toy_example_2d_2():

    def field_vector_u(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0]*x[1] + 2*x[0], x[1]

    def field_vector_v(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[0]*x[1] + 1, x[0]**2

    def ground_jac_product_u_v(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[0]*x[1]**2 + 6*x[0]*x[1], 2 * (x[0]**2)*x[1] + 4*x[0]**2

    def ground_jac_product_v_u(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[0]*(x[1]**2) + x[1] + 4*x[0]*x[1] + 2 + x[0]**3, x[0]**2

    # init:
    u   = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))
    v   = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))

    ground_u_v = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))
    ground_v_u = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))

    # construction:
    for i in range(0, 20):
        for j in range(0, 20):
            u.field[i, j, 0, 0, :]   = field_vector_u(1, [i, j])
            v.field[i, j, 0, 0, :]   = field_vector_v(1, [i, j])
            ground_u_v.field[i, j, 0, 0, :] = ground_jac_product_u_v(1, [i, j])
            ground_v_u.field[i, j, 0, 0, :] = ground_jac_product_v_u(1, [i, j])

    jac_prod_u_v = SVF.jacobian_product(u, v)
    jac_prod_v_u = SVF.jacobian_product(v, u)

    assert isinstance(jac_prod_u_v, SVF)
    pp = 1
    assert_array_almost_equal(jac_prod_u_v.field[pp:-pp, pp:-pp, 0, 0, :], ground_u_v.field[pp:-pp, pp:-pp, 0, 0, :])
    assert_array_almost_equal(jac_prod_v_u.field[pp:-pp, pp:-pp, 0, 0, :], ground_v_u.field[pp:-pp, pp:-pp, 0, 0, :])


def test_jacobian_product_toy_example_3d():

    def field_vector_u(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0]*x[2], x[1], 2*x[0] + x[2]

    def field_vector_v(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[1], x[0]*x[2], 3*x[2]

    def ground_jac_product_u_v(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[1], x[0]*x[2]**2 +2*x[0]**2 + x[0]*x[2], 6*x[0] + 3*x[2]

    def ground_jac_product_v_u(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[1]*x[2] + 3*x[0]*x[2], x[0]*x[2], 4*x[1] + 3*x[2]

    # init:
    u   = SVF.from_array(np.zeros([20, 20, 20, 1, 3]))
    v   = SVF.from_array(np.zeros([20, 20, 20, 1, 3]))

    ground_u_v = SVF.from_array(np.zeros([20, 20, 20, 1, 3]))
    ground_v_u = SVF.from_array(np.zeros([20, 20, 20, 1, 3]))

    for i in range(0, 20):
        for j in range(0, 20):
            for k in range(0, 20):
                u.field[i, j, k, 0, :]   = field_vector_u(1, [i, j, k])
                v.field[i, j, k, 0, :]   = field_vector_v(1, [i, j, k])
                ground_u_v.field[i, j, k, 0, :] = ground_jac_product_u_v(1, [i, j, k])
                ground_v_u.field[i, j, k, 0, :] = ground_jac_product_v_u(1, [i, j, k])

    jac_prod_u_v = SVF.jacobian_product(u, v)
    jac_prod_v_u = SVF.jacobian_product(v, u)

    assert isinstance(jac_prod_u_v, SVF)
    pp = 1
    assert_array_almost_equal(jac_prod_u_v.field[pp:-pp, pp:-pp, 0, 0, :], ground_u_v.field[pp:-pp, pp:-pp, 0, 0, :])
    assert_array_almost_equal(jac_prod_v_u.field[pp:-pp, pp:-pp, 0, 0, :], ground_v_u.field[pp:-pp, pp:-pp, 0, 0, :])


test_jacobian_product_toy_example_2d_1()
test_jacobian_product_toy_example_2d_2()
test_jacobian_product_toy_example_3d()


''' Test iterative jacobian product '''


def test_iterative_jacobian_product_toy_example_2d_1():
    pass



'''
def test_iterative_jacobian_product_toy_example_2d_1():

    def field_vector_u(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0]**2 + 2*x[0] + x[1],   3 * x[0] + 2 * x[1]

    def ground_jac_product_u_3_times(t, x):
        t = float(t); x = [float(y) for y in x]
        return - x[0]**2 + x[0] + x[1], 2 * x[0]**2 + 4 * x[0] + 2 * x[1]

    def ground_jac_product_v_u(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[0]*x[1] - 2*x[0]**2 + 2*x[1], 3 * x[1] + x[0]

    # init:
    u   = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))
    v   = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))

    ground_u_v = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))
    ground_v_u = SVF.from_array(np.zeros([20, 20, 1, 1, 2]))

    # construction:
    for i in range(0, 20):
        for j in range(0, 20):
            u.field[i, j, 0, 0, :]   = field_vector_u(1, [i, j])
            v.field[i, j, 0, 0, :]   = field_vector_v(1, [i, j])
            ground_u_v.field[i, j, 0, 0, :] = ground_jac_product_u_v(1, [i, j])
            ground_v_u.field[i, j, 0, 0, :] = ground_jac_product_v_u(1, [i, j])

    jac_prod_u_v = SVF.jacobian_product(u, v)
    jac_prod_v_u = SVF.jacobian_product(v, u)

    assert isinstance(jac_prod_u_v, SVF)
    pp = 1
    assert_array_almost_equal(jac_prod_u_v.field[pp:-pp, pp:-pp, 0, 0, :], ground_u_v.field[pp:-pp, pp:-pp, 0, 0, :])
    assert_array_almost_equal(jac_prod_v_u.field[pp:-pp, pp:-pp, 0, 0, :], ground_v_u.field[pp:-pp, pp:-pp, 0, 0, :])


'''

'''


def test_jacobian_toy_field_3(open_fig=open_f):
    clear_cache()



    svf_f        = svf.generate_id_svf()
    jac_f_ground = image.Image.initialise_jacobian_from_image(svf_f)

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_f_ground.__class__.__name__ == 'Image'

    for i in range(0, 20):
        for j in range(0, 20):
            svf_f.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_f_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_f_numeric = svf_f.compute_jacobian()

    if open_fig:
        see_2_fields(jac_f_ground.field[..., 0:2], jac_f_numeric.field[..., 0:2],
                     title_input_0='ground truth 3', title_input_1='numeric approximation 3')

        plt.ion()
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

    square_size = range(1, 19)  # the behaviour on the boundary is tested later.
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, 0, 0, :],
                              jac_f_numeric.field[square_size,square_size, 0, 0, :])


def test_jacobian_toy_field_4(open_fig=open_f):

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5 * x[0] * x[1], 0.5 * (x[0] ** 2) * x[1]

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5 * x[1], 0.5 * x[0],\
               2 * 0.5 * x[0] * x[1], 0.5 * (x[0] ** 2)

    svf_f        = svf.generate_id_svf()
    jac_f_ground = image.Image.initialise_jacobian_from_image(svf_f)

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_f_ground.__class__.__name__ == 'Image'

    for i in range(0, 20):
        for j in range(0, 20):
            # svf defined by the field F
            svf_f.field[i, j, 0, 0, :] = field_f(1, [i, j])
            # ground truth jacobian
            jac_f_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_f_numeric = svf_f.compute_jacobian()

    if open_fig:
        see_2_fields(jac_f_ground.field[..., 0:2], jac_f_numeric.field[..., 0:2],
                     title_input_0='ground truth 4', title_input_1='numeric approximation 4')

        plt.ion()
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

    # the smaller the better for trigonometric functions.
    alpha = 0.05

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return np.sin(alpha * x[0]), np.cos(alpha * x[1])

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return alpha * np.cos(alpha * x[0]), 0.,\
               0.0, -alpha * np.sin(alpha * x[1])

    svf_f        = svf.generate_id_svf()
    jac_f_ground = image.Image.initialise_jacobian_from_image(svf_f)

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_f_ground.__class__.__name__ == 'Image'

    for i in range(0, 20):
        for j in range(0, 20):
            svf_f.field[i, j, 0, 0, :] = field_f(1, [i, j])
            jac_f_ground.field[i, j, 0, 0, :] = jacobian_f(1, [i, j])

    jac_f_numeric = svf_f.compute_jacobian()

    if open_fig:
        see_2_fields(jac_f_ground.field[..., 0:2], jac_f_numeric.field[..., 0:2],
                     title_input_0='ground truth 5', title_input_1='numeric approximation 5')

        plt.ion()
        plt.show()
        time.sleep(seconds_fig)
        plt.close()
        plt.ioff()

    square_size = range(1, 19)
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, 0, 0, :],
                              jac_f_numeric.field[square_size,square_size, 0, 0, :],
                              decimal=4)


### Jacobian tests for the class Image 3d ###


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

    svf_f        = svf.generate_id_svf(shape=(20, 20, 20, 0, 3))
    jac_f_ground = image.Image.initialise_jacobian_from_image(svf_f)

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_f_ground.__class__.__name__ == 'Image'

    for i in range(0, 20):
        for j in range(0, 20):
            for z in range(0, 20):
                svf_f.field[i, j, z, 0, :] = field_f(1, [i, j, z])
                jac_f_ground.field[i, j, z, 0, :] = jacobian_f(1, [i, j, z])

    jac_f_numeric = svf_f.compute_jacobian()

    square_size = range(1, 19)
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, square_size, 0, :],
                              jac_f_numeric.field[square_size,square_size, square_size, 0, :],
                              decimal=9)


def test_jacobian_toy_field_3d_2():

    clear_cache()

    def field_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.2 * (x[0] ** 2) * x[1], 0.3 * x[1] * x[2], 0.8 * x[0] * x[1] * x[2]

    def jacobian_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.4 * x[0] * x[1], 0.2 * x[0] ** 2,   0.0,       \
               0.0,               0.3 * x[2],        0.3 * x[1],\
               0.8 * x[1] * x[2], 0.8 * x[0] * x[2], 0.8 * x[0] * x[1],

    svf_f        = svf.generate_id_svf(shape=(20, 20, 20, 0, 3))
    jac_f_ground = image.Image.initialise_jacobian_from_image(svf_f)

    assert svf_f.__class__.__name__ == 'SVF'
    assert jac_f_ground.__class__.__name__ == 'Image'

    for i in range(1, 19):
        for j in range(1, 19):
            for z in range(1, 19):
                svf_f.field[i, j, z, 0, :] = field_f(1, [i, j, z])
                jac_f_ground.field[i, j, z, 0, :] = jacobian_f(1, [i, j, z])

    jac_f_numeric = svf_f.compute_jacobian()

    square_size = range(2, 17)
    assert_array_almost_equal(jac_f_ground.field[square_size, square_size, square_size, 0, :],
                              jac_f_numeric.field[square_size,square_size, square_size, 0, :],
                              decimal=9)


### Jacobian determinant tests ###


def test_jacobian_determinant_toy_field_2d_1():
    # build a toy field: with 2 time points
    field = np.zeros([2, 2, 1, 2, 4])

    field[0, 0, 0, 0, :] = [1., 2., 3., 6.]
    field[1, 0, 0, 0, :] = [1., 0., 0., 1.]
    field[0, 1, 0, 0, :] = [0., 1., 1., 0.]
    field[1, 1, 0, 0, :] = [2., 1., 6., 2.]

    field[0, 0, 0, 1, :] = [2., 1., 6., 2.]
    field[1, 0, 0, 1, :] = [0., 1., 1., 0.]
    field[0, 1, 0, 1, :] = [1., 0., 0., 1.]
    field[1, 1, 0, 1, :] = [2., 1., 6., 3.]

    field_ground_truth_det = np.zeros([2, 2, 1, 2])

    field_ground_truth_det[0, 0, 0, 0] = 0.
    field_ground_truth_det[1, 0, 0, 0] = 1.
    field_ground_truth_det[0, 1, 0, 0] = -1.
    field_ground_truth_det[1, 1, 0, 0] = -2.

    field_ground_truth_det[0, 0, 0, 1] = -2.
    field_ground_truth_det[1, 0, 0, 1] = -1.
    field_ground_truth_det[0, 1, 0, 1] = 1.
    field_ground_truth_det[1, 1, 0, 1] = 0.

    m = field.reshape([2, 2, 1, 2, 2, 2])

    # verify ground truth is the same
    assert_array_equal(field_ground_truth_det, np.linalg.det(m))

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
