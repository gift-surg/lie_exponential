from math import pi
import numpy as np
from fractions import Fraction
from scipy.misc import factorial
from scipy.misc import comb as binomial


# ----Auxiliary function for so and se ----


def mod_pipi(alpha):
    """
    mod_pipi(alpha) \n
    :param alpha: angle in rad
    :return: equivalent alpha in (-pi, pi]
    """
    if alpha > 2 * pi:
        alpha %= (2 * pi)
    elif alpha < -2 * pi:
        alpha %= (-2 * pi)

    if alpha > pi:
        alpha -= 2 * pi
    elif alpha <= -pi:
        alpha += 2 * pi

    return alpha

# ------------- auxiliary method for the BCH ----------------


def fact(n):
    """
    Covers the predefined function factorial from scipy.misc
    :param n:
    :return: factorial of n type float
    """
    return float(factorial(n, True))


def bern(n):
    """
    bern(n) \n
    :param: integer n
    :return: nth Bernoulli number (type Fraction)
    (iterative algorithm, do not uses polynomials)
    """
    if n == 1:
        ans = Fraction(1, 2)
    elif n % 2 == 1:
        ans = Fraction(0, 1)
    else:
        n += 1
        a = [0] * (n + 1)
        for m in range(n + 1):
            a[m] = Fraction(1, m + 1)
            for j in range(m - 1, 0, -1):
                a[j - 1] = j * (a[j - 1] - a[j])
        ans = a[0]
    return ans  # type Fraction.
    # return float(ans) # type Float


def bernoulli_poly(x, n):
    """
    bernoulli_poly(x,n) \n
    :param: x,n value for the unknown and degree of the polynomial.
    :return: j-th bernoulli polynomial evaluate at x (unknown of the poly).
    """
    return sum([binomial(n, k) * bern(n - k) * (x ** k) for k in range(n)])


def bernoulli_numb_via_poly(n):
    """
    bernoulli_numb(n) \n
    :param: integer n
    :return: nth Bernoulli number
    (uses first type bernoulli polynomials)
    """
    return bernoulli_poly(0, n)


def bch_right_jacobian(r):
    """
    BCH_right_jacobian(r) \n
    :param r: element of lie algebra so2_a in restricted form
    :return: Jacobian (equation [57] Tom tech memo)
    """
    theta = r[0]
    dtx = r[1]
    dty = r[2]
    J = np.array([0.0] * 9).reshape(3, 3)
    half_theta = theta * 0.5
    tan_half_theta = np.tan(theta * 0.5)
    prec = abs(np.spacing(theta))

    if abs(theta) > prec:
        factor1 = (half_theta - tan_half_theta) / (theta * tan_half_theta)
        factor2 = half_theta / tan_half_theta
    else:
        factor1 = theta / 12.0
        factor2 = 1 - (theta ** 2) / 12.0

    J[0, 0] = 1
    J[1, 0] = - factor1 * dtx + 0.5 * dty
    J[2, 0] = - 0.5 * dtx - dty * factor1
    J[1, 1] = factor2
    J[2, 2] = factor2
    J[2, 1] = 0.5 * theta
    J[1, 2] = -0.5 * theta

    return J


def split_the_time(t, x, len_range=None, number_of_intervals=5, epsilon=0):
    """

    :param t: list or tuple relative to the time if unordered it will be ordered
    :param x: values corresponding to the time (same length of t), if t is unordered it will follow the same reordering.
    :param len_range: interval of the data that will be splitted.
    :param number_of_intervals: number of interval in which we want x to be splitted
    :return: x_splitted in intervals
    """

    if not len(t) == len(x):
        raise TypeError('t and x must have the same dimension')

    if not sorted(t) == t:
        t, x = (list(z) for z in zip(*sorted(zip(t, x))))

    if len_range is None:
        if epsilon > 0:
            starting_range = t[0]- epsilon
            ending_range   = t[len(t) - 1] + epsilon
        else:
            starting_range = np.floor(t[0])
            ending_range   = np.ceil(t[len(t) - 1])

    else:
        starting_range = len_range[0]
        ending_range   = len_range[1]

    steps = np.linspace(starting_range, ending_range, num=number_of_intervals + 1)
    steps[len(steps) - 1] += 0.1

    x_splitted = [
        [x[i] for i in range(len(x)) if steps[j - 1] <= t[i] < steps[j]]
        for j in range(1, number_of_intervals + 1)
    ]

    return x_splitted


def custom_transposer(d, num_col):
    """

    :param d: list of lists of len n*num_col
    :param num_col: list of lists in a different order (as transposed)

    e.g.

    custom_transposer([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]], num_col = 2)
    ->  [[1,1],[3,3],[5,5],[2,2],[4,4],[6,6]]

    custom_transposer([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]], num_col = 3)
    ->  [[1,1],[4,4],[2,2],[5,5],[3,3],[6,6]]

    """

    if not num_col % len(d):
        raise TypeError('dimensions are not compatible')

    ans = []

    for r in range(num_col):  # reminder
        for q in range(0,len(d),num_col):  # quotient
            ans = ans + [d[q + r]]

    return ans


# ---------- list management utils methods ---------------


def remove_k(l, k):
    l_new = l[:]; l_new.pop(k)
    return l_new


# ---------- data management utils methods ---------------

def get_in_out_liers(data, coeff=0.6745, return_values=True):
    """
    Input: 1d numpy array
    Output: position of the outliers in the vector
    if return_values=False it returns the indexes
    """
    median = np.median(data)
    diff = np.sum(np.abs(data - median))
    mad = np.median(diff)

    thresh = coeff * 0.674491 * mad

    out_liers_index = [k for k in range(len(data)) if data[k] > thresh]
    in_liers_index = [k for k in range(len(data)) if data[k] > thresh]

    in_liers_values = np.delete(data, out_liers_index)
    out_liers_values = [data[j] for j in out_liers_index]

    if return_values:
        return out_liers_values, in_liers_values
    else:
        return out_liers_index, in_liers_index


# ---------- vectors and matrices defined on a grid manipulators ---------------


def matrix_vector_field_product(j_input, v_input):
    """
    :param j_input: matrix m x n x (4 or 9) as for example a jacobian column major
    :param v_input: matrix m x n x (2 or 3) to be multiplied by the matrix point-wise.
    :return: m x n  x (2 or 3) whose each element is the result of the product of the
     matrix (i,j,:) multiplied by the corresponding element in the vector v (i,j,:).

    In tensor notation for n = 1: R_{i,j,k} = \sum_{l=0}^{2} M_{i,j,l+3k} v_{i,j,l}

    ### equivalent code in a more readable version:

    # dimensions of the problem:
    d = v_input.shape[-1]
    vol = list(v_input.shape[:-1])

    # repeat v input 3 times, one for each row of the input matrix 3x3 or 2x2 in corresponding position:
    v = np.tile(v_input, [1]*d + [d])

    # element-wise product:
    j_times_v = np.multiply(j_input, v)

    # Sum the three blocks in the third dimension:
    return np.sum(j_times_v.reshape(vol + [d, d]), axis=d+1).reshape(vol + [d])

    """
    assert len(j_input.shape) == len(v_input.shape)

    d = v_input.shape[-1]
    vol = list(v_input.shape[:d])
    extra_ones = len(v_input.shape) - (len(vol) + 1)

    temp = j_input.reshape(vol + [1]*extra_ones + [d, d])  # transform in squared block with additional ones
    return np.einsum('...kl,...l->...k', temp, v_input)


def matrix_fields_product(a_input, b_input):
    """
    Multiplies the matrix a_input[i,j,:] times b_input[i,j,:] for each i, j.
    works for any dimension
    :param a_input:
    :param b_input:
    :return:
    """
    # test
    np.testing.assert_array_equal(a_input.shape, b_input.shape)

    d = int(np.sqrt(a_input.shape[-1]))
    vol = list(a_input.shape[:d])
    extra_ones = len(a_input.shape) - (len(vol) + 1)

    temp_a = a_input.reshape(vol + [1]*extra_ones + [d, d])  # transform in squared block with additional ones
    temp_b = b_input.reshape(vol + [1]*extra_ones + [d, d])

    return np.einsum('...kl,...lm', temp_a, temp_b).reshape(vol + [1]*extra_ones + [d*d])


def matrix_fields_product_iterative(a_input, n=1):
    """
    Matrix products, for matrices defined at each point of a grid, row major.
    Multiplies the matrix a_input[i,j,:] by itself n times for each i,j.
    :param a_input: matrix field
    :param n: number of iterations
    :return: a_input^n point-wise
    """
    ans = a_input[...]
    for _ in range(1, n):
        ans = matrix_fields_product(ans, a_input)

    return ans


def id_matrix_field(domain):
    """
    From a domain of dimension dim =2,3, it returns the identity field
    that at each point of the domain has the (row mayor) vectorized identity
    matrix.
    :param domain: a squared or cubed domain
    :return:
    """
    dim = len(domain)
    if dim not in [2, 3]:
        assert IOError

    shape = list(domain) + [1]*(4 - dim) + [dim**2]
    flat_id = np.eye(dim).reshape(1, dim**2)
    return np.repeat(flat_id, np.prod(domain)).reshape(shape, order='F')

# ---------- Graphical methods ---------------


def grid_generator(x_size=101,
                   y_size=101,
                   x_step=10,
                   y_step=10,
                   line_thickness=1):

    m = np.zeros([x_size, y_size])
    # preliminary slow version:
    for x in range(x_size):
        for y in range(y_size):
            if 0 <= x % x_step < line_thickness or 0 <= y % y_step < line_thickness:
                m[x, y] = 1

    return m

'''
def trim_2d(array, passe_partout_size, return_copy=False):
        """
        :param array: array input to be trimmed
        :param passe_partout_size: passepartout value
        :param return_copy: False by default, if you want to adjust the existing field. True otherwise.
        :return: the same field trimmed by the value of the passepartout in each dimension.
        """

        if return_copy:

            new_field = copy.deepcopy(array)

            new_field.field = array.field[passe_partout_size:-passe_partout_size,
                                         passe_partout_size:-passe_partout_size,
                                         ...]
            return new_field

        else:

            self.field = self.field[passe_partout_size:-passe_partout_size,
                                    passe_partout_size:-passe_partout_size,
                                    ...]
'''

# ---------- SVF and SDISP generators ---------------

'''
def generate_svf(kind='', random_output=False, domain_input=(), parameters=()):
    """
    NOTE: all the functions parameters are optional but the default return an error
    as sanity check.

    :param kind: can be 'SE2', 'GAUSS' or 'ADNII'
    :param random: if the parameters are the parameter of a random or a fixed
    :param parameters: according to the random variables and the kind provides the
            se2 parameters, the sigma of the transformation, the index of the adnii image.
    :return: one svf, in accordance with the input data
    """

    svf_0 = None
    disp_0 = None

    if kind is 'SE2':

        if random_output:
            # generate matrices from parameters -> epsilon, interval_theta (x, y), omega (xa, ya, xb, yb)
            epsilon = parameters[0]
            interval_theta = parameters[1:3]
            omega = parameters[3:]
            m_0 = se2_g.randomgen_custom_center(interval_theta=interval_theta,
                                                omega=omega,
                                                epsilon_zero_avoidance=epsilon)
            dm_0 = se2_g.log(m_0)

        else:
            # generate matrices -> theta, tx, ty
            theta, tx, ty = parameters[0], parameters[1], parameters[2]
            m_0 = se2_g.se2_g(theta, tx, ty)
            dm_0 = se2_g.log(m_0)

        # Generate svf and disp
        svf_0   = SVF.generate_from_matrix(domain_input, dm_0.get_matrix, affine=np.eye(4))
        disp_0  = SDISP.generate_from_matrix(domain_input, m_0.get_matrix - np.eye(3), affine=np.eye(4))

    elif kind is 'GAUSS':
        pass
    elif kind is 'ADNII':
        pass
    else:
        raise IOError('The option inserted for kind is not available.')




    if disp_0 is not None:
        return svf_0, disp_0
    else:
        return svf_0
'''
