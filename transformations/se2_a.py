from math import sin, cos, sqrt
from random import uniform
import numpy as np
import se2_g
from utils.aux_functions import mod_pipi


class se2_a(object):
    """
    Class for algebra se2 quotient over equivalent relation given by exp.
    NOTE: se2_a is an improper name for this class!
    It is the quotient algebra in which exp is well defined.
    Quotient projection is applied to each new instance by default.

    sum and subtract,
    scalar product,
    Lie bracket
    """
    def __init__(self, theta, tx, ty):

        self.rotation_angle = mod_pipi(theta)

        if self.rotation_angle == theta:
            self.tx = tx
            self.ty = ty
        else:
            modfact = self.rotation_angle / theta
            self.tx = modfact*tx
            self.ty = modfact*ty

    def __quotient_projection__(self):
        """
        Maps the elements of se2_a in the quotient over the
        equivalence relation defined by exp, in order to have
        exp well defined.
        a ~ b iff exp(a) == exp(b)
        Here se2_a is intended as the quotient se2_a over the
        above equivalence relation.
        :param self: element of se2_a
        """
        theta_quot = mod_pipi(self.rotation_angle)
        if self.rotation_angle == theta_quot:
            tx_quot = self.tx
            ty_quot = self.ty
        else:
            modfact = mod_pipi(self.rotation_angle) / self.rotation_angle
            tx_quot = self.tx * modfact
            ty_quot = self.ty * modfact
        return se2_a(theta_quot, tx_quot, ty_quot)

    quot = property(__quotient_projection__)

    def __get_matrix__(self):
        # apply the quotient before transforming the element into a matrix.
        self_quotient = self.quot

        theta = self_quotient.rotation_angle
        tx = self_quotient.tx
        ty = self_quotient.ty

        a1 = [0, -theta, tx]
        a2 = [theta, 0, ty]
        a3 = [0, 0, 0]
        return np.array([a1, a2, a3])

    get_matrix = property(__get_matrix__)

    def __get_restricted__(self):
        self_quotient = self.quot
        theta = self_quotient.rotation_angle
        tx = self_quotient.tx
        ty = self_quotient.ty
        return [theta, tx, ty]

    get = property(__get_restricted__)

    def __add__(self, element2):
        alpha1 = self.rotation_angle
        x1 = self.tx
        y1 = self.ty
        alpha2 = element2.rotation_angle
        x2 = element2.tx
        y2 = element2.ty
        alpha_sum = alpha1+alpha2
        return se2_a(alpha_sum, x1 + x2, y1 + y2).quot

    def __sub__(self, element2):
        alpha1 = self.rotation_angle
        x1 = self.tx
        y1 = self.ty
        alpha2 = element2.rotation_angle
        x2 = element2.tx
        y2 = element2.ty
        alpha_subs = alpha1 - alpha2
        return se2_a(alpha_subs, x1 - x2, y1 - y2).quot

    def __rmul__(self, scalar):
        """
        Alternative for the scalar product
        """
        tx = self.tx
        ty = self.ty
        alpha = self.rotation_angle
        return se2_a(scalar*alpha, scalar*tx, scalar*ty).quot

    def norm(self, norm_type='standard', lamb=1):
        """
        norm(self, typology, lamb for the standard norm type)

        norm_type get the type of the norm we want to deal with:
        norm_type in {'', 'vector_len'}

        :param norm_type:
        :param lamb:
        :return:
        """
        if norm_type == 'lamb':
            ans = sqrt(self.rotation_angle**2 + lamb*(self.tx**2 + self.ty**2))  # lamb has to be modified in aux_func
        elif norm_type == 'translation':
            ans = sqrt(self.tx ** 2 + self.ty ** 2)
        elif norm_type == 'fro':
            ans = sqrt(2*self.rotation_angle**2 + self.tx ** 2 + self.ty ** 2)
        else:
            ans = -1
        return ans


def randomgen(intervals=(0, 10), norm_type='fro', lamb=1):
    """
    Montecarlo sampling to build a random element in se2_a.
    it uses 3 possible norm_type, lamb, translation, or fro

    :param intervals:
    :param norm_type:
    :param lamb: weight of the radius: sqrt(theta**2 + lamb * rho**2)
    :return: se2_a(theta, tx, ty) such that
    a < se2_a(theta, tx, ty).norm(norm_type) < b
    """
    # montecarlo sampling to get a < se2_a(theta, tx, ty).norm(norm_type) < b
    norm_belongs_to_interval = False
    num_attempts = 0
    a = intervals[0]
    b = intervals[1]

    theta, tx, ty = 0, 0, 0

    if norm_type == 'lamb':

        while not norm_belongs_to_interval and num_attempts < 1000:
            num_attempts += 1
            theta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
            rho = uniform(a, b)

            if a <= sqrt(theta**2 + lamb * rho**2) <= b:
                # keep theta, compute tx ty according to rho and a new eta
                eta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
                tx = rho * cos(eta)
                ty = rho * sin(eta)

                norm_belongs_to_interval = True

    elif norm_type == 'translation':

        # we pick rho to satisfy the condition a < se2_a(theta, tx, ty).norm(norm_type) < b
        # where norm is the the norm of the translation.
        # no Montecarlo is needed.
        theta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
        rho = uniform(a, b)
        eta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
        tx = rho * cos(eta)
        ty = rho * sin(eta)

    elif norm_type == 'fro':

        while not norm_belongs_to_interval and num_attempts < 1000:
            num_attempts += 1
            theta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
            rho = uniform(a, b)

            if a <= sqrt(2*theta**2 + rho**2) <= b:
                # keep theta, compute tx ty according to rho and a new eta
                eta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
                tx = rho * cos(eta)
                ty = rho * sin(eta)

                norm_belongs_to_interval = True

    if num_attempts == 1000 and not norm_belongs_to_interval:
        raise Exception("randomgen_standard_norm in se2_g with lambda = 1: safety recursion number has been reached"
                        "montecarlo method has failed in finding an element after 1000 attempt.")

    else:
        return se2_a(theta, tx, ty)


def is_a_matrix_in_se2_a(m_input, relax=False):
    ans = False
    if type(m_input) == np.ndarray and m_input.shape == (3, 3):
        m = m_input.copy()
        if relax:
            m = np.around(m, 6)
        if m[0, 0] == m[1, 1] == 0 and m[1, 0] == -m[0, 1] and m[2, 0] == m[2, 1] == m[2, 2] == 0:
            ans = True
    return ans


def lie_bracket(element1, element2):
    """
    lie_bracket(A,B) \n
    :param element1 :
    :param element2 : elements of se2 already in the quotient
    :return: their Lie bracket
    Note: no need to put quotient. The resultant angle is zero.
    """
    # if str(type(element1))[-5:] != str(type(element2))[-5:] != 'se2_a':
    #    raise TypeError('warning: wrong input data format, 2 se2_a elements expected.')

    element1 = element1.quot
    element2 = element2.quot
    alpha1 = element1.rotation_angle
    alpha2 = element2.rotation_angle
    x1 = element1.tx
    y1 = element1.ty
    x2 = element2.tx
    y2 = element2.ty
    return se2_a(0, alpha2 * y1 - alpha1 * y2, alpha1 * x2 - alpha2 * x1)


def lie_multi_bracket(l):
    """
    lie_bracket(l) \n
    :param l: L list of element of se2
    :return: [L0 ,[L1,[L2, [... [L(n-1),Ln]...]
    """
    if len(l) <= 1:
        return 0
    elif len(l) == 2:
        return lie_bracket(l[0], l[1])
    else:
        le = len(l)
        return lie_multi_bracket(l[0:le - 2] + [lie_bracket(l[le - 2], l[le - 1])])


def scalarpr(const, element):
    """
    scalarpr(const, element) \n
    :param const: scalar float value, element of se2
    :param element: element of the lie algebra.
    :return: const*element as scalar product in the Lie algebra
    """
    # how to define the scalar product directly in the class? external product with float element...
    alpha = element.rotation_angle
    x = element.tx
    y = element.ty
    return se2_a(const * alpha, const * x, const * y)


def matrix2se2_a(a, eat_em_all=False):
    """
    matrix2se2(a, relax = False) \n
    :param a: matrix of se(2) np.array([[0,0,0],[tx,0,-theta],[ty,theta,0]])
            optional relax, if it relax the input condition, allowing any kind of matrix
    :param eat_em_all: if True, any kind of matrix can be utilized and converted into se2, cutting the extra info.
    :return: corresponding element in SE2 .
    """
    if not eat_em_all and not is_a_matrix_in_se2_a(a):
        raise Exception("matrix2se2_a in se2_a: the inserted element is not a matrix in se2_a  ")
    else:
        theta = a[1, 0]
        x = a[0, 2]
        y = a[1, 2]
        return se2_a(theta, x, y)


def list2se2_a(a):
    if not type(a) == list or not len(a) == 3:
        raise TypeError("list2se2_g in se2_g: list of dimension 3 expected")
    return se2_a(a[0], a[1], a[2])


def exp(element):
    """
    exp(element) \n
    algebra exponential 
    :param element: instance of se2
    :return: corresponding element in Lie group SE2 
    """
    '''
    m_element = element.get_matrix
    ans = se2_g.matrix2se2_g(lin.expm(m_element), True)
    '''
    element = element.quot
    theta = element.rotation_angle
    v1 = element.tx
    v2 = element.ty
    esp = abs(np.spacing(0))
    if abs(theta) <= 10*esp:
        factor1 = 1 - (theta**2)/6.0
        factor2 = theta/2.0
    else:
        factor1 = sin(theta)/theta
        factor2 = (1 - cos(theta))/theta
    x1 = factor1*v1 - factor2*v2
    x2 = factor2*v1 + factor1*v2

    return se2_g.se2_g(theta, x1, x2)


# trim noise for matrices.
def trimmer_to_se2_a_matrix():
    pass


def exp_for_matrices(m, eat_em_all=True):
    """
    exp_m(m) \n
    algebra exponential for matrices
    :param m: instance of se2 in matrix form
    :param eat_em_all: if any matrix can be its input. It will be forced to be exponentiated, cutting the extra info.
    :return: corresponding element in Lie group SE2, matrix form again
    """
    if not eat_em_all and not is_a_matrix_in_se2_a(m):
        raise Exception("exp_for_matrices in se2_a: the inserted element is not a matrix in se2_a  ")

    # first step is to quotient the input element (if the input is a sum)
    ans = exp(se2_a(m[1, 0], m[0, 2], m[1, 2]))
    return ans.get_matrix


def bracket_for_matrices(m1, m2, eat_em_all=True):
    """
    Compute lie bracket for matrices
    :param m1:
    :param m2:
    :param eat_em_all: how strict we are on input!
    :return:
    """
    if not eat_em_all:
        if not (is_a_matrix_in_se2_a(m1) and is_a_matrix_in_se2_a(m2)):
            raise Exception("bracket_for_matrices in se2_a: the inserted element is not a matrix in se2_a  ")

    return m1*m2 - m2*m1


def lie_multi_bracket_for_matrices(l):
    """
    lie_bracket(L) \n
    :param l: list of matrices
    :return: [L0 ,[L1,[L2, [... [L(n-1),Ln]...]
    """
    if len(l) <= 1:
        return 0
    elif len(l) == 2:
        return bracket_for_matrices(l[0], l[1])
    else:
        num = len(l)
        return lie_multi_bracket_for_matrices(l[0:num - 2] + [lie_bracket(l[num - 2], l[num - 1])])
