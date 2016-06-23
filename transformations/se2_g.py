from math import sin, cos, sqrt
from numpy.random import uniform, choice
import numpy as np

import se2_a
from utils.aux_functions import mod_pipi


class se2_g(object):
    """
    Class for group of SE(2) Lie group of rotations and translation in 2 dimensions.
    """

    def __init__(self, theta, tx, ty):
        self.rotation_angle = mod_pipi(theta)
        self.tx = tx
        self.ty = ty

    def __get_matrix__(self):
        a1 = [cos(self.rotation_angle), -sin(self.rotation_angle), self.tx]
        a2 = [sin(self.rotation_angle),  cos(self.rotation_angle), self.ty]
        a3 = [0, 0, 1]
        return np.array([a1, a2, a3])

    get_matrix = property(__get_matrix__)

    def __get_restricted__(self):
        return [self.rotation_angle, self.tx, self.ty]

    get = property(__get_restricted__)

    def __mul__(self, element2):
        x1 = self.tx
        x2 = element2.tx
        y1 = self.ty
        y2 = element2.ty
        theta_1 = mod_pipi(self.rotation_angle)
        c1 = cos(theta_1)
        s1 = sin(theta_1)

        alpha = mod_pipi(self.rotation_angle + element2.rotation_angle)
        x = x1 + x2 * c1 - y2 * s1
        y = y1 + x2 * s1 + y2 * c1

        return se2_g(alpha, x, y)

    def inv(self):
        alpha = self.rotation_angle
        xv = self.tx
        yv = self.ty
        xn = - cos(alpha)*xv - sin(alpha)*yv
        yn = + sin(alpha)*xv - cos(alpha)*yv
        return se2_g(-alpha, xn, yn)

    def norm(self, norm_type='standard', lamb=1):
        """
        norm(self, typology, order = None, axes = None)

        norm_type get the type of the norm we want to deal with:
        norm_type in {'', 'vector_len'}

        """
        if norm_type == 'standard' or norm_type == 'group_norm':
            ans = sqrt(self.rotation_angle**2 + lamb*(self.tx**2 + self.ty**2))
        elif norm_type == 'translation':
            ans = sqrt(self.tx ** 2 + self.ty ** 2)
        elif norm_type == 'fro':
            ans = sqrt(3 + self.tx ** 2 + self.ty ** 2)
        else:
            ans = -1
        return ans


def randomgen_custom(interval_theta=(),
                     interval_tx=(),
                     interval_ty=(),
                     avoid_zero_rotation=True):
    """
    GIGO method.
    :param interval_theta:  (interval_theta[0], interval_theta[1])
    :param interval_tx:     (interval_tx[0], interval_tx[1])
    :param interval_ty:     (interval_ty[0], interval_ty[1])
    :return:
    se2_g(uniform(interval_theta[0], interval_theta[1]),
          uniform(interval_tx[0], interval_tx[1]),
          uniform(interval_ty[0], interval_ty[1]) )
    """
    # avoid theta = zero

    theta = uniform(interval_theta[0], interval_theta[1])

    if avoid_zero_rotation:
        if interval_theta[0] < 0 < interval_theta[1]:
            epsilon = 0.001  # np.spacing(0)
            theta = choice([uniform(interval_theta[0], 0 - epsilon), uniform(0 + epsilon, interval_theta[1])])

    if len(interval_tx) == 1:
        tx = interval_tx[0]
    else:
        tx = uniform(interval_tx[0], interval_tx[1])

    if len(interval_ty) == 1:
        ty = interval_ty[0]
    else:
        ty = uniform(interval_ty[0], interval_ty[1])

    return se2_g(theta, tx, ty)


def randomgen_custom_center(interval_theta=(-np.pi/2, np.pi/2),
                            omega=(1, 6, 1, 6),
                            avoid_zero_rotation=True,
                            epsilon_zero_avoidance=0.001):
    """
    An element of SE(2) defines a rotation (from SO(2)) away from the origin.
    The center of the rotation is the fixed point of the linear map.
    This method provides an element of SE(2), random, with rotation parameter sampled over a uniform distribution
    in the interval interval_theta, and with center of rotation in the squared subset of the cartesian plane
    (x0,x1)x(y0,y1)
    :param interval_theta:
    :param omega:
    :param avoid_zero_rotation:
    :param epsilon_zero_avoidance:
    :return:
    """

    theta  = uniform(interval_theta[0], interval_theta[1])

    if avoid_zero_rotation:
        if interval_theta[0] < 0 < interval_theta[1]:
            epsilon = epsilon_zero_avoidance  # np.spacing(0)
            theta = choice([uniform(interval_theta[0], 0 - epsilon), uniform(0 + epsilon, interval_theta[1])])

    x_c  = uniform(omega[0], omega[1])
    y_c  = uniform(omega[2], omega[3])

    tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
    ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

    return se2_g(theta, tx, ty)


def randomgen(intervals=(), lamb=1):
    if len(intervals) == 2:
        a = intervals[0]
        b = intervals[1]
        if b < a or a < 0 or b < 0:
            raise Exception("randomgen_standard_norm in se2_g  "
                            "given interval (a,b) must have a < b for a, b positive ")
    elif len(intervals) == 0:
        a = 0
        b = 10
    else:
        raise Exception("randomgen_standard_norm in se2_g: "
                        "the list of intervals inserted must have dimension 2 or 0")

    if lamb < 0:
        raise Exception("randomgen_standard_norm in se2_g with lambda < 0: "
                        "negative lambda is not accepted. ")
    elif lamb == 0:
        if a > np.pi:
            raise Exception("randomgen_standard_norm in se2_g with lambda = 0: "
                            "intervals (a,b) and (-pi, pi) can not be disjoint ")
        else:
            a_theta_pos = a
            b_theta_pos = min(b, np.pi)
            a_theta_neg = -a
            b_theta_neg = max(-b, -np.pi + abs(np.spacing(-np.pi)))
            theta = choice([uniform(a_theta_pos, b_theta_pos), uniform(a_theta_neg, b_theta_neg)])
            rho = uniform(0, 10)
            eta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
            tx = rho * cos(eta)
            ty = rho * sin(eta)
    else:
        a_theta = max(-a, -np.pi + abs(np.spacing(-np.pi)))
        b_theta = min(a, np.pi)
        theta = uniform(a_theta, b_theta)
        rho = uniform(sqrt((a**2 - theta**2)/lamb), sqrt((b**2 - theta**2)/lamb))
        eta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
        tx = rho * cos(eta)
        ty = rho * sin(eta)
    return se2_g(theta, tx, ty)


def randomgen_translation(intervals=()):
    if len(intervals) == 2:
        a = intervals[0]
        b = intervals[1]
        if b < a or a < 0 or b < 0:
            raise Exception("randomgen_standard_norm in se2_g with: "
                            "given interval (a,b) must have a < b for a, b positive ")
    elif len(intervals) == 0:
        a = 0
        b = 10
    else:
        raise Exception("randomgen_translation_norm in se2_g: "
                        "the list of intervals inserted must have dimension 2 or 0")
    theta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    rho = uniform(a, b)
    eta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = rho * cos(eta)
    ty = rho * sin(eta)
    return se2_g(theta, tx, ty)


def randomgen_fro(intervals=()):
    if len(intervals) == 2:
        a = intervals[0]
        b = intervals[1]
        if b < a or a < sqrt(3):
            raise Exception("randomgen_fro_norm in se2_g with: given interval (a,b) "
                            "must have a < b for a, b positive greater than sqrt(3) ")
    elif len(intervals) == 0:
        a = sqrt(3)
        b = 10
    else:
        raise Exception("randomgen_translation_norm in se2_g: "
                        "the list of intervals inserted must have dimension 2 or 0")
    theta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    rho = uniform(sqrt(round(a**2 - 3, 15)), sqrt(round(b**2 - 3, 15)))
    eta = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    tx = rho * cos(eta)
    ty = rho * sin(eta)
    return se2_g(theta, tx, ty)


def is_a_matrix_in_se2_g(m_input, relax=False):
    ans = False
    if type(m_input) == np.ndarray:
        m = m_input.copy()
        if relax:
            m = np.around(m, 6)
        if m.shape == (3, 3) and m[0, 0] == m[1, 1] and m[1, 0] == -m[0, 1] and m[2, 0] == m[2, 1] == 0 and m[2, 2] == 1:
            ans = True
    return ans


def matrix2se2_g(A, eat_em_all=False):
    """
    matrix2se2_g(M) \n
    :param: matrix of SE(2) np.array([[1,0],[t,R]]). t translation (in IR^(2x1)), R rotation (in SO(2))
            optional relax, if it relax the input condition, allowing any kind of matrix
            Relax, if we want some relaxation in the input.
    :return: corresponding element in SE2 .
    """
    if not eat_em_all and not is_a_matrix_in_se2_g(A):
        raise Exception("matrix2se2_g in se2_g: the inserted element is not a matrix in se2_g  ")
    else:
        theta = np.arctan2(A[1, 0], A[0, 0])
        x = A[0, 2]
        y = A[1, 2]
        return se2_g(theta, x, y)


def list2se2_g(a):
    if not type(a) == list or not len(a) == 3:
        raise TypeError("list2se2_g in se2_g: list of dimension 3 expected")
    return se2_g(a[0], a[1], a[2])


def log(element):
    """
    log(element) \n
    group logarithm 
    :param: instance of SE2
    :return: corresponding element in Lie algebra se2 
    """
    theta = mod_pipi(element.rotation_angle)
    v1 = element.tx
    v2 = element.ty
    c = cos(theta)
    prec = abs(np.spacing(0.0))
    if abs(c - 1.0) <= 10*prec:
        x1 = v1
        x2 = v2
        theta = 0
    else:
        factor = (theta / 2.0) * sin(theta) / (1.0 - c)
        x1 = factor * v1 + (theta / 2.0) * v2
        x2 = factor * v2 - (theta / 2.0) * v1
    return se2_a.se2_a(theta, x1, x2)


def log_for_matrices(m):
    """
    log(element) \n
    group logarithm for matrices and not for the close form
    :param: instance of SE2 in matrix form!
    :return: corresponding element in Lie algebra se2
    """
    ans = log(se2_g(mod_pipi(np.arctan2(m[1, 0], m[0, 0])), m[0, 2], m[1, 2]))
    return ans.get_matrix