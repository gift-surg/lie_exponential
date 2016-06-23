"""
Classes for projective real Lie algebras and groups, general and special, of dimension d.
Default value d=2.

https://en.wikipedia.org/wiki/Projective_linear_group
"""
import numpy as np
from scipy.linalg import expm, logm


class ProjectiveAlgebra(object):
    """
    Real projective general/special linear Lie algebra of dimension d.
    Each element is a (d+1)x(d+1) real matrix defined up to a constant.
    Its exponential is given through the numerical approximation expm.
    If special trace must be zero.
    """
    def __init__(self, d=2, m=np.zeros([3, 3]), special=False):
        """
        """
        if np.array_equal(m.shape, [d+1]*2):
            self.matrix = m  # - np.min(m)*np.ones([d+1, d+1])
            self.dim = d
            self.special = special
        else:
            raise IOError
        if special is True and not abs(np.trace(m)) < 1e-4:
            raise IOError('input matrix is not in the special projective group')

    def shape(self):
        return self.matrix.shape

    shape = property(shape)

    @staticmethod
    def randomgen(d=2, scale_factor=None, sigma=1.0, special=False):
        """
        Generate a random element in the projective linear algebra
        :param d:
        :param scale_factor:
        :param sigma:
        :param special:
        :return:
        """
        random_h = sigma*np.random.randn(d+1,  d+1)

        if scale_factor is not None:
            random_h = scale_factor * random_h

        if special:
            random_h[0, 0] = -1 * np.sum(np.diagonal(random_h)[1:])

        return ProjectiveAlgebra(d=d, m=random_h, special=special)

    def exponentiate(self):
        return ProjectiveGroup(self.dim, expm(self.matrix), special=self.special)

    def ode_solution(self, init_cond=np.array([0, 0, 1]), affine_coordinates=True):
        s = expm(self.matrix).dot(init_cond)
        if affine_coordinates:
            return s[0:self.dim]/s[self.dim]  # the projective coordinate is the last one
        else:
            return s


class ProjectiveGroup(object):
    """
    Real projective general/special linear Lie group of dimension 2.
    Each element is a (d+1)x(d+1) real matrix with non-zero determinant defined up to an element of
    the scalar transformations given by c*I for c constant and I identity matrix.
    It is meant to be generated as exponential of an element of the class pgl_2 with its method exponentiate.
    """
    def __init__(self, d=2, m=np.eye(3), special=False):
        if np.array_equal(m.shape, [d+1]*2) and not np.linalg.det(m) == 0:
            self.matrix = m
            self.dim = d
            self.special = special
        else:
            raise IOError

    def shape(self):
        return self.matrix.shape

    shape = property(shape)

    def centered_matrix(self, c):
        """
        :param c: center = np.array([x_c, y_c])
        Returns: the matrix correspondent to self, centered in c. Non destructive over c.
        """
        h = self.matrix[:]
        h_prime = np.zeros_like(self.matrix)
        d = self.dim
        if isinstance(c, list):
            c = np.array(c)
        # den = one - Bc
        den = 1 - h[d, :-1].dot(c)

        # A prime
        h_prime[:-1, :-1] = (h[:-1, :-1] + np.kron(c.reshape(1, 2), h[d, :-1].reshape(2, 1)))/den
        # B prime
        h_prime[d, :-1] = (h[d, :-1])/den
        # T prime
        h_prime[:-1, 2] = (-(h[:-1, :-1]).dot(c) - h[d, :-1].dot(c) * c + c)/den

        h_prime[d, d] = 1

        return h_prime

    def centered(self, c):
        """
        Non destructive, provide a new homography as self, translated on c.
        :param c:
        :return:
        """
        return ProjectiveGroup(d=self.dim, m=self.centered_matrix(c))

    def centering(self, c):
        """
        destructive, center the given homography to the given center
        :param c:
        :return:
        """
        self.matrix = self.centered_matrix(c)

    @staticmethod
    def randomgen(d=2, center=None, scale_factor=None, sigma=1.0, special=False):
        """
        H = [A, T; B, 1]
        :param d:
        :param center: if we want to center the given matrix.
        :param scale_factor:
        :param sigma:
        :param special:
        :return:
        """
        random_h = sigma*np.random.randn(d+1,  d+1)
        # select one equivalence class.
        random_h[-1, -1] = 1
        # Ensure its matrix logarithm will have real entries.
        random_h = expm(random_h)

        if scale_factor is not None:
            random_h = scale_factor * random_h

        if center is not None:
            random_h = ProjectiveGroup(d=d, m=random_h).centered_matrix(center)

        if special:
            random_h[0, 0] = -1 * np.sum(np.diagonal(random_h)[1:])

        random_h[-1, -1] = 1.

        return ProjectiveGroup(d=d, m=random_h, special=special)

    def logaritmicate(self):
        return ProjectiveAlgebra(self.dim, logm(self.matrix)[:], special=self.special)


# methods to get above elements as matrices: it masks the classes!
def get_random_hom_g(d=2, center=None, scale_factor=None, sigma=1.0, special=False):
    """
    :param d: dimension of the homography in pgl by default or in psl
    :param center: center of the homography, if any
    :param scale_factor: scale factor of the homography
    :param sigma: sigma for the random values of the initial matrix.
    :param special: if the homography is in psl (True) or in pgl (False, default)
    :return: [h_g, h_a]random homography (in the GROUP) and the corresponding in the algebra h_g = expm(h_a)
    """
    if special:
        h_g = ProjectiveGroup.randomgen(d=d, center=center, scale_factor=scale_factor, sigma=sigma)
        h_a = h_g.logaritmicate()

    else:
        h_g = ProjectiveGroup.randomgen(d=d, center=center, scale_factor=scale_factor, sigma=sigma)
        h_a = h_g.logaritmicate()

    return h_g, h_a


def get_random_hom_a_matrices(d=2, scale_factor=None, sigma=1.0, special=False):
    """
    :param d: dimension of the homography in pgl by default or in psl
    :param center: center of the homography, if any
    :param scale_factor: scale factor of the homography
    :param sigma: sigma for the random values of the initial matrix.
    :param special: if the homography is in psl (True) or in pgl (False, default)
    :return: [h_g, h_a]random homography (in the GROUP) and the corresponding in the algebra h_g = expm(h_a)
    """

    h_a = ProjectiveAlgebra.randomgen(d=d, scale_factor=scale_factor, sigma=sigma, special=special)
    h_g = h_a.exponentiate()

    return h_a.matrix, h_g.matrix



def get_random_hom_matrices(d=2, center=None, random_kind='diag', scale_factor=None, sigma=1.0, special=False):
    """
    :param d: dimension of the homography in pgl by default or in psl
    :param center: center of the homography, if any
    :param random_kind
    :param scale_factor: scale factor of the homography
    :param sigma: sigma for the random values of the initial matrix.
    :param special: if the homography is in psl (True) or in pgl (False, default)
    :return: [h_g, h_a]random homography (in the GROUP) and the corresponding in the algebra h_g = expm(h_a)
    AS MATRICES!
    """
    if special:
        h_g = ProjectiveGroup.randomgen(d=d, center=center, scale_factor=scale_factor, sigma=sigma)
        h_a = h_g.logaritmicate()

    else:
        h_g = ProjectiveGroup.randomgen(d=d, center=center, scale_factor=scale_factor, sigma=sigma)
        h_a = h_g.logaritmicate()

    return h_g.matrix, h_a.matrix

'''
 if random_kind == 'diag':
                    random_h[0, 0] = 1 - (random_h[0, 1] * y_c + random_h[0, 2] * z_c)/float(x_c)
                    random_h[1, 1] = 1 - (random_h[1, 0] * x_c + random_h[1, 2] * z_c)/float(y_c)
                    random_h[2, 2] = 1 - (random_h[2, 0] * x_c + random_h[2, 1] * y_c)/float(z_c)
                elif random_kind == 'skew':
                    random_h[0, 2] = ((1 - random_h[0, 0]) * x_c - random_h[0, 1] * y_c) / float(z_c)
                    random_h[1, 0] = ((1 - random_h[1, 1]) * y_c - random_h[1, 2] * z_c) / float(x_c)
                    random_h[2, 1] = ((1 - random_h[2, 2]) * z_c - random_h[2, 0] * x_c) / float(y_c)
                elif random_kind == 'transl':
                    random_h[0, 2] = (1 - random_h[0, 0]) * x_c - random_h[0, 1] * y_c
                    random_h[1, 2] = - random_h[1, 0] * x_c + (1 - random_h[1, 1]) * y_c
                    random_h[2, 2] = 1  # - random_h[2, 0] * x_c - random_h[2, 1] * y_c + 1

                elif random_kind == 'shift':

                    two_minus_Bc      = 2 - random_h[2, :-1].dot(np.array([x_c, y_c]).T)
                    T_plus_c_minus_Ac = random_h[:-1, 2] + np.array([x_c, y_c]).T - random_h[-1, :-1].dot(np.array([x_c, y_c]).T)

                    # A prime
                    random_h[:-1, :-1] = random_h[:-1, :-1]/float(two_minus_Bc)
                    # B prime
                    random_h[2, :-1] = random_h[2, :-1]/float(two_minus_Bc)
                    # T prime
                    random_h[:-1, 2] = T_plus_c_minus_Ac

                    random_h[2, 2] = 1

                elif random_kind == 's':


                    one_minus_Bc = 1 - random_h[2, :-1].dot(np.array([x_c, y_c]).T)

                    # A prime
                    random_h[:-1, :-1] = random_h[:-1, :-1]/float(two_minus_Bc)
                    # B prime
                    random_h[2, :-1] = random_h[2, :-1]/float(two_minus_Bc)
                    # T prime
                    random_h[:-1, 2] = T_plus_c_minus_Ac

                    random_h[2, 2] = 1

                    pass

                else:
                    raise IOError('kind not recognized')

                    if random_kind == 'diag':
                    random_h[0, 0] = 1 - (random_h[0, 1] * y_c + random_h[0, 2] * z_c + random_h[0, 3] * w_c)/float(x_c)
                    random_h[1, 1] = 1 - (random_h[1, 0] * x_c + random_h[1, 2] * z_c + random_h[1, 3] * w_c)/float(y_c)
                    random_h[2, 2] = 1 - (random_h[2, 0] * x_c + random_h[2, 1] * y_c + random_h[2, 3] * w_c)/float(z_c)
                    random_h[3, 3] = 1 - (random_h[3, 0] * x_c + random_h[3, 1] * y_c + random_h[3, 2] * z_c)/float(w_c)
                elif random_kind == 'skew':
                    random_h[0, 3] = ((1 - random_h[0, 0]) * x_c - random_h[0, 1] * y_c - random_h[0, 2] * z_c)/float(w_c)
                    random_h[1, 0] = ((1 - random_h[1, 1]) * y_c - random_h[1, 2] * z_c - random_h[1, 3] * w_c)/float(x_c)
                    random_h[2, 1] = ((1 - random_h[2, 2]) * z_c - random_h[2, 3] * w_c - random_h[2, 0] * x_c)/float(y_c)
                    random_h[3, 2] = ((1 - random_h[3, 3]) * w_c - random_h[3, 0] * x_c - random_h[3, 1] * y_c)/float(z_c)
                elif random_kind == 'transl':
                    random_h[0, 3] = (1 - random_h[0, 0]) * x_c - random_h[0, 1] * y_c - random_h[0, 2] * z_c
                    random_h[1, 3] = - random_h[1, 0] * x_c + (1 - random_h[1, 1]) * y_c - random_h[1, 2] * z_c
                    random_h[2, 3] = - random_h[2, 0] * x_c - random_h[2, 1] * y_c + (1 - random_h[2, 2]) * z_c
                    random_h[3, 3] = 1 #  - random_h[3, 0] * x_c - random_h[3, 1] * y_c - random_h[3, 2] * z_c  + 1
                elif random_kind == 'shift':

                    two_minus_Bc      = 2 - random_h[3, :-1].dot(np.array([x_c, y_c, z_c]).T)
                    T_plus_c_minus_Ac = random_h[:-1, 3] + np.array([x_c, y_c]).T - random_h[-1, :-1].dot(np.array([x_c, y_c, z_c]).T)

                    # A prime
                    random_h[:-1, :-1] = random_h[:-1, :-1]/float(two_minus_Bc)
                    # B prime
                    random_h[3, :-1] = random_h[3, :-1]/float(two_minus_Bc)
                    # T prime
                    random_h[:-1, 3] = T_plus_c_minus_Ac

                    random_h[3, 3] = 1

                else:
                    raise IOError('kind not recognized')
'''