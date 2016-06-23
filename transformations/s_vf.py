import numpy as np
import copy
from scipy.misc import factorial as fact
from scipy.linalg import expm
import nibabel as nib

from scipy.integrate import ode

from utils.image import Image
from utils.aux_functions import matrix_vector_field_product, matrix_fields_product_iterative

from transformations.s_disp import SDISP


class SVF(Image):
    """
    Class for operations on stationary velocity fields (elements in the Lie algebra).
    An SVF is an Image with more operations (methods) provided by the lie algebra structure
    and by the log euclidean framework.
    It is associated with s_disp.py as the correspondent elements in the Lie group through
    exp function.
    """

    # SVF manager methods #

    def __init__(self, input_nifti_image):
        """
        From nifti_image to SVF.
        By default svf are initialized with nifti images of adequate shape.
        The difference with class IMAGE is not in the attributes
        but in the methods:
        an svf can access more operations respect to the IMAGE, and
        get a warning when a sum or a scalar multiplication is performed.
        """
        if len(input_nifti_image.get_data().shape) == 5:
            if input_nifti_image.get_data().shape[3] == 1:  # time points
                super(SVF, self).__init__(input_nifti_image)
            else:
                raise TypeError('Inserted nifty images does not corresponds to a stationary vector field')
        else:
            raise TypeError('Inserted nifty images does not corresponds to a vector field')

    @classmethod
    def from_nifti_image(cls, input_nifti_image):
        """
        Redundant version of the __init__ that creates object svf from nifti_image.
        Just to remember who is the input for the initialization.
        :param input_nifti_image: input nifti image from which we want to obtain the SVF.
        """
        return cls(input_nifti_image)

    @classmethod
    def from_image(cls, input_image):
        """
        Create object from element of the class Image.
        :param input_image: input nifti image from which we want to compute the image.
        """
        return cls(input_image.nib_image)

    @classmethod
    def from_array(cls, array, affine=np.eye(4), homogeneous=False):
        image = nib.Nifti1Image(array, affine=affine)
        return cls(image)

    ### SVF operations ###

    def __add__(self, other):
        """
        Inner sum of the vector field structure. Images can be summed only with images
        and only if the transformation is the same.
        :param self: first addend
        :param other: second addend
        :return: addition of images with same shape and same affine matrix
        """

        if not isinstance(other, self.__class__):
            raise TypeError('unsupported operand type(s) for +: Image and ' + other.__class__.__name__)

        return SVF.from_array_with_header(self.field + other.field,
                                            self.nib_image.get_header(),
                                            affine=self.voxel_2_mm)

    def __sub__(self, other, warn=False):
        """
        Inner sum of the vector field structure
        :param self:
        :param other:
        :return: subtraction of images
        """
        if not len(self.shape) == len(other.shape):
            raise TypeError('unsupported operand type(s) for -: Images of different sizes.')

        return SVF.from_array_with_header(self.field - other.field,
                                            self.nib_image.get_header(),
                                            affine=self.voxel_2_mm)

    def __rmul__(self, alpha):
        """
        operation of scalar multiplication
        :param alpha:
        :return:
        """
        return SVF.from_array_with_header(alpha * self.field,
                                            self.nib_image.get_header(),
                                            affine=self.voxel_2_mm)

    @staticmethod
    def jacobian_product(left_input, right_input):
        """
        Compute the jacobian product between two 2d or 3d SVF self and right: J_{self}(right)
        the results is a second SVF.
        :param left_input : svf
        :param right_input: svf
        :return: J_{right}(left) : jacobian product between 2 svfs
        """
        if not isinstance(right_input, type(left_input)):
            raise TypeError('Jacobian product is an inner operation between SVF. Check involved elements are SVF')

        # non-destructive algorithm.
        left  = copy.deepcopy(left_input)
        right = copy.deepcopy(right_input)

        result_field = matrix_vector_field_product(SDISP.compute_jacobian(right).field, left.field)

        return SVF.from_field(result_field)

    @staticmethod
    def iterative_jacobian_product(v, n):
        """
        :param v: input SVF
        :param n: number of iterations
        :return: a new SVF defined by the jacobian product J_v^(n-1) v
        """

        jv_field = SDISP.compute_jacobian(v).field[...]
        v_field = v.field[...]

        jv_n_prod_v = matrix_vector_field_product(matrix_fields_product_iterative(jv_field, n-1), v_field)

        return SVF.from_field(jv_n_prod_v)

    @staticmethod
    def lie_bracket(left, right):
        """
        Compute the Lie bracket of two velocity fields.

        Parameters:
        -----------
        :param left: Left velocity field.
        :param right: Right velocity field.
        Order of Lie bracket: [left,right] = Jac(left)*right - Jac(right)*left
        :return Return the resulting velocity field
        """

        return SVF.jacobian_product(left, right) - SVF.jacobian_product(right, left)

    ### Functions methods (exp and related) ###

    def exponential(self, algorithm='ss', s_i_o=3, input_num_steps=None):
        """
        Compute the exponential of this velocity field using the
        scaling and squaring approach.

        Scaling and squaring:
        (1) -- Scaling step:
        divides data time .

        (2) -- Squaring step:
        Do the squaring step to perform the integration
        The exponential is num_steps times recursive composition of
        the field with itself, which is equivalent to integration over
        the unit interval.

        Polyaffine scaling and squaring

        Euler method

        Midpoint method

        Euler modified

        Trapezoidal method

        Runge Kutta method

        -> These method has been rewritten externally as an external function in utils exp_svf

        :param algorithm: algorithm name
        :param s_i_o: spline interpolation order
        :param input_num_steps: num steps of
        :param : It returns a displacement, element of the class disp.
        """

        v = copy.deepcopy(SVF.from_field(self.field, header=self.nib_image.get_header()))
        phi = copy.deepcopy(SDISP.generate_zero(shape=self.shape, header=self.nib_image.get_header()))

        ''' automatic computation of the optimal number of steps: '''
        if input_num_steps is None:

            norm = np.linalg.norm(v.field, axis=v.field.ndim - 1)
            max_norm = np.max(norm[:])

            if max_norm < 0:
                raise ValueError('Maximum norm is invalid.')
            if max_norm == 0:
                return phi
            pix_dims = np.asarray(v.zooms)
            min_size = np.min(pix_dims[pix_dims > 0])
            num_steps = max(0, np.ceil(np.log2(max_norm / (min_size / 2))).astype('int')) + 3
        else:
            num_steps = input_num_steps

        ''' Collection of numerical method: '''

        # scaling and squaring:
        if algorithm == 'ss':

            # (1)
            init = 1 << num_steps  # equivalent to 1 * pow(2, num_steps)
            phi.field = v.field / init

            # (2)
            for _ in range(0, num_steps):
                phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

        # Scaling and squaring exponential integrators
        elif algorithm == 'gss_ei':

            # (1)
            init = 1 << num_steps
            phi.field = v.field / init

            # (1.5)
            jv = SDISP.compute_jacobian(phi)

            if v.dim == 2:

                v_matrix = np.array([0.0] * 3 * 3).reshape([3, 3])

                for x in range(phi.field.shape[0]):
                    for y in range(phi.field.shape[1]):
                        # skew symmetric part
                        v_matrix[0:2, 0:2] = jv.field[x, y, 0, 0, :].reshape([2, 2])
                        # translation part
                        v_matrix[0, 2], v_matrix[1, 2] = phi.field[x, y, 0, 0, 0:2] #+ \
                                                         #jv.field[x, y, 0, 0, :].reshape([2, 2]).dot([x, y])

                        # translational part of the exp is the answer:
                        phi.field[x, y, 0, 0, :] = expm(v_matrix)[0:2, 2]

            elif v.dim == 3:

                v_matrix = np.array([0.0] * 4 * 4).reshape([4, 4])

                for x in range(phi.field.shape[0]):
                    for y in range(phi.field.shape[1]):
                        for z in range(phi.field.shape[2]):

                            # skew symmetric part
                            v_matrix[0:3, 0:3] = jv.field[x, y, z, 0, :].reshape([3, 3])

                            # translation part
                            v_matrix[0, 3], v_matrix[1, 3], v_matrix[2, 3] = phi.field[x, y, z, 0, 0:3]

                            phi.field[x, y, z, 0, :] = expm(v_matrix)[0:3, 3]

            else:
                raise TypeError("Problem in the number of dimensions!")

            # (2)
            for _ in range(0, num_steps):
                phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

        # Affine scaling and squaring exponential integrators modified.
        elif algorithm == 'gss_ei_mod':

            # (1) copy the reduced v in phi, future solution of the ODE.
            init = 1 << num_steps
            phi.field = v.field / init

            # (1.5)
            jv = SDISP.compute_jacobian(phi)

            if v.dim == 2:

                for x in range(phi.shape[0]):
                    for y in range(phi.shape[1]):

                        j = jv.field[x, y, 0, 0, :].reshape([2, 2])
                        tr = phi.field[x, y, 0, 0, 0:2]
                        j_tr = j.dot(tr)
                        phi.field[x, y, 0, 0, :] = tr + 0.5 * j_tr  # + 1/6. * J.dot(J_tr)

            elif v.dim == 3:

                for x in range(phi.field.shape[0]):
                    for y in range(phi.field.shape[1]):
                        for z in range(phi.field.shape[2]):

                            j = jv.field[x, y, z, 0, :].reshape([3, 3])
                            tr = phi.field[x, y, z, 0, 0:3]
                            j_tr = j.dot(tr)
                            phi.field[x, y, z, 0, :] = tr + 0.5 * j_tr  # + 1/6. * j.dot(j_tr)

            else:
                raise TypeError("Problem in dimensions number!")

            # (2)
            for _ in range(0, num_steps):
                phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

        # scaling and squaring approximated exponential integrators
        elif algorithm == 'gss_aei':

            # (1)
            if num_steps == 0:
                phi.field = v.field
            else:
                init = 1 << num_steps
                phi.field = v.field / init

            # (1.5)  phi = 1 + v + 0.5jac*v
            jv = np.squeeze(SDISP.compute_jacobian(phi).field)
            v_sq = np.squeeze(phi.field)
            jv_prod_v = matrix_vector_field_product(jv, v_sq).reshape(list(v.vol_ext) + [1]*(4-v.dim) + [v.dim])

            phi.field += 0.5*jv_prod_v

            # (2)
            for _ in range(0, num_steps):
                phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

        elif algorithm == 'midpoint':

            if input_num_steps is None:
                num_steps = 10
            else:
                num_steps = input_num_steps

            if num_steps == 0:
                h = 1.0
            else:
                h = 1.0 / num_steps
            for i in range(num_steps):

                phi_pos = SDISP.deformation_from_displacement(phi)
                phi_tilda = SDISP.from_array(phi.field)
                phi_tilda.field = phi.field + (h / 2) * SDISP.compose_with_deformation_field(v, phi_pos,
                                                                                             s_i_o=s_i_o).field

                phi_tilda_pos = SDISP.deformation_from_displacement(phi_tilda)

                phi.field += h * SDISP.compose_with_deformation_field(v, phi_tilda_pos, s_i_o=s_i_o).field

        # Series method
        elif algorithm == 'series':

            # Automatic step selector:
            if input_num_steps is None:
                norm = np.linalg.norm(v.field, axis=v.field.ndim - 1)
                max_norm = np.max(norm[:])
                toll = 1e-3
                k = 10
                while max_norm / fact(k) >  toll:
                    k += 1
                num_steps = k
                print 'automatic steps selector for series method: ' + str(k)

            else:
                num_steps = input_num_steps

            phi.field = v.field[...]  # final output is phi.

            for k in range(2, num_steps):
                jac_v = SVF.iterative_jacobian_product(v, k)
                phi.field = phi.field[...] + jac_v.field[...] / fact(k)

        # Series method
        elif algorithm == 'series_mod':  # jacobian computed in the improper way

            jac_v = copy.deepcopy(v)
            phi.field = v.field[...]  # final output is phi.

            for k in range(1, input_num_steps):
                jac_v = SVF.jacobian_product(jac_v, v)
                phi.field = phi.field[...] + jac_v.field[...] / fact(k)

        # Euler method
        elif algorithm == 'euler':

            if input_num_steps is None:
                num_steps = 10
            else:
                num_steps = input_num_steps
            if num_steps == 0:
                h = 1.0
            else:
                h = 1.0 / num_steps
            for i in range(num_steps):
                phi_def = SDISP.deformation_from_displacement(phi)
                phi.field += h * SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field

        # Euler approximated exponential integrator
        elif algorithm == 'euler_aei':

            v.field = v.field / num_steps

            jv = np.squeeze(SDISP.compute_jacobian(v).field)
            v_sq = np.squeeze(v.field)
            jv_prod_v = matrix_vector_field_product(jv, v_sq).reshape(list(v.vol_ext) + [1]*(4-v.dim) + [v.dim])

            v.field += 0.5*jv_prod_v

            for _ in range(num_steps):
                phi = SDISP.composition(v, phi, s_i_o=s_i_o)

        # Euler modified
        elif algorithm == 'euler_mod':

            if input_num_steps is None:
                num_steps = 10
            else:
                num_steps = input_num_steps

            if num_steps == 0:
                h = 1.0
            else:
                h = 1.0 / num_steps

            for i in range(num_steps):
                # Code can be optimized if we allow vector operations for displacement field, but not deformation!

                phi_def = SDISP.deformation_from_displacement(phi)

                psi_1 = SDISP.from_array(phi.field)
                psi_2 = SDISP.from_array(phi.field)

                psi_1.field = phi.field + h * SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field
                psi_1_def   = SDISP.deformation_from_displacement(psi_1)

                psi_2.field = SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field + \
                        SDISP.compose_with_deformation_field(v, psi_1_def, s_i_o=s_i_o).field

                phi.field += (h/2) * psi_2.field

        # Heun
        elif algorithm == 'heun':

            if input_num_steps is None:
                num_steps = 10
            else:
                num_steps = input_num_steps
            if num_steps == 0:
                h = 1.0
            else:
               h = 1.0 / num_steps

            for i in range(num_steps):
                phi_def = SDISP.deformation_from_displacement(phi)

                psi_1 = SDISP.from_array(phi.field)
                psi_2 = SDISP.from_array(phi.field)

                psi_1.field = phi.field + h * (2. / 3) * SDISP.compose_with_deformation_field(v, phi_def,
                                                                                              s_i_o=s_i_o).field
                psi_1_def   = SDISP.deformation_from_displacement(psi_1)

                psi_2.field = SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field + \
                              3 * SDISP.compose_with_deformation_field(v, psi_1_def, s_i_o=s_i_o).field

                phi.field += (h / 4) * psi_2.field

        # Heun modified
        elif algorithm == 'heun_mod':

            if input_num_steps is None:
                num_steps = 10
            else:
                num_steps = input_num_steps

            if num_steps == 0:
                h = 1.0
            else:
                h = 1.0 / num_steps
            for i in range(num_steps):
                phi_def = SDISP.deformation_from_displacement(phi)

                psi_1 = SDISP.from_array(phi.field)
                psi_2 = SDISP.from_array(phi.field)
                psi_3 = SDISP.from_array(phi.field)

                psi_1.field = phi.field + (h / 3) * SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field
                psi_1_def   = SDISP.deformation_from_displacement(psi_1)

                psi_2.field = phi.field + h * (2. / 3) * SDISP.compose_with_deformation_field(v, psi_1_def,
                                                                                              s_i_o=s_i_o).field
                psi_2_def   = SDISP.deformation_from_displacement(psi_2)

                psi_3.field = SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field + \
                              3 * SDISP.compose_with_deformation_field(v, psi_2_def, s_i_o=s_i_o).field

                phi.field += (h / 4) * psi_3.field

        # Runge Kutta 4
        elif algorithm == 'rk4':

            if input_num_steps is None:
                num_steps = 10
            else:
                num_steps = input_num_steps

            if num_steps == 0:
                h = 1.0
            else:
                h = 1.0 / num_steps

            for i in range(num_steps):

                phi_def = SDISP.deformation_from_displacement(phi)

                r_1 = SDISP.from_array(phi.field)
                r_2 = SDISP.from_array(phi.field)
                r_3 = SDISP.from_array(phi.field)
                r_4 = SDISP.from_array(phi.field)

                psi_1 = SDISP.from_array(phi.field)
                psi_2 = SDISP.from_array(phi.field)
                psi_3 = SDISP.from_array(phi.field)

                r_1.field   = h * SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field

                psi_1.field = phi.field + .5 * r_1.field
                psi_1_def  = SDISP.deformation_from_displacement(psi_1)
                r_2.field   = h  * SDISP.compose_with_deformation_field(v, psi_1_def, s_i_o=s_i_o).field

                psi_2.field = phi.field + .5 * r_2.field
                psi_2_def  = SDISP.deformation_from_displacement(psi_2)
                r_3.field   = h  * SDISP.compose_with_deformation_field(v, psi_2_def, s_i_o=s_i_o).field

                psi_3.field = phi.field + r_3.field
                psi_3_def  = SDISP.deformation_from_displacement(psi_3)
                r_4.field = h  * SDISP.compose_with_deformation_field(v, psi_3_def, s_i_o=s_i_o).field

                phi.field += (1. / 6) * (r_1.field + 2 * r_2.field + 2 * r_3.field + r_4.field)

        # Generalized scaling and squaring with runge kutta
        elif algorithm == 'gss_rk4':

            norm = np.linalg.norm(v.field, axis=v.field.ndim - 1)
            max_norm = np.max(norm[:])

            if max_norm < 0:
                raise ValueError('Maximum norm is invalid.')
            if max_norm == 0:
                return phi

            if input_num_steps is None:
                # automatic computation of the optimal number of steps:
                pix_dims = np.asarray(v.zooms)
                min_size = np.min(pix_dims[pix_dims > 0])
                num_steps = max(0, np.ceil(np.log2(max_norm / (min_size / 2))).astype('int')) + 2  # adaptative method.
            else:
                num_steps = input_num_steps

            # (1)
            init = 1 << num_steps  # equivalent to 1 * pow(2, num_steps)
            v.field = v.field / init  # LET IT LIKE THAT! No augmented assignment!!

            # rk steps:
            input_num_steps_rk4 = 7
            h = 1.0 / input_num_steps_rk4

            for i in range(input_num_steps_rk4):

                phi_def = SDISP.deformation_from_displacement(phi)

                r_1 = SDISP.from_array(phi.field)
                r_2 = SDISP.from_array(phi.field)
                r_3 = SDISP.from_array(phi.field)
                r_4 = SDISP.from_array(phi.field)

                psi_1 = SDISP.from_array(phi.field)
                psi_2 = SDISP.from_array(phi.field)
                psi_3 = SDISP.from_array(phi.field)

                r_1.field   = h * SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field

                psi_1.field = phi.field + .5 * r_1.field
                psi_1_def  = SDISP.deformation_from_displacement(psi_1)
                r_2.field   = h  * SDISP.compose_with_deformation_field(v, psi_1_def, s_i_o=s_i_o).field

                psi_2.field = phi.field + .5 * r_2.field
                psi_2_def  = SDISP.deformation_from_displacement(psi_2)
                r_3.field   = h  * SDISP.compose_with_deformation_field(v, psi_2_def, s_i_o=s_i_o).field

                psi_3.field = phi.field + r_3.field
                psi_3_def  = SDISP.deformation_from_displacement(psi_3)
                r_4.field = h  * SDISP.compose_with_deformation_field(v, psi_3_def, s_i_o=s_i_o).field

                phi.field += (1. / 6) * (r_1.field + 2 * r_2.field + 2 * r_3.field + r_4.field)

            # (2)
            for _ in range(num_steps):
                phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

        else:
            raise TypeError('Error: wrong algorithm name. You inserted ' + algorithm)

        return phi

    def exponential_scipy(self,
                          integrator='vode',
                          method='bdf',
                          max_steps=7,
                          interpolation_method='cubic',
                          passepartout=3,
                          return_integral_curves=False,
                          verbose=False):
        """
        Compute the exponential of this velocity field using scipy libraries.
        :param integrator: vode, zvode, lsoda, dopri5, dopri853, see scipy documentations:
            http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.integrate.ode.html
        :param method: adams (non-stiff) or bdf (stiff)
        :param interpolation_method:
        :param max_steps: maximum steps of integrations. If less steps are required, it prints a warning message.
        :param passepartout:
        :param return_integral_curves:
        :param verbose:
        :param : It returns a displacement phi, element of the class disp, as
                    phi(x) = psi(x) - x
                where psi is the chosen integrator.

        """
        v = copy.deepcopy(SVF.from_field(self.field, header=self.nib_image.get_header()))
        phi = copy.deepcopy(SDISP.generate_zero(shape=self.shape,
                                                header=self.nib_image.get_header(),
                                                affine=self.mm_2_voxel))

        flows_collector = []

        # transform v in a function suitable for ode library :
        def vf(t, x):
            return list(v.one_point_interpolation(point=x, method=interpolation_method))

        t0, t_n = 0, 1
        dt = (t_n - t0)/float(max_steps)

        r = ode(vf).set_integrator(integrator, method=method, max_step=max_steps)

        for i in range(0 + passepartout, phi.vol_ext[0] - passepartout + 1):
            for j in range(0 + passepartout, phi.vol_ext[1] - passepartout + 1):  # cycle on the point of the grid.

                y = []
                r.set_initial_value([i, j], t0).set_f_params()  # initial conditions are point on the grid
                while r.successful() and r.t + dt < t_n:
                    r.integrate(r.t+dt)
                    y.append(r.y)

                # flow of the svf at the point [i,j]
                fl = np.array(np.real(y))

                # subtract id on the run to return a displacement.
                if fl.shape[0] > 0:
                    phi.field[i, j, 0, 0, :] = fl[fl.shape[0]-1, :] - np.array([i, j])
                else:
                    phi.field[i, j, 0, 0, :] = - np.array([i, j])

                if verbose:
                    print 'Integral curve at grid point ' + str([i, j]) + ' is computed.'

                # In some cases as critical points, or when too closed to the closure of the domain
                # the number of steps can be reduced by the algorithm.
                if fl.shape[0] < max_steps - 2 and verbose:  # the first step is not directly considered
                    print "--------"
                    print "Warning!"  # steps jumped for the point
                    print "--------"

                if return_integral_curves:
                    flows_collector += [fl]

        if return_integral_curves:
            return phi, flows_collector
        else:
            return phi

    @classmethod
    def log_composition(cls, svf_im0, svf_im1, kind=('ground',), answer='dom'):
        """
        From two stationary velocity fields in the tangent space he returns their composition in the Lie group.
        No log is computed on the resulting def (deformation field)
        :param svf_im0:
        :param svf_im1:
        :param kind: 'ground'
            ground truth for the composition, it returns the value \exp(svf_im0)\circ \exp(svf_im1)
        :param answer:
        # TODO select if you want the answer in the domain or not!

        kind: 'bch0'
            composition using the bch 0: it returns
            \exp(svf_im0)\circ \exp(svf_im1) \simeq \exp(svf_im0) + \exp(svf_im1)
        kind: 'bch1'
            composition using the bch 1: it returns
            \exp(svf_im0)\circ \exp(svf_im1) \simeq \exp(svf_im0) + \exp(svf_im1) + .5 [\exp(svf_im0), \exp(svf_im1)]
        kind: 'pt'
            composition using parallel transport: it returns
            \exp(svf_im0)\circ \exp(svf_im1) \simeq ...
        kind: 'pt_warp'
            composition using parallel transport warped: it returns
            \exp(svf_im0)\circ \exp(svf_im1) \simeq ...

        kind: numerical closed, if len = 2 - [kind_log, kind_exp], uses a numerical method for the
            log and a numerical method for the exp, in the formula exp(log(svf_im0) o log(svf_im1))

        answer: 'svf' returns approximation of log(exp(svf_im0) o exp(svf_im1))
             or 'def' returns approximation of exp(svf_im0) o exp(svf_im1), more manageable to compute the ground truth.

        :return: exp(svf_im0) o exp(svf_im1) or log(exp(svf_im0) o exp(svf_im1)) according to the flag answer.
        It is in the Lie group and not in the Lie algebra, since the computation of the error is in the Lie group.
        """
        # TODO: refactor this code within debugging!

        str_error_kind = 'Error: wrong input data for the chosen kind of composition.'
        str_error_answer = 'Error: wrong input data for the chosen answer of composition.'

        if len(kind) == 1:
            # Numerical methods BCH based and related

            if kind == 'bch0':
                svf_bch0 = copy.deepcopy(svf_im0)
                svf_bch0.field.data += svf_im1.field.data
                def_result = svf_bch0.exponential()

            elif kind == 'bch1':
                vel_bch1 = copy.deepcopy(svf_im0)
                vel_bch1.field.data += svf_im1.field.data
                vel_bch1.field.data += 0.5 * svf_im0.lie_bracket(svf_im1).field.data
                def_result = vel_bch1.exponential()

            elif kind == 'bch1.5':
                vel_bch2 = copy.deepcopy(svf_im0)
                vel_bch2.field.data += svf_im1.field.data
                vel_bch2.field.data += 0.5 * cls.lie_bracket(svf_im0, svf_im1).field.data
                vel_bch2.field.data += (1 / 12.0) * cls.lie_bracket(svf_im0, cls.lie_bracket(svf_im0, svf_im1)).field

                def_result = vel_bch2.exponential()

            elif kind == 'bch2':
                vel_bch2 = copy.deepcopy(svf_im0)
                vel_bch2.field.data += svf_im1.field.data
                vel_bch2.field.data += 0.5 * cls.lie_bracket(svf_im0, svf_im1).field.data
                vel_bch2.field.data += (1 / 12.0) * (cls.lie_bracket(svf_im0, cls.lie_bracket(svf_im0, svf_im1)).field +
                                                     cls.lie_bracket(svf_im1,
                                                                       cls.lie_bracket(svf_im1, svf_im0)).field.data)
                def_result = vel_bch2.exponential()

            elif kind == 'pt':
                tmp_vel = copy.deepcopy(svf_im0)
                tmp_vel.field.data /= 2

                tmp_def_a = tmp_vel.exponential()
                tmp_vel.field.data = -tmp_vel.field.data
                tmp_def_b = tmp_vel.exponential()
                tmp_def_c = SDISP.composition(SDISP.composition(tmp_def_a, svf_im1.exponential()), tmp_def_b)

                vel_pt = copy.deepcopy(svf_im0)
                vel_pt.field.data += tmp_def_c.data
                def_result = vel_pt.exponential()

            elif kind == 'pt_alternate':

                tmp_vel = copy.deepcopy(svf_im0)
                tmp_vel.field.data /= 2

                tmp_def_a = tmp_vel.exponential()
                tmp_vel.field.data = -tmp_vel.field.data
                tmp_def_b = tmp_vel.exponentiatial()
                tmp_def_c = SDISP.composition(SDISP.composition(tmp_def_a, svf_im1.exponential()), tmp_def_b)

                vel_pt = copy.deepcopy(svf_im0)
                vel_pt.field.data += tmp_def_c.data
                def_result = vel_pt.exponential()

            else:
                raise TypeError(str_error_kind)

        elif len(kind) == 2:

            if kind[0] == 'euler' and kind[1] == 'inv_ss':
                def_im0 = svf_im0.exponential(kind=kind[0])
                def_im1 = svf_im1.exponential(kind=kind[0])
                if answer == 'disp':
                    def_result = SDISP.composition(def_im0, def_im1)
                elif answer == 'svf':
                    def_result = SDISP.composition(def_im0, def_im1).logarithm(kind=kind[1])
                else:
                    raise TypeError(str_error_answer)
            else:
                raise TypeError(str_error_kind)

        else:
            raise TypeError(str_error_kind)

        return def_result
