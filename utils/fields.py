import numpy as np
import scipy.ndimage.filters as fil
from scipy.interpolate import griddata, Rbf
import copy
from utils.aux_functions import id_matrix_field


class Field(object):
    """
    class for scalar and vector fields.
    Can be array as:
    (x,y)       : 2d scalar field
    (x,y,z)     : 3d scalar field
    (x,y,1,t)   : time varying 2d scalar field (time always need to be at the third point)
    (x,y,z,t)   : time varying 3d scalar field
    (x,y,z,t,d) : time varying (t>0) or stationary (t=0) vector field
    In this last case d can be a vector of any dimension, not necessarily a multiple of the volume (e.g. jacobian).

    A vector field can be defined in lagrangian or eulerian coordinate system.
     - a vector field in Lagrangian coordinates is sometimes called displacement
     - a vector field in Eulerian coordinates is sometimes called deformation

    By default is lagrangian, but this must be considered with some care when an svf is exponentiated, and when the
    composition of two def is performed.
    """

    def __init__(self, array, homogeneous=False):
        self.field = array
        self.homogeneous = homogeneous

        self.__set_attributes()

    def __set_attributes(self):
        self.time_points = 1
        self.vol_ext = self.field.shape
        self.shape = self.field.shape

        if len(self.vol_ext) > 3:
            self.time_points = self.vol_ext[3]
            if self.vol_ext[2] > 1:
                self.vol_ext = self.vol_ext[:3]
            else:
                self.vol_ext = self.vol_ext[:2]

        self.dim = sum(np.array(self.vol_ext) > 1)

    @classmethod
    def from_array(cls, array, homogeneous=False):
        """
        Alternative initialization to generate a field.
        :param array:
        :param homogeneous:
        :return:
        """
        return cls(array, homogeneous=homogeneous)

    ### Vector space operations: ###

    def __add__(self, other):
        """
        Inner sum of the vector field structure
        :param self: first addend
        :param other: second addend
        :return: addition of images with same shape and same affine matrix

        TODO can this be refactored as a class method?
        Using a dictionary to store the types of each input and calling the
        type of the external function.

        """
        return Field(self.field + other.field)

    def __sub__(self, other):
        """
        Inner sum of the vector field structure
        :param self:
        :param other:
        :return: subtraction of images
        """
        return Field(self.field - other.field)

    def __rmul__(self, alpha):
        """
        operation of scalar multiplication
        :param alpha:
        :return:
        """
        return Field(alpha * self.field)

    @classmethod
    def trim(cls, self, passe_partout_size, return_copy=False):
        """
        :param self: must be an object of the class field, or children.
        :param passe_partout_size: passepartout value
        :param return_copy: False by default, if you want to adjust the existing field. True otherwise.
        :return: the same field trimmed by the value of the passepartout in each dimension.
        """

        if return_copy:

            new_field = copy.deepcopy(self)
            if new_field.dim == 2:
                new_field.field = self.field[passe_partout_size:-passe_partout_size,
                                             passe_partout_size:-passe_partout_size,
                                             ...]
            else:
                new_field.field = self.field[passe_partout_size:-passe_partout_size,
                                             passe_partout_size:-passe_partout_size,
                                             ...]
            return cls.from_array(new_field.field)

        else:
            if self.dim == 2:
                self.field = self.field[passe_partout_size:-passe_partout_size,
                                        passe_partout_size:-passe_partout_size,
                                        ...]
            else:
                self.field = self.field[passe_partout_size:-passe_partout_size,
                                        passe_partout_size:-passe_partout_size,
                                        passe_partout_size:-passe_partout_size,
                                        ...]

    ### Affine-homogeneous coordinates methods ###

    def to_homogeneous(self):
        """
        Adds the homogeneous coordinates to the given affine field.
        It changes the provided data structure instead of creating a new one.

        :return: field in homogeneous coordinates.

        NOTE: it works only for vector fields, so for fields with one more coordinate.
        If the given vector field was already affine nothing happen.
        """
        if not len(self.shape) == 5:
                raise TypeError('Method to_homogeneous of class Field is only for 5 dim fields')
        if not self.homogeneous:
            slice_shape = list(self.shape[:])
            slice_shape[4] = 1

            self.field = np.append(self.field, np.ones(slice_shape), axis=4)

            self.homogeneous = True
            self.__set_attributes()

    def to_affine(self):
        """
        Removes the homogeneous coordinates to the given homogeneous field.
        It changes the provided data structure instead of creating a new one.

        :return: field in affine coordinates.

        NOTE: it works only for vector fields, so for fields with one more coordinate.
        If the given vector field was already affine nothing happen.
        """
        if not len(self.shape) == 5:
                raise TypeError('Method to_affine of class Field is only for 5 dim fields')
        if self.homogeneous:
            self.field = self.field[..., :-1]

            self.homogeneous = False
            self.__set_attributes()

    ### Jacobian methods ###

    @classmethod
    def initialise_jacobian(cls, field_input):
        """
        Create a jacobian field image from the specified target field.
        Sets the data to 0.

        Parameters:
        -----------
        :param field_input: The target field.
        :return: Return the created jacobian field object. Each jacobian is stored in a vector of size 9
                 in row major order
        """
        # Initialise image with zeros of appropriate shape:
        sh = list(field_input.shape)
        while len(sh) < 5:
            sh.extend([1])
        sh[-1] = field_input.dim ** 2

        array = np.zeros(sh, dtype=np.float64)

        return cls.from_array(array)

    @classmethod
    def compute_jacobian(cls, input_obj, affine=np.eye(4), is_displacement=False):
        """
        :param input_obj: The field (or children) whose jacobian we need to compute
        :param affine: The affine transformation optionally associated to the field.
        :param is_displacement: if the identity matrix should be added to each jacobian matrix
        See itk documentation:
        http://www.itk.org/Doxygen/html/classitk_1_1DisplacementFieldJacobianDeterminantFilter.html

        On the diagonal it possess the sample distances for each dimension.
        Jacobian matrix at each point of the grid is stored in a vector of size 9 in row major order.
        """
        # NOTE: It works only for 1 time point - provisional - do with multiple time point
        # once tv_disp and tv_vf are implemented
        # if self.time_points > 1:
        #    raise TypeError('Jacobian works only for 1 time points Images.')
        if not input_obj.dim == input_obj.shape[4]:
            raise TypeError('Jacobian works only for vector fields of reasonable size.')

        jacobian = cls.initialise_jacobian(input_obj)

        # affine = self.voxel_2_mm
        dims = []

        for i in range(input_obj.dim):
            dims.append(affine[i, i])

        output_data = jacobian.field.squeeze()

        for i in range(input_obj.dim):
            grad = np.gradient(np.squeeze(input_obj.field[..., i]), *dims)

            for j in range(input_obj.dim):
                output_data[..., i * input_obj.dim + j] = grad[j].squeeze()

        if is_displacement:
            jacobian.field += id_matrix_field(input_obj.shape[:input_obj.dim])

        return jacobian

    @classmethod
    def compute_jacobian_determinant(cls, input_obj, is_displacement=False):
        """
        :param input_obj: The Field or children whose jacobian we need to compute.
        :param is_displacement: add the identity to the jacobian matrix before the computation of the
        jacobian determinant.
        If it is none, it is allocated.
        Jacobian matrix at each point of the grid is stored in a vector of size 9 in row major order.
        !! It works only for 1 time point - provisional !!
        """
        jacobian_image = cls.compute_jacobian(input_obj, is_displacement=is_displacement)

        sh = list(input_obj.shape)
        while len(sh) < 5:
            sh.extend([1])
        sh = sh[:-1]

        sh.extend([input_obj.dim, input_obj.dim])

        v = jacobian_image.field.reshape(sh)
        array_jac_det = np.linalg.det(v)

        return cls.from_array(array_jac_det)

    ### Norm methods ###

    def norm(self, passe_partout_size=1, normalized=False):
        """
        This returns the L2-norm as the field is a discretization of some continuous field.
        Based on the norm function from numpy.linalg of ord=2 for the vectorized matrix.
        The result can be computed with a passe partout
        (the discrete domain is reduced on each side by the same value, keeping the proportion
        of the original image) and can be normalized with the size of the domain.

        -> F vector field from a compact \Omega to R^d
        \norm{F} = (\frac{1}{|\Omega|}\int_{\Omega}|F(x)|^{2}dx)^{1/2}
        Discretisation:
        \Delta\norm{F} = \frac{1}{\sqrt{dim(x)dim(y)dim(z)}}\sum_{v \in \Delta\Omega}|v|^{2})^{1/2}
                       = \frac{1}{\sqrt{XYZ}}\sum_{i,j,k}^{ X,Y,Z}|a_{i,j,k}|^{2})^{1/2}

        -> f scalar field from \Omega to R, f is an element of the L^s space
        \norm{f} = (\frac{1}{|\Omega|}\int_{\Omega}f(x)^{2}dx)^{1/2}
        Discretisation:
        \Delta\norm{F} = \frac{1}{\sqrt{XYZ}}\sum_{i,j,k}^{ X,Y,Z} a_{i,j,k}^{2})^{1/2}

        Parameters:
        ------------
        :param passe_partout_size: size of the passe partout (rectangular mask, with constant offset on each side).
        :param normalized: if the result is divided by the normalization constant.
        """
        if passe_partout_size > 0:
            if self.dim == 2:
                masked_field = self.field[passe_partout_size:-passe_partout_size,
                                          passe_partout_size:-passe_partout_size,
                                          ...]
            else:
                masked_field = self.field[passe_partout_size:-passe_partout_size,
                                          passe_partout_size:-passe_partout_size,
                                          passe_partout_size:-passe_partout_size,
                                          ...]
        else:
            masked_field = self.field

        if normalized:
            # volume of the field after masking (to compute the normalization factor):
            mask_vol = \
                (np.array(self.field.shape[0:self.dim])
                 - np.array([2 * passe_partout_size] * self.dim)).clip(min=1)

            return np.linalg.norm(masked_field.ravel(), ord=2) / np.sqrt(np.prod(mask_vol))
        else:
            return np.linalg.norm(masked_field.ravel(), ord=2)

    @staticmethod
    def norm_of_difference_of_fields(field_a, field_b, passe_partout_size=1, normalized=False):
        """
        Norm of the difference of two images.
        :param field_a:
        :param field_b:
         :param passe_partout_size: size of the passe partout (rectangular mask, with constant offset on each side).
        :param normalized: if the result is divided by the normalization constant.
        """
        return (field_a - field_b).norm(passe_partout_size=passe_partout_size, normalized=normalized)

    ### Generator methods ###

    @classmethod
    def generate_zero(cls, shape):
        array = np.zeros(shape)
        return cls(array)

    @classmethod
    def generate_id(cls, shape):
        """
        shape must have the form (x,y,z,t,d).
        The field is in matrix coordinates. The values of the domain is
        0 <= x < shape[0]
        0 <= y < shape[1]
        0 <= z < shape[2]  (optional if greater than 1)
        and d must have the appropriate dimension ALWAYS related to matrix coordinates.

        Parameters
        -------------
        :param shape: shape of a standard image (4 dim, [x,y,z,t]) or vector field (5 dim, [x,y,z,t,d]).
        :return:
        """

        if not len(shape) == 5:
            raise IOError("shape must be of the standard form (x,y,z,t,d) of len 5.")

        domain = [shape[j] for j in range(3) if shape[j] > 1]
        dim = len(domain)
        time_points = shape[3]

        if not dim == shape[4]:
            raise IOError("To have the identity, shape must be of the standard form (x,y,z,t,d) "
                          "with d corresponding to the dimension.")

        if dim == 2:
            x = range(shape[0])
            y = range(shape[1])
            gx, gy = np.meshgrid(x, y)
            gx, gy = gx.T, gy.T

            id_field = np.zeros(list(gx.shape) + [1, time_points, 2])
            id_field[..., 0, :, 0] = np.repeat(gx, time_points).reshape(domain + [time_points])
            id_field[..., 0, :, 1] = np.repeat(gy, time_points).reshape(domain + [time_points])

        elif dim == 3:
            x = range(shape[0])
            y = range(shape[1])
            z = range(shape[2])
            gx, gy, gz = np.meshgrid(x, y, z)
            gx, gy, gz = gy, gx, gz  # swap!

            id_field = np.zeros(list(domain) + [time_points, 3])
            id_field[..., :, 0] = np.repeat(gx, time_points).reshape(domain + [time_points])
            id_field[..., :, 1] = np.repeat(gy, time_points).reshape(domain + [time_points])
            id_field[..., :, 2] = np.repeat(gz, time_points).reshape(domain + [time_points])

        else:
            raise IOError("Dimensions allowed: 2, 3")

        return cls(id_field)

    @classmethod
    def generate_random_smooth(cls, shape,
                               mean=0, sigma=5,
                               sigma_gaussian_filter=2):

        array = np.random.normal(mean, sigma, shape)

        # Filter data, per slices!

        if sigma_gaussian_filter > 0:
            len_shape = len(shape)
            if len_shape < 3:
                array = fil.gaussian_filter(array, sigma_gaussian_filter)

            elif len_shape == 4:
                if shape[3] > 1:
                    # Not easy to implement this for multiple time points!
                    # Find some way to validate this  once implemented. Can
                    # lead to major mistakes.
                    raise IndexError('not defined for multiple time points')
                array = fil.gaussian_filter(array, sigma_gaussian_filter)
            elif len_shape == 5:
                if shape[3] > 1:
                    raise IndexError('not defined for multiple time points')
                for i in range(shape[4]):
                    array[..., 0, i] = fil.gaussian_filter(array[..., 0, i], sigma_gaussian_filter)
            else:
                raise TypeError('Invalid shape.')

        return Field.from_array(array)

    @classmethod
    def generate_id_from_obj(cls, obj):

        if not cls.__name__ == obj.__class__.__name__:
            # verify that the class calling the method generate_id_from_obj is the same as the input object.
            # Image.generate_id_from_obj(f) works <-> f is not of class Field
            raise TypeError('Input must be a ' + str(cls.__name__) + '. Received ' + str(type(obj)) + ' instead.')

        output_field = cls.generate_id(shape=obj.shape)
        return output_field

    @classmethod
    def generate_from_matrix(cls, input_vol_ext, input_matrix):
        """
        Given a domain in matrix coordinates and a transformation matrix, it provides
        the corresponding vector fields
        :param input_vol_ext: domain of the vector field.
        :param input_matrix: in homogeneous coordinates.
        :param affine: additional affine transformation to move from affine to
        homogeneous coordinates. (coincides with the voxel2mm case)
        :return: vector field generated by a matrix.

        NOTE: Vectors in the vector field are (x,y,z,1) where the homogeneous coordinate
         is added with to_homogeneous method.

        Lot of improvements to be done here.
        """
        dim = len(input_vol_ext)

        if dim == 2:

            f = cls.generate_id(shape=list(input_vol_ext) + [1, 1, 2])

            f.to_homogeneous()

            # TODO: code to be vectorized!
            x_intervals, y_intervals = input_vol_ext
            for i in range(x_intervals):
                for j in range(y_intervals):
                    f.field[i, j, 0, 0, :] = input_matrix.dot(f.field[i, j, 0, 0, :])

            f.to_affine()

            return f

        elif dim == 3:

            # TODO: the rotation axis is perpendicular to the plane z=0.
            # Fine for our purposes. a generalized 3d rotation will require the class se3 to be implemented.

            f = cls.generate_zero(shape=list(input_vol_ext) + [1, 3])
            x_intervals, y_intervals, z_intervals = input_vol_ext

            # Create the slice at the ground of the domain (x,y,z) , z = 0, as a 2d rotation:
            base_slice = cls.generate_id(shape=list(input_vol_ext[:2]) + [1, 1, 2])
            base_slice.to_homogeneous()

            for i in range(x_intervals):
                for j in range(y_intervals):
                    base_slice.field[i, j, 0, 0, :] = input_matrix.dot(base_slice.field[i, j, 0, 0, :])

            # Copy the slice at the ground on all the others:
            for k in range(z_intervals):
                f.field[..., k, 0, :2] = base_slice.field[..., 0, 0, :2]

            return f

        else:
            raise TypeError("Dimensions allowed: 2 or 3")

    @classmethod
    def generate_from_projective_matrix_algebra(cls, input_vol_ext, input_h):
        """
        See formula to generate these type of field.
        :param input_vol_ext:
        :param input_h: projective matrix in an algebra
        :return:
        """
        d = len(input_vol_ext)
        np.testing.assert_array_equal(input_h.shape, [d+1, d+1])

        if d == 2:

            idd = cls.generate_id(shape=list(input_vol_ext) + [1, 1, d])
            vf = cls.generate_zero(shape=list(input_vol_ext) + [1, 1, d])

            idd.to_homogeneous()

            x_intervals, y_intervals = input_vol_ext
            for i in range(x_intervals):
                for j in range(y_intervals):
                    vf.field[i, j, 0, 0, 0] = input_h[0, :].dot(idd.field[i, j, 0, 0, :]) - i * input_h[2, :].dot(idd.field[i, j, 0, 0, :])
                    vf.field[i, j, 0, 0, 1] = input_h[1, :].dot(idd.field[i, j, 0, 0, :]) - j * input_h[2, :].dot(idd.field[i, j, 0, 0, :])

            return vf

        elif d == 3:

            idd = cls.generate_id(shape=list(input_vol_ext) + [1, d])
            vf = cls.generate_zero(shape=list(input_vol_ext) + [1, d])

            idd.to_homogeneous()

            x_intervals, y_intervals, z_intervals = input_vol_ext
            for i in range(x_intervals):
                for j in range(y_intervals):
                    for k in range(z_intervals):
                        vf.field[i, j, k, 0, 0] = input_h[0, :].dot(idd.field[i, j, k, 0, :]) - i * input_h[3, :].dot(idd.field[i, j, k, 0, :])
                        vf.field[i, j, k, 0, 1] = input_h[1, :].dot(idd.field[i, j, k, 0, :]) - j * input_h[3, :].dot(idd.field[i, j, k, 0, :])
                        vf.field[i, j, k, 0, 2] = input_h[2, :].dot(idd.field[i, j, k, 0, :]) - k * input_h[3, :].dot(idd.field[i, j, k, 0, :])

            return vf

        else:
            raise TypeError("Dimensions allowed: 2 or 3")

    @classmethod
    def generate_from_projective_matrix_group(cls, input_vol_ext, input_exp_h):
        """
        See formula to generate these type of field.
        :param input_vol_ext:
        :param input_exp_h: projective matrix in a group
        :return:
        """
        d = len(input_vol_ext)
        np.testing.assert_array_equal(input_exp_h.shape, [d+1, d+1])

        if d == 2:

            vf = cls.generate_zero(shape=list(input_vol_ext) + [1, 1, d])

            x_intervals, y_intervals = input_vol_ext
            for i in range(x_intervals):
                for j in range(y_intervals):

                    s = input_exp_h.dot(np.array([i, j, 1]))[:]
                    if abs(s[2]) > 1e-5:
                        # subtract the id to have the result in displacement coordinates
                        vf.field[i, j, 0, 0, :] = (s[0:2]/float(s[2])) - np.array([i, j])

            return vf

        elif d == 3:

            vf = cls.generate_id(shape=list(input_vol_ext) + [1, d])

            x_intervals, y_intervals, z_intervals = input_vol_ext
            for i in range(x_intervals):
                for j in range(y_intervals):
                    for k in range(z_intervals):

                        s = input_exp_h.dot(np.array([i, j, k, 1]))[:]
                        if abs(s[3]) > 1e-5:
                            vf.field[i, j, k, 0, :] = (s[0:3]/float(s[3])) - np.array([i, j, k])

            return vf

        else:
            raise TypeError("Dimensions allowed: 2 or 3")

    def one_point_interpolation(self, point, method='linear', as_float=True):
        """
        For the moment only 2d and in matrix coordinates.
        :param point: [x,y] in matrix coordinates
        :param method: interpolation method. See griddata documentation
        :return:
        """
        if not len(point) == 2:
            raise IOError("Input expected is a 2d point for a 2d field.")

        # rename for clarity
        x_p, y_p = point[0], point[1]

        # all of the grid point in 2 columns:
        points = np.array([[i*1.0, j*1.0] for i in range(self.vol_ext[0]) for j in range(self.vol_ext[1])])
        values_x = np.array([self.field[i, j, 0, 0, 0]
                             for i in range(self.vol_ext[0])
                             for j in range(self.vol_ext[1])]).T
        values_y = np.array([self.field[i, j, 0, 0, 1]
                             for i in range(self.vol_ext[0])
                             for j in range(self.vol_ext[1])]).T

        grid_x = griddata(points, values_x, (x_p, y_p), method=method)
        grid_y = griddata(points, values_y, (x_p, y_p), method=method)

        if as_float:
            v_at_point = (float(grid_x), float(grid_y))
        else:
            v_at_point = (grid_x, grid_y)
        return v_at_point

    def one_point_interpolation_rdf(self, point, epsilon=50, as_float=True):
        """
        For the moment only 2d and in matrix coordinates.
        Secondary method for the interpolation, with radial basis function:
        :param point: [x,y] in matrix coordinates
        :param epsilon: epsilon value of the Rdf, see Rdf documentation.
        :return:
        """
        if not len(point) == 2:
            raise IOError("Input expected is a 2d point for a 2d field.")

        # rename for clarity
        x_p, y_p = point[0], point[1]

        sh = self.shape
        x_grid, y_grid = np.mgrid[0:sh[0], 0:sh[1]]

        v_x = self.field[x_grid, y_grid, 0, 0, 0]
        v_y = self.field[x_grid, y_grid, 0, 0, 1]

        rbf_x_in = Rbf(x_grid, y_grid, v_x, epsilon=epsilon)
        rbf_y_in = Rbf(x_grid, y_grid, v_y, epsilon=epsilon)

        if as_float:
            v_at_point = (float(rbf_x_in(x_p, y_p)), float(rbf_y_in(x_p, y_p)))
        else:
            v_at_point = (rbf_x_in(x_p, y_p), rbf_y_in(x_p, y_p))
        return v_at_point
