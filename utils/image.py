import numpy as np
import nibabel as nib
from numpy import linalg as la
from os import path
from scipy import ndimage
import warnings

from utils.fields import Field


class Image(Field):
    """
    A wrapper around the nibabel image implementation, based on the class field.
    """

    def __init__(self, nifti_image):
        """
        Set attributes of the Image and of the Field related.
        Here an image is a data set with a matrix an header and an affine translation.
        The matrix is compatible with the data structure Field, father of Image.
        """

        super(Image, self).__init__(nifti_image.get_data())  # Use python Image to import images!
        self.nib_image = nifti_image

        [self.voxel_2_mm, code] = self.nib_image.get_sform(True)
        if code <= 0:
            self.voxel_2_mm = self.nib_image.get_qform()

        self.mm_2_voxel = la.inv(self.voxel_2_mm)
        self.zooms = self.nib_image.get_header().get_zooms()

        # Matrix image data attributes:
        self.is_matrix_data = False
        self.num_matrix_rows = 0
        self.num_matrix_cols = 0

        self.load_matrix_data_attributes()

    def load_matrix_data_attributes(self):
        """
        Load the matrix data attributes
        """
        (code, params, name) = self.nib_image.get_header().get_intent('code')
        if code == 1004 and len(params) >= 2:
            self.num_matrix_rows = params[0]
            self.num_matrix_cols = params[1]
            self.is_matrix_data = True

    # Set the intent flags which should tell us that the image
    # is a matrix image. The matrix is always stored in row major
    # format (according to NIFTI standards). The intent_1 and
    # intent_2 fields tell us the dimensions
    def set_matrix_data_attributes(self, rows, cols):
        # Set the matrix rows and column sizes
        self.num_matrix_rows = rows
        self.num_matrix_cols = cols
        self.is_matrix_data = True
        self.nib_image.get_header().set_intent(1004,
                                               (self.num_matrix_rows, self.num_matrix_cols),
                                               name='Matrix')

    def update_field(self):
        """
        In nibabel, the loaded field are in the cache.
        Any modification of the self.field does not change the image but only the data loaded in the cache.
        To make modifications on the data of the object, run this function.
        """
        data = self.nib_image.get_data()
        data[...] = self.field

    def update_affine(self):
        """
        In nibabel, the loaded field are in the cache.
        Any modification of the self.field does not change the image but only the data loaded in the cache.
        To make modifications on the data of the object, run this function.
        """
        affine = self.nib_image.affine()
        affine[...] = self.affine

    def update_transformation(self, mat):
        """
        Update the transformation matrix
        :param mat: The new transformation matrix
        """
        self.voxel_2_mm = mat
        self.mm_2_voxel = la.inv(self.voxel_2_mm)
        self.nib_image.get_header().set_sform(mat, 1)

    ### Vector space operations: ###
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

        if not len(self.shape) == len(other.shape):
            raise TypeError('unsupported operand type(s) for +: Images of different sizes.')

        if not np.prod([self.shape[j] == other.shape[j] for j in range(len(self.shape))]):
            raise TypeError('unsupported operand type(s) for +: Images of different sizes.')

        if not (self.voxel_2_mm == other.voxel_2_mm).all():
            raise TypeError('unsupported operand type(s) for +: Images with different positions in the space.')

        return Image.from_array_with_header(self.field + other.field,
                                            self.nib_image.get_header(),
                                            affine=self.voxel_2_mm)

    def __sub__(self, other, warn=False):
        """
        Inner sum of the vector field structure
        :param self:
        :param other:
        :return: subtraction of images
        """
        if warn:
            if not isinstance(other, self.__class__):
                warnings.warn('Warning: you are subtracting .' + self.__class__.__name__ + 'with ' + other.__class__.__name__)

        if not len(self.shape) == len(other.shape):
            raise TypeError('unsupported operand type(s) for -: Images of different sizes.')

        if not np.prod([self.shape[j] == other.shape[j] for j in range(len(self.shape))]):
            raise TypeError('unsupported operand type(s) for -: Images of different sizes.')

        if not (self.voxel_2_mm == other.voxel_2_mm).all():
            raise TypeError('unsupported operand type(s) for -: Images with different positions in the space.')

        return Image.from_array_with_header(self.field - other.field,
                                            self.nib_image.get_header(),
                                            affine=self.voxel_2_mm)

    def __rmul__(self, alpha):
        """
        operation of scalar multiplication
        :param alpha:
        :return:
        """
        return Image.from_array_with_header(alpha * self.field,
                                            self.nib_image.get_header(),
                                            affine=self.voxel_2_mm)

    ### Composition and resampling methods OLD ###

    @classmethod
    def field_conversion_method(cls, img, get_deformation_field=True):
        """
        Return deformation from displacement field if get_position_field=True.
        Return displacement from deformation field if get_position_field=False.
        :param img:
        :param get_deformation_field:
        :return:
        """

        data = np.zeros_like(img.field, dtype=np.float32)

        # Matrix to go from voxel to world space
        voxel_2_xyz = img.voxel_2_mm
        vol_ext = img.vol_ext

        ans_img = cls.from_array_with_header(data, header=img.nib_image.get_header())

        voxels = np.mgrid[[slice(i) for i in vol_ext]]
        voxels = [d.reshape(vol_ext, order='F') for d in voxels]
        mms = [voxel_2_xyz[i][3] + sum(voxel_2_xyz[i][k] * voxels[k]
                                       for k in range(len(voxels)))
               for i in range(len(voxel_2_xyz) - (4 - len(vol_ext)))]

        input_data = np.squeeze(img.field)
        field_data = np.squeeze(data)
        mms = np.squeeze(mms)
        if get_deformation_field:
            for i in range(data.shape[-1]):
                # Output is the deformation/position field
                field_data[..., i] = input_data[..., i] + mms[i]
        else:
            for i in range(data.shape[-1]):
                # Output is the displacement field
                field_data[..., i] = input_data[..., i] - mms[i]

        return ans_img

    @classmethod
    def deformation_from_displacement(cls, img):
        """
        Return deformation from displacement field if get_position_field=True.
        As field_conversion_method with get_position_field=True
        :param img:
        :return:
        """
        return cls.field_conversion_method(img, get_deformation_field=True)

    @classmethod
    def displacement_from_deformation(cls, img):
        """
        Return deformation from displacement field if get_position_field=True.
        As field_conversion_method with get_position_field=True
        :param img:
        :return:
        """
        return cls.field_conversion_method(img, get_deformation_field=False)

    @classmethod
    def compose_with_deformation_field(cls, left, right_def, s_i_o=3):
        """
        Mind the nomenclature:
        Position = deformation (2 names for the same things)
        displacement(x) = deformation(x) - identity(x).
        Composition between vector fields. Based on DisplacementFieldComposer()
        :param cls: s_disp object.
        :param s_i_o: spline interpolation order
        :return: external composition with displacement fields.
        """
        d = np.zeros(right_def.field.shape)
        result = cls.from_array_with_header(d, header=right_def.nib_image.get_header())
        vol_ext = right_def.vol_ext[:right_def.field.shape[-1]]
        right_field = [right_def.field[..., i].reshape(vol_ext, order='F')
                       for i in range(right_def.field.shape[-1])]

        field = np.squeeze(result.field)

        for i in range(field.shape[-1]):
            ndimage.map_coordinates(np.squeeze(left.field[..., i]),
                                    right_field,
                                    field[..., i],
                                    mode='nearest',
                                    order=s_i_o,
                                    prefilter=True)
        return result

    @classmethod
    def compose_with_displacement_field(cls, left, right_disp, s_i_o=3):
        """

        Mind the nomenclature:
        Position = deformation (2 names for the same things)
        displacement(x) = deformation(x) - identity(x).
        Composition between vector fields. Based on DisplacementFieldComposer()
        :param cls: s_disp object.
        :param s_i_o: spline interpolation order
        :return: external composition with displacement fields.
        """
        d = np.zeros(right_disp.field.shape)
        result = cls.from_array_with_header(d, header=right_disp.nib_image.get_header())
        vol_ext = right_disp.vol_ext[:right_disp.field.shape[-1]]
        right_field = [right_disp.field[..., i].reshape(vol_ext, order='F')
                       for i in range(right_disp.field.shape[-1])]

        field = np.squeeze(result.field)

        for i in range(field.shape[-1]):
            ndimage.map_coordinates(np.squeeze(left.field[..., i]),
                                    right_field,
                                    field[..., i],
                                    mode='nearest',
                                    order=s_i_o, prefilter=True)

        return result

    @classmethod
    def composition(cls, left, right, s_i_o=3):
        """
        :param left:
        :param right:
        :param s_i_o: spline interpolation order
        :return:
        """
        right_disp = cls.field_conversion_method(right)
        result = cls.compose_with_displacement_field(left, right_disp, s_i_o=s_i_o)
        result.field += right.field
        return result

    ### Image manager methods: ###

    def save(self, filename):
        """
        Save the file
        :param filename: Full path and filename for the saved file
        """
        self.update_field()
        name = path.expanduser(filename)
        nib.save(self.nib_image, name)

    @classmethod
    def from_array_with_header(cls, field, header, affine=np.eye(4)):
        """
        Create Image (or children) from data and header.
        Header is directly inserted by the user.
        :param field: The field to generate the image.
        :param header: The image header type Nifty1Header.
        """
        image = nib.Nifti1Image(field, header=header, affine=affine)
        return cls(image)

    @classmethod
    def from_array(cls, array, affine=np.eye(4), homogeneous=False):
        """
        Create image (or children) from array. Header is the nibabel default.
        :param array: The array to generate the image.
        """
        if not isinstance(affine, np.ndarray):
            raise TypeError('affine transformation must be np.ndarray')
        if not affine.shape[0] == affine.shape[1] == 4:
            raise TypeError('affine transformation must be 4 dimensional')
        image = nib.Nifti1Image(array, affine=affine)
        return cls(image)

    @classmethod
    def from_file(cls, imagepath):
        """
        Create object from image file
        :param imagepath: The path to the image file
        """
        image = nib.load(imagepath)
        return cls(image)

    @classmethod
    def from_field(cls, field, affine=np.eye(4), header=None):
        """
        Create object from field
        :param field: input field from which we want to compute the image.
        """
        if header is None:
            image = nib.Nifti1Image(field, affine=affine)
        else:
            image = nib.Nifti1Image(field, affine=affine, header=header)
        return cls(image)

    ### Normed space methods: ###

    def norm(self, passe_partout_size=1, normalized=True):
        """
        This returns the L2-norm of the discretised image.
        Based on the norm function from numpy.linalg of ord=2 for the vectorized matrix.
        The result can be computed with a passe partout (the discrete domain is reduced on each side)
        and can be normalized with the size of the domain.

        -> F vector field from a compact \Omega to R^d
        \norm{F} = (\frac{1}{|\Omega|}\int_{\Omega}|F(x)|^{2}dx)^{1/2}
        Discretization:
        \Delta\norm{F} = \frac{1}{\sqrt{dim(x)dim(y)dim(z)}}\sum_{v \in \Delta\Omega}|v|^{2})^{1/2}
                       = \frac{1}{\sqrt{XYZ}}\sum_{i,j,k}^{ X,Y,Z}|a_{i,j,k}|^{2})^{1/2}

        -> f scalar field from \Omega to R, f is an element of the L^s space
        \norm{f} = (\frac{1}{|\Omega|}\int_{\Omega}f(x)^{2}dx)^{1/2}
        Discretization:
        \Delta\norm{F} = \frac{1}{\sqrt{XYZ}}\sum_{i,j,k}^{ X,Y,Z} a_{i,j,k}^{2})^{1/2}

        Parameters:
        ------------
        :param passe_partout_size: size of the passe partout (rectangular mask, with constant offset on each side).
        :param normalized: if the result is divided by the normalization constant.
        """
        if passe_partout_size > 0:
            if self.dim == 2:
                masked_im = self.field[passe_partout_size:-passe_partout_size,
                                        passe_partout_size:-passe_partout_size,
                                        ...]
            else:
                masked_im = self.field[passe_partout_size:-passe_partout_size,
                                        passe_partout_size:-passe_partout_size,
                                        passe_partout_size:-passe_partout_size,
                                        ...]
        else:
            masked_im = self.field[...]

        if normalized:
            # shape of the field after masking:
            mask_shape = \
                (np.array(self.field.shape[0:self.dim])
                 - np.array([2 * passe_partout_size] * self.dim)).clip(min=1)

            return np.linalg.norm(masked_im.ravel(), ord=2) / np.sqrt(np.prod(mask_shape))
        else:
            return np.linalg.norm(masked_im.ravel(), ord=2)

    @staticmethod
    def norm_of_difference_of_images(im_a, im_b, passe_partout_size=1, normalized=False):
        """
        Norm of the difference of two images.
        :param im_a:
        :param im_b:
         :param passe_partout_size: size of the passe partout (rectangular mask, with constant offset on each side).
        :param normalized: if the result is divided by the normalization constant.
        """
        return (im_a - im_b).norm(passe_partout_size=passe_partout_size, normalized=normalized)

    ### Similarity measures here ###

    # TODO: add similarity measures for images here.

    ### Jacobian computation methods ###

    @staticmethod
    def norm_of_difference_of_jac(im_a, im_b, passe_partout_size=2):
        jac_im_a = im_a.compute_jacobian()
        jac_im_b = im_b.compute_jacobian()
        error = Image.norm_of_difference_of_fields(jac_im_a, jac_im_b, passe_partout_size=passe_partout_size)
        return error

    ### Generative methods ### see comments in class Field.

    @classmethod
    def generate_zero(cls, shape, affine=np.eye(4), header=None):
        input_field = Field.generate_zero(shape)
        if header is None:
            return cls(nib.Nifti1Image(input_field.field, affine=affine))
        else:
            return cls(nib.Nifti1Image(input_field.field, affine=affine, header=header))

    @classmethod
    def generate_id(cls, shape, affine=np.eye(4), header=None):
        """
        Generate id of an image. At the moment it does not take into account the affine matrix.
        To upgrade do it with Pankaj code in change_coordinate_system.
        :param shape:
        :param affine:
        :param header:
        :return:
        """
        # TODO: refactor with Pankaj code! See how the identity is created in change_coordinate_system
        # use this generate_id for images to change coordinate systems afterwards.
        input_field = Field.generate_id(shape)
        if header is None:
            return cls(nib.Nifti1Image(input_field.field, affine=affine))
        else:
            return cls(nib.Nifti1Image(input_field.field, affine=affine, header=header))

    @classmethod
    def generate_random_smooth(cls, shape, mean=0, sigma=5,
                               sigma_gaussian_filter=2,
                               affine=np.eye(4), header=None):

        input_field = super(Image, cls).generate_random_smooth(shape, mean=mean, sigma=sigma,
                                                               sigma_gaussian_filter=sigma_gaussian_filter)

        if header is None:
            return cls(nib.Nifti1Image(input_field.field, affine=affine))
        else:
            return cls(nib.Nifti1Image(input_field.field, affine=affine, header=header))

    @classmethod
    def generate_from_matrix(cls, input_vol_ext, input_matrix, affine=np.eye(4), header=None):

        input_field = Field.generate_from_matrix(input_vol_ext=input_vol_ext,
                                                 input_matrix=input_matrix)

        if header is None:
            return cls(nib.Nifti1Image(input_field.field, affine=affine))
        else:
            return cls(nib.Nifti1Image(input_field.field, affine=affine, header=header))

    @classmethod
    def generate_from_projective_matrix_algebra(cls, input_vol_ext, input_h, affine=np.eye(4), header=None):

        input_field = Field.generate_from_projective_matrix_algebra(input_vol_ext=input_vol_ext, input_h=input_h)

        if header is None:
            return cls(nib.Nifti1Image(input_field.field, affine=affine))
        else:
            return cls(nib.Nifti1Image(input_field.field, affine=affine, header=header))

    @classmethod
    def generate_from_projective_matrix_group(cls, input_vol_ext, input_exp_h, affine=np.eye(4), header=None):

        input_field = Field.generate_from_projective_matrix_group(input_vol_ext=input_vol_ext, input_exp_h=input_exp_h)

        if header is None:
            return cls(nib.Nifti1Image(input_field.field, affine=affine))
        else:
            return cls(nib.Nifti1Image(input_field.field, affine=affine, header=header))
