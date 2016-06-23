import numpy as np
from utils.image import Image
from scipy import ndimage


class RegError(Exception):
    """
    Exception class override
    """

    def __init__(self, v):
        super(RegError, self).__init__(v)
        self.message = v

    def __str__(self):
        return repr(self.message)


def generate_identity_deformation(def_image, image=None):
    """
    Helper method to generate an identity transform
    :param def_image: The deformation field image.
    :param image: The image whose geometry we will use.
    If none, then the geometry of the deformation field is used
    """

    # Matrix to go from voxel to world space
    if image is not None:
        voxel_2_xyz = image.voxel_2_mm
        vol_ext = image.vol_ext
    else:
        voxel_2_xyz = def_image.voxel_2_mm
        vol_ext = def_image.vol_ext

    voxels = np.mgrid[[slice(i) for i in vol_ext]]
    voxels = [d.reshape(vol_ext, order='F') for d in voxels]
    mms = [voxel_2_xyz[i][3] + sum(voxel_2_xyz[i][k] * voxels[k]
                                   for k in range(len(voxels)))
           for i in range(len(voxel_2_xyz) - (4 - len(vol_ext)))]

    field = np.squeeze(def_image.field)
    mms = np.squeeze(mms)

    for i in range(field.shape[-1]):
        field[..., i] = mms[i]


def generate_random_smooth_deformation(volume_size,
                                       max_deformation=3,
                                       sigma=1):
    """
    Generate a random smooth deformation
    param: def_image: Deformation field image. It will be updated with the deformation.
    max_deformation_in_voxels: Maximum amount of deformation in voxels.
    The method ensures that the jacobian determinant of the deformation is positive.
    """
    if sigma <= 0:
        sigma = max_deformation/3

    if len(volume_size) > 3:
        volume_size = volume_size[0:3]

    dims = list()
    dims.extend(volume_size)
    while len(dims) < 4:
        dims.extend([1])
    dims.extend([len(volume_size)])

    # Inititalise with zero
    field = np.zeros(dims, dtype=np.float32)
    def_field = Image.from_array(field)
    generate_identity_deformation(def_field)
    # Generate a random displacement field
    displacement = max_deformation * 2 * \
        (np.random.random_sample(def_field.field.shape) - 0.5)
    # Smooth the displacement field
    for i in range(0, displacement.shape[4]):
        displacement[:,:,:,0,i] = ndimage.filters.gaussian_filter(displacement[:,:,:,0,i], sigma=sigma)

    disp_s = def_field.field.squeeze() + displacement.squeeze()
    done = False
    while not done:
        if len(volume_size) == 2:
            grad = np.gradient(disp_s[..., 0])
            x_x = grad[0]
            x_y = grad[1]
            grad = np.gradient(disp_s[..., 1])
            y_x = grad[0]
            y_y = grad[1]

            jac_det = x_x * y_y - x_y * y_x

            if np.min(jac_det) < 0.1:
                for i in range(0, displacement.shape[4]):
                    displacement[:,:,:,0,i] = ndimage.filters.gaussian_filter(displacement[:,:,:,0,i], sigma=sigma)
                disp_s = def_field.field.squeeze() + displacement.squeeze()
            else:
                done = True
        else:
            grad = np.gradient(disp_s[..., 0])
            x_x = grad[0]
            x_y = grad[1]
            x_z = grad[2]
            grad = np.gradient(disp_s[..., 1])
            y_x = grad[0]
            y_y = grad[1]
            y_z = grad[2]
            grad = np.gradient(disp_s[..., 2])
            z_x = grad[0]
            z_y = grad[1]
            z_z = grad[2]

            jac_det = x_x * (y_y*z_z - y_z*z_y) - \
                x_y * (y_x*z_z - y_z*z_x) + x_z * (y_x*z_y - y_y*z_x)

            if np.min(jac_det) < 0.1:
                for i in range(0, displacement.shape[4]):
                    displacement[:,:,:,0,i] = ndimage.filters.gaussian_filter(displacement[:,:,:,0,i], sigma=sigma)
                disp_s = def_field.field.squeeze() + displacement.squeeze()
            else:
                done = True

    def_field.field += displacement
    return def_field


def field_conversion_method(field_image, image=None,
                            get_position_field=True):

    array = np.zeros_like(field_image.field, dtype=np.float32)

    # Matrix to go from voxel to world space
    if image is not None:
        voxel_2_xyz = image.voxel_2_mm
        vol_ext = image.vol_ext
        im_field = Image.from_array(array)
    else:
        voxel_2_xyz = field_image.voxel_2_mm
        vol_ext = field_image.vol_ext
        im_field = Image.from_array(array)

    voxels = np.mgrid[[slice(i) for i in vol_ext]]
    voxels = [d.reshape(vol_ext, order='F') for d in voxels]
    mms = [voxel_2_xyz[i][3] + sum(voxel_2_xyz[i][k] * voxels[k]
                                   for k in range(len(voxels)))
           for i in range(len(voxel_2_xyz) - (4 - len(vol_ext)))]

    input_data = np.squeeze(field_image.field)
    field_array = np.squeeze(array)
    mms = np.squeeze(mms)
    if get_position_field:
        for i in range(array.shape[-1]):
            # Output is the deformation/position field
            field_array[..., i] = input_data[..., i] + mms[i]
    else:
        for i in range(array.shape[-1]):
            # Output is the displacement field
            field_array[..., i] = input_data[..., i] - mms[i]

    return im_field


def generate_displacement_from_deformation(pos_image, image=None, ):
    """
    Generate the displacement field image from position field image
    :rtype : Image
    :param image: Image whose geometry we will use.
    :param pos_image: The input position field image
    :return: Displacement field image
    """
    return field_conversion_method(pos_image, image, get_position_field=False)


def generate_position_from_displacement(disp_image, image=None):
    """
    Generate the position field image from displacement field image
    :rtype : Image
    :param image: Image whose geometry we will use.
    :param disp_image: The input displacement field image
    :return: Position field image
    """
    return field_conversion_method(disp_image, image)


def computation_jacobian_matrix_from_displacement(displacement_field):
    """
    Generate the jacobian matrix from a displacement field
    :param displacement_field: The input displacement field
    :return: The jacobian image matrix
    """

    input_shape = displacement_field.field.shape
    if len(input_shape) != 5:
        raise RuntimeError("The input does not seem to be a displacement field")

    position_field = generate_position_from_displacement(displacement_field)

    return compute_jacobian_field(position_field)


def read_affine_transformation(input_aff):
    """
    Read an affine transformation file.
    The function expects a 4x4 transformation matrix.
    :rtype : Numpy 4x4 matrix
    :param input_aff: File object or file path
    Returns a numpy affine transformation matrix.
    """

    if isinstance(input_aff, basestring):
        file_obj = open(input_aff, 'r')
        matrix = read_affine_transformation(file_obj)
        file_obj.close()
        return matrix

    elif isinstance(input_aff, file):
        file_content = input_aff.read().strip()
        file_content = file_content.replace('\r\n', ';')
        file_content = file_content.replace('\n', ';')
        file_content = file_content.replace('\r', ';')

        mat = np.matrix(file_content)
        if mat.shape != (4, 4):
            raise RegError('Input affine transformation '
                           'should be a 4x4 matrix.')
        return mat

    raise TypeError('Input must be a file object or a file name.')


def is_power2(num):
    """
    Check if a number is power of 2
    :param num: Input integer
    :rtype: Boolean
    Returns true if the input is a power of 2, else false
    """

    return num != 0 and ((num & (num - 1)) == 0)


def compute_variance(array):
    """
    Compute the variance of a numpy array. This can also be a masked array
    :param array: A numpy (masked) array
    :return: The computed variance
    """
    return np.ma.var(array)


def initialise_field(im, affine=None):
    """
    Create a field image from the specified target image.
    Sets the field to 0.
    Parameters:
    -----------
    :param im: The target image. Mandatory.
    :param affine: The initial affine transformation
    :return: Return the created field object
    """
    vol_ext = np.array(im.vol_ext)
    dims = list()
    dims.extend(vol_ext)
    while len(dims) < 4:
        dims.extend([1])
    num_dims = len(vol_ext[vol_ext>1])
    dims.extend([num_dims])

    # Inititalise with zero
    field = np.zeros(dims, dtype=np.float32)
    field = Image.from_field(field, im.get_header())

    # We have supplied an affine transformation
    if affine is not None:
        if affine.shape != (4, 4):
            raise RegError('Input affine transformation '
                           'should be a 4x4 matrix.')
        # The updated transformation
        transform = affine * im.voxel_2_mm
        field.update_transformation(transform)

    return field


def initialise_jacobian_field(im, affine=None):
    """
    Create a jacobian field image from the specified target image/field.
    Sets the field to 0.
    Parameters:
    -----------
    :param im: The target image/field. Mandatory.
    :param affine: The initial affine transformation
    :return: Return the created jacobian field object. Each jacobian is stored in a vector of size 9 in row major order
    """
    vol_ext = np.array(im.vol_ext)
    dims = list()
    dims.extend(vol_ext)
    while len(dims) < 4:
        dims.extend([1])
    num_dims = len(vol_ext[vol_ext>1])
    dims.extend([num_dims**2])

    # Inititalise with zero
    field = np.zeros(dims, dtype=np.float32)
    jacfield = Image.from_field(field, im.get_header())
    jacfield.set_matrix_field_attributes(num_dims, num_dims)

    # We have supplied an affine transformation
    if affine is not None:
        if affine.shape != (4, 4):
            raise RegError('Input affine transformation '
                           'should be a 4x4 matrix.')
        # The updated transformation
        transform = affine * im.voxel_2_mm
        jacfield.update_transformation(transform)

    return jacfield


def ndmesh(*xi, **kwargs):
    """
    n-dimensional mesh code stripped from numpy. This ensures meshgrid can
    be called from older versions of numpy which only supported 2-D meshgrid
    calls.
    """

    if len(xi) < 2:
        msg = 'meshgrid() takes 2 or more arguments (%d given)' % int(len(xi) > 0)
        raise ValueError(msg)

    args = np.atleast_1d(*xi)
    ndim = len(args)
    copy_ = kwargs.get('copy', True)

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1::]) for i, x in enumerate(args)]

    shape = [x.size for x in output]

    # Return the full N-D matrix (not only the 1-D vector)
    if copy_:
        mult_fact = np.ones(shape, dtype=int)
        return [x * mult_fact for x in output]
    else:
        return np.broadcast_arrays(*output)


def compute_spatial_gradient(image, derivative=None):
    """
    Compute the spatial gradient of the image using finite differences
    :param image: The image whose gradient we need to compute
    :param derivative: The derivative image. If it is none, it is allocated
    """

    if derivative is None:
        derivative = initialise_field(image)

    transform = image.voxel_2_mm
    dims = []
    for i in range(len(image.vol_ext)):
        dims.append(transform[i, i])

    grad = np.gradient(image.field, *dims)
    output_field = derivative.field.squeeze()
    for i in range(derivative.field.shape[-1]):
        output_field[..., i] = grad[i].squeeze()

    return derivative


def compute_jacobian_field(field, jacobian=None):
    """
    Compute the spatial gradient of the field using finite differences
    :param field: The field whose jacobian we need to compute
    :param jacobian: The jacobian image. If it is none, it is allocated. Each jacobian is stored in a vector of size 9 in row major order
    """

    if jacobian is None:
        jacobian = initialise_jacobian_field(field)

    transform = field.voxel_2_mm
    dims = []
    num_dims = sum(np.array(field.vol_ext)>1)
    for i in range(num_dims):
        dims.append(transform[i, i])

    output_field = jacobian.field.squeeze()
    for i in range(num_dims):
        grad = np.gradient(np.squeeze(field.field[...,i]), *dims)
        for j in range(num_dims):
            output_field[..., i*num_dims+j] = grad[j].squeeze()

    return jacobian


