from scipy import ndimage

from image import *
from helper_2 import generate_identity_deformation, \
    generate_displacement_from_deformation, \
    generate_position_from_displacement


class ImageResampler(object):
    """
    Class to resample image according to a deformation field
    """
    def __init__(self, order=1, prefilter=False):
        """
        Linear interpolation by default.
        """
        self.order = order
        self.prefilter = prefilter

    def resample(self, source, deformation, warped):
        """
        Resample the source image in the space of target.
        Parameters:
        :param source: The source image.
        :param deformation: The deformation field
        :param warped: Warped image. Should be allocated in the space of target
        and must have the same number of time points as the source image.
        """

        assert source.time_points == warped.time_points

        # Matrix to go from world to voxel space
        ijk_2_voxel = source.mm_2_voxel
        vol_ext = warped.vol_ext

        def_data = [deformation.field[..., i].reshape(vol_ext, order='F')
                    for i in range(deformation.shape[-1])]

        # Create the sampling grid in which the source image will be evaluated.
        def_data = [ijk_2_voxel[i][3] + sum(ijk_2_voxel[i][k] * def_data[k]
                    for k in range(len(def_data)))
                    for i in range(len(vol_ext))]

        if source.time_points == 1:
            ndimage.map_coordinates(source.field, np.asarray(def_data),
                                    warped.field, order=self.order,
                                    prefilter=self.prefilter)
        else:
            for i in range(source.time_points):
                ndimage.map_coordinates(source.field[i], def_data,
                                        warped.field[i], order=self.order,
                                        prefilter=self.prefilter)


class NearestNeighbourResampler(ImageResampler):
    """
    Set to nearest neighbourhood resampling
    """
    def __init__(self):
        super(NearestNeighbourResampler, self).__init__(order=0)


class CubicSplineResampler(ImageResampler):
    """
    Set to cubic spline resampling
    """
    def __init__(self):
        super(CubicSplineResampler, self).__init__(order=2, prefilter=True)


class PositionFieldComposer(object):
    """
    Compose position fields using linear interpolation.
    """
    def __init__(self):
        self.order = 1

    def set_order(self, order):
        if order > 0:
            self.order = order
        else:
            self.order = order

    def compose(self, left, right):
        """
        Compose position fields.
        Parameters:
        -----------
        :param left: Outer field.
        :param right: Inner field
        Order of composition: left(right(x))
        :return The composed position field
        """
        left_displ = generate_displacement_from_deformation(left)

        disp_composer = DisplacementFieldComposer()
        disp_composer.order = self.order
        result = disp_composer.compose_with_position_field(left_displ, right)
        result.field += right.field
        return result


class DisplacementFieldComposer(object):
    """
    Compose displacement fields using linear interpolation.
    """
    def __init__(self):
        self.order = 1

    def compose(self, left, right):
        """
        Compose displacement fields.
        Parameters:
        -----------
        :param left: Outer displacement field.
        :param right: Inner displacement field.
        Order of composition: (Id+left)(Id+right)(x)
        :return Return the composed displacement field
        """
        right_pos = generate_position_from_displacement(right)

        result = self.compose_with_position_field(left, right_pos)
        result.field += right.field
        return result

    def compose_with_position_field(self, left, right_pos):
        """
        Compose displacement fields.
        Parameters:
        -----------
        :param left: Outer displacement field.
        :param right_pos: Inner position field.
        Order of composition: left(right_pos(x))
        :return Return the composed displacement field
        """
        d = np.zeros(right_pos.field.shape)
        result = Image.from_array_with_header(d, header=None)
        vol_ext = right_pos.vol_ext[:right_pos.field.shape[-1]]
        right_field = [right_pos.field[..., i].reshape(vol_ext, order='F')
                       for i in range(right_pos.field.shape[-1])]

        field = np.squeeze(result.field)

        for i in range(field.shape[-1]):
            ndimage.map_coordinates(np.squeeze(left.field[..., i]),
                                    right_field,
                                    field[..., i],
                                    mode='nearest',
                                    order=self.order, prefilter=True)
        return result
