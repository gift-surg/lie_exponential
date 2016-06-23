import numpy as np
from numpy.testing import assert_array_equal
import copy
import matplotlib.pyplot as plt

from transformations.s_vf import SVF
from transformations.s_disp import SDISP

from visualizer.fields_at_the_window import see_2_fields, see_field


shape = (15, 15, 1, 1, 2)

sigma_init = 4
sigma_gaussian_filter = 2

svf_0   = SVF.generate_random_smooth(shape=shape,
                                         sigma=sigma_init,
                                         sigma_gaussian_filter=sigma_gaussian_filter)


sdisp_0 = svf_0.exponential(algorithm='ss')
sdisp_0_inv = (-1*svf_0).exponential(algorithm='ss')


comp_0 = SDISP.composition(1*svf_0, -1*svf_0)
comp_1 = SDISP.composition(-1*svf_0, +1*svf_0)

comp_sdisp_0_inv = SDISP.composition(sdisp_0, sdisp_0_inv)
comp_sdisp_inv_0 = SDISP.composition(sdisp_0_inv, sdisp_0)

id_field = SDISP.generate_zero(shape)

np.testing.assert_array_almost_equal(id_field.field, comp_sdisp_0_inv.field, decimal=0)

# sfv_0 is provided in Lagrangian coordinates!

see_field(svf_0, fig_tag=1, input_color='b')
see_field(-1*svf_0, fig_tag=1, input_color='r')
see_field(comp_0, fig_tag=1, input_color='g', title_input='v composed with -v (positive blue, negative red, composition green)')

see_field(svf_0, fig_tag=2, input_color='b')
see_field(-1*svf_0, fig_tag=2, input_color='r')
see_field(comp_1, fig_tag=2, input_color='g', title_input='-v composed with v (positive blue, negative red, composition green)')

# displacement

see_field(sdisp_0, fig_tag=3, input_color='b')
see_field(sdisp_0_inv, fig_tag=3, input_color='r')
see_field(comp_sdisp_0_inv, fig_tag=3, input_color='g', title_input='exp(v) composed with exp(-v) (positive blue, negative red, composition green)')


see_field(sdisp_0, fig_tag=4, input_color='b')
see_field(sdisp_0_inv, fig_tag=4, input_color='r')
see_field(comp_sdisp_inv_0, fig_tag=4, input_color='g', title_input='exp(-v) composed with exp(v) (positive blue, negative red, composition green)')


plt.show()

