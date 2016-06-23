"""
Class test to see if and how the generative methods for projective svfs works
"""


import numpy as np
from sympy.core.cache import clear_cache

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

from utils.fields import Field
from utils.image import Image

from utils.projective_algebras import ProjectiveGroup, ProjectiveAlgebra
from transformations.s_vf import SVF
from transformations.s_disp import SDISP
from visualizer.fields_comparisons import see_overlay_of_n_fields


def test_invariance_under_linear_translations():

    random_seed = 5

    if random_seed > 0:
        np.random.seed(random_seed)

    domain = (10, 10)

    # Algebra
    # generate a 2 projective vector field up to a linear translation and verify that they are the same:
    h_a = ProjectiveAlgebra.randomgen(d=2)

    svf_h1 = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a.matrix)
    svf_h2 = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a.matrix - 17*np.eye(3))

    assert_array_almost_equal(svf_h1.field, svf_h2.field)

    # Group
    # generate a 2 projective vector field up to a scalar factor and verify that they are the same:
    h_g = h_a.exponentiate()

    # generate the corresponding disp
    disp_h1 = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g.matrix)
    disp_h2 = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=5*h_g.matrix)

    assert_array_almost_equal(disp_h1.field, disp_h2.field)

test_invariance_under_linear_translations()
