from utils import log_composition as lcp

# Bossa algorithm to compute the logarithm:

import numpy as np
import transformations.se2_a as se2_a
import transformations.se2_g as se2_g


def log_algorithm_se2(p, composition='ground', toll=0.01):
    """
    p: must be a matrix in se2, otherwise it is converted into a matrix!
    log_computation using variance of the bossa algorithm.
    Do not work. Still don't know who is exp(u) - id

    It takes into account only matrices!
    Here a ground truth is known.
    """

    if isinstance(p, se2_g.se2_g):
        p = p.get_matrix
    elif not se2_g.is_a_matrix_in_se2_g(p, relax=True):
        raise TypeError('warning: wrong input data format, se2_a element expected.')

    v = se2_a.se2_a(0, 0, 0)
    max_iter = 10000
    cont = 0

    while abs(se2_a.exp(v).norm() - p.norm()) > toll and cont < max_iter:
        dv = (se2_a.exp(se2_a.scalarpr(-1, v)))*p - np.identity(3)  # This operation is obviously not defined. How to solve it?
        v = lcp.log_composition_matrix_se2_a(v, dv, kind=composition)
        cont += 1

        print se2_a.exp(v).norm() - p.norm()

    ans = v

    return ans


