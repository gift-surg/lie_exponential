
import numpy as np
import matplotlib.pyplot as plt

from transformations.s_vf import SVF
from transformations.s_disp import SDISP
from utils.image import Image


def f_v(t, x):

    """
    A 2-dimensional stationary velocity field.
    :return:
    """
    t = float(t); x = [float(y) for y in x]
    return np.array([x[0] - 0.5 * x[1], x[1] - 0.5 * x[0]])


def f_phi(t, x):
    """
    Integral of f_v.
    Solution of the original ordinary differential equation.

    :return:
    """
    t = float(t); x = [float(y) for y in x]
    return np.array([t * x[0], t * x[1]])


array_v     = np.zeros([15, 15, 1, 1, 2])
array_f_phi = np.zeros([15, 15, 1, 1, 2])
for i in range(15):
    for j in range(15):
        array_v[i, j, 0, 0, :]     = f_v(1, [i, j])
        array_f_phi[i, j, 0, 0, :] = f_phi(1, [i, j])


svf_0   = SVF.from_array(array_v)
sdisp_0 = SDISP.from_array(array_f_phi)



