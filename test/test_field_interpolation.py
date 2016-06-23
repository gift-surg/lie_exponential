import numpy as np
import matplotlib.pyplot as plt

import scipy as spy
from scipy.interpolate import Rbf, griddata
from scipy import integrate
from scipy.integrate import ode

from utils.fields import Field
from transformations.s_vf import SVF

from visualizer.fields_at_the_window import see_field
from visualizer.fields_comparisons import see_2_fields_separate_and_overlay, \
    see_overlay_of_n_fields, see_n_fields_special

# TODO: implement the test class for the one_point_interpolation method, for different functions where
# the ground truth is known.


# Function input:
def function_1(t, x):
    t = float(t); x = [float(y) for y in x]
    return 0.5*x[1], -0.5 * x[0] + 0.8 * x[1]

# Initialize the field with the function input:
field_0 = Field.generate_zero(shape=(20, 20, 1, 1, 2))

for i in range(0, 20):
        for j in range(0, 20):
            field_0.field[i, j, 0, 0, :] = function_1(1, [i, j])

x1 = 3.22
y1 = 2.2

print 'ground truth =  ' + str(function_1(1, [x1, y1]))
print
print 'values interpolated nearest = ' + str(field_0.one_point_interpolation(point=(x1, y1), method='nearest'))
print 'values interpolated linear  = ' + str(field_0.one_point_interpolation(point=(x1, y1), method='linear'))
print 'values interpolated cubic   = ' + str(field_0.one_point_interpolation(point=(x1, y1), method='cubic'))
print 'values interpolated rdf     = ' + str(field_0.one_point_interpolation_rdf(point=(x1, y1), epsilon=50))