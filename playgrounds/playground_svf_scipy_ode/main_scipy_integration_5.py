"""
Integral vector field from integral curves.
Structure of the bi-points shown in the previous file is here stored in a new vector field.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.integrate import ode

from transformations.s_vf import SVF
from visualizer.fields_comparisons import see_2_fields_separate_and_overlay


# ----- COMPUTE AND PLOT EXPONENTIAL OF SVF USING SCIPY, command line by line: --- #

# Initialize the random field with the function input:
svf_0  = SVF.generate_random_smooth(shape=(20, 20, 1, 1, 2))

# Initialize the displacement field that will be computed using the integral curves.
disp_0 =  SVF.generate_id_from_obj(svf_0)


# Vector field function, from the :
def vf(t, x):
    global svf_0
    return list(svf_0.one_point_interpolation(point=x, method='cubic'))


t0, tEnd = 0, 1
steps = 20.
dt = (tEnd - t0)/steps

r = ode(vf).set_integrator('dopri5', method='bdf', max_step=dt)


fig = plt.figure(num=1)
ax = fig.add_subplot(111)

# Plot vector field
id_field = svf_0.__class__.generate_id_from_obj(svf_0)

input_field_copy = copy.deepcopy(svf_0)

ax.quiver(id_field.field[..., 0, 0, 0],
           id_field.field[..., 0, 0, 1],
           input_field_copy.field[..., 0, 0, 0],
           input_field_copy.field[..., 0, 0, 1], color='r', alpha=0.9,
           linewidths=0.01, width=0.05, scale=1, scale_units='xy', units='xy', angles='xy')

print 'Beginning of the integral curves computations'


# Plot integral curves
passepartout = 4


for i in range(0 + passepartout, disp_0.vol_ext[0] - passepartout + 1):
    for j in range(0 + passepartout, disp_0.vol_ext[0] - passepartout + 1):  # cycle on the point of the grid.

        Y, T = [], []
        r.set_initial_value([i, j], t0).set_f_params()  # initial conditions are point on the grid
        while r.successful() and r.t + dt < tEnd:
            r.integrate(r.t+dt)
            Y.append(r.y)

        S = np.array(np.real(Y))

        disp_0.field[i, j, 0, 0, :] = S[S.shape[0]-1, :]

        ax.plot(S[:, 0], S[:, 1], color='b', lw=1)

        ax.plot(i, j, 'go', alpha=0.5)
        ax.plot(S[S.shape[0]-1, 0], S[S.shape[0]-1, 1], 'bo', alpha=0.5)
        if S.shape[0] < steps-2:  # the first step is not directly considered
            print "--------"
            print "Warning!"  # steps jumped for the point
            print "--------"

print 'End of the integral curves computations'

plt.xlim([0, 20])
plt.ylim([0, 20])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid()

see_2_fields_separate_and_overlay(disp_0, svf_0,
                                  fig_tag=2,
                                  title_input_0='disp',
                                  title_input_1='svf',
                                  title_input_both='overlay')


# ----- COMPUTE AND PLOT THE SAME: command from svf methods --- #
'''
# Initialize the random field with the function input:
#svf_1  = SVF.generate_random_smooth(shape=(20, 20, 1, 1, 2))

# Initialize the displacement field that will be computed using the integral curves.
disp_1 =  svf_0.exponential_scipy(verbose=True, passepartout=4)


see_2_fields_separate_and_overlay(disp_1, svf_0,
                                  fig_tag=3,
                                  title_input_0='disp',
                                  title_input_1='svf',
                                  title_input_both='overlay',
                                  window_title_input='embedded',
                                  subtract_id_0=True,
                                  subtract_id_1=False)

'''
plt.show()