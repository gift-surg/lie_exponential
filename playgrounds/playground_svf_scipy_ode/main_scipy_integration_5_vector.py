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

r = ode(vf).set_integrator('vode', method='bdf', max_step=dt)


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

# Attempt to vectorize the previous module:

Y = []
# initial conditions are point on the grid
r.set_initial_value(np.array([[5, 5], [5, 6], [5, 7], [6, 5], [6, 6], [6, 7]]), t0).set_f_params()
while r.successful() and r.t + dt < tEnd:
    # Somehow believe that this can not be vectorized!
    # The vectorization can be done in other situation: ask on stack overflow!
    print 'spam11'
    print r.successful()
    print r.t + dt < tEnd
    print 'spam22'
    r.integrate(r.t+dt)
    Y.append(r.y)

S = np.array(np.real(Y))
# shape of S is (time, number initial point, 2 are coordinates)

print S

print S.shape
print S[S.shape[0]-1, :].shape
print disp_0.field[5:7, 5:8, 0, 0, :].shape
print 'spam'

cont = 0
for i in [5, 6]:
    for j in [5, 6, 7]:
        print S[0, cont, :]
        disp_0.field[i, j, 0, 0, :] = S[0, cont, :]
        cont += 1

'''
ax.plot(S[:, 0], S[:, 1], color='b', lw=1)

ax.plot([5, 6, 7], [5, 6, 7], 'go', alpha=0.5)
ax.plot(S[S.shape[0]-1, 0], S[S.shape[0]-1, 1], 'bo', alpha=0.5)
'''
if S.shape[0] < steps-2:  # the first step is not directly considered
    print "--------"
    print "Warning!"  # steps jumped for the point
    print "--------"

print 'End of the integral curves computations'
'''
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid()
'''


see_2_fields_separate_and_overlay(disp_0, svf_0,
                                  fig_tag=2,
                                  title_input_0='disp',
                                  title_input_1='svf',
                                  title_input_both='overlay')


plt.show()