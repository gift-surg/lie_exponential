"""
Integral curve at each point of the grid from time 0 to time 1.
Numerical approximation of the 1-parameter subgroup.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.integrate import ode
from utils.fields import Field


# Initialize the field with the function input:
field_0 = Field.generate_random_smooth(shape=(20, 20, 1, 1, 2))


# Vector field function:
def vf(t, x):
    global field_0
    return list(field_0.one_point_interpolation(point=x, method='cubic'))

t0, tEnd, dt = 0, 20, 0.1
ic = [[i, j] for i in range(10, 11) for j in range(5, 16)]
colors = ['b'] * len(ic)

r = ode(vf).set_integrator('vode', method='bdf', max_step=dt)

fig = plt.figure(num=1)
ax = fig.add_subplot(111)

# Plot vector field
id_field = field_0.__class__.generate_id_from_obj(field_0)

input_field_copy = copy.deepcopy(field_0)

ax.quiver(id_field.field[..., 0, 0, 0],
           id_field.field[..., 0, 0, 1],
           input_field_copy.field[..., 0, 0, 0],
           input_field_copy.field[..., 0, 0, 1], color='r', alpha=0.8,
           linewidths=0.01, width=0.05, scale=1, scale_units='xy', units='xy', angles='xy', )

print 'Beginning of the integral curves computations'

# Plot integral curves
for k in range(len(ic)):
    Y, T = [], []
    r.set_initial_value(ic[k], t0).set_f_params()
    # first step dt = 0:
    #r.integrate(r.t)
    #Y.append(r.y)
    # subsequent steps:
    while r.successful() and r.t + dt < tEnd:
        r.integrate(r.t+dt)
        Y.append(r.y)

    S = np.array(np.real(Y))
    ax.plot(S[:, 0], S[:, 1], color=colors[k], lw=1)

print 'End of the integral curves computations'


plt.xlim([0, 20])
plt.ylim([0, 20])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid()
plt.show()











