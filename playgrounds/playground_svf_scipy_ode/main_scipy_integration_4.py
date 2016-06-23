"""
Integral vector field from integral curves.
Emphasis on the initial and on the final point of the integral curve in transparency.
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

t0, tEnd = 0, 1
steps = 20.
dt = (tEnd - t0)/steps
ic = [[i, j] for i in range(2, 18) for j in range(2, 18)]
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
           input_field_copy.field[..., 0, 0, 1], color='r', alpha=0.9,
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
    #print 'final point of ' + str(ic[k]) + ':'
    #print S[steps-2, 0], S[steps-2, 1]  # -1 because it omits the first step, -2 because it starts from zero
    #print ic[k]
    #print 'size of S : ' + str(S.shape[0])
    #print 'content : ' + str([steps-2, 0])
    # ic and S are the searched bi-points.
    ax.plot(ic[k][0], ic[k][1], 'go', alpha=0.5)
    ax.plot(S[S.shape[0]-1, 0], S[S.shape[0]-1, 1], 'bo', alpha=0.5)
    if S.shape[0] < steps-2:
        print "Warning!"  # steps jumped for the point
        print "--------"

print 'End of the integral curves computations'


plt.xlim([0, 20])
plt.ylim([0, 20])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid()
plt.show()












