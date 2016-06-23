"""
Interpolation of an SVF generated by a function and 3 of its integral curves for given initial points.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.integrate import ode
from utils.fields import Field


# Function input:
def function_1_bis(t, x):
    t = float(t); x = [float(y) for y in x]
    return list([0.5*x[1], -0.5 * x[0] + 0.8 * x[1]])


def function_1(t, x):
    # complex (conjugate) eigenvalue: spirals.
    t = float(t); x = [float(z) for z in x]
    a, b, c, d = 0.7, -3, 3, -0.4
    alpha = 0.1
    return alpha * (a*x[0] + b*x[1] + 30), alpha * (c*x[0] + d*x[1] - 15)

# Initialize the field with the function input:
field_0 = Field.generate_zero(shape=(20, 20, 1, 1, 2))

for i in range(0, 20):
        for j in range(0, 20):
            field_0.field[i, j, 0, 0, :] = function_1(1, [i, j])

x1 = 5.22
y1 = 1.2

print 'ground truth =  ' + str(function_1(1, [x1, y1]))
print
print 'values interpolated nearest = ' + str(field_0.one_point_interpolation(point=(x1, y1), method='nearest'))
print 'values interpolated linear  = ' + str(field_0.one_point_interpolation(point=(x1, y1), method='linear'))
print 'values interpolated cubic   = ' + str(field_0.one_point_interpolation(point=(x1, y1), method='cubic'))
print 'values interpolated rdf     = ' + str(field_0.one_point_interpolation_rdf(point=(x1, y1), epsilon=30))


# Vector field function:

def vf(t, x):
    global field_0
    return list(field_0.one_point_interpolation(point=x, method='cubic'))

t0, tEnd, dt = 0, 50, 0.5
ic = [[5, 7], [6, 8], [8, 8]]
colors = ['r', 'b', 'g']

r = ode(vf).set_integrator('vode', method='bdf', max_step=dt)

fig = plt.figure(num=1)
ax = fig.add_subplot(111)

# Plot integral curves
for k in range(len(ic)):

    Y, T = [], []
    r.set_initial_value(ic[k], t0).set_f_params()
    while r.successful() and r.t + dt < tEnd:
        r.integrate(r.t+dt)
        Y.append(r.y)

    S = np.array(np.real(Y))
    ax.plot(S[:, 0], S[:, 1], color=colors[k], lw=1.25)

# Plot vector field
id_field = field_0.__class__.generate_id_from_obj(field_0)

input_field_copy = copy.deepcopy(field_0)


ax.quiver(id_field.field[..., 0, 0, 0],
           id_field.field[..., 0, 0, 1],
           input_field_copy.field[..., 0, 0, 0],
           input_field_copy.field[..., 0, 0, 1],
           linewidths=0.01, width=0.03, scale=1, scale_units='xy', units='xy', angles='xy', )

plt.xlim([0, 20])
plt.ylim([0, 20])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

print 'finish!'
plt.show()








