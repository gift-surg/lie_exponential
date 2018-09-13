"""
Playing with vector fields and meshgrid.
Group of transformation se2.

SVF and its displacement in Lagrangian coordinates.
"""

import numpy as np
from numpy import cos, sin, pi

from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt
from transformations.s_vf import SVF

# Coordinate system

x_min = -10
x_max = 10
y_min = -10
y_max = 10
x_intervals = 21
y_intervals = 21

x = np.linspace(x_min, x_max, x_intervals)
y = np.linspace(y_min, y_max, y_intervals)
gx, gy = np.meshgrid(x, y)

gx, gy = gx.T, gy.T  # matrix to cartesian coordinate

# Transformation: rotation and translation

theta = - pi / 3
tx, ty = 2, 2

m  = np.array([[cos(theta), -sin(theta), tx],
               [sin(theta), cos(theta), ty],
               [0,          0,          1]])

# corresponding tangent vector of the transformation

c = cos(theta)
prec = abs(np.spacing(0.0))
if abs(c - 1.0) <= 10 * prec:
    dtx = tx
    dty = ty
    theta = 0
else:
    factor = (theta / 2.0) * sin(theta) / (1.0 - c)
    dtx = factor * tx + (theta / 2.0) * ty
    dty = factor * ty - (theta / 2.0) * tx

dm = np.array([[0,     -theta, dtx],
               [theta, 0,      dty],
               [0,     0,      0]])


# ---- create the plane with values from the meshgrid
# and the transformed plane through m and dm
# add the third projective coordinate:

plane = np.zeros(list(gx.shape) + [1, 1, 3])
print list(gx.shape) + [1, 1, 3]
plane[..., 0, 0, 0] = gx
plane[..., 0, 0, 1] = gy
plane[..., 0, 0, 2] = np.ones(gx.shape) # z coordinate = 1 as projective coordinate

dm_plane = np.zeros(plane.shape)
m_plane = np.zeros(plane.shape)

# Apply the transformation in matrix form:
for i in range(x_intervals):
    for j in range(y_intervals):
        dm_plane[i, j, 0, 0, :] = dm.dot(plane[i, j, 0, 0, :])
        m_plane[i, j, 0, 0, :] = m.dot(plane[i, j, 0, 0, :])


# move from numpy array to svf:
id_svf = SVF.from_array(plane)
m_svf  = SVF.from_array(m_plane)
dm_svf = SVF.from_array(dm_plane)

# do some terminal tests

print 'Origin of the plane:'
print plane[11, 11, 0, 0, :]

print plane.shape
print (plane[..., 0, 0, 0] == gx).all()
print (plane[..., 0, 0, 1] == gy).all()
print (plane[..., 0, 0, 2] == np.ones(gx.shape)).all()
print type(id_svf)


# --- Show must go on!

if 1:
    #plot in 2d, as quiver
    sub_interval = True

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.8, 10.8), dpi=80)
    plt.subplots_adjust(left=0.04, right=0.88, bottom=0.08, top=0.94)

    # Plot the id
    ax_id = axes.flat[0]
    im_id = ax_id.quiver(id_svf.field[..., 0, 0, 0],  id_svf.field[..., 0, 0, 1],
                         id_svf.field[..., 0, 0, 0] - id_svf.field[..., 0, 0, 0],
                         id_svf.field[..., 0, 0, 1] - id_svf.field[..., 0, 0, 1],
                         color='g', linewidths=0.2, units='xy',angles='xy',scale=1.5,scale_units='xy')
    if sub_interval:
        ax_id.set_xlim([-14, 14])
        ax_id.set_ylim([-14, 14])
    ax_id.set_title('identity')

    # Plot m
    ax_m = axes.flat[1]
    im_m = ax_m.quiver(id_svf.field[..., 0, 0, 0],  id_svf.field[..., 0, 0, 1],
                       m_svf.field[..., 0, 0, 0] - id_svf.field[..., 0, 0, 0],
                       m_svf.field[..., 0, 0, 1] - id_svf.field[..., 0, 0, 1],
                       color='b', linewidths=0.2, units='xy',angles='xy',scale=1,scale_units='xy')
    if sub_interval:
        ax_m.set_xlim([-4, 4])
        ax_m.set_ylim([-4, 4])
    ax_m.set_title('Vector field m(x) - x')

    ax_m.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_m.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_m.set_axisbelow(True)

    # Plot dm
    ax_dm = axes.flat[2]
    im_dm = ax_dm.quiver(id_svf.field[..., 0, 0, 0],  id_svf.field[..., 0, 0, 1],
                         dm_svf.field[..., 0, 0, 0], dm_svf.field[..., 0, 0, 1],
                         color='r',linewidths=0.2, units='xy',angles='xy',scale=1,scale_units='xy')
    if sub_interval:
        ax_dm.set_xlim([-4, 4])
        ax_dm.set_ylim([-4, 4])
    ax_dm.set_title('Vector field dm')

    ax_dm.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_dm.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_dm.set_axisbelow(True)

    # Plot m, dm
    ax_m_and_dm = axes.flat[3]

    ax_m_and_dm.quiver(id_svf.field[..., 0, 0, 0],  id_svf.field[..., 0, 0, 1],
                       m_svf.field[..., 0, 0, 0] - id_svf.field[..., 0, 0, 0],
                       m_svf.field[..., 0, 0, 1] - id_svf.field[..., 0, 0, 1],
                       color='b', linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')

    ax_m_and_dm.quiver(id_svf.field[..., 0, 0, 0],  id_svf.field[..., 0, 0, 1],
                       dm_svf.field[..., 0, 0, 0], dm_svf.field[..., 0, 0, 1],
                       color='r', linewidths=0.2, units='xy', angles='xy', scale=1, scale_units='xy')

    if sub_interval:
        ax_m_and_dm.set_xlim([-4, 4])
        ax_m_and_dm.set_ylim([-4, 4])
    ax_m_and_dm.set_title('Both, scaled')

    ax_m_and_dm.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_m_and_dm.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax_m_and_dm.set_axisbelow(True)

plt.show()


