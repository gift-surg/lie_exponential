"""
Comparing Cauchy and Flow problem for the same initial value problem.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from scipy.integrate import ode

from transformations.s_vf import SVF

from utils.path_manager import path_to_results_folder

from visualizer.fields_at_the_window import see_field


### Visualization methods ###

def compare_cauchy_flow_integrators(list_of_2_vect_fields,
                                    list_of_2_integral_curves,
                                     list_of_alpha_for_obj=(0.8, 0.8),
                                     alpha_for_integral_curves=0.4,
                                     window_title_input='quiver',
                                     titles=('Cauchy problem', 'Flow problem'),
                                     fig_tag=2, scale=1,
                                     subtract_id=(False, False),
                                     input_color=('r', 'r'),
                                     color_integral_curves='k',
                                     see_tips=False):

    fig = plt.figure(fig_tag, figsize=(10, 6), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    fig.canvas.set_window_title(window_title_input)

    for j in range(2):

        if not len(list_of_2_vect_fields[j].shape) == 5:
            raise TypeError('Wrong input size for a the field' + str(list_of_2_vect_fields[j]))

        id_field = list_of_2_vect_fields[j].__class__.generate_id_from_obj(list_of_2_vect_fields[j])
        input_field_copy = copy.deepcopy(list_of_2_vect_fields[j])

        if subtract_id[j]:
            input_field_copy -= id_field

        # add subplot here:
        ax  = fig.add_subplot(2, 1, j+1)

        ax.quiver(id_field.field[:, :, 0, 0, 0],
                  id_field.field[:, :, 0, 0, 1],
                  input_field_copy.field[:, :, 0, 0, 0],
                  input_field_copy.field[:, :, 0, 0, 1],
                  color=input_color[j], width=0.04, scale=scale,
                  scale_units='xy', units='xy', angles='xy',
                  alpha=list_of_alpha_for_obj[j])

        for ic in list_of_2_integral_curves[j]:
            ax.plot(ic[:, 0], ic[:, 1],
                    color=color_integral_curves, lw=1, alpha=alpha_for_integral_curves)
            if see_tips:
                ax.plot(ic[0, 0], ic[0, 1],
                        'go', alpha=0.3)
                ax.plot(ic[ic.shape[0]-1, 0],
                        ic[ic.shape[0]-1, 1],
                        'mo', alpha=0.5)

        ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.set_axisbelow(True)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(titles[j])

    fig.set_tight_layout(True)


###########################
### Path control panel: ###
###########################

fullpath = os.path.join(path_to_results_folder, 'figures')
filename_1 = os.path.join(fullpath, 'comparison_cauchy_flow_problem.png')

############################
### Values control panel ###
############################

shape = (25, 10, 1, 1, 2)    # domain of the svf

ic_c = [[8, 8]] #,[8, 7],[8, 8], [14, 4], [14, 3], [14, 2]]              # initial conditions for the Cauchy problem
t_init_c, t_end_c = -0.3, 50    # step size parameter for the Cauchy problem
dt_c = 0.1                   # time step interval for the Cauchy problem

passepartout = [2, 2]        # x an y dimension of the passepartout
t_f, t_end_f = 0, 1          # step size parameter for the Flow problem
dt_f = 0.01                  # time step interval for the Flow problem

see_field_only = True        # visualizer methods
show_final     = True

verbose = True

###########
## START ##
###########

### Generating function:

def f_vcon(t, x):
    # real eigenvalue both positive or both negative: stable node.
    t = float(t); x = [float(z) for z in x]
    sigma = 0.2
    I = 0.99
    w = 1.4
    alpha = 0.5
    tx, ty = -5, 0

    return list([alpha * (x[1] + tx), alpha * (-1 * sigma * x[1] + I + w * np.cos(x[0])  + ty)])


### Generate the vector field from the function and visualise it if required:

svf_1 = SVF.generate_zero(shape=shape)

for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            svf_1.field[i, j, 0, 0, :] = f_vcon(1, [i, j])


if see_field_only:
    see_field(svf_1, scale=1, input_color='r', fig_tag=1, annotate='', annotate_position=[-4, 1])

### initialize the problem with Scipy ###

r = ode(f_vcon).set_integrator('vode', method='bdf', max_step=dt_c)

### Solve the Cauchy problem: compute the integral curve passing through a point

print 'Beginning of the integral curves computations, Cauchy problem'

integral_curves_collector = []

for k in range(len(ic_c)):
    Y, T = [], []
    r.set_initial_value(ic_c[k], t_init_c).set_f_params()
    while r.successful() and r.t + dt_c < t_end_c:
        r.integrate(r.t+dt_c)
        Y.append(r.y)

    integral_curves_collector += [np.array(np.real(Y))]

print 'End of the integral curves computations, Cauchy problem'

### Solve the Flow problem: compute the integral at each point of the grid

print 'Beginning of the integral curves computations, Flow problem'

flows_collector = []

for i in range(0 + passepartout[0], shape[0] - passepartout[0] + 1):
    for j in range(0 + passepartout[1], shape[1] - passepartout[1] + 1):  # cycle on the point of the grid.

        y = []
        r.set_initial_value([i, j], 0).set_f_params()  # initial conditions are point on the grid
        while r.successful() and r.t + dt_f < t_end_f:
            r.integrate(r.t+dt_f)
            y.append(r.y)

        # flow of the svf at the point [i,j]
        fl = np.array(np.real(y))

        if verbose:
            print 'Integral curve at grid point ' + str([i, j]) + ' is computed.'

        # In some cases as critical points, or when too closed to the closure of the domain
        # the number of steps can be reduced by the algorithm.
        if fl.shape[0] < t_end_f - 2 and verbose:  # the first step is not directly considered
            print "--------"
            print "Warning!"  # steps jumped for the point
            print "--------"

        flows_collector += [fl]

print 'End of the integral curves computations, Flow problem'

if verbose:
    print len(integral_curves_collector)
    print len(integral_curves_collector[0])

    print len(flows_collector)
    print len(flows_collector[0]), len(flows_collector[1]), len(flows_collector[2])


if show_final:
    compare_cauchy_flow_integrators([svf_1, svf_1], [integral_curves_collector, flows_collector])
    plt.savefig(filename_1, format='png', dpi=400)  # dpi ignored if pdf
    plt.show()


