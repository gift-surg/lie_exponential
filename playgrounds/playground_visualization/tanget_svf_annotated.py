"""
Integration from a curve perspective. Not from SVF but from the curve itself.
As tangent_svf_visualisation with annotations!
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


### Visualization methods ###


def plot_integral_curves(arrows_tails,
                         arrows_heads,
                         list_of_integral_curves,
                         list_of_alpha_for_obj=(1, 1, 1),
                         alpha_for_integral_curves=0.5,
                         window_title_input='Tangent fields',
                         titles='',
                         fig_tag=2, scale=1,
                         input_color=('r', 'r', 'r'),
                         color_integral_curves='k',
                         see_tips=False):

    fig = plt.figure(fig_tag, figsize=(6, 4), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    fig.canvas.set_window_title(window_title_input)

    num_initial_points = len(list_of_integral_curves)

    ax  = fig.add_subplot(111)

    for k in range(num_initial_points):

        ic = list_of_integral_curves[k]

        ax.quiver(arrows_tails[0, 0, k],
                  arrows_tails[0, 1, k],
                  arrows_heads[0, 0, k],
                  arrows_heads[0, 1, k],
                  color=input_color[k], width=0.04, scale=scale,
                  scale_units='xy', units='xy', angles='xy',
                  alpha=list_of_alpha_for_obj[k])

        ax.plot(ic[:, 0], ic[:, 1],
                color=color_integral_curves, lw=1, alpha=alpha_for_integral_curves)

        ax.plot(arrows_tails[:, 0, k], arrows_tails[:, 1, k], 'bo', ms=3)

        if see_tips:
            ax.plot(ic[0, 0], ic[0, 1],
                    'go', alpha=0.3)
            ax.plot(ic[ic.shape[0]-1, 0],
                    ic[ic.shape[0]-1, 1],
                    'mo', alpha=0.5)

    ax.set_title(titles)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)

    ax.set_xlim([-0.5, 11])
    ax.set_ylim([8, 12])

    ax.set_xlabel('')
    ax.set_ylabel('')

    fig.set_tight_layout(True)

    plt.setp( ax.get_xticklabels(), visible=False)
    plt.setp( ax.get_yticklabels(), visible=False)


if __name__ == '__main__':

    ############################
    ### Values control panel ###
    ############################

    # domain of the frame
    shape = (15, 10)

    # initial conditions for the Cauchy problem
    ic_c = [[0, 10]]

    num_initial_conditions = len(ic_c)

    # Curve length for the visualisation
    t_0, t_n = 0, 4

    # number of infinitesimal curve subdivisions and time step interval
    n = 102
    dt_n = (t_n - t_0)/float(n)

    # number of tangent vectors to be visualised and step interval
    N = 20
    dt_N = (t_n - t_0)/float(N)

    ### Generating function:
    def f_vcon(t, x):
        t = float(t)
        x = [float(z) for z in x]
        sigma = 0.2
        ii = 0.99
        w = 1.4
        alpha = 0.5
        tx, ty = -5, 0

        return list([alpha * (x[1] + tx), alpha * (-1 * sigma * x[1] + ii + w * np.cos(x[0])  + ty)])

    ### Initialize the problem with Scipy ###
    r = ode(f_vcon).set_integrator('vode', method='bdf', max_step=dt_n)

    print 'Beginning of the integral curves computations'

    integral_curves_collector = []

    euler = False

    # divide for scaling and squaring:
    steps_tangent_points = [0, 6, 12, 25, 50, 101]
    scaling_exponents = [32, 16, 8, 4, 2, 0]  # 1/2^(scaling_exponent)

    print steps_tangent_points

    ### Tangent vector field parameter
    tangent_points_coordinates  = np.zeros([len(steps_tangent_points), 2, num_initial_conditions])
    tangent_vectors_coordinates = np.zeros([len(steps_tangent_points), 2, num_initial_conditions])

    for k in range(num_initial_conditions):  # loops on initial conditions
        Y = []
        r.set_initial_value(ic_c[k], t_0).set_f_params()
        while r.successful() and r.t + dt_n < t_n:
            r.integrate(r.t + dt_n)
            Y.append(r.y)

        # list of array: one element of the list for each integral curve
        integral_curves_collector += [np.array(np.real(Y))]

        ### extract tangent points from integral curves collector:
        cont = 0
        for pt_index in steps_tangent_points:
            print pt_index
            tangent_points_coordinates[cont, :, k]  = integral_curves_collector[k][pt_index, :]
            tangent_vectors_coordinates[cont, :, k] = f_vcon(0, tangent_points_coordinates[cont, :, k])
            cont += 1

    print 'End of the integral curves computations'

    tangent_vectors_coordinates[0, :, 0] *= 4

    plot_integral_curves(tangent_points_coordinates,
                         tangent_vectors_coordinates,
                         integral_curves_collector)

    plt.show()

    if False:
        print len(integral_curves_collector)
        print len(integral_curves_collector[0])

        print integral_curves_collector
        print tangent_points_coordinates
        print tangent_vectors_coordinates
