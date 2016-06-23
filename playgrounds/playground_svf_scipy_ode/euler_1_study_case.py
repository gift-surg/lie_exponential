"""
Pedagogical explanation of the Euler method.
"""

import numpy as np
import matplotlib.pyplot as plt


def euler_met_1(f, xa, xb, ya, n, verbose=False, y_ground=None, return_all=True):
    """
    First Order ODE (y' = f(x, y)) Solver using Euler method
    :param f: input function
    :param xa: left extreme of the value of independent variable
    :param xb: left extreme of value of independent variable
    :param ya: initial value of dependent variable
    :param n : number of steps
    :param verbose :
    :param y_ground :
    :param return_all : for debug purposes

    :return : value of y at xb.

    ans has three or five column:
    step, x, w, [ground, truncating error]

    Simplified version:

      h = (xb - xa) / float(n)
      x = xa
      w = ya
      for i in range(n):
          w += h * f(x, w)
          x += h
      return y

    """
    h = (xb - xa) / float(n)
    x = xa
    w = ya

    if return_all:
        if y_ground is None:
            ans = np.zeros([n+1, 3])
        else:
            ans = np.zeros([n+1, 5])

        ans[0, 0] = 0
        ans[0, 1] = x
        ans[0, 2] = w
        if y_ground is not None:
            ans[0, 3] = y_ground(x)
            ans[0, 4] = np.abs(w - y_ground(x))

        if verbose:
            print ans[0, :]

    for i in range(1, n+1):
        w += h * f(x, w)
        x += h

        if return_all:
            ans[i, 0] = i
            ans[i, 1] = x
            ans[i, 2] = w
            if y_ground is not None:
                ans[i, 3] = y_ground(x)
                ans[i, 4] = np.abs(w - y_ground(x))

            if verbose:
                print ans[i, :]

    if return_all:
        return ans
    else:
        return w

if __name__ == "__main__":

    # Functions collection:

    def f1(x, y): return y * np.cos(x)
    def y1_ground(x): return np.exp(np.sin(x))


    # parameter of f2:
    alpha = 7  # increase from 2 to 12

    def f2(x, y):
        return alpha * y - (alpha + 1) * np.exp(-x)

    def y2_ground(x):
        return np.exp(-x)

    def f3(x, y):
        l = 50  # to be considered within the perturbation on the initial condition
        return l*y - l
    def y3_ground(x): return 1.0

    ### Cauchy problem 1 ###
    if 0:

        x0 = 0
        x1 = 5
        y0 = 1
        num_steps = [5, 10, 15, 20, 25, 30, 50]

        # Figure
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 5), dpi=100)
        fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.1)
        xx = np.linspace(x0, x1, 10000)
        yy = [y1_ground(x) for x in xx]
        ax[0].plot(xx, yy, '-', label='ground', color='g')
        for num_step in num_steps:

            ans1 = euler_met_1(f1, x0, x1, y0, num_step, y_ground=y1_ground)

            ax[0].plot(ans1[:, 1], ans1[:, 2], '--+', label='steps = ' + str(num_step))
            # Figure of error versus iteration numbers.
            ax[1].plot(ans1[:, 0], ans1[:, 4], '--+', label='Error')

        # Labels, grids legend and titles
        ax[0].grid(True)
        ax[0].set_title('Euler method vs ground truth')
        ax[0].set_xlabel(r'$x$')
        ax[0].set_ylabel(r'$y$')
        ax[0].legend(loc='upper right')
        # add mathematical formula
        ax[0].text(0.5, 0.5, r'Ground: $y(x)=\exp(\sin(x))$')
        ax[0].text(0.5, 0.1, r'ODE: $\frac{dy}{dx}=y \cdot \cos(x)$, $y(0)=1$')

        # Labels, grids legend and titles
        ax[1].grid(True)
        ax[1].set_title('Number of steps versus errors')
        ax[1].set_xlabel(r'Steps')
        ax[1].set_ylabel(r'Error')

    ### Cauchy problem 2 ###
    if 1:

        x0 = 0
        x1 = 1
        y0 = 1
        num_steps = [3, 5, 10, 25, 50]

        # Figure:
        fig, ax2 = plt.subplots(ncols=2, nrows=1, figsize=(12, 5), dpi=130)
        fig.subplots_adjust(left=0.075, right=0.80, top=0.9, bottom=0.1)

        xx = np.linspace(x0, x1, 1000)
        yy = [y2_ground(x) for x in xx]
        ax2[0].plot(xx, yy, '-', label='ground', color='g')

        for num_step in num_steps:
            ans2 = euler_met_1(f2, x0, x1, y0, num_step, y_ground=y2_ground)

            ax2[0].plot(ans2[:, 1], ans2[:, 2], '--+', label='step ' + str(num_step))

            # Figure of error versus iteration numbers.
            ax2[1].plot(ans2[:, 0], ans2[:, 4], '--+', label='Error')

        # Labels, grids legend and titles
        ax2[0].grid(True)
        ax2[0].set_title('Euler method vs ground truth')
        ax2[0].set_xlabel(r'$x$')
        ax2[0].set_ylabel(r'$y$')
        ax2[0].legend(loc='lower left')
        # add mathematical formula

        fig.text(.82, .8,  r'ODE:')
        fig.text(.82, .72,  r'$\frac{dy}{dx}=\alpha y - (\alpha + 1) \exp(-x)$')
        fig.text(.82, .67,  r'$y(0) = 1$')
        fig.text(.82, .58,  r'Ground: ')
        fig.text(.82, .5,  r'$y(x)=\exp(-x)$')

        fig.text(.82, .38,  r'Parameter: ')
        fig.text(.82, .31, r'$\alpha = $ ' + str(alpha))

        # Labels, grids legend and titles
        ax2[1].grid(True)
        ax2[1].set_title('Number of steps versus errors')
        ax2[1].set_xlabel(r'Steps')
        ax2[1].set_ylabel(r'Error')


    ### Cauchy problem 3 ###
    if 0:

        x0 = 0
        x1 = 1

        perturbation = 0.000001   # regulate with the parameter function l

        y0 = 1 + perturbation
        num_steps = [3, 5, 7]

        # Figure:
        fig, ax2 = plt.subplots(ncols=2, nrows=1, figsize=(12, 5), dpi=100)
        fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.1)

        xx = np.linspace(x0, x1, 1000)
        yy = [y3_ground(x) for x in xx]
        ax2[0].plot(xx, yy, '-', label='ground', color='g')

        for num_step in num_steps:
            ans3 = euler_met_1(f3, x0, x1, y0, num_step, y_ground=y3_ground)

            ax2[0].plot(ans3[:, 1], ans3[:, 2], '--+', label='step ' + str(num_step))

            # Figure of error versus iteration numbers.
            ax2[1].plot(ans3[:, 0], ans3[:, 4], '--+', label='Error')

        # Labels, grids legend and titles
        y_val = ax2[0].get_ylim()
        ax2[0].set_ylim([0.4, max(y_val[1], 2)])
        ax2[0].grid(True)
        ax2[0].set_title('Euler method vs ground truth')
        ax2[0].set_xlabel(r'$x$')
        ax2[0].set_ylabel(r'$y$')
        ax2[0].legend(loc='lower left')
        # add mathematical formula
        #ax2[0].text(0.5, 0.5, r'Ground: $y(x)=\exp(\sin(x))$')
        #ax2[0].text(0.5, 0.1, r'ODE: $\frac{dy}{dx}=y \cdot \cos(x)$, $y(0)=1$')

        # Labels, grids legend and titles
        ax2[1].grid(True)
        ax2[1].set_title('Number of steps versus errors')
        ax2[1].set_xlabel(r'Steps')
        ax2[1].set_ylabel(r'Error')

    plt.show()
