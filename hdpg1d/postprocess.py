"""
A module for postprocessing the numerical results from HDPG1d solver.
"""
import matplotlib.pyplot as plt
import numpy as np


def errorL2(solution):
    errorL2 = 0.
    # solve the uniform case
    x = np.linspace(0, 1, solution.n_ele + 1)
    U, _ = solution.solve_local([], x)
    errorL2 = np.abs(U[solution.p * solution.n_ele - 1] -
                     np.sqrt(solution.kappa))
    return errorL2


def uniConv(solution):
    p = np.arange(2, 3)
    n_ele = 2**np.arange(1, 9)
    uniError = np.zeros((n_ele.size, p.size))
    for i in range(p.size):
        solution.p = p[i]
        for j, n in enumerate(n_ele):
            solution.n_ele = n
            uniError[j, i] = errorL2(solution)
    return n_ele, uniError


def convHistory(solution):
    plt.loglog(solution.trueError[0, 0:-1],
               solution.trueError[1, 0:-1], '-ro')
    # plt.axis([1, 250, 1e-13, 1e-2])
    n_ele, errorL2 = uniConv(solution)
    plt.loglog(n_ele, errorL2, '-o')
    plt.loglog(solution.estError[0, :],
               solution.estError[1, :], '--', color='#1f77b4')
    plt.xlabel('Number of elements', fontsize=17)
    plt.ylabel('Error', fontsize=17)
    plt.grid()
    plt.legend(('Adaptive', 'Uniform', 'Estimator'), loc=3, fontsize=15)
    plt.show()
