"""
A module for postprocessing the numerical results.
"""
import matplotlib.pyplot as plt
import numpy as np


def convHistory(trueError, estError):
    plt.loglog(trueError[0, 0:-1], np.abs(trueError[1, 0:-1]), '-ro')
    plt.axis([1, 250, 1e-13, 1e-2])
    # plt.loglog(n_ele, errorL2, '-o')
    plt.loglog(estError[0, :], estError[1, :], '--', color='#1f77b4')
    plt.xlabel('Number of elements', fontsize=17)
    plt.ylabel('Error', fontsize=17)
    plt.grid()
    plt.legend(('Adaptive', 'Estimator'), loc=3, fontsize=15)
    # plt.savefig('conv{}p{}_4_test.pdf'.format(15, test.p))
    plt.show()
