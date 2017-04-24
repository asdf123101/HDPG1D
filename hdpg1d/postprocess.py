"""
A module for postprocessing the numerical results from HDPG1d solver.
"""
import matplotlib.pyplot as plt
import numpy as np


class utils(object):
    def __init__(self, solution):
        self.solution = solution
        exactNumEle = 200
        exactPolyOrder = 5
        self.solution.n_ele = exactNumEle
        self.solution.p = exactPolyOrder
        x = np.linspace(0, 1, exactNumEle + 1)
        self.exactSol = self.solution.solve_local(
            [], x)[0].A1[exactNumEle * exactPolyOrder - 1]

    def errorL2(self):
        errorL2 = 0.
        n_ele = self.solution.n_ele
        p = self.solution.p
        # solve the uniform case
        x = np.linspace(0, 1, n_ele + 1)
        U, _ = self.solution.solve_local([], x)
        errorL2 = np.abs(U[p * n_ele - 1] - self.exactSol)
        return errorL2

    def uniConv(self):
        p = np.arange(2, 3)
        n_ele = 2**np.arange(1, 9)
        uniError = np.zeros((n_ele.size, p.size))
        for i in range(p.size):
            self.solution.p = p[i]
            for j, n in enumerate(n_ele):
                self.solution.n_ele = n
                uniError[j, i] = self.errorL2()
        return n_ele, uniError

    def convHistory(self):
        trueError = self.solution.trueError
        estError = self.solution.estError
        plt.loglog(trueError[0, 0:-1],
                   trueError[1, 0:-1], '-ro')
        # plt.axis([1, 250, 1e-13, 1e-2])
        n_ele, errorL2 = self.uniConv()
        plt.loglog(n_ele, errorL2, '-o')
        plt.loglog(estError[0, :],
                   estError[1, :], '--', color='#1f77b4')
        plt.xlabel('Number of elements', fontsize=17)
        plt.ylabel('Error', fontsize=17)
        plt.grid()
        plt.legend(('Adaptive', 'Uniform', 'Estimator'), loc=3, fontsize=15)
        plt.show()
