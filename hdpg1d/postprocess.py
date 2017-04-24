"""
A module for postprocessing the numerical results from HDPG1d solver.
"""
import matplotlib.pyplot as plt
import numpy as np


class utils(object):
    def __init__(self, solution):
        self.solution = solution
        exactNumEle = 200
        exactBasisFuncs = 5
        self.solution.numEle = exactNumEle
        self.solution.numBasisFuncs = exactBasisFuncs
        x = np.linspace(0, 1, exactNumEle + 1)
        self.exactSol = self.solution.solve_local(
            [], x)[0].A1[exactNumEle * exactBasisFuncs - 1]

    def errorL2(self):
        errorL2 = 0.
        n_ele = self.solution.numEle
        p = self.solution.numBasisFuncs
        # solve the uniform case
        x = np.linspace(0, 1, n_ele + 1)
        U, _ = self.solution.solve_local([], x)
        errorL2 = np.abs(U[p * n_ele - 1] - self.exactSol)
        return errorL2

    def uniConv(self):
        numBasisFuncs = np.arange(2, 3)
        numEle = 2**np.arange(1, 9)
        uniError = np.zeros((numEle.size, numBasisFuncs.size))
        for i in range(numBasisFuncs.size):
            self.solution.numBasisFuncs = numBasisFuncs[i]
            for j, n in enumerate(numEle):
                self.solution.numEle = n
                uniError[j, i] = self.errorL2()
        return numEle, uniError

    def convHistory(self):
        trueError = self.solution.trueError
        estError = self.solution.estError
        plt.loglog(trueError[0, 0:-1],
                   trueError[1, 0:-1], '-ro')
        # plt.axis([1, 250, 1e-13, 1e-2])
        numEle, errorL2 = self.uniConv()
        plt.loglog(numEle, errorL2, '-o')
        plt.loglog(estError[0, :],
                   estError[1, :], '--', color='#1f77b4')
        plt.xlabel('Number of elements', fontsize=17)
        plt.ylabel('Error', fontsize=17)
        plt.grid()
        plt.legend(('Adaptive', 'Uniform', 'Estimator'), loc=3, fontsize=15)
        plt.show()
