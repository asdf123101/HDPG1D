"""
A module for postprocessing the numerical results from HDPG1d solver.
"""
import matplotlib.pyplot as plt
import numpy as np


class utils(object):
    def __init__(self, solution):
        self.solution = solution
        exactNumEle = 500
        exactBasisFuncs = 5
        self.solution.coeff.numEle = exactNumEle
        self.solution.coeff.pOrder = exactBasisFuncs - 1
        self.solution.mesh = np.linspace(0, 1, exactNumEle + 1)
        # approximate the exact solution for general problems
        # self.exactSol = self.solution.solveLocal()[0][exactNumEle * exactBasisFuncs - 1]
        # for the reaction diffusion test problem, we know the exact solution
        self.exactSol = np.sqrt(self.solution.kappa)

    def errorL2(self):
        errorL2 = 0.
        numEle = self.solution.coeff.numEle
        numBasisFuncs = self.solution.coeff.pOrder + 1
        # solve on the uniform mesh
        self.solution.mesh = np.linspace(0, 1, numEle + 1)
        U = self.solution.solveLocal()[0]
        errorL2 = np.abs(U[numBasisFuncs * numEle - 1] - self.exactSol)
        return errorL2

    def uniConv(self):
        numBasisFuncs = np.arange(
            self.solution.numBasisFuncs, self.solution.numBasisFuncs + 1)
        numEle = 2**np.arange(1, 9)
        uniError = np.zeros((numEle.size, numBasisFuncs.size))
        for i in range(numBasisFuncs.size):
            self.solution.coeff.pOrder = numBasisFuncs[i] - 1
            for j, n in enumerate(numEle):
                self.solution.coeff.numEle = n
                uniError[j, i] = self.errorL2()
        return numEle, uniError

    def convHistory(self):
        """Plot the uniform and adaptive convergence history"""
        plt.figure(2)
        trueErrorList = self.solution.trueErrorList
        trueErrorList[1] = np.abs(trueErrorList[1] - self.exactSol)
        estErrorList = self.solution.estErrorList
        plt.loglog(trueErrorList[0],
                   trueErrorList[1], '-ro')
        numEle, errorL2 = self.uniConv()
        plt.loglog(numEle, errorL2, '-o')
        plt.loglog(estErrorList[0],
                   estErrorList[1], '--', color='#1f77b4')
        plt.xlabel('Number of elements', fontsize=17)
        plt.ylabel('Error', fontsize=17)
        plt.grid()
        plt.legend(('Adaptive', 'Uniform', 'Estimator'), loc=3, fontsize=15)
        plt.show()
