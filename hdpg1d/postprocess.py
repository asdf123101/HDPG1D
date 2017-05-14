"""
A module for postprocessing the numerical results from HDPG1d solver.
"""
import matplotlib.pyplot as plt
import numpy as np


class utils(object):
    exactNumEle = 300
    exactBasisFuncs = 5

    def __init__(self, solution):
        self.solution = solution
        self.numEle = self.exactNumEle
        self.exactSoln = self.solveExact()

    def solveExact(self):
        self.solution.coeff.numEle = self.exactNumEle
        self.solution.coeff.pOrder = self.exactBasisFuncs - 1
        self.solution.mesh = np.linspace(0, 1, self.exactNumEle + 1)
        # approximate the exact solution for general problems
        self.solution.solveLocal()
        exactSoln = self.solution.separateSoln(self.solution.primalSoln)[0][
            self.exactNumEle * self.exactBasisFuncs - 1]
        # for the reaction diffusion test problem, we know the exact solution
        # self.exactSol = np.sqrt(self.solution.coeff.diffusion)
        return exactSoln

    def errorL2(self):
        errorL2 = 0.
        numEle = self.solution.coeff.numEle
        self.solution.numEle = numEle
        numBasisFuncs = self.solution.coeff.pOrder + 1
        # solve on the uniform mesh
        self.solution.mesh = np.linspace(0, 1, numEle + 1)
        self.solution.solveLocal()
        gradState, _ = self.solution.separateSoln(self.solution.primalSoln)
        errorL2 = np.abs(
            gradState[numBasisFuncs * numEle - 1] - self.exactSoln)
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
        print("Please note that the error is calculated using {} "
              "elements with polynomial order {}."
              .format(self.exactNumEle, self.exactBasisFuncs))
        plt.figure(2)
        trueErrorList = self.solution.trueErrorList
        trueErrorList[1] = np.abs(trueErrorList[1] - self.exactSoln)
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
