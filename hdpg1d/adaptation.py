import numpy as np
from numpy import concatenate as cat
import matplotlib.pyplot as plt
import warnings
from .preprocess import shape, discretization, boundaryCondition

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# supress the deprecation warning
warnings.filterwarnings("ignore", ".*GUI is implemented.*")


class hdpg1d(object):
    """
    1D HDG solver
    """

    def __init__(self, coeff):
        self.numEle = coeff.numEle
        self.numBasisFuncs = coeff.pOrder + 1
        self.tau_pos = coeff.tauPlus
        self.tau_neg = coeff.tauMinus
        self.c = coeff.convection
        self.kappa = coeff.diffusion
        self.coeff = coeff
        self.mesh = np.linspace(0, 1, self.numEle + 1)
        self.u = []
        self.estErrorList = [[], []]
        self.trueErrorList = [[], []]

    def plotU(self, counter):
        """Plot solution u with smooth higher oredr quadrature"""
        uSmooth = np.array([])
        uNode = np.zeros(self.numEle + 1)
        xSmooth = np.array([])
        u = self.u[int(len(self.u) / 2):len(self.u)]

        # quadrature rule
        gorder = 10 * self.numBasisFuncs
        xi, wi = np.polynomial.legendre.leggauss(gorder)
        shp, shpx = shape(xi, self.numBasisFuncs)
        for j in range(1, self.numEle + 1):
            xSmooth = np.hstack((xSmooth, (self.mesh[(j - 1)] + self.mesh[j]) / 2 + (
                self.mesh[j] - self.mesh[j - 1]) / 2 * xi))
            uSmooth = np.hstack(
                (uSmooth, shp.T.dot(u[(j - 1) * self.numBasisFuncs:j * self.numBasisFuncs])))
            uNode[j - 1] = u[(j - 1) * self.numBasisFuncs]
        uNode[-1] = u[-1]
        plt.figure(1)
        plt.plot(xSmooth, uSmooth, '-', color='C3')
        plt.plot(self.mesh, uNode, 'C3.')
        plt.xlabel('$x$', fontsize=17)
        plt.ylabel('$u$', fontsize=17)
        # plt.axis([-0.05, 1.05, 0, 1.3])
        plt.grid()
        plt.pause(5e-1)
        plt.clf()

    def meshAdapt(self, index):
        """Given the index list, adapt the mesh"""
        inValue = np.zeros(len(index))
        for i in np.arange(len(index)):
            inValue[i] = (self.mesh[index[i]] +
                          self.mesh[index[i] - 1]) / 2
        self.mesh = np.sort(np.insert(self.mesh, 0, inValue))

    def solveLocal(self):
        """Solve the primal problem"""
        if 'matLocal' in locals():
            # if matLocal exists,
            # only change the mesh instead of initializing again
            matLocal.mesh = self.mesh
        else:
            matLocal = discretization(self.coeff, self.mesh)
        matGroup = matLocal.matGroup()
        A, B, _, C, D, E, F, G, H, L, R = matGroup
        # solve
        K = -cat((C.T, G), axis=1)\
            .dot(np.linalg.inv(np.bmat([[A, -B], [B.T, D]]))
                 .dot(cat((C, E)))) + H
        F_hat = np.array([L]).T - cat((C.T, G), axis=1)\
            .dot(np.linalg.inv(np.bmat([[A, -B], [B.T, D]])))\
            .dot(np.array([cat((R, F))]).T)
        uFace = np.linalg.solve(K, F_hat)
        u = np.linalg.inv(np.bmat([[A, -B], [B.T, D]]))\
            .dot(np.array([np.concatenate((R, F))]).T -
                 cat((C, E)).dot(uFace))
        return u.A1, uFace.A1

    def solveAdjoint(self):
        """Solve the adjoint problem"""
        # solve in the enriched space
        self.coeff.pOrder += 1
        if 'matAdjoint' in locals():
            matAdjoint.mesh = self.mesh
        else:
            matAdjoint = discretization(self.coeff, self.mesh)
        self.coeff.pOrder = self.coeff.pOrder - 1
        matGroup = matAdjoint.matGroup()
        A, B, _, C, D, E, F, G, H, L, R = matGroup
        # add adjoint LHS conditions
        F = np.zeros(len(F))
        R[-1] = -boundaryCondition(1)[1]

        # assemble global matrix LHS
        LHS = np.bmat([[A, -B, C],
                       [B.T, D, E],
                       [C.T, G, H]])
        # solve
        U = np.linalg.solve(LHS.T, cat((R, F, L)))
        return U[0:2 * len(C)], U[len(C):len(U)]

    def residual(self, U, hat_U, z, hat_z):
        enrich = 1
        if 'matResidual' in locals():
            matResidual.mesh = self.mesh
        else:
            matResidual = discretization(self.coeff, self.mesh, enrich)
        matGroup = matResidual.matGroup()
        A, B, BonU, C, D, E, F, G, H, L, R = matGroup
        LHS = np.bmat([[A, -B, C],
                       [BonU, D, E]])
        RHS = cat((R, F))
        residual = np.zeros(self.numEle)
        numEnrich = self.numBasisFuncs + enrich
        for i in np.arange(self.numEle):
            primalResidual = (LHS.dot(cat((U, hat_U))) - RHS).A1
            uLength = self.numEle * numEnrich
            stepLength = i * numEnrich
            uDWR = primalResidual[stepLength:stepLength + numEnrich].dot(
                (1 - z)[stepLength:stepLength + numEnrich])
            qDWR = primalResidual[uLength + stepLength:uLength +
                                  stepLength + numEnrich]\
                .dot((1 - z)[uLength + stepLength:uLength +
                             stepLength + numEnrich])
            residual[i] = uDWR + qDWR
        # sort residual index
        com_index = np.argsort(np.abs(residual))
        # select \theta% elements with the large error
        theta = 0.15
        refine_index = com_index[
            int(self.numEle * (1 - theta)):len(residual)] + 1
        return np.abs(np.sum(residual)), refine_index

    def adaptive(self):
        tol = 1e-10
        estError = 10
        counter = 0
        ceilCounter = 30
        while estError > tol and counter < ceilCounter:
            print("Iteration {}. Target function error {:.3e}.".format(
                counter, estError))
            # solve
            u, uFace = self.solveLocal()
            adjoint, adjointFace = self.solveAdjoint()
            self.u = u
            # plot the solution at certain counter
            if counter in [0, 4, 9, 19]:
                self.plotU(counter)
            # record error
            self.trueErrorList[0].append(self.numEle)
            self.trueErrorList[1].append(
                u[self.numEle * self.numBasisFuncs - 1])
            estError, index = self.residual(u, uFace, adjoint, adjointFace)
            self.estErrorList[0].append(self.numEle)
            self.estErrorList[1].append(estError)
            # adapt
            index = index.tolist()
            self.meshAdapt(index)
            self.numEle = self.numEle + len(index)
            counter += 1
