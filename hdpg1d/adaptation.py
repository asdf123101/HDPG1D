import numpy as np
from numpy import concatenate as cat
import matplotlib.pyplot as plt
import warnings
from matplotlib import rc
from .preprocess import shape, discretization

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
        self.mesh = np.linspace(0, 1, self.numEle + 1)
        self.c = coeff.convection
        self.kappa = coeff.diffusion
        self.coeff = coeff
        self.u = []
        self.estErrorList = []
        self.trueErrorList = []

    def bc(self, case, t=None):
        # boundary condition
        if case == 0:
            # advection-diffusion
            bc = [0, 0]
        if case == 1:
            # simple convection
            # bc = np.sin(2*np.pi*t)
            # adjoint boundary
            bc = [0, 1]
        return bc

    def forcing(self, x):
        # f = np.cos(2*np.pi*x)
        # f = 4*pi**2*sin(2*pi*x)
        f = 1
        return f

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
        plt.axis([-0.05, 1.05, 0, 1.3])
        plt.grid()
        plt.pause(1e-1)
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
            # only change the mesh instead of initialize again
            matLocal.mesh = self.mesh
        else:
            matLocal = discretization(self.coeff, self.mesh)
        matLocal.matGen()
        # solve
        K = -cat((matLocal.C.T, matLocal.G), axis=1)\
            .dot(np.linalg.inv(np.bmat([[matLocal.A, -matLocal.B], [matLocal.B.T, matLocal.D]]))
                 .dot(cat((matLocal.C, matLocal.E)))) + matLocal.H
        F_hat = np.array([matLocal.L]).T - cat((matLocal.C.T, matLocal.G), axis=1)\
            .dot(np.linalg.inv(np.bmat([[matLocal.A, -matLocal.B], [matLocal.B.T, matLocal.D]])))\
            .dot(np.array([cat((matLocal.R, matLocal.F))]).T)
        uFace = np.linalg.solve(K, F_hat)
        u = np.linalg.inv(np.bmat([[matLocal.A, -matLocal.B], [matLocal.B.T, matLocal.D]]))\
            .dot(np.array([np.concatenate((matLocal.R, matLocal.F))]).T -
                 cat((matLocal.C, matLocal.E)).dot(uFace))
        # self.u = u.A1
        return u.A1, uFace.A1

    def solveAdjoint(self):
        """Solve the adjoint problem"""
        # solve in the enriched space
        self.coeff.pOrder += 1
        if 'matAdjoint' in locals():
            matAdjoint.mesh = self.mesh
        else:
            matAdjoint = discretization(self.coeff, self.mesh)
        matAdjoint.matGen()
        self.coeff.pOrder = self.coeff.pOrder - 1
        # add adjoint LHS conditions
        matAdjoint.F = np.zeros(len(matAdjoint.F))
        matAdjoint.R[-1] = -self.bc(1)[1]

        # assemble global matrix LHS
        LHS = np.bmat([[matAdjoint.A, -matAdjoint.B, matAdjoint.C],
                       [matAdjoint.B.T, matAdjoint.D, matAdjoint.E],
                       [matAdjoint.C.T, matAdjoint.G, matAdjoint.H]])
        # solve
        U = np.linalg.solve(LHS.T, np.concatenate(
            (matAdjoint.R, matAdjoint.F, matAdjoint.L)))
        return U[0:2 * len(matAdjoint.C)], U[len(matAdjoint.C):len(U)]

    def residual(self, U, hat_U, z, hat_z):
        numEle = self.numEle
        p = self.numBasisFuncs + 1

        p_l = p - 1
        # order of gauss quadrature
        gorder = 2 * p
        # shape function and gauss quadrature
        xi, wi = np.polynomial.legendre.leggauss(gorder)
        shp, shpx = shape(xi, p)
        shp_l, shpx_l = shape(xi, p_l)
        # ---------------------------------------------------------------------
        # advection constant
        con = self.c

        # diffusion constant
        kappa = self.kappa

        z_q, z_u, z_hat = np.zeros(p * numEle), \
            np.zeros(p *
                     numEle), np.zeros(numEle - 1)

        q, u, lamba = np.zeros(p_l * numEle), \
            np.zeros(p_l * numEle), np.zeros(numEle - 1)
        for i in np.arange(p * numEle):
            z_q[i] = z[i]
            z_u[i] = z[i + p * numEle]

        for i in np.arange(p_l * numEle):
            q[i] = U[i]
            u[i] = U[i + p_l * numEle]

        for i in np.arange(numEle - 1):
            z_hat[i] = hat_z[i]

        # add boundary condtions to U_hat
        U_hat = np.zeros(numEle + 1)
        for i, x in enumerate(hat_U):
            U_hat[i + 1] = x
        U_hat[0] = self.bc(0)[0]
        U_hat[-1] = self.bc(0)[1]

        # L, easy in 1d
        L = np.zeros(numEle + 1)

        # R, easy in 1d
        RR = np.zeros(p * numEle)

        # elemental forcing vector
        dist = np.zeros(numEle)
        F = np.zeros(p * numEle)
        for i in range(1, numEle + 1):
            dist[i - 1] = self.mesh[i] - self.mesh[i - 1]
            f = dist[i - 1] / 2 * shp.dot(
                wi * self.forcing(self.mesh[i - 1] + 1 / 2 * (1 + xi) * dist[i - 1]))
            F[(i - 1) * p:(i - 1) * p + p] = f

        # elemental h
        h = np.zeros((2, 2))
        h[0, 0], h[-1, -1] = -con - self.tau_pos, con - self.tau_neg
        # mappinng matrix
        map_h = np.zeros((2, numEle), dtype=int)
        map_h[:, 0] = np.arange(2)
        for i in np.arange(1, numEle):
            map_h[:, i] = np.arange(
                map_h[2 - 1, i - 1], map_h[2 - 1, i - 1] + 2)
        # assemble H and eliminate boundaries
        H = np.zeros((numEle + 1, numEle + 1))
        for i in range(numEle):
            for j in range(2):
                m = map_h[j, i]
                for k in range(2):
                    n = map_h[k, i]
                    H[m, n] += h[j, k]
        H = H[1:numEle][:, 1:numEle]

        # elemental g
        g = np.zeros((2, p_l))
        g[0, 0], g[-1, -1] = self.tau_pos, self.tau_neg
        # mapping matrix
        map_g_x = map_h
        map_g_y = np.arange(p_l * numEle, dtype=int).reshape(numEle, p_l).T
        # assemble global G
        G = np.zeros((numEle + 1, p_l * numEle))
        for i in range(numEle):
            for j in range(2):
                m = map_g_x[j, i]
                for k in range(p_l):
                    n = map_g_y[k, i]
                    G[m, n] += g[j, k]
        G = G[1:numEle, :]

        # elemental c
        c = np.zeros((p_l, 2))
        c[0, 0], c[-1, -1] = -1, 1

        # mapping matrix
        map_e_x = np.arange(p_l * numEle, dtype=int).reshape(numEle, p_l).T
        map_e_y = map_h

        # assemble global C
        C = np.zeros((p_l * numEle, numEle + 1))
        for i in range(numEle):
            for j in range(p_l):
                m = map_e_x[j, i]
                for k in range(2):
                    n = map_e_y[k, i]
                    C[m, n] += c[j, k]
        C = C[:, 1:numEle]

        # L, easy in 1d
        L = np.zeros(numEle - 1)

        # residual vector
        R = np.zeros(self.numEle)
        for i in np.arange(self.numEle):
            a = dist[i] / 2 * 1 / kappa * \
                ((shp.T).T).dot(np.diag(wi).dot(shp_l.T))

            b = ((shpx.T) * np.ones((gorder, p))
                 ).T.dot(np.diag(wi).dot(shp_l.T))

            b_t = ((shpx_l.T) * np.ones((gorder, p_l))
                   ).T.dot(np.diag(wi).dot(shp.T))

            d = dist[i] / 2 * shp.dot(np.diag(wi).dot(shp_l.T))
            d[0, 0] += self.tau_pos
            d[-1, -1] += self.tau_neg

            h = np.zeros((2, 2))
            h[0, 0], h[-1, -1] = -con - self.tau_pos, con - self.tau_neg

            g = np.zeros((2, p_l))
            g[0, 0], g[-1, -1] = self.tau_pos, self.tau_neg

            e = np.zeros((p, 2))
            e[0, 0], e[-1, -1] = -con - self.tau_pos, con - self.tau_neg

            c = np.zeros((p, 2))
            c[0, 0], c[-1, -1] = -1, 1

            m = np.zeros((2, p_l))
            m[0, 0], m[-1, -1] = -1, 1
            # local error
            R[i] = (np.concatenate((a.dot(q[p_l * i:p_l * i + p_l]) + -b.dot(u[p_l * i:p_l * i + p_l]) + c.dot(U_hat[i:i + 2]),
                                    b_t.T.dot(q[p_l * i:p_l * i + p_l]) + d.dot(u[p_l * i:p_l * i + p_l]) + e.dot(U_hat[i:i + 2]))) - np.concatenate((RR[p * i:p * i + p], F[p * i:p * i + p]))).dot(1 - np.concatenate((z_q[p * i:p * i + p], z_u[p * i:p * i + p])))

        com_index = np.argsort(np.abs(R))
        # select \theta% elements with the large error
        theta = 0.15
        refine_index = com_index[int(self.numEle * (1 - theta)):len(R)]

        # global error
        R_g = (C.T.dot(q) + G.dot(u) + H.dot(U_hat[1:-1])).dot(1 - z_hat)
        return np.abs(np.sum(R) + np.sum(R_g)), refine_index + 1

    def adaptive(self):
        tol = 1e-12
        estError = 10
        counter = 0
        ceilCounter = 100
        trueErrorList = [[], []]
        estErrorList = [[], []]
        while estError > tol or counter > ceilCounter:
            # solve
            u, uFace = self.solveLocal()
            adjoint, adjointFace = self.solveAdjoint()
            self.u = u
            # plot the solution at certain counter
            if counter in [0, 4, 9, 19]:
                self.plotU(counter)
            # record error
            trueErrorList[0].append(self.numEle)
            trueErrorList[1].append(np.abs(
                u[self.numEle * self.numBasisFuncs - 1] - np.sqrt(self.kappa)))
            estError, index = self.residual(u, uFace, adjoint, adjointFace)
            estErrorList[0].append(self.numEle)
            estErrorList[1].append(estError)
            # adapt
            index = index.tolist()
            self.meshAdapt(index)
            self.numEle = self.numEle + len(index)
            counter += 1
        # save error
        self.trueErrorList = trueErrorList
        self.estErrorList = estErrorList
