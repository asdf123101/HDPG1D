import numpy as np
from collections import namedtuple
from scipy.linalg import block_diag


def shape(x, p):
    """generate p shape functions and its first order derivative
    (order p-1) at the given location x"""
    A = np.array([np.linspace(-1, 1, p)]).T**np.arange(p)
    C = np.linalg.inv(A).T
    x = np.array([x]).T
    shp = C.dot((x**np.arange(p)).T)
    shpx = C[:, 1::1].dot((x**np.arange(p - 1) * np.arange(1, p)).T)
    return shp, shpx


def forcing(x):
    f = 1
    return f


def boundaryCondition(case, t=None):
    # boundary condition
    if case == 0:
        # primal problem
        bc = [0, 0]
    if case == 1:
        # adjoint problem
        bc = [0, 1]
    return bc


class discretization(object):
    """Given the problem statement, construct the discretization matrices"""

    def __init__(self, coeff, mesh, enrich=None):
        self.mesh = mesh
        self.coeff = coeff
        self.enrich = enrich
        # the following init are for the sake of simplicity
        self.numBasisFuncs = coeff.pOrder + 1
        self.tau_pos = coeff.tauPlus
        self.tau_neg = coeff.tauMinus
        self.conv = coeff.convection
        self.kappa = coeff.diffusion
        self.n_ele = len(mesh) - 1
        self.dist = self.distGen()
        # shape function and gauss quadrature
        self.gqOrder = 5 * self.numBasisFuncs
        self.xi, self.wi = np.polynomial.legendre.leggauss(self.gqOrder)
        self.shp, self.shpx = shape(self.xi, self.numBasisFuncs)
        # enrich the space if the enrich argument is given
        if enrich is not None:
            self.shpEnrich, self.shpxEnrich = shape(
                self.xi, self.numBasisFuncs + enrich)

    def distGen(self):
        dist = np.zeros(self.n_ele)
        for i in range(1, self.n_ele + 1):
            dist[i - 1] = self.mesh[i] - self.mesh[i - 1]
        return dist

    def matGroup(self):
        lhsMat = self.lhsGen()
        F, L, R = lhsMat.F, lhsMat.L, lhsMat.R
        eleMat = self.eleGen()
        A, B, BonU, D = eleMat.A, eleMat.B, eleMat.BonU, eleMat.D
        faceMat = self.interfaceGen()
        C, E, G, H = faceMat.C, faceMat.E, faceMat.G, faceMat.H
        matGroup = namedtuple(
            'matGroup', ['A', 'B', 'BonU', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'R'])
        return matGroup(A, B, BonU, C, D, E, F, G, H, L, R)

    def lhsGen(self):
        """Generate matrices associated with left hand side"""
        if self.enrich is not None:
            # when enrich is given
            # provide matrices used in the DWR residual calculation
            numBasisFuncs = self.numBasisFuncs + self.enrich
            shp = self.shpEnrich
        else:
            numBasisFuncs = self.numBasisFuncs
            shp = self.shp
        # forcing vector F
        F = np.zeros(numBasisFuncs * self.n_ele)
        for i in range(1, self.n_ele + 1):
            f = self.dist[i - 1] / 2 \
                * shp.dot(self.wi * forcing(self.mesh[i - 1] +
                                            1 / 2 * (1 + self.xi) *
                                            self.dist[i - 1]))
            F[(i - 1) * numBasisFuncs:i * numBasisFuncs] = f
        F[0] += (self.conv + self.tau_pos) * boundaryCondition(0)[0]
        F[-1] += (-self.conv + self.tau_neg) * boundaryCondition(0)[1]
        # L, easy in 1d
        L = np.zeros(self.n_ele - 1)
        # R, easy in 1d
        R = np.zeros(numBasisFuncs * self.n_ele)
        R[0] = boundaryCondition(0)[0]
        R[-1] = -boundaryCondition(0)[1]
        lhsMat = namedtuple('lhsMat', ['F', 'L', 'R'])
        return lhsMat(F, L, R)

    def eleGen(self):
        """Generate matrices associated with elements"""
        if self.enrich is not None:
            numBasisFuncsEnrich = self.numBasisFuncs + self.enrich
            shpEnrich = self.shpEnrich
            shpxEnrich = self.shpxEnrich
            b = (self.shpx.T * np.ones((self.gqOrder, self.numBasisFuncs))
                 ).T.dot(np.diag(self.wi).dot(shpEnrich.T))
            # BonU is only used in calculating DWR residual
            BonU = block_diag(*[b] * (self.n_ele)).T
        else:
            numBasisFuncsEnrich = self.numBasisFuncs
            shpEnrich = self.shp
            shpxEnrich = self.shpx
            BonU = None
        a = 1 / self.kappa * shpEnrich.dot(np.diag(self.wi).dot(self.shp.T))
        A = np.repeat(self.dist, self.numBasisFuncs) / \
            2 * block_diag(*[a] * (self.n_ele))
        b = (shpxEnrich.T * np.ones((self.gqOrder, numBasisFuncsEnrich))
             ).T.dot(np.diag(self.wi).dot(self.shp.T))
        B = block_diag(*[b] * (self.n_ele))
        d = shpEnrich.dot(np.diag(self.wi).dot(self.shp.T))
        # assemble D
        d_face = np.zeros((numBasisFuncsEnrich, self.numBasisFuncs))
        d_face[0, 0] = self.tau_pos
        d_face[-1, -1] = self.tau_neg
        D = np.repeat(self.dist, self.numBasisFuncs) / 2 * \
            block_diag(*[d] * (self.n_ele)) + \
            block_diag(*[d_face] * (self.n_ele))
        eleMat = namedtuple('eleMat', ['A', 'B', 'BonU', 'D'])
        return eleMat(A, B, BonU, D)

    def interfaceGen(self):
        """Generate matrices associated with interfaces"""
        if self.enrich is not None:
            # when enrich is given
            # provide matrices used in the DWR residual calculation
            numBasisFuncs = self.numBasisFuncs + self.enrich
        else:
            numBasisFuncs = self.numBasisFuncs
        tau_pos, tau_neg, conv = self.tau_pos, self.tau_neg, self.conv
        # elemental h
        h = np.zeros((2, 2))
        h[0, 0], h[-1, -1] = -conv - tau_pos, conv - tau_neg
        # mappinng matrix
        map_h = np.zeros((2, self.n_ele), dtype=int)
        map_h[:, 0] = np.arange(2)
        for i in np.arange(1, self.n_ele):
            map_h[:, i] = np.arange(
                map_h[2 - 1, i - 1], map_h[2 - 1, i - 1] + 2)
        # assemble H and eliminate boundaries
        H = np.zeros((self.n_ele + 1, self.n_ele + 1))
        for i in range(self.n_ele):
            for j in range(2):
                m = map_h[j, i]
                for k in range(2):
                    n = map_h[k, i]
                    H[m, n] += h[j, k]
        H = H[1:self.n_ele][:, 1:self.n_ele]

        # elemental e
        e = np.zeros((numBasisFuncs, 2))
        e[0, 0], e[-1, -1] = -conv - tau_pos, conv - tau_neg
        # mapping matrix
        map_e_x = np.arange(numBasisFuncs * self.n_ele,
                            dtype=int).reshape(self.n_ele,
                                               numBasisFuncs).T
        map_e_y = map_h
        # assemble global E
        E = np.zeros((numBasisFuncs * self.n_ele, self.n_ele + 1))
        for i in range(self.n_ele):
            for j in range(numBasisFuncs):
                m = map_e_x[j, i]
                for k in range(2):
                    n = map_e_y[k, i]
                    E[m, n] += e[j, k]
        E = E[:, 1:self.n_ele]

        # elemental c
        c = np.zeros((numBasisFuncs, 2))
        c[0, 0], c[-1, -1] = -1, 1
        # assemble global C
        C = np.zeros((numBasisFuncs * self.n_ele, self.n_ele + 1))
        for i in range(self.n_ele):
            for j in range(numBasisFuncs):
                m = map_e_x[j, i]
                for k in range(2):
                    n = map_e_y[k, i]
                    C[m, n] += c[j, k]
        C = C[:, 1:self.n_ele]

        # elemental g
        g = np.zeros((2, numBasisFuncs))
        g[0, 0], g[-1, -1] = tau_pos, tau_neg
        # mapping matrix
        map_g_x = map_h
        map_g_y = np.arange(numBasisFuncs * self.n_ele,
                            dtype=int).reshape(self.n_ele,
                                               numBasisFuncs).T
        # assemble global G
        G = np.zeros((self.n_ele + 1, numBasisFuncs * self.n_ele))
        for i in range(self.n_ele):
            for j in range(2):
                m = map_g_x[j, i]
                for k in range(numBasisFuncs):
                    n = map_g_y[k, i]
                    G[m, n] += g[j, k]
        G = G[1:self.n_ele, :]
        faceMat = namedtuple('faceMat', ['C', 'E', 'G', 'H'])
        return faceMat(C, E, G, H)
