import numpy as np
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


def bc(case, t=None):
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


class discretization(object):
    """Given the problem statement, construct the discretization matrices"""

    def __init__(self, coeff, mesh, enrich=None):
        self.mesh = mesh
        self.coeff = coeff
        self.enrich = enrich
        # p is the number of basis functions
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

    def matGen(self):
        self.lhsGen()
        self.eleGen()
        self.interfaceGen()

    def lhsGen(self):
        """Generate matrices associated with left hand side"""
        # elemental forcing vector F
        F = np.zeros(self.numBasisFuncs * self.n_ele)
        for i in range(1, self.n_ele + 1):
            f = self.dist[i - 1] / 2 * self.shp.dot(
                self.wi * forcing(self.mesh[i - 1] + 1 / 2 * (1 + self.xi) * self.dist[i - 1]))
            F[(i - 1) * self.numBasisFuncs:i * self.numBasisFuncs] = f
        F[0] += (self.conv + self.tau_pos) * bc(0)[0]
        F[-1] += (-self.conv + self.tau_neg) * bc(0)[1]
        # L, easy in 1d
        L = np.zeros(self.n_ele - 1)
        # R, easy in 1d
        R = np.zeros(self.numBasisFuncs * self.n_ele)
        R[0] = bc(0)[0]
        R[-1] = -bc(0)[1]
        self.F, self.L, self.R = F, L, R

    def eleGen(self):
        """Generate matrices associated with elements"""
        a = 1 / self.kappa * self.shp.dot(np.diag(self.wi).dot(self.shp.T))
        A = np.repeat(self.dist, self.numBasisFuncs) / \
            2 * block_diag(*[a] * (self.n_ele))
        b = (self.shpx.T * np.ones((self.gqOrder, self.numBasisFuncs))
             ).T.dot(np.diag(self.wi).dot(self.shp.T))
        B = block_diag(*[b] * (self.n_ele))
        d = self.shp.dot(np.diag(self.wi).dot(self.shp.T))
        # assemble global D
        d_face = np.zeros((self.numBasisFuncs, self.numBasisFuncs))
        d_face[0, 0] = self.tau_pos
        d_face[-1, -1] = self.tau_neg
        D = np.repeat(self.dist, self.numBasisFuncs) / 2 * \
            block_diag(*[d] * (self.n_ele)) + \
            block_diag(*[d_face] * (self.n_ele))
        self.A, self.B, self.D = A, B, D

    def interfaceGen(self):
        """Generate matrices associated with interfaces"""
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
        e = np.zeros((self.numBasisFuncs, 2))
        e[0, 0], e[-1, -1] = -conv - tau_pos, conv - tau_neg
        # mapping matrix
        map_e_x = np.arange(self.numBasisFuncs * self.n_ele,
                            dtype=int).reshape(self.n_ele, self.numBasisFuncs).T
        map_e_y = map_h
        # assemble global E
        E = np.zeros((self.numBasisFuncs * self.n_ele, self.n_ele + 1))
        for i in range(self.n_ele):
            for j in range(self.numBasisFuncs):
                m = map_e_x[j, i]
                for k in range(2):
                    n = map_e_y[k, i]
                    E[m, n] += e[j, k]
        E = E[:, 1:self.n_ele]

        # elemental c
        c = np.zeros((self.numBasisFuncs, 2))
        c[0, 0], c[-1, -1] = -1, 1
        # assemble global C
        C = np.zeros((self.numBasisFuncs * self.n_ele, self.n_ele + 1))
        for i in range(self.n_ele):
            for j in range(self.numBasisFuncs):
                m = map_e_x[j, i]
                for k in range(2):
                    n = map_e_y[k, i]
                    C[m, n] += c[j, k]
        C = C[:, 1:self.n_ele]

        # elemental g
        g = np.zeros((2, self.numBasisFuncs))
        g[0, 0], g[-1, -1] = tau_pos, tau_neg
        # mapping matrix
        map_g_x = map_h
        map_g_y = np.arange(self.numBasisFuncs * self.n_ele,
                            dtype=int).reshape(self.n_ele, self.numBasisFuncs).T
        # assemble global G
        G = np.zeros((self.n_ele + 1, self.numBasisFuncs * self.n_ele))
        for i in range(self.n_ele):
            for j in range(2):
                m = map_g_x[j, i]
                for k in range(self.numBasisFuncs):
                    n = map_g_y[k, i]
                    G[m, n] += g[j, k]
        G = G[1:self.n_ele, :]

        self.C, self.E, self.G, self.H = C, E, G, H
