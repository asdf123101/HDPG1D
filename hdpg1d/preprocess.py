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
    # f = np.cos(2*np.pi*x)
    # f = 4*pi**2*sin(2*pi*x)
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


def discretization(coeff, mesh):
    """Given the problem statement, construct the discretization matrice"""
    # p is the number of basis functions
    p = coeff.pOrder + 1
    tau_pos = coeff.tauPlus
    tau_neg = coeff.tauMinus

    # order of gauss quadrature
    gorder = 2 * p
    # shape function and gauss quadrature
    xi, wi = np.polynomial.legendre.leggauss(gorder)
    shp, shpx = shape(xi, p)

    con = coeff.convection
    kappa = coeff.diffusion

    n_ele = len(mesh) - 1
    dist = np.zeros(n_ele)
    # elemental forcing vector
    F = np.zeros(p * n_ele)
    for i in range(1, n_ele + 1):
        dist[i - 1] = mesh[i] - mesh[i - 1]
        f = dist[i - 1] / 2 * shp.dot(
            wi * forcing(mesh[i - 1] + 1 / 2 * (1 + xi) * dist[i - 1]))
        F[(i - 1) * p:(i - 1) * p + p] = f
    F[0] += (con + tau_pos) * bc(0)[0]
    F[-1] += (-con + tau_neg) * bc(0)[1]

    d = shp.dot(np.diag(wi).dot(shp.T))

    # assemble global D
    d_face = np.zeros((p, p))
    d_face[0, 0] = tau_pos
    d_face[-1, -1] = tau_neg
    D = np.repeat(dist, p) / 2 * block_diag(*
                                            [d] * (n_ele)) + block_diag(*[d_face] * (n_ele))

    a = 1 / kappa * shp.dot(np.diag(wi).dot(shp.T))
    A = np.repeat(dist, p) / 2 * block_diag(*[a] * (n_ele))
    b = (shpx.T * np.ones((gorder, p))).T.dot(np.diag(wi).dot(shp.T))
    B = block_diag(*[b] * (n_ele))

    # elemental h
    h = np.zeros((2, 2))
    h[0, 0], h[-1, -1] = -con - tau_pos, con - tau_neg
    # mappinng matrix
    map_h = np.zeros((2, n_ele), dtype=int)
    map_h[:, 0] = np.arange(2)
    for i in np.arange(1, n_ele):
        map_h[:, i] = np.arange(
            map_h[2 - 1, i - 1], map_h[2 - 1, i - 1] + 2)
    # assemble H and eliminate boundaries
    H = np.zeros((n_ele + 1, n_ele + 1))
    for i in range(n_ele):
        for j in range(2):
            m = map_h[j, i]
            for k in range(2):
                n = map_h[k, i]
                H[m, n] += h[j, k]
    H = H[1:n_ele][:, 1:n_ele]

    # elemental g
    g = np.zeros((2, p))
    g[0, 0], g[-1, -1] = tau_pos, tau_neg
    # mapping matrix
    map_g_x = map_h
    map_g_y = np.arange(p * n_ele, dtype=int).reshape(n_ele, p).T
    # assemble global G
    G = np.zeros((n_ele + 1, p * n_ele))
    for i in range(n_ele):
        for j in range(2):
            m = map_g_x[j, i]
            for k in range(p):
                n = map_g_y[k, i]
                G[m, n] += g[j, k]
    G = G[1:n_ele, :]

    # elemental e
    e = np.zeros((p, 2))
    e[0, 0], e[-1, -1] = -con - tau_pos, con - tau_neg
    # mapping matrix
    map_e_x = np.arange(p * n_ele, dtype=int).reshape(n_ele, p).T
    map_e_y = map_h
    # assemble global E
    E = np.zeros((p * n_ele, n_ele + 1))
    for i in range(n_ele):
        for j in range(p):
            m = map_e_x[j, i]
            for k in range(2):
                n = map_e_y[k, i]
                E[m, n] += e[j, k]
    E = E[:, 1:n_ele]

    # L, easy in 1d
    L = np.zeros(n_ele - 1)

    # elemental c
    c = np.zeros((p, 2))
    c[0, 0], c[-1, -1] = -1, 1

    # assemble global C
    C = np.zeros((p * n_ele, n_ele + 1))
    for i in range(n_ele):
        for j in range(p):
            m = map_e_x[j, i]
            for k in range(2):
                n = map_e_y[k, i]
                C[m, n] += c[j, k]
    C = C[:, 1:n_ele]
    # L, easy in 1d
    L = np.zeros(n_ele - 1)

    # R, easy in 1d
    R = np.zeros(p * n_ele)
    R[0] = bc(0)[0]
    R[-1] = -bc(0)[1]
    return A, B, C, D, E, F, G, H, L, R, m
