import numpy as np
from numpy import sin, cos, pi
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


class hdpg1d(object):
    """
    1d advection hdg solver outlined in 'an implicit HHDG method for
    confusion'. Test case: /tau = 1, convection only, linear and higher order.
    Please enter number of elements and polynomial order, i.e., HDG1d(10,2)
    """

    def __init__(self, numEle, numPolyOrder):
        self.numEle = numEle
        self.numBasisFuncs = numPolyOrder + 1
        self.tau_pos = 1e-6
        self.tau_neg = 1e-6
        self.c = 0
        self.kappa = 1e-6
        self.estError = []
        self.trueError = []

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

    def shape(self, x, p):
        """ evaluate shape functions at give locations"""
        # coeffient matrix
        A = np.array([np.linspace(-1, 1, p)]).T**np.arange(p)
        C = np.linalg.inv(A).T
        x = np.array([x]).T
        shp = C.dot((x**np.arange(p)).T)
        shpx = C[:, 1::1].dot((x**np.arange(p - 1) * np.arange(1, p)).T)
        return shp, shpx

    def forcing(self, x):
        # f = np.cos(2*np.pi*x)
        # f = 4*pi**2*sin(2*pi*x)
        f = 1
        return f

    def mesh(self, n_ele, index, x):
        """generate mesh"""
        # if n_ele < 1 or n_ele > self.numEle:
        #   raise RuntimeError('Bad Element number')
        in_value = np.zeros(len(index))
        for i in np.arange(len(index)):
            in_value[i] = (x[index[i]] + x[index[i] - 1]) / 2

        x_c = np.sort(np.insert(x, 0, in_value))

        x_i = np.linspace(x_c[n_ele - 1], x_c[n_ele], num=self.numBasisFuncs)
        dx = x_c[n_ele] - x_c[n_ele - 1]
        return x_i, dx, x_c

    def exact(self, x):
        """solve the problem in an enriched space to simulate exact soltuion"""
        self.numEle = 1000
        self.numBasisFuncs = 3
        x = np.linspace(0, 1, self.numEle + 1)
        self.exactSol = self.solve_local([], x)

    def matrix_gen(self, index, x):
        n_ele = self.numEle

        # order of polynomial shape functions
        p = self.numBasisFuncs

        # order of gauss quadrature
        gorder = 2 * p
        # shape function and gauss quadrature
        xi, wi = np.polynomial.legendre.leggauss(gorder)
        shp, shpx = self.shape(xi, p)
        # ---------------------------------------------------------------------
        # advection constant
        con = self.c

        # diffusion constant
        kappa = self.kappa

        # number of nodes (solution U)
        n_ele = self.numEle + len(index)
        # elemental forcing vector
        F = np.zeros(p * n_ele)
        for i in range(1, n_ele + 1):
            x_i, dx_i, _ = self.mesh(i, index, x)
            f = dx_i / 2 * \
                shp.dot(wi * self.forcing(x[0] + 1 / 2 * (1 + xi) * dx_i))
            F[(i - 1) * p:(i - 1) * p + p] = f
        F[0] += (con + self.tau_pos) * self.bc(0)[0]
        F[-1] += (-con + self.tau_neg) * self.bc(0)[1]

        # elemental d
        d = shp.dot(np.diag(wi).dot(shp.T))

        # elemental a
        a = 1 / kappa * shp.dot(np.diag(wi).dot(shp.T))

        # elemental b
        b = (shpx.T * np.ones((gorder, p))).T.dot(np.diag(wi).dot(shp.T))

        # elemental h
        h = np.zeros((2, 2))
        h[0, 0], h[-1, -1] = -con - self.tau_pos, con - self.tau_neg
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
        g[0, 0], g[-1, -1] = self.tau_pos, self.tau_neg
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
        e[0, 0], e[-1, -1] = -con - self.tau_pos, con - self.tau_neg
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
        R[0] = self.bc(0)[0]
        R[-1] = -self.bc(0)[1]
        return d, m, E, G, H, F, L, a, b, C, R

    def solve_local(self, index, x):
        """ solve the 1d advection equation wit local HDG"""
        d, _, E, G, H, F, L, a, b, C, R = self.matrix_gen(index, x)
        # find dx
        dx = np.zeros(self.numEle + len(index))

        for i in range(1, self.numEle + len(index) + 1):
            x_i, dx_i, x_n = self.mesh(i, index, x)
            dx[i - 1] = dx_i

        # assemble global D
        bb = np.zeros((self.numBasisFuncs, self.numBasisFuncs))
        bb[0, 0] = self.tau_pos
        bb[-1, -1] = self.tau_neg
        D = np.repeat(dx, self.numBasisFuncs) / 2 * block_diag(*[d] * (
            self.numEle + len(index))) + block_diag(*[bb] * (self.numEle + len(index)))
        # assemble global A
        A = np.repeat(dx, self.numBasisFuncs) / 2 * block_diag(*
                                                               [a] * (self.numEle + len(index)))

        # assemble global B
        B = block_diag(*[b] * (self.numEle + len(index)))
        # solve U and \lambda
        K = -np.concatenate((C.T, G), axis=1).dot(np.linalg.inv(
            np.bmat([[A, -B], [B.T, D]])).dot(np.concatenate((C, E)))) + H
        F_hat = np.array([L]).T - np.concatenate((C.T, G), axis=1).dot(np.linalg.inv(
            np.bmat([[A, -B], [B.T, D]]))).dot(np.array([np.concatenate((R, F))]).T)
        lamba = np.linalg.solve(K, F_hat)
        U = np.linalg.inv(np.bmat([[A, -B], [B.T, D]])).dot(
            np.array([np.concatenate((R, F))]).T - np.concatenate((C, E)).dot(lamba))
        return U, lamba

    def solve_adjoint(self, index, x, u, u_hat):
        self.numBasisFuncs = self.numBasisFuncs + 1
        d, _, E, G, H, F, L, a, b, C, R = self.matrix_gen(index, x)

        # add boundary
        F = np.zeros(len(F))
        R[-1] = -self.bc(1)[1]

        # find dx
        dx = np.zeros(self.numEle + len(index))
        for i in range(1, self.numEle + len(index) + 1):
            x_i, dx_i, x_n = self.mesh(i, index, x)
            dx[i - 1] = dx_i
        # assemble global D
        bb = np.zeros((self.numBasisFuncs, self.numBasisFuncs))
        bb[0, 0] = self.tau_pos
        bb[-1, -1] = self.tau_neg
        D = np.repeat(dx, self.numBasisFuncs) / 2 * block_diag(*[d] * (
            self.numEle + len(index))) + block_diag(*[bb] * (self.numEle + len(index)))

        # assemble global A
        A = np.repeat(dx, self.numBasisFuncs) / 2 * block_diag(*
                                                               [a] * (self.numEle + len(index)))

        # assemble global B
        B = block_diag(*[b] * (self.numEle + len(index)))

        # # assemble global matrix LHS
        LHS = np.bmat([[A, -B, C], [B.T, D, E], [C.T, G, H]])

        # solve U and \lambda
        U = np.linalg.solve(LHS.T, np.concatenate((R, F, L)))

        return U[0:2 * self.numBasisFuncs * (self.numEle + len(index))], U[2 * self.numBasisFuncs * (self.numEle + len(index)):len(U)]

    def diffusion(self):
        """solve 1d convection with local HDG"""

        # begin and end time
        t, T = 0, 1

        # time marching step for diffusion equation
        dt = 1e-3

        d, m, E, G, H, F, L, a, b, C, R = self.matrix_gen()
        # add time derivatives to the space derivatives (both are
        # elmental-wise)
        d = d + 1 / dt * m
        # assemble global D
        D = block_diag(*[d] * self.numEle)
        # assemble global A
        A = block_diag(*[a] * self.numEle)
        # assemble global B
        B = block_diag(*[b] * self.numEle)

        # initial condition
        X = np.zeros(self.numBasisFuncs * self.numEle)
        for i in range(1, self.numEle + 1):
            x = self.mesh(i)
            X[(i - 1) * self.numBasisFuncs:(i - 1) *
              self.numBasisFuncs + self.numBasisFuncs] = x
        U = np.concatenate((pi * cos(pi * X), sin(pi * X)))

        # assemble M
        M = block_diag(*[1 / dt * m] * self.numEle)

        # time marching
        while t < T:
            # add boundary conditions
            F_dynamic = F + \
                M.dot(U[self.numEle * self.numBasisFuncs:2 *
                        self.numEle * self.numBasisFuncs])

            # assemble global matrix LHS
            LHS = np.bmat([[A, -B, C], [B.T, D, E], [C.T, G, H]])

            # solve U and \lambda
            U = np.linalg.solve(LHS, np.concatenate((R, F_dynamic, L)))

            # plot solutions
            plt.clf()
            plt.plot(X, U[self.numEle * self.numBasisFuncs:2 *
                          self.numEle * self.numBasisFuncs], '-r.')
            plt.plot(X, sin(pi * X) * np.exp(-pi**2 * t))
            plt.ylim([0, 1])
            plt.grid()
            plt.pause(1e-3)

        plt.close()
        print("Diffusion equation du/dt - du^2/d^2x = 0 with u_exact ="
              ' 6sin(pi*x)*exp(-pi^2*t).')
        plt.figure(1)
        plt.plot(X, U[self.numEle * self.numBasisFuncs:2 *
                      self.numEle * self.numBasisFuncs], '-r.')
        plt.plot(X, sin(pi * X) * np.exp(-pi**2 * T))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(('Numberical', 'Exact'), loc='upper left')
        plt.title('Simple Diffusion Equation Solution at t = {}'.format(T))
        plt.grid()
        plt.savefig('diffusion', bbox_inches='tight')
        plt.show(block=False)
        return U

    def residual(self, U, hat_U, z, hat_z, dx, index, x_c):
        n_ele = self.numEle

        # order of polynomial shape functions
        p = self.numBasisFuncs

        p_l = p - 1
        # order of gauss quadrature
        gorder = 2 * p
        # shape function and gauss quadrature
        xi, wi = np.polynomial.legendre.leggauss(gorder)
        shp, shpx = self.shape(xi, p)
        shp_l, shpx_l = self.shape(xi, p_l)
        # ---------------------------------------------------------------------
        # advection constant
        con = self.c

        # diffusion constant
        kappa = self.kappa

        z_q, z_u, z_hat = np.zeros(self.numBasisFuncs * self.numEle), \
            np.zeros(self.numBasisFuncs *
                     self.numEle), np.zeros(self.numEle - 1)

        q, u, lamba = np.zeros(p_l * self.numEle), \
            np.zeros(p_l * self.numEle), np.zeros(self.numEle - 1)

        for i in np.arange(self.numBasisFuncs * self.numEle):
            z_q[i] = z[i]
            z_u[i] = z[i + self.numBasisFuncs * self.numEle]

        for i in np.arange(p_l * self.numEle):
            q[i] = U[i]
            u[i] = U[i + p_l * self.numEle]

        for i in np.arange(self.numEle - 1):
            z_hat[i] = hat_z[i]

        # add boundary condtions to U_hat
        U_hat = np.zeros(self.numEle + 1)
        for i, x in enumerate(hat_U):
            U_hat[i + 1] = x
        U_hat[0] = self.bc(0)[0]
        U_hat[-1] = self.bc(0)[1]

        # L, easy in 1d
        L = np.zeros(n_ele + 1)

        # R, easy in 1d
        RR = np.zeros(p * n_ele)

        # elemental forcing vector
        F = np.zeros(p * n_ele)
        for i in range(1, n_ele + 1):
            f = dx[i - 1] / 2 * \
                shp.dot(
                    wi * self.forcing(x_c[0] + 1 / 2 * (1 + xi) * dx[i - 1]))
            F[(i - 1) * p:(i - 1) * p + p] = f

        # elemental h
        h = np.zeros((2, 2))
        h[0, 0], h[-1, -1] = -con - self.tau_pos, con - self.tau_neg
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
        g = np.zeros((2, p_l))
        g[0, 0], g[-1, -1] = self.tau_pos, self.tau_neg
        # mapping matrix
        map_g_x = map_h
        map_g_y = np.arange(p_l * n_ele, dtype=int).reshape(n_ele, p_l).T
        # assemble global G
        G = np.zeros((n_ele + 1, p_l * n_ele))
        for i in range(n_ele):
            for j in range(2):
                m = map_g_x[j, i]
                for k in range(p_l):
                    n = map_g_y[k, i]
                    G[m, n] += g[j, k]
        G = G[1:n_ele, :]

        # elemental c
        c = np.zeros((p_l, 2))
        c[0, 0], c[-1, -1] = -1, 1

        # mapping matrix
        map_e_x = np.arange(p_l * n_ele, dtype=int).reshape(n_ele, p_l).T
        map_e_y = map_h

        # assemble global C
        C = np.zeros((p_l * n_ele, n_ele + 1))
        for i in range(n_ele):
            for j in range(p_l):
                m = map_e_x[j, i]
                for k in range(2):
                    n = map_e_y[k, i]
                    C[m, n] += c[j, k]
        C = C[:, 1:n_ele]

        # L, easy in 1d
        L = np.zeros(n_ele - 1)

        # residual vector
        R = np.zeros(self.numEle)
        for i in np.arange(self.numEle):
            a = dx[i] / 2 * 1 / kappa * \
                ((shp.T).T).dot(np.diag(wi).dot(shp_l.T))

            b = ((shpx.T) * np.ones((gorder, p))
                 ).T.dot(np.diag(wi).dot(shp_l.T))

            b_t = ((shpx_l.T) * np.ones((gorder, p_l))
                   ).T.dot(np.diag(wi).dot(shp.T))

            d = dx[i] / 2 * shp.dot(np.diag(wi).dot(shp_l.T))
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
        self.numBasisFuncs = self.numBasisFuncs - 1

        # global error
        R_g = (C.T.dot(q) + G.dot(u) + H.dot(U_hat[1:-1])).dot(1 - z_hat)
        return np.abs(np.sum(R) + np.sum(R_g)), refine_index + 1

    def adaptive(self):
        x = np.linspace(0, 1, self.numEle + 1)
        index = []
        U, hat_U = self.solve_local(index, x)
        U_adjoint, hat_adjoint = self.solve_adjoint(index, x, U, hat_U)
        X = np.zeros(self.numBasisFuncs * self.numEle)
        dx = np.zeros(self.numEle + len(index))
        for i in range(1, self.numEle + 1):
            x_i, dx_i, x_n = self.mesh(i, index, x)
            X[(i - 1) * self.numBasisFuncs:(i - 1) *
              self.numBasisFuncs + self.numBasisFuncs] = x_i
            dx[i - 1] = dx_i

        numAdaptive = 28
        trueError = np.zeros((2, numAdaptive))
        estError = np.zeros((2, numAdaptive))
        for i in np.arange(numAdaptive):
            est_error, index = self.residual(
                U, hat_U, U_adjoint, hat_adjoint, dx, index, x)
            index = index.tolist()
            U, hat_U = self.solve_local(index, x)
            U_adjoint, hat_adjoint = self.solve_adjoint(index, x, U, hat_U)
            self.numBasisFuncs = self.numBasisFuncs - 1
            X = np.zeros(self.numBasisFuncs * (self.numEle + len(index)))
            dx = np.zeros(self.numEle + len(index))
            for j in range(1, self.numEle + len(index) + 1):
                x_i, dx_i, x_n = self.mesh(j, index, x)
                X[(j - 1) * self.numBasisFuncs:(j - 1) *
                  self.numBasisFuncs + self.numBasisFuncs] = x_i
                dx[j - 1] = dx_i
            x = x_n
            estError[0, i] = self.numEle
            estError[1, i] = est_error

            self.numEle = self.numEle + len(index)

            # U_1d = np.zeros(len(U))
            # for j in np.arange(len(U)):
            #     U_1d[j] = U[j]
            # Unum = np.array([])
            # Xnum = np.array([])
            # Qnum = np.array([])
            # for j in range(1, self.numEle + 1):
            #     # Gauss quadrature
            #     gorder = 10 * self.numBasisFuncs
            #     xi, wi = np.polynomial.legendre.leggauss(gorder)
            #     shp, shpx = self.shape(xi, self.numBasisFuncs)
            #     Xnum = np.hstack((Xnum, (X[(j - 1) * self.numBasisFuncs + self.numBasisFuncs - 1] + X[(j - 1) * self.numBasisFuncs]) / 2 + (
            #         X[(j - 1) * self.numBasisFuncs + self.numBasisFuncs - 1] - X[(j - 1) * self.numBasisFuncs]) / 2 * xi))
            #     Unum = np.hstack(
            #         (Unum, shp.T.dot(U_1d[int(len(U) / 2) + (j - 1) * self.numBasisFuncs:int(len(U) / 2) + j * self.numBasisFuncs])))
            #     Qnum = np.hstack(
            #         (Qnum, shp.T.dot(U_1d[int((j - 1) * self.numBasisFuncs):j * self.numBasisFuncs])))
            # if i in [0, 4, 9, 19]:
            #     plt.plot(Xnum, Unum, '-', color='C3')
            #     plt.plot(X, U[int(len(U) / 2):len(U)], 'C3.')
            #     plt.xlabel('$x$', fontsize=17)
            #     plt.ylabel('$u$', fontsize=17)
            #     plt.axis([-0.05, 1.05, 0, 1.3])
            #     plt.grid()
            #     # plt.savefig('u_test_{}.pdf'.format(i+1))
            #     plt.show()
            #     plt.clf()
            trueError[0, i] = self.numEle
            trueError[1, i] = np.abs(
                U[self.numEle * self.numBasisFuncs - 1] - np.sqrt(self.kappa))
            self.numBasisFuncs = self.numBasisFuncs + 1
        self.trueError = trueError
        self.estError = estError
