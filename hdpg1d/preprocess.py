import os
import json
import ast
import operator as op
import numpy as np
from collections import namedtuple
from scipy.linalg import block_diag

# load the configuration file
installDir = os.path.split(__file__)[0]
cfgPath = os.path.join(installDir, "config")
for loc in cfgPath, os.curdir, os.path.expanduser("~"):
    try:
        with open(os.path.join(loc, "config.json")) as source:
            configdata = json.load(source)
    except IOError:
        pass

# evaluate the input json function with only these math operators
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}


def eval_expr(expr):
    return eval_(ast.parse(expr, mode='eval').body)


def eval_(node):
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, "x"):
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)


def queryYesNo(question, default="yes"):
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        print(question + prompt, end='')
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' "
                  "(or 'y' or 'n').\n")


def setDefaultCoefficients():
    question = "Do you want to use the default parameters?"
    isDefault = queryYesNo(question, "yes")
    if (isDefault):
        diffDefault = configdata["coefficients"]["diffusion"]
        convDefault = configdata["coefficients"]["convection"]
        reactionDefault = configdata["coefficients"]["reaction"]
        pOrderDefault = configdata["coefficients"]["pOrder"]
        numEleDefault = configdata["coefficients"]["numEle"]
        tauPlusDefault = configdata["coefficients"]["tauPlus"]
        tauMinusDefault = configdata["coefficients"]["tauMinus"]
        coeff = coefficients(diffDefault, convDefault, reactionDefault,
                             pOrderDefault, numEleDefault,
                             tauPlusDefault, tauMinusDefault)
    else:
        coeff = coefficients.fromInput()
    return coeff


def shape(x, p):
    """generate p shape functions and its first order derivative
    at the given location x. x can be an array"""
    A = np.array([np.linspace(-1, 1, p)]).T**np.arange(p)
    C = np.linalg.inv(A).T
    x = np.array([x]).T
    shp = C.dot((x**np.arange(p)).T)
    shpx = C[:, 1::1].dot((x**np.arange(p - 1) * np.arange(1, p)).T)
    return shp, shpx


def forcing(x):
    f = np.zeros(len(x))
    for i, forcingItem in enumerate(x):
        forcingExpr = configdata["forcing"]
        # replace the 'x' in the json file with the function parameters
        f[i] = eval_expr(forcingExpr.replace("x", str(forcingItem)))
    return f


def boundaryCondition(case):
    if case == 'primal':
        # primal problem
        bcLeft = configdata["boundary"]["left"]
        bcRight = configdata["boundary"]["right"]
        bc = [bcLeft, bcRight]
    if case == 'adjoint':
        # adjoint problem
        bc = [0, 1]
    return bc


class coefficients:
    def __init__(self, diff, conv, reaction, pOrder, numEle, tauPlus, tauMinus):
        if diff == 0:
            # set the diffusion constant to a small number
            # to avoid division by zero error
            diff = 1e-16
        self.DIFFUSION = diff
        self.CONVECTION = conv
        self.REACTION = reaction
        self.pOrder = pOrder
        self.numEle = numEle
        self.TAUPLUS = tauPlus
        self.TAUMINUS = tauMinus

    @classmethod
    def fromInput(cls):
        while True:
            try:
                print("Please provide the following coefficients.")
                diff = float(input("Diffusion constant (float): "))
                conv = float(input("Covection constant (float): "))
                reaction = float(input("Reaction constant (float): "))
                pOrder = int(input("Order of polynomials (int): "))
                numEle = int(input("Number of elements (int): "))
                tauPlus = float(input("Stablization parameter plus (float): "))
                tauMinus = float(
                    input("Stablization parameter minus (float): "))
            except ValueError:
                print("Sorry, wrong data type.")
                continue
            else:
                print("Something is wrong. Exit.")
                break
        return cls(diff, conv, reaction, pOrder, numEle, tauPlus, tauMinus)


class discretization(object):
    """Given the problem statement and current mesh,
    construct the discretization matrices"""

    def __init__(self, coeff, mesh, enrich=None):
        self.mesh = mesh
        self.coeff = coeff
        self.enrich = enrich
        # the following init are for the sake of simplicity
        self.numBasisFuncs = coeff.pOrder + 1
        self.tau_pos = coeff.TAUPLUS
        self.tau_neg = coeff.TAUMINUS
        self.conv = coeff.CONVECTION
        self.kappa = coeff.DIFFUSION
        self.numEle = len(mesh) - 1
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
        dist = np.zeros(self.numEle)
        for i in range(1, self.numEle + 1):
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
        F = np.zeros(numBasisFuncs * self.numEle)
        for i in range(1, self.numEle + 1):
            f = self.dist[i - 1] / 2 \
                * shp.dot(self.wi * forcing(self.mesh[i - 1] +
                                            1 / 2 * (1 + self.xi) *
                                            self.dist[i - 1]))
            F[(i - 1) * numBasisFuncs:i * numBasisFuncs] = f
        F[0] += (self.conv + self.tau_pos) * boundaryCondition('primal')[0]
        F[-1] += (-self.conv + self.tau_neg) * boundaryCondition('primal')[1]
        # L, easy in 1d
        L = np.zeros(self.numEle - 1)
        # R, easy in 1d
        R = np.zeros(numBasisFuncs * self.numEle)
        R[0] = boundaryCondition('primal')[0]
        R[-1] = -boundaryCondition('primal')[1]
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
            # BonQ is only used in calculating DWR residual
            BonQ = block_diag(*[b] * (self.numEle)).T
        else:
            numBasisFuncsEnrich = self.numBasisFuncs
            shpEnrich = self.shp
            shpxEnrich = self.shpx
            BonQ = None
        a = 1 / self.coeff.DIFFUSION * \
            shpEnrich.dot(np.diag(self.wi).dot(self.shp.T))
        A = np.repeat(self.dist, self.numBasisFuncs) / \
            2 * block_diag(*[a] * (self.numEle))
        b = (shpxEnrich.T * np.ones((self.gqOrder, numBasisFuncsEnrich))
             ).T.dot(np.diag(self.wi).dot(self.shp.T))
        B = block_diag(*[b] * (self.numEle))
        d = self.coeff.REACTION * \
            shpEnrich.dot(np.diag(self.wi).dot(self.shp.T))
        # assemble D
        dFace = np.zeros((numBasisFuncsEnrich, self.numBasisFuncs))
        dFace[0, 0] = self.tau_pos
        dFace[-1, -1] = self.tau_neg
        dConv = -self.conv * (shpxEnrich.T * np.ones((self.gqOrder,
                                                      numBasisFuncsEnrich)))\
            .T.dot(np.diag(self.wi).dot(self.shp.T))
        D = np.repeat(self.dist, self.numBasisFuncs) / 2 * \
            block_diag(*[d] * (self.numEle)) + \
            block_diag(*[dFace] * (self.numEle)) +\
            block_diag(*[dConv] * (self.numEle))
        eleMat = namedtuple('eleMat', ['A', 'B', 'BonU', 'D'])
        return eleMat(A, B, BonQ, D)

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
        map_h = np.zeros((2, self.numEle), dtype=int)
        map_h[:, 0] = np.arange(2)
        for i in np.arange(1, self.numEle):
            map_h[:, i] = np.arange(
                map_h[2 - 1, i - 1], map_h[2 - 1, i - 1] + 2)
        # assemble H and eliminate boundaries
        H = np.zeros((self.numEle + 1, self.numEle + 1))
        for i in range(self.numEle):
            for j in range(2):
                m = map_h[j, i]
                for k in range(2):
                    n = map_h[k, i]
                    H[m, n] += h[j, k]
        H = H[1:self.numEle][:, 1:self.numEle]

        # elemental e
        e = np.zeros((numBasisFuncs, 2))
        e[0, 0], e[-1, -1] = -conv - tau_pos, conv - tau_neg
        # mapping matrix
        map_e_x = np.arange(numBasisFuncs * self.numEle,
                            dtype=int).reshape(self.numEle,
                                               numBasisFuncs).T
        map_e_y = map_h
        # assemble global E
        E = np.zeros((numBasisFuncs * self.numEle, self.numEle + 1))
        for i in range(self.numEle):
            for j in range(numBasisFuncs):
                m = map_e_x[j, i]
                for k in range(2):
                    n = map_e_y[k, i]
                    E[m, n] += e[j, k]
        E = E[:, 1:self.numEle]

        # elemental c
        c = np.zeros((numBasisFuncs, 2))
        c[0, 0], c[-1, -1] = -1, 1
        # assemble global C
        C = np.zeros((numBasisFuncs * self.numEle, self.numEle + 1))
        for i in range(self.numEle):
            for j in range(numBasisFuncs):
                m = map_e_x[j, i]
                for k in range(2):
                    n = map_e_y[k, i]
                    C[m, n] += c[j, k]
        C = C[:, 1:self.numEle]

        # elemental g
        g = np.zeros((2, numBasisFuncs))
        g[0, 0], g[-1, -1] = tau_pos, tau_neg
        # mapping matrix
        map_g_x = map_h
        map_g_y = np.arange(numBasisFuncs * self.numEle,
                            dtype=int).reshape(self.numEle,
                                               numBasisFuncs).T
        # assemble global G
        G = np.zeros((self.numEle + 1, numBasisFuncs * self.numEle))
        for i in range(self.numEle):
            for j in range(2):
                m = map_g_x[j, i]
                for k in range(numBasisFuncs):
                    n = map_g_y[k, i]
                    G[m, n] += g[j, k]
        G = G[1:self.numEle, :]
        faceMat = namedtuple('faceMat', ['C', 'E', 'G', 'H'])
        return faceMat(C, E, G, H)
