# content of test_module.py
import pytest
import math
import matplotlib
matplotlib.use('Agg')           # supress figures in the following modules
from hdpg1d.coefficients import coefficients as coeff
from hdpg1d.adaptation import hdpg1d


class TestClass(object):
    def test_zeroDivision(self, monkeypatch):
        coeffTest = coeff(*([0] * 7))
        assert coeffTest.DIFFUSION != 0

    @pytest.mark.parametrize("coeffInput, expected", [
        ([1e-4, 0, 1, 2, 2, 1, 1], 1e-2),  # diffusion reaction
        ([0, 1, 0, 2, 2, 1, 1], 0),        # convection
        # ([1, 1, 0, 2, 2, 1, 1], 1)         # diffusion convection
    ])
    def test_solveAdaptive(self, coeffInput, expected):
        coeffTest = coeff(*coeffInput)
        hdpgTest = hdpg1d(coeffTest)
        hdpgTest.adaptive()
        # get the target function value
        # and compare to the expected value
        soln = hdpgTest.trueErrorList[1][-1]
        assert math.isclose(soln, expected, rel_tol=1e-5, abs_tol=1e-10)

    @pytest.mark.parametrize("coeffInput, expected", [
        ([1e-4, 0, 1, 2, 2, 1, 1], 1e-2),  # diffusion reaction
        ([0, 1, 0, 2, 2, 1, 1], 0),        # convection
        # ([1, 1, 0, 2, 2, 1, 1], 1)         # diffusion convection
    ])
    def test_solvePrimal(self, coeffInput, expected):
        # test the primal solver with a refined mesh and poly order
        coeffInput[3] = 5
        coeffInput[4] = 200
        coeffTest = coeff(*coeffInput)
        hdpgTest = hdpg1d(coeffTest)
        hdpgTest.solvePrimal()
        soln = hdpgTest.primalSoln[hdpgTest.numEle *
                                   hdpgTest.numBasisFuncs - 1]
        assert math.isclose(soln, expected, rel_tol=1e-5, abs_tol=1e-10)
