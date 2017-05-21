# content of test_module.py
import pytest
import math
import matplotlib
matplotlib.use('Agg')           # supress figures in the following modules
from hdpg1d.preprocess import coefficients as coeff
from hdpg1d.adaptation import hdpg1d


testData = [
    ([1e-4, 0, 1, 2, 2, 1, 1, 1e-10, 50], 1e-2),  # diffusion reaction
    ([0, 1, 0, 2, 2, 1, 1, 1e-10, 50], 0),        # convection
    # ([1, 1, 0, 2, 2, 1, 1,1e-10,50], 1)         # diffusion convection)
]


class TestClass(object):
    @pytest.fixture(scope="module", params=testData)
    def coeffGen(self, request):
        coeffTest = coeff(*request.param[0])
        expected = request.param[1]
        yield coeffTest, expected  # teardown

    def test_zeroDivision(self, monkeypatch):
        coeffTest = coeff(*([0] * 9))
        assert coeffTest.DIFFUSION != 0

    def test_solveAdaptive(self, coeffGen):
        coeffTest, expected = coeffGen
        hdpgTest = hdpg1d(coeffTest)
        hdpgTest.adaptive()
        # get the target function value
        # and compare to the expected value
        soln = hdpgTest.trueErrorList[1][-1]
        assert math.isclose(soln, expected, rel_tol=1e-5, abs_tol=1e-10)

    def test_solvePrimal(self, coeffGen):
        coeffTest, expected = coeffGen
        # test the primal solver on a refined mesh and higher poly order
        coeffTest.pOrder = 5
        coeffTest.numEle = 300
        hdpgTest = hdpg1d(coeffTest)
        hdpgTest.solvePrimal()
        soln = hdpgTest.primalSoln[hdpgTest.numEle *
                                   hdpgTest.numBasisFuncs - 1]
        assert math.isclose(soln, expected, rel_tol=1e-5, abs_tol=1e-10)
