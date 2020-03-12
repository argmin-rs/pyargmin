
import pytest
import numpy as np
from numpy.testing import assert_allclose

import argmin


def test_rosen():
    # Adapted from scipy.optimize.tests.test_optimize

    scipy_opt = pytest.importorskip('scipy.optimize')

    class Rosen():
        def apply(self, x):
            res = scipy_opt.rosen(x)
            print('cost', x, res)
            return res

        def gradient(self, x):
            res = scipy_opt.rosen_der(x)
            print('grad', x, res)
            return res

    prob = Rosen()

    x0 = np.array([-1.2, 1.0])

    solver = argmin.landweber(1e-3)

    executor = argmin.executor(prob, solver, x0)
    x_opt = executor.run()
    print("Result:", x_opt)
    #assert_allclose(x_opt, np.array([1, 1]), rtol=1e-4)
