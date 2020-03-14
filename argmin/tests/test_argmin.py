
import pytest
import numpy as np
from numpy.testing import assert_allclose

import argmin


@pytest.mark.parametrize('solver_name', ['lbfgs'])
def test_rosen_lbfgs(solver_name):
    # Adapted from scipy.optimize.tests.test_optimize

    scipy_opt = pytest.importorskip('scipy.optimize')

    class Rosen():
        def apply(self, x):
            res = scipy_opt.rosen(x)
            print('Roman.apply', x, res)
            return res

        def gradient(self, x):
            res = scipy_opt.rosen_der(x)
            print('Rosen.gradient', x, res)
            return res

    prob = Rosen()

    x0 = np.array([-1.2, 1.0])

    if solver_name == 'lbfgs':
        solver = argmin.lbfgs(10)
    else:
        raise NotImplementedError

    executor = argmin.executor(prob, solver, x0, max_iter=100)
    x_opt = executor.run()
    assert_allclose(x_opt, np.array([1, 1]), rtol=1e-4)
