#!/usr/bin/python

# Copyright 2019 Stefan Kroboth
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

"""
fu
"""

import numpy as np
import argmin


def blah(inp):
    print(inp)


class Problem:
    """blah"""

    def __init__(self):
        """Constructor (duh)"""
        self.a = 1.0
        self.b = 100.0

    def apply(self, param):
        """apply"""
        out = (self.a - param[0])**2 + self.b * (param[1] - param[0]**2)**2
        print(out)
        return out

    def gradient(self, param):
        x = param[0]
        y = param[1]
        out = np.array(
                [-2.0 * self.a + 4.0 * self.b * x**3 - 4.0 * self.b * x * y
                 + 2.0 * x, 2.0 * self.b * (y - x**2)])
        return out


prob = Problem()

#  argmin.closure(prob)
#  argmin.closure3(blah)

solver = argmin.landweber(0.1)
print(solver)
solver.set_omega(3.0)

executor = argmin.executor(prob, solver, np.array([1.2, 1.2]))
print(executor)
executor.run()
