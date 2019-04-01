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

import numpy
import pyargmin as argmin


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
        return param[0]*self.a + param[1]*self.b


prob = Problem()

argmin.closure(prob)
argmin.closure3(blah)
