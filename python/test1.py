#!/usr/bin/python
"""
fu
"""

import numpy
import pyargmin as argmin

class Problem:
    """blah"""

    def __init__(self):
        """Constructor (duh)"""
        self.a = 1.0
        self.b = 100.0

    def apply(self, x):
        """apply"""
        return x[0]*self.a + x[1]*self.b


def blah():
    """blah"""
    print("fu")


def blah2(num):
    """blah"""
    print(num)

prob = Problem()

#  argmin.closure(blah)
#  argmin.closure2(blah2)
argmin.closure3(prob)
