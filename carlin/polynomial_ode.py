r"""
A class to represent a system of polynomial ordinary differential equations (ODEs).

AUTHOR:

- Marcelo Forets (May 2017 at VERIMAG - UGA)
"""

#************************************************************************
#       Copyright (C) 2016 Marcelo Forets <mforets@nonlinearnotes.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# any later version.
#                  http://www.gnu.org/licenses/
#************************************************************************

class PolynomialODE(SageObject):
    """
    This class represents a finite set of polynomial ODEs.
    """
    def __init__(self, f=None, n=None, k=None):
        self._funcs = f
        self._dim = n
        self._degree = k
        
    def __repr__(self):
        return "A Polynomial ODE in n = %s variables"%self.dim
    
    def funcs(self):
        """
        Get the right-hand side functions list.
        """
        return self._funcs

    def dim(self):
        """
        Get the dimesion (number of variables).
        """
        return self._dim

    def degree(self):
        """
        Get the degree (max over the degree of all polynomials).
        """
        return self._degree
    deg=degree