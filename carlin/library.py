r"""
Library of commonly used or famous polynomial ODE systems.

AUTHOR:

- Marcelo Forets (May 2017)
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

# Sage objects: Rings and Polynomials
from sage.rings.integer import Integer
from sage.rings.all import RR, QQ
from sage.rings.real_double import RDF
from sage.rings.polynomial.polynomial_ring import polygens 

def vanderpol(mu, omega):
    """
    The Van der Pol oscillator is a non-conservative system with non-linear damping.
    
    It is defined as:

    .. MATH::

        \begin{aligned}
        x' &= y \\
        y' &= -\omega^2  x - (x^2 - 1) \mu y
        \end{aligned}

    where `\omega` is the natural frequency and `\mu` is the damping parameter.
    For additional information see the :wiki:`Van_der_Pol_oscillator`.

    EXAMPLES::

        sage: from carlin.library import vanderpol
        sage: vanderpol(SR.var('mu'), SR.var('omega'))
        ([x1, -omega^2*x0 - (x0^2 - 1)*mu*x1], 2, 3)
    """
    # dimension of state-space
    n=2

    # define the vector of symbolic variables
    x = polygens(QQ, ['x'+str(i) for i in range(n)])

    # vector field (n-dimensional)
    f = [None] * n

    f[0] = x[1]
    f[1] = - omega**2 * x[0] + mu * (1 - x[0]**2) * x[1]

    # order of the ODE
    k = 3

    return f, n, k