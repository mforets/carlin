r"""
Library of commonly used or famous polynomial ODE systems.

The following functions are available:

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :func:`~vanderpol`            | `Van der Pol oscillator <https://en.wikipedia.org/wiki/Van_der_Pol_oscillator>`_
    :func:`~scalar_cubic`         | A scalar ODE with a cubic term
    :func:`~scalar_quadratic`     | A scalar ODE with a quadratic term

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

def vanderpol(mu=1, omega=1):
    r"""
    The Van der Pol oscillator is a non-conservative system with non-linear damping.
    
    It is defined as:

    .. MATH::

        \begin{aligned}
        x' &= y \\
        y' &= -\omega^2  x - (x^2 - 1) \mu y
        \end{aligned}

    where `\omega` is the natural frequency and `\mu` is the damping parameter.
    For additional information see the :wikipedia:`Van_der_Pol_oscillator`.

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

def scalar_cubic(a, b):
    r"""
    A scalar ODE with a cubic term.
    
    It is defined as:

    .. MATH::

        x'(t) = -ax(t) + bx(t)^3

    where `a` and `b` are paremeters of the ODE.

    EXAMPLES::

        sage: from carlin.library import scalar_cubic
        sage: scalar_cubic()
        ([x0^3 - x0], 1, 3)

    Compute the Carleman embedding truncated at order `N=4`::

        sage: from carlin.transformation import get_Fj_from_model, truncated_matrix 
        sage: Fj = get_Fj_from_model(*scalar_cubic())
        sage: matrix(truncated_matrix(4, *Fj, input_format="Fj_matrices").toarray())
        [-1.0  0.0  1.0  0.0]
        [ 0.0 -2.0  0.0  2.0]
        [ 0.0  0.0 -3.0  0.0]
        [ 0.0  0.0  0.0 -4.0]
    """
    # define the vector of symbolic variables
    x = polygens(QQ, ['x'+str(0)])
    f = [None] * 1
    f[0] = - a * x[0] + b * x[0]**3
    return f, 1, 3

def scalar_quadratic(a, b):
    r"""
    A scalar ODE with a quadratic term.

    It is defined as:

    .. MATH::

        x'(t) = ax(t) + bx(t)^2

    where `a` and `b` are paremeters of the ODE.

    EXAMPLES::

        sage: from carlin.library import scalar_cubic
        sage: scalar_quadratic()
        ([x0^3 - x0], 1, 3)

    Compute the Carleman embedding truncated at order `N=4`::

        sage: from carlin.transformation import get_Fj_from_model, truncated_matrix 
        sage: Fj = get_Fj_from_model(*scalar_cubic())
        sage: matrix(truncated_matrix(4, *Fj, input_format="Fj_matrices").toarray())
        [-1.0  0.0  1.0  0.0]
        [ 0.0 -2.0  0.0  2.0]
        [ 0.0  0.0 -3.0  0.0]
        [ 0.0  0.0  0.0 -4.0]
    """
    # define the vector of symbolic variables
    x = polygens(QQ, ['x'+str(0)])
    f = [None] * 1
    f[0] =  a * x[0] + b * x[0]**2
    return f, 1, 2
