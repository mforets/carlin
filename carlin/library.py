r"""
Library of commonly used or famous polynomial ODE systems.

The following functions are available:

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :func:`~arrowsmith_and_place_fig_3_5e_page_79`     | Nonlinear two-dimensional system with an hyperbolic fixed point
    :func:`~biomodel_2`                                | Nine-dimensional polynomial ODE form a biological model
    :func:`~chen_seven_dim`                            | Seven-dimensional nonlinear system of quadratic order
    :func:`~scalar_cubic`                              | Scalar ODE with a cubic term
    :func:`~scalar_quadratic`                          | Scalar ODE with a quadratic term
    :func:`~vanderpol`                                 | `Van der Pol oscillator <https://en.wikipedia.org/wiki/Van_der_Pol_oscillator>`_

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

from carlin.polynomial_ode import PolynomialODE

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
        sage: vanderpol(SR.var('mu'), SR.var('omega')).funcs()
        [x1, -omega^2*x0 - (x0^2 - 1)*mu*x1]
    """
    # dimension of state-space
    n=2

    # define the vector of symbolic variables
    x = polygens(QQ, ['x'+str(i) for i in range(n)])

    # vector field (n-dimensional)
    f = [None] * n

    f[0] = x[1]
    f[1] = - omega**2 * x[0] + mu * (1 - x[0]**2) * x[1]

    # the order is k=3
    return PolynomialODE(f, n, k=3)

def scalar_cubic(a=1, b=1):
    r"""
    A scalar ODE with a cubic term.
    
    It is defined as:

    .. MATH::

        x'(t) = -ax(t) + bx(t)^3

    where `a` and `b` are paremeters of the ODE.

    EXAMPLES::

        sage: from carlin.library import scalar_cubic
        sage: C = scalar_cubic()
        sage: C.funcs()
        [x0^3 - x0]

    Compute the Carleman embedding truncated at order `N=4`::

        sage: from carlin.transformation import get_Fj_from_model, truncated_matrix 
        sage: Fj = get_Fj_from_model(Q.funcs(), Q.dim(), Q.degree())
        sage: matrix(truncated_matrix(4, *Fj, input_format="Fj_matrices").toarray())
        [-1.0  0.0  1.0  0.0]
        [ 0.0 -2.0  0.0  2.0]
        [ 0.0  0.0 -3.0  0.0]
        [ 0.0  0.0  0.0 -4.0]
    """
    # define the vector of symbolic variables
    x = polygens(QQ, ['x'+str(0)])
    f = [None] * 1
    f[0] = -a * x[0] + b * x[0]**3
    return PolynomialODE(f, n=1, k=3)

def scalar_quadratic(a=1, b=1):
    r"""
    A scalar ODE with a quadratic term.

    It is defined as:

    .. MATH::

        x'(t) = ax(t) + bx(t)^2

    where `a` and `b` are paremeters of the ODE.

    EXAMPLES::

        sage: from carlin.library import scalar_quadratic
        sage: Q = scalar_quadratic()
        A Polynomial ODE in n = 1 variables
        sage: Q.funcs()
        [x0^2 + x0]

    Compute the Carleman embedding truncated at order `N=4`::

        sage: from carlin.transformation import get_Fj_from_model, truncated_matrix 
        sage: Fj = get_Fj_from_model(Q.funcs(), Q.dim(), Q.degree())
        sage: matrix(truncated_matrix(4, *Fj, input_format="Fj_matrices").toarray())
        [1.0 1.0 0.0 0.0]
        [0.0 2.0 2.0 0.0]
        [0.0 0.0 3.0 3.0]
        [0.0 0.0 0.0 4.0]
    """
    # define the vector of symbolic variables
    x = polygens(QQ, ['x'+str(0)])
    f = [None] * 1
    f[0] =  a * x[0] + b * x[0]**2
    return PolynomialODE(f, 1, 2)

def arrowsmith_and_place_fig_3_5e_page_79():
    r"""
    Nonlinear two-dimensional system with an hyperbolic fixed point.

    It is defined as:

    .. MATH::

        \begin{aligned}
         x' &= x^2+(x+y)/2 \\
         y' &= (-x+3y)/2
        \end{aligned}

    Taken from p. 79 of the book by Arrowsmith and Place, Dynamical Systems:
    Differential Equations, maps and chaotic behaviour.
    """
    # dimension of state-space 
    n=2

    # vector of variables
    x = polygens(QQ, ['x'+str(i) for i in range(n)])

    # ODE and order k=2
    f = [x[0]^2+(x[0]+x[1])/2, (-x[0] +3*x[1])/2]    

    return PolynomialODE(f, n, k=2)

def biomodel_2():
    r"""
    This is a nine-dimensional polynomial ODE used as benchmark model in
    `the Flow star tool <https://ths.rwth-aachen.de/research/projects/hypro/biological-model-ii/>`_.

    The model is adapted from E. Klipp, R. Herwig, A. Kowald, C. Wierling, H. Lehrach.
    Systems Biology in Practice: Concepts, Implementation and Application. Wiley-Blackwell, 2005.
    """
    # dimension of state-space
    n = 9

    # vector of variables
    x = polygens(QQ, ['x'+str(i) for i in range(n)])

    f = [None]*n
    f[0] = 3*x[2] - x[0]*x[5]
    f[1] = x[3] - x[1]*x[5]
    f[2] = x[0]*x[5] - 3*x[2]
    f[3] = x[1]*x[5] - x[3]
    f[4] = 3*x[2] + 5*x[0] - x[4]
    f[5] = 5*x[4] + 3*x[2] + x[3] - x[5]*(x[0]+x[1]+2*x[7]+1)
    f[6] = 5*x[3] + x[1] - 0.5*x[6]
    f[7] = 5*x[6] - 2*x[5]*x[7] + x[8] - 0.2*x[7];
    f[8] = 2*x[5]*x[7] - x[8]

    return PolynomialODE(f, n, k=2)

def chen_seven_dim(u=0):
    r"""
    This is a seven-dimensional nonlinear system of quadratic order.

    It appears as ``'example_nonlinear_reach_04_sevenDim_nonConvexRepr.m'`` in 
    the tool CORA 2016, in the examples for continuous nonlinear systems.

    .. NOTE:

    There is an independent term, `u`, in the fourth equation with value `2.0` that has
    been neglected here for convenience (hence we take `u=0` by default).
    """
    # dimension of state-space
    n = 7

    # vector of variables
    x = polygens(QQ, ['x'+str(i) for i in range(n)])

    f = [None]*n

    f[0] = 1.4*x[2]-0.9*x[0]
    f[1] = 2.5*x[4]-1.5*x[1]
    f[2] = 0.6*x[6]-0.8*x[2]*x[1]
    f[3] = -1.3*x[3]*x[2] + u
    f[4] = 0.7*x[0]-1.0*x[3]*x[4]
    f[5] = 0.3*x[0]-3.1*x[5]
    f[6] = 1.8*x[5]-1.5*x[6]*x[1]

    return PolynomialODE(f, n, k=2)
