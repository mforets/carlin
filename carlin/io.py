r"""
Carleman linearization input/output functions.

Load and export polynomial ODEs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :func:`~export_model_to_mat`   | Export model to a MAT file as the sequence of sparse `F_j` matrices
    :func:`~load_model`            | Read an ODE system from a text file

Transformation functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :func:`~get_Fj_from_model`     | Transform a model into standard form as a sum of Kronecker products

Solve polynomial ODEs
~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :func:`~solve_ode_exp`  | Solve Carleman linearized ODE.
    :func:`~plot_truncated` | Solve and return the graphics in phase space. 

AUTHOR:

- Marcelo Forets (Dec 2016 at VERIMAG - UGA)
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

#=============
# Dependencies
#=============

# Working numerical libraries: NumPy
import numpy as np

# Working numerical libraries: SciPy
import scipy as sp
from numpy import inf
import scipy.sparse.linalg
from scipy.sparse import dok_matrix

# Sage objects: Rings and Polynomials
from sage.rings.integer import Integer
from sage.rings.real_mpfr import RR
from sage.rings.rational_field import Q as QQ
from sage.rings.real_double import RDF
from sage.rings.polynomial.polynomial_ring import polygens
from sage.modules.free_module_element import vector

from carlin.utils import kron_power, lift

#==========================
# Functions to load a model
#==========================

def load_model(model_filename):
    r"""
    Read an input ODE system.

    INPUT:

    - ``model_filename`` -- string with the model filename

    OUTPUT:

    - ``f`` -- list of multivariate polynomials which describes the system of ODEs,
      each component in the polynomial ring `\mathbb{Q}[x_1,\ldots,x_n]`

    - ``n`` -- integer, dimension of f

    - ``k`` -- integer, degree of f
    """
    from sage.structure.sage_object import load
    
    # should define n and f
    load(model_filename)

    k = max( [fi.degree() for fi in f] )

    return [f, n, k]

def get_Fj_from_model(model_filename=None, f=None, n=None, k=None):
    r"""
    Transform an input model of a polynomial vector field into standard
    form as a sum of Kronecker products.

    The model can be given either as an external file (``model_filename``), or
    as the data `f`, `n` and `k`.

    INPUT:

    - ``model_filename`` -- string containing the filename

    OUTPUT:

    - ``F`` -- list of sparse matrices `F_1, \ldots, F_k`.
      These are formatted in dictionary-of-keys (dok) form.

    - ``n`` -- dimension of the state-space of the system

    - ``k`` -- degree of the system

    EXAMPLES:

    Let's create a two-dimensional system of second order::

        sage: x = polygens(QQ, ['x0', 'x1'])
        sage: f = [-2*x[0] + x[1] + x[0]**2, x[0] + 3/2*x[0]*x[1]]
        sage: from carlin.polynomial_ode import PolynomialODE
        sage: T = PolynomialODE(f, 2, 2); T
        A Polynomial ODE in n = 2 variables

    There are two `F_j` matrices in sparse representation, which can be visualizeda
    with the ``toarray`` method::

        sage: from carlin.io import get_Fj_from_model
        sage: F, _, _ = get_Fj_from_model(T.funcs(), T.dim(), T.degree())
        sage: F
        [<2x2 sparse matrix of type '<type 'numpy.float64'>'
                with 3 stored elements in Dictionary Of Keys format>,
         <2x4 sparse matrix of type '<type 'numpy.float64'>'
                with 2 stored elements in Dictionary Of Keys format>]
        sage: F[0].toarray()
        array([[-2.,  1.],
               [ 1.,  0.]])
       sage: F[1].toarray()
       array([[ 1. ,  0. ,  0. ,  0. ],
              [ 0. ,  1.5,  0. ,  0. ]])

    TESTS:

    Check if a polynomial with a non-numeric coefficient is allowed::

    sage: from carlin.io import get_Fj_from_model
    sage: x = polygens(QQ, ["x0"]); mu = SR.var("mu")
    sage: from carlin.polynomial_ode import PolynomialODE
    sage: T = PolynomialODE([mu*x[0]^2 - x[0]], 1, 2)
    sage: get_Fj_from_model(T.funcs(), T.dim(), T.degree())
    Traceback (most recent call last):
    ...
    NotImplementedError: the coefficients of the polynomial should be numeric
    """
    if model_filename is not None and f is None:
        got_model_by_filename = True
    elif model_filename is not None and f is not None and n is not None and k is None:
        k = n; n = f; f = model_filename;
        got_model_by_filename = False
    else:
        raise ValueError("either the model name or the vector field (f, n, k) should be specified")

    if got_model_by_filename:
        [f, n, k] = load_model(model_filename)

    # create the collection of sparse matrices Fj
    F = [dok_matrix((n, n**i), dtype=np.float64) for i in range(1, k+1)]

    # read the powers appearing in each monomial
    try:
        dictionary_f = [fi.dict() for fi in f]

    except AttributeError:
        # if the polynomial has symbolic coefficients, we have to parse the 
        # expression tree
        raise NotImplementedError("the coefficients of the polynomial should be numeric")

    if (n>1):
        from carlin.transformation import get_index_from_key
        
        for i, dictionary_f_i in enumerate(dictionary_f):
            for key in dictionary_f_i.iterkeys():
                row = i;
                j = sum(key)
                column = get_index_from_key(list(key), j, n)
                F[j-1].update({tuple([row,column]): dictionary_f_i.get(key)})

    elif (n==1):
        #the scalar case is treated separately. the problem arises from using
        #sum(1) (note it doesn't break if one uses from scipy import sum)
        for i, dictionary_f_i in enumerate(dictionary_f):
            for key in dictionary_f_i.iterkeys():
                row = i;
                j = key
                column = 0 # because Fj are 1x1 in the scalar case
                F[j-1].update({tuple([row,column]): dictionary_f_i.get(key)})

    return F, n, k

#===============================================
# Functions to export a model
#===============================================

def export_model_to_mat(model_filename, F=None, n=None, k=None, **kwargs):
    r"""
    Export ODE model to a Matlab ``.mat`` format.

    INPUT:

    The model can be given either as a model in text file, or as the tuple `(F, n, k)`.

    - ``model_filename`` -- string with the model filename. If `(F, n, k)` is not provided,
      then such model should be reachable from the current path

    - ``F`` -- list of sparse matrices `F_1, \ldots, F_k`

    - ``n`` -- dimension of state-space

    - ``k`` -- order of the system

    OUTPUT:

    List containing the solution of the 1st order ODE `x'(t) = A_N x(t)`, with initial 
    condition `x(0) = x_0`. The output filename is ``model_filename``,
    replacing the ``.sage`` extension with the ``.mat`` extension.
    """
    from scipy.io import savemat

    if '.sage' in model_filename:
        mat_model_filename = model_filename.replace(".sage", ".mat")
    elif '.mat' in model_filename:
        mat_model_filename = model_filename
    else:
        raise ValueError("Expected .sage or .mat file format in model filename.")

    if F is None:
        [F, n, k] = get_Fj_from_model(model_filename)

    savemat(mat_model_filename, {'F': F, 'n': n, 'k': k})
    return

#===============================================
# Functions to solve ODE's
#===============================================

def solve_ode_exp(AN, x0, N, tini=0, T=1, NPOINTS=100):
    r"""
    Solve 1st order ODE with initial condition in Kronecker power form.

    INPUT:

    - ``AN`` -- matrix, it can be Sage dense or NumPy sparse in COO format

    - ``x0`` -- vector, initial point

    - ``N`` -- integer, order of the truncation

    - ``tini`` -- (optional, default: 0) initial time

    - ``T`` -- (optional, default: 1) final time

    - ``NPOINTS`` -- (optional, default: 100) number of points computed

    OUTPUT:

    List containing the solution of the 1st order ODE `x'(t) = A_N x(t)`, with initial 
    condition `x(0) = x_0`. The solution is computed the matrix exponential `e^{A_N t_i}` directly.

    NOTES:

    For high-dimensional systems prefer using ``AN`` in sparse format. In this case,
    the matrix exponential is computed using `scipy.sparse.linalg.expm_multiply 
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html>`_.

    EXAMPLES::

        sage: from carlin.transformation import get_Fj_from_model, truncated_matrix
        sage: from carlin.library import quadratic_scalar
        sage: S = quadratic_scalar()
        sage: (f, n, k) = S.funcs(), S.dim(), S.degree()
        sage: Fjnk = get_Fj_from_model(f, n, k)
    
    Consider a fourth order truncation::

        sage: AN_sparse = truncated_matrix(4, *Fjnk, input_format="Fj_matrices")
        sage: AN_sparse.toarray()
        array([[ 1.,  1.,  0.,  0.],
               [ 0.,  2.,  2.,  0.],
               [ 0.,  0.,  3.,  3.],
               [ 0.,  0.,  0.,  4.]])

    We can solve the linear ODE using the sparse matrix ``AN_sparse``::

        sage: from carlin.io import solve_ode_exp
        sage: ans = solve_ode_exp(AN_sparse, x0=[0.1], N=4, tini=0, \
                    T=1, NPOINTS=20)

    It can also be solved using Sage matrices (although this is often less performant,
    since it works with dense matrices)::

        sage: AN_dense = matrix(AN_sparse.toarray())
        sage: ans = solve_ode_exp(AN_dense, x0=[0.1], N=4, tini=0, \
                    T=1, NPOINTS=20)
    """
    # transform to [x0, x0^[2], ..., x0^[N]]
    y0 = vector(lift(x0, N))

    # compute solution
    if "sage.matrix" in str(type(AN)):
        #t_dom = [tini + (T-tini)/(NPOINTS-1)*i for i in range(NPOINTS)]
        t_dom = np.linspace(tini, T, num=NPOINTS)
        sol = [AN.exp() * np.exp(ti) * y0 for ti in t_dom]

    elif "scipy.sparse" in str(type(AN)):
        from scipy.sparse.linalg import expm_multiply
        sol = expm_multiply(AN, np.array(y0), start=tini, stop=T, \
                            num=NPOINTS, endpoint=True)

    else:
        raise ValueError("invalid matrix type")

    return sol

def plot_truncated(model, N, x0, tini, T, NPOINTS, xcoord=0, ycoord=1, **kwargs):
    """
    Solve and return graphics in phase space of a given model.

    INPUT:

    - ``model`` -- PolynomialODE, defining the tuple `(f, n, k)`

    - ``N`` -- integer; truncation order

    - ``x0`` -- vector; initial condition

    - ``tini`` -- initial time of simulation

    - ``T`` -- final time of simulation

    - ``NPOINTS`` -- number of points sampled

    - ``xcoord`` -- (default: `0`), x-coordinate in plot

    - ``ycoord`` -- (default: `1`), y coordinate in plot

    NOTES:

    By default, returns a plot in the plane `(x_1, x_2)`. All other keyword arguments
    passes are sent to the `list_plot` command (use to set line color, style, etc.)

    EXAMPLES::

        sage: from carlin.library import vanderpol
        sage: from carlin.io import plot_truncated
        sage: G = plot_truncated(vanderpol(1, 1), 2, [0.1, 0], 0, 5, 100)
        sage: G.show(gridlines=True, axes_labels = ['$x_1$', '$x_2$']) # not tested
        Graphics object consisting in 1 graphics primitive

    All other keyword arguments are passed to the ``list_plot`` function. For example, specify color and 
    maximum and minimum values for the axes::

        sage: G = plot_truncated(vanderpol(1, 1), 2, [0.1, 0], 0, 5, 100, color='green', xmin=-1, xmax=1, ymin=-1, ymax=1)
        sage: G.show(gridlines=True, axes_labels = ['$x_1$', '$x_2$']) # not tested
        Graphics object consisting in 1 graphics primitive
    """
    from carlin.transformation import truncated_matrix
    from carlin.io import solve_ode_exp, get_Fj_from_model
    from sage.plot.plot import list_plot

    f, n, k = model.funcs(), model.dim(), model.degree()
    Fjnk = get_Fj_from_model(f, n, k)

    # this is a sparse matrix in coo format
    AN = truncated_matrix(N, *Fjnk, input_format='Fj_matrices')

    # solve the linea ODE using SciPy's sparse matrix solver
    sol = solve_ode_exp(AN, x0, N, tini=tini, T=T, NPOINTS=NPOINTS)
    sol_x1 = [sol[i][xcoord] for i in range(NPOINTS)]
    sol_x2 = [sol[i][ycoord] for i in range(NPOINTS)]

    return list_plot(zip(sol_x1, sol_x2), plotjoined=True, **kwargs)
