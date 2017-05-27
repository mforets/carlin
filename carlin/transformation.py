r"""
Carleman linearization of polynomial differential equations in SageMath.

Reduction methods
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :func:`~quadratic_reduction`          | Reduce from any order in standard form to quadratic order
    :func:`~transfer_matrices`            | Compute the higher order transfer matrices `A^{i}_{i+j-1}`

Linearization
~~~~~~~~~~~~~~~~~~

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :func:`~linearize`            | Compute Carleman linearization and export to a MAT file
    :func:`~truncated_matrix`     | Finite order `N` Carleman linearization


Kronecker power and linear algebra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :func:`~get_index_from_key`   | Return first occurrence of given key over a Kronecker power
    :func:`~get_key_from_index`   | Return multi-index of Kronecker power given an index and the order
    :func:`~kron_prod`            | Compute the Kronecker product of x and y
    :func:`~kron_power`           | Receives a `n\times 1` vector and computes its Kronecker power `x^{[i]}`
    :func:`~log_norm`             | Compute the logarithmic norm of a matrix

Error computation
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :func:`~characteristics`      | Information about the norms of the matrices in `F`
    :func:`~error_function`       | Compute the error function of a truncated ODE
    :func:`~plot_error_function`  | Plot the estimated error of the linearized ODE as a function of time

AUTHOR:

- Marcelo Forets (Dec 2016 at VERIMAG - France)

MF acknowledges the hospitality at Max-Planck Institute for Software
Systems, Saarbrucken, Germany, where part of this package was written (Apr 2016).
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

#===============================================
# Dependencies
#===============================================

# Working numerical libraries: NumPy
import numpy as np
from numpy.linalg import norm

# Working numerical libraries: SciPy
import scipy as sp
from scipy import inf
from scipy.io import savemat
from scipy.sparse import kron, eye
import scipy.sparse.linalg

# Carleman input/output libraries
from carlin.io import get_Fj_from_model

# Toolbox for operations on polytopes
from polyhedron_tools.misc import polyhedron_to_Hrep, chebyshev_center, radius

# Sage objects: Rings, Polynomials, Linear algebra
from sage.rings.all import RR, QQ
from sage.rings.real_double import RDF
from sage.rings.polynomial.polynomial_ring import polygens

from sage.modules.free_module_element import vector

from sage.functions.other import real_part, imag_part
from sage.functions.log import log, exp

#===============================================
# Functions to compute Carleman linearization
#===============================================

def transfer_matrices(N, F, n, k):
    r"""
    Higher order transfer matrices `A^{i}_{i+j-1}`.

    INPUT:

    - ``N`` -- order of truncation 

    - ``F`` -- list, sequence of matrices `F_j`

    - ``n`` -- the dimension of the state-space

    - ``k`` -- the order of the polynomial vector field. It is equal to ``len(F)``

    OUTPUT:

    - ``A`` -- the transfer matrices `A^{i}_{i+j-1}` that correspond to `i = 1, \ldots, N`. 
               It is given as a list of lists. Each inner list has dimension `k`.
    """
    A = []

    # first row is trivial
    A.append(F)

    for i in range(1, N):
        newRow = []
        for j in range(k):
            L = kron(A[i-1][j], eye(n))
            R = kron(eye(n**i), F[j])
            newRow += [np.add(L, R)]
        A.append(newRow)

    return A

def truncated_matrix(N, *args, **kwargs):
    r"""
    Finite order Carleman linearization.

    INPUT:

    - ``N`` -- order of truncation

    - ``input_format`` -- if it is not present, then it is assumed to be
      ``'model_filename'`` (file in text format); other options are ``'transfer_matrices'``
      for `(A, n, k)` and ``'Fj_matrices'`` for `(F, n, k)`. In the latter cases, these 
      should be given as separate arguments.

    OUTPUT:

    The transfer matrices `A^{i}_{i+j-1}` that correspond to `i = 1, \ldots, N`.
    It is given as a list of lists, and each inner list has dimension `k`.
    """
    from scipy.sparse import bmat

    if kwargs['input_format'] == 'model_filename' or 'input_format' not in kwargs.keys():
        # assuming 'model_filename'
        model_filename = args[0]
        [F, n, k] = get_Fj_from_model(model_filename)
        A = transfer_matrices(N, F, n, k)
    else:
        if kwargs['input_format'] == 'transfer_matrices':
            A = args[0]
            n = args[1]
            k = args[2]
        elif kwargs['input_format'] == 'Fj_matrices':
            F = args[0]
            n = args[1]
            k = args[2]
            A = transfer_matrices(N, F, n, k)
        else:
            raise ValueError('invalid input format')

    BN_list = []

    for i in range(N):

        n3_i = max(N-i-k, 0)
        n2_i = N-i-n3_i

        newBlockRow = A[i][0:n2_i]

        for j in range(i):
            newBlockRow.insert(0, None)

        for j in range(n3_i):
            newBlockRow.append(None)

        BN_list.append(newBlockRow)

    BN = bmat(BN_list)

    return BN

def quadratic_reduction(F, n, k):
    r"""
    Reduce a `k`-th order system of polynomial ODE's into a quadratic one.

    INPUT:

    - ``F`` -- list of matrices defining the system of polynomial ODE's

    - ``n`` -- integer, system's dimension

    - ``k`` -- integer, order of the polynomial ODE

    OUTPUT:

    The list ``[F_tilde, nquad, kquad]`` corresponding to the reduced list of `F_j`'s,
    dimension and order respectively.
    """
    from scipy.sparse import bmat, lil_matrix

    A = transfer_matrices(k-1, F, n, k)

    # LINEAR PART
    F1_tilde_list = []
    for i in range(k-1):
        newRow = A[i][0:k-i-1]
        for j in range(i):
            newRow.insert(0, None)
        F1_tilde_list.append(newRow)

    F1_tilde = bmat(F1_tilde_list)

    # QUADRATIC PART
    F2_tilde_list = []

    for i in range(k-1):
        newRow = []
        for h in range(k-1):

            for j in range(k-2):
                newRow.append(lil_matrix((n**(i+1), n**(h+j+2))))

            if h>i:
                newRow.append(lil_matrix((n**(i+1), n**(h+k))))
            else:
                newRow.append(A[i][k-i-1+h])


        F2_tilde_list.append(newRow)

    F2_tilde = bmat(F2_tilde_list)

    F_tilde = [F1_tilde, F2_tilde]

    kquad = 2

    #nquad is expected to be : (n^k-n)/(n-1)
    nquad = F1_tilde.shape[0]

    return [F_tilde, nquad, kquad]

def error_function(model_filename, N, x0):
    """
    Compute the error function of a truncated ODE.
    
    INPUT:

    - ``model_filename`` -- string containing the model in text format

    - ``N`` -- truncation order

    - ``x0`` -- initial point, a list
    """
    from numpy.linalg import norm
    from sage.symbolic.ring import SR
    
    [F, n, k] = get_Fj_from_model(model_filename)
    [Fquad, nquad, kquad] = quadratic_reduction(F, n, k)
    ch = characteristics(Fquad, nquad, kquad);

    norm_F1_tilde = ch['norm_Fi_inf'][0]
    norm_F2_tilde = ch['norm_Fi_inf'][1]

    x0_hat = [kron_power(x0, i+1) for i in range(k-1)]
    
    #transform to flat list
    x0_hat = [item for sublist in x0_hat for item in sublist]

    norm_x0_hat = norm(x0_hat, ord=inf)

    beta0 = ch['beta0_const']*norm_x0_hat

    Ts = 1/norm_F1_tilde*log(1+1/beta0)

    t = SR.var('t')
    error = norm_x0_hat*exp(norm_F1_tilde*t)/(1+beta0-beta0*exp(norm_F1_tilde*t))*(beta0*(exp(norm_F1_tilde*t)-1))**N

    return [Ts, error]

def plot_error_function(model_filename, N, x0, Tfrac=0.8):
    """
    Plot the estimated error of the linearized ODE as a function of time.

    INPUT:

    - ``model_filename`` -- string containing the model in text format

    - ``N`` -- truncation order

    - ``x0`` -- initial point, a list

    - ``Tfrac`` -- (optional, default: `0.8`): fraction of the convergence radius,
      to specify the plotting range in the time axis

    NOTE:

    This function calls ``error_function`` for the error computations.
    """
    from sage.plot.graphics import Graphics
    from sage.plot.plot import plot
    from sage.plot.line import line

    [Ts, eps] = error_function(model_filename, N, x0)
    P = Graphics()
    P = plot(eps, 0, Ts*Tfrac, axes_labels = ["$t$", r"$\mathcal{E}(t)$"])
    P += line([[Ts, 0], [Ts, eps(t=Ts * Tfrac)]], linestyle='dotted', color='black')

    return P

#===============================================
# Functions to export Carleman linearization
#===============================================

def linearize(model_filename, target_filename, N, x0, **kwargs):
    r"""
    Compute Carleman linearization and export to a MAT file.

    INPUT:

    - ``model_filename`` -- string containing the model in text format
    
    - ``target_filename`` -- string with the name of the output file in MAT format
    
    - ``N`` -- truncation order
    
    - ``x0`` -- initial point, can be either a list or a polyhedron; see the code for 
      further details
    
    NOTES:
    
    This function is self-contained; it transforms to canonical quadratic form, then
    computes Carleman linearization together with the error estimates and exports the resulting
    matrix `A_N` and characteristics to a MAT file.
    """
    
    dic = dict()
    dic['model_name'] = model_filename
    dic['N'] = N

    print 'Obtaining the canonical representation...',
    [F, n, k] = get_Fj_from_model(model_filename)
    print 'done'

    dic['n'] = n
    dic['k'] = k

    print 'Computing matrix BN...',
    B_N = truncated_matrix(N, model_filename)
    print 'done'

    dic['BN'] = B_N

    print 'Computing the quadratic reduction...',
    [Fquad, nquad, kquad] = quadratic_reduction(F, n, k)
    print 'done'

    print 'Computing the characteristics of the model...',
    ch = characteristics(Fquad, nquad, kquad);
    print 'done'

    norm_F1_tilde = ch['norm_Fi_inf'][0]
    norm_F2_tilde = ch['norm_Fi_inf'][1]

    dic['norm_F1_tilde'] = norm_F1_tilde
    dic['norm_F2_tilde'] = norm_F2_tilde

    dic['log_norm_F1_inf'] = ch['log_norm_F1_inf']

    if 'polyhedron' in str(type(x0)):

        norm_initial_states = polyhedron_sup_norm(x0).n()

        if (norm_initial_states >= 1):
            norm_x0_hat = norm_initial_states**(k-1)
        elif (norm_initial_states < 1):
            norm_x0_hat = norm_initial_states

        # when the initial set is a polytope, this is the maximum norm
        # of the initial states in the lifted (quadratic) system
        dic['norm_x0_tilde'] = norm_x0_hat

        beta0 = ch['beta0_const']*norm_x0_hat
        dic['beta0'] = beta0

        Ts = 1./norm_F1_tilde*log(1+1./beta0)
        dic['Ts'] = Ts

        [F, g] = polyhedron_to_Hrep(x0)

        #dic['F'] = F; dic['g'] = g.column()
        dic['x0'] = {'F' : F, 'g' : g.column()}

        cheby_center_X0 = chebyshev_center(x0)
        dic['x0_cc'] = vector(cheby_center_X0).column()

    else: # assuming that x0 is a list object

        #x0_hat = [kron_power(x0, i+1) for i in range(k-1)]
        #transform to flat list
        #x0_hat = [item for sublist in x0_hat for item in sublist]
        #norm_x0_hat = np.linalg.norm(x0_hat, ord=inf)

        #use crossnorm property
        nx0 = np.linalg.norm(x0, ord=inf)
        norm_x0_hat = max([nx0**i for i in range(1, k)])

        dic['norm_x0_tilde'] = norm_x0_hat

        beta0 = ch['beta0_const']*norm_x0_hat
        dic['beta0'] = beta0

        Ts = 1/norm_F1_tilde*log(1+1/beta0)
        dic['Ts'] = Ts

        dic['x0'] = vector(x0).column()


    # if required, write additional data from dictionary
    if 'append' in kwargs.keys():

        # the object extra_data should be a list of dictionaries
        # each dictionary must constain the field 'name', that we use as a
        # Matlab variable identifier.
        extra_data = kwargs['append']
        if not isinstance(extra_data, list):
            dkey = extra_data.pop('name')
            dic[dkey] = extra_data
        else:
            for new_data in extra_data:
                # field 'name' is required for Matlab identifier
                new_key = new_data.pop('name')
                dic[new_key] = new_data

    print 'Exporting to ', target_filename, '...',
    savemat(target_filename, dic)
    print 'done'

    return

#===============================================
# Auxiliary mathematical functions
#===============================================

def kron_prod(x, y):
    r""" Compute the Kronecker product of x and y.

    INPUT:

    - ``x`` -- vector or list 

    - ``y`` -- vector or list

    OUTPUT:

    A list is returned, corresponding to the Kronecker product `x\otimes y`.
    """
    return [x[i]*y[j] for i in range(len(x)) for j in range(len(y))]

def kron_power(x, i):
    r""" Receives a `n\times 1` vector and computes its Kronecker power `x^{[i]}`.

    INPUT:

    - ``x`` -- list or vector

    - ``i`` -- integer

    OUTPUT:

    A list corresponding to the `i`-th Kronecker power of `x`, namely `x^{[i]}`.
    """
    if (i > 2):
        return kron_prod(x, kron_power(x,i-1))
    elif (i == 2):
        return kron_prod(x,x)
    elif (i == 1):
        return x
#   elif (i==0):
#        return 1
    else:
        raise ValueError('index i should be an integer >= 1')

def get_key_from_index(i, j, n):
    r"""
    Return multi-index of Kronecker power given an index and the order.

    INPUT:

    - ``i`` -- integer, index in the canonial Kronecker power enumeration

    - ``j`` -- integer, order of the Kronecker power

    - ``n`` -- integer, number of dimensions

    EXAMPLES:

    Take `x^{[2]}` for `x=(x_1, x_2)` and compute the exponent vector of the element
    in position `1`::

        sage: from carlin.transformation import get_key_from_index
        sage: get_key_from_index(1, 2, 2)
        [1, 1]
    """
    x = polygens(QQ, ['x'+str(1+k) for k in range(n)])
    x_power_j = kron_power(x, j)
    d = x_power_j[i].dict()
    return list(d.items()[0][0])

def get_index_from_key(key, j, n):
    r"""
    Return first occurrence of given key over a Kronecker power.

    INPUT:

    - ``key`` -- list or tuple, key corresponding to the exponent vector in the
      Kronecker power

    - ``j`` -- integer, order of the Kronecker power

    - ``n`` -- integer, number of dimensions

    NOTES:

    - We assume `n \geq 2`. Notice that if `n=1`, we would return always that ``first_occurence = 0``.

    EXAMPLES:

    Take `x^{[2]}` for `x=(x_1, x_2)` and compute retrive the first ocurrence of
    the given key::

        sage: from carlin.transformation import get_index_from_key
        sage: get_index_from_key([1, 1], 2, 2)
        1
    """
    x = polygens(QQ, ['x'+str(1+k) for k in range(n)])
    x_power_j = kron_power(x, j)

    for i, monomial in enumerate(x_power_j):
        if ( list(monomial.dict().keys()[0]) == key):
            first_occurence = i
            break

    return first_occurence

def log_norm(A, p='inf'):
    r"""
    Compute the logarithmic norm of a matrix.

    INPUT:

    - ``A`` -- a rectangular (Sage dense) matrix of order `n`. The coefficients can be either real or complex

    - ``p`` -- (default: ``'inf'``). The vector norm; possible choices are ``1``, ``2``, or ``'inf'``

    OUTPUT:

    - ``lognorm`` -- the log-norm of `A` in the `p`-norm
    """

    # parse the input matrix
    if 'scipy.sparse' in str(type(A)):
        # cast into numpy array (or ndarray)
        A = A.toarray()
        n = A.shape[0]
    elif 'numpy.array' in str(type(A)) or 'numpy.ndarray' in str(type(A)):
        n = A.shape[0]
    else:
        # assuming sage matrix
        n = A.nrows();

    # computation, depending on the chosen norm p
    if (p == 'inf' or p == oo):
        z = max( real_part(A[i][i]) + sum( abs(A[i][j]) for j in range(n)) - abs(A[i][i]) for i in range(n))
        return z

    elif (p == 1):
        n = A.nrows();
        return max( real_part(A[j][j]) + sum( abs(A[i][j]) for i in range(n)) - abs(A[j][j]) for j in range(n))

    elif (p == 2):

        if not (A.base_ring() == RR or A.base_ring() == CC):
            return 1/2*max((A+A.H).eigenvalues())
        else:
            # Alternative, always numerical
            z = 1/2*max( np.linalg.eigvals( np.matrix(A+A.H, dtype=complex) ) )
            return real_part(z) if imag_part(z) == 0 else z

    else:
        raise NotImplementedError('value of p not understood or not implemented')

def characteristics(F, n, k, ord=inf):
    r"""
    Information about the norms of the matrices in `F`.

    INPUT:

    - ``F`` -- list of matrices in some NumPy sparse format, for which the 
     ``toarray`` method is available

    - ``n`` -- dimension on state-space

    - ``k`` -- order of the system

    - ``ord`` -- order of the `p`-th norm, for `1 \leq p < \infty`, and ``p='inf'``
      for `p=\infty`

    OUTPUT:

    Dictionary ``c`` containing ``norm_Fi_inf``, ``log_norm_F1_inf`` and ``beta0_const``.
    """
    c = dict()

    c['norm_Fi_inf'] = [norm(F[i].toarray(), ord=ord) for i in range(k)]

    if ord == inf:
        c['log_norm_F1_inf'] = log_norm(F[0], p='inf')
    else:
        raise NotImplementedError("log norm error should be supremum (='inf')")

    if k > 1:
        if c['norm_Fi_inf'][0] != 0:
            c['beta0_const'] = c['norm_Fi_inf'][1]/c['norm_Fi_inf'][0]
        else:
            c['beta0_const'] = 'inf'

    return c
