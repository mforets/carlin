#===============================================
# Dependencies
#===============================================

# NumPy
import numpy as np
from numpy.linalg import norm

# SciPy
import scipy as sp
from scipy import inf
from scipy.io import savemat
from scipy.sparse import kron, eye
import scipy.sparse.linalg

# Toolbox for operations on polytopes
from polyhedron_tools.misc import polyhedron_to_Hrep, chebyshev_center, radius

# Sage objects: Rings, Polynomials, Linear algebra and all that
from sage.rings.all import RR, QQ
from sage.rings.real_double import RDF
from sage.rings.polynomial.polynomial_ring import polygens
from sage.modules.free_module_element import vector
from sage.functions.other import real_part, imag_part
from sage.functions.log import log, exp

#===============================================
# Kronecker operations
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

def kron_power(x, i):
    r""" Receives a `n\times 1` vector and computes its Kronecker power `x^{[i]}`.

    INPUT:

    - ``x`` -- list or vector

    - ``i`` -- integer

    OUTPUT:

    A list corresponding to the `i`-th Kronecker power of `x`, namely `x^{[i]}`.

    EXAMPLES::

        sage: from carlin.utils import kron_power
        sage: kron_power([1, 2], 3)
        [1, 2, 2, 4, 2, 4, 4, 8]
        sage: x, y = SR.var('x, y')
        sage: kron_power([x, y], 2)
        [x^2, x*y, x*y, y^2]
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

def lift(x0, N):
    y0 = kron_power(x0, 1)
    for i in range(2, N+1):
        y0 = y0 + kron_power(x0, i)
    return y0

#===============================================
# Matrix operations
#===============================================

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
