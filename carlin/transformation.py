r"""
Carleman linearization of polynomial differential equations in SageMath.

- Marcelo Forets  (first draft: 10 Dic 2016)
                  (last modified: 10 Feb 2017)
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

# Working numerical libraries
import numpy as np
import scipy
from scipy import inf
from scipy.io import savemat
import scipy.sparse as sp
from scipy.sparse import kron, eye
import scipy.sparse.linalg

# Carleman input/output libraries
from carlin.io import get_Fj_from_model

# Toolbox for operations on polytopes
from polyhedron_toolbox import polyhedron_to_Hrep, chebyshev_center, radius

#===============================================
# Functions to compute Carleman linearization
#===============================================

def transfer_matrices(N, F, n, k):
    r""" Finite order transfer matrices `A_ij`.

    INPUTS:

    - "N" : order of truncation.

    - "F" : sequence of matrices $F_j$ (list).

    - "n" : the dimension of the state-space.

    - "k" : the order of the polynomial vector field. It is equal to len(F).


    OUTPUTS:

    - "A" : the transfer matrices $A^{i}_{i+j-1}$ that correspond
            to $i = 1, \ldots , N$. It is given as a list of lists.
            Each inner list has dimension $k$.

    """

    A = []

    # first row is trivial
    A.append(F)

    for i in range(1, N):
        newRow = []
        for j in range(k):
            L = kron(A[i-1][j], eye(n))
            R = kron(eye(n^(i)), F[j])
            newRow += [np.add(L, R)]
        A.append(newRow)

    return A

def truncated_matrix(N, *args, **kwargs):
    r""" Finite order Carleman linearization. 

    INPUTS:

    - "N" : order of truncation.

    - "input_format" : sequence of matrices $F_j$ (list).

    - "F" : the dimension of the state-space.

    - "input_format" : the order of the polynomial vector field. It is equal to len(F).


    OUTPUTS:

    - "A" : the transfer matrices $A^{i}_{i+j-1}$ that correspond to
            $i = 1, \ldots , N$. It is given as a list of lists.
            Each inner list has dimension $k$.
    """

    from scipy.sparse import bmat

    if 'input_format' not in kwargs.keys():
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
            raise ValueError('Input format not understood.')


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
                newRow.append(lil_matrix((n^(i+1), n^(h+j+2))))

            if h>i:
                newRow.append(lil_matrix((n^(i+1), n^(h+k))))
            else:
                newRow.append(A[i][k-i-1+h])


        F2_tilde_list.append(newRow)

    F2_tilde = bmat(F2_tilde_list)

    F_tilde = [F1_tilde, F2_tilde]

    kquad = 2
    #nquad should be: (n^k-n)/(n-1)
    nquad = F1_tilde.shape[0]

    return [F_tilde, nquad, kquad]

def error_function(model_filename, N, x0):

    import scipy as sp
    from scipy import inf
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

#===============================================
# Functions to export Carleman linearization
#===============================================

def carleman_export(model_filename, target_filename, N, x0, **kwargs):
    r""" Compute Carleman linearization and export to a MAT file.
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
            norm_x0_hat = norm_initial_states^(k-1)
        elif (norm_initial_states < 1):
            norm_x0_hat = norm_initial_states

        # when the initial set is a polytope, this is the maximum norm
        # of the initial states in the lifted (quadratic) system
        dic['norm_x0_tilde'] = norm_x0_hat

        beta0 = ch['beta0_const']*norm_x0_hat
        dic['beta0'] = beta0

        Ts = 1/norm_F1_tilde*log(1+1/beta0)
        dic['Ts'] = Ts

        [F, g] = PolyhedronToHSpaceRep(x0)

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
        norm_x0_hat = max([nx0^i for i in range(1, k)])

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

def kron_prod(x,y):
    r""" Compute the Kronecker product of two vectors x and y.
    
    A list is returned. The method len should be available.
    """
    return [x[i]*y[j]  for i in range(len(x)) for j in range(len(y))]


def kron_power(x, i):
    r""" Receives a nx1 vector and computes its Kronecker power x^[i]. Assuming that i >= 1.
    """
    if (i > 2):
        return kron_prod(x, kron_power(x,i-1))
    elif (i == 2):
        return kron_prod(x,x)
    elif (i == 1):
        return x
    #elif (i==0):
#        return 1
    else:
        raise ValueError('Index i should be an integer >= 1')


def get_key_from_index(i, j, n):

    x = polygen(QQ, ['x'+str(1+k) for k in range(n)])
    x_power_j = kron_power(x, j)
    d = x_power_j[i].dict()

    return list(d.items()[0][0])


def get_index_from_key(key, j, n):
    r"""

    NOTES:

    - We assume n >= 2. Notice that if n=1, we would return always that:
    first_occurence = 0.

    TO-DO:

    - Include some bounds check?

    - Case j = sum(key) = 0. kron_power(x, 0) = 1

    """

    x = polygen(QQ, ['x'+str(1+k) for k in range(n)])
    x_power_j = kron_power(x, j)

    for i, monomial in enumerate(x_power_j):
        if ( list(monomial.dict().keys()[0]) == key):
            first_occurence = i
            break

    return first_occurence


def log_norm(A, p='inf'):
    r"""Compute the logarithmic norm of a matrix.

    INPUTS:

    * "A" - A rectangular (Sage dense) matrix of order n. The coefficients can be either real or complex.

    * "p" - (default: 'inf'). The vector norm; possible choices are 1, 2, or 'inf'.

    OUTPUT:

    * "lognorm" - The log-norm of A in the p norm.

    TO-DO:

    - Add support for a Numpy array for all values of p. (added - not tested).

    - Add support for an arbitrary p >= 1 vector norm. (how?)

    - Check assumed shape. (not sure if I want this)

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
        raise ValueError('Value of p not understood or not implemented.')

def characteristics(F, n, k):
    r""" c = characteristics(F, n, k)
    where c is a dictionary containing information about the norms of the matrices in F.

    INPUTS:

    - "F" : list of matrices in a Numpy sparse format. The method toarray should be available.

    TO-DO:

    - Accept an optional parameter (params) that specifies:
        - norms chosen
    """

    import scipy as sp
    from scipy import inf
    from numpy.linalg import norm

    c = dict()

    c['norm_Fi_inf'] = [norm(F[i].toarray(), ord=inf) for i in range(k)]

    c['log_norm_F1_inf'] = log_norm(F[0], p='inf')

    if k > 1:
        if c['norm_Fi_inf'][0] != 0:
            c['beta0_const'] = c['norm_Fi_inf'][1]/c['norm_Fi_inf'][0]
        else:
            c['beta0_const'] = 'inf'

    return c
    