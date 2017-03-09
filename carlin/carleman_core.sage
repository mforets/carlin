r"""
This is the core model for Carleman linearization.
It contains:
- functions to compute Carleman linearization
- functions to export the results

AUTHORS:

- Marcelo Forets (2016-12) First version

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
# Working numerical libraries
#===============================================

import numpy as np

import scipy
from scipy import inf
from scipy.io import savemat

import scipy.sparse as sp
from scipy.sparse import kron, eye
import scipy.sparse.linalg

#===============================================
# Carleman libraries
#===============================================

from src.carleman_utils import get_Fj_from_model, kron_power, characteristics

#===============================================
# Libraries for handling polytopes
#===============================================

from lib.polyFunctions_core import polyhedron_sup_norm, PolyhedronToHSpaceRep, chebyshev_center

#===============================================
# Functions to compute Carleman linearization
#===============================================

def transfer_matrices(N, F, n, k):
    r""" A = transfer_matrices(N, F, n, k)

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

    for i in [1..N-1]:
        newRow = []
        for j in range(k):
            L = kron(A[i-1][j], eye(n))
            R = kron(eye(n^(i)), F[j])
            newRow += [np.add(L, R)]
        A.append(newRow)

    return A


def truncated_matrix(N, *args, **kwargs):
    r""" BN = truncated_matrix(N, *args, **kwargs)

    INPUTS:

    - "N" : order of truncation.

    - "input_format" : sequence of matrices $F_j$ (list).

    - "F" : the dimension of the state-space.

    - "input_format" : the order of the polynomial vector field. It is equal to len(F).


    OUTPUTS:

    - "A" : the transfer matrices $A^{i}_{i+j-1}$ that correspond to
            $i = 1, \ldots , N$. It is given as a list of lists.
            Each inner list has dimension $k$.

    TO-DO:

    - Update docstring.

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

    var('t')
    eps(t) = norm_x0_hat*exp(norm_F1_tilde*t)/(1+beta0-beta0*exp(norm_F1_tilde*t))*(beta0*(exp(norm_F1_tilde*t)-1))^N

    return [Ts, eps]


#===============================================
# Functions to export Carleman linearization
#===============================================

def export_model_to_mat(model_filename, F=None, n=None, k=None, **kwargs):

    from scipy.io import savemat

    if '.sage' in model_filename:
        mat_model_filename = model_filename.replace(".sage", ".mat")
    elif '.mat' in model_filename:
        mat_model_filename = model_filename
    else:
        raise ValueError("Expected .sage or .mat file format in model filename.")


    got_Fj = False if F is None else True

    if not got_Fj:
        [F, n, k] = get_Fj_from_model(model_filename)

    d = dict()

    d['F'] = F
    d['n'] = n
    d['k'] = k

    savemat(mat_model_filename, d)

    return



def carleman_export(model_filename, target_filename, N, x0, **kwargs):
    r""" Compute Carleman linearization and export to a MAT file.

    INPUTS:

    OUTPUTS:

    EXAMPLES:



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
        norm_x0_hat = max([nx0^i for i in [1.0..k-1]])

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
