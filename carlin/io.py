r"""
Carleman linearization input/output functions.

Features:

- functions to load model polynomial ODE's
- auxiliary mathematical functions

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
from scipy import inf
import scipy.sparse.linalg
from scipy.sparse import dok_matrix

# Sage objects: Rings and Polynomials
from sage.rings.integer import Integer
from sage.rings.all import RR, QQ
from sage.rings.real_double import RDF
from sage.rings.polynomial.polynomial_ring import polygens 
from sage.modules.free_module_element import vector

#==========================
# Functions to load a model
#==========================

def load_model(model_filename):
    r""" Read an input ODE system.

    INPUT:

    - ``model_filename`` -- string with the model filename

    OUTPUT:

    - ``f`` -- list of multivariate polynomials which describes the system of ODE's,
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
    r""" Transform an input model of a polynomial vector field into standard
    form as a sum of Kronecker products.
    
    The model can be given either as an external file (``model_filename``), or
    as the tuple ``(f, n, k)``.

    INPUT:

    - ``model_filename`` -- string containing the filename

    OUTPUT:

    - ``F`` -- F is a list of sparse matrices `F_1, ..., F_k`.
     These are formatted in dok (dictionary-of-keys) form.

    - ``n`` -- dimension of the state-space of the system

    - ``k`` -- degree of the system
    """
    if model_filename is not None and f is None:
        got_model_by_filename = True
    elif model_filename is not None and f is not None and n is not None and k is None:
        k = n; n = f; f = model_filename;
        got_model_by_filename = False
    else:
        raise ValueError("Either the model name or the vector field (f, n, k) should be specified.")

    if got_model_by_filename:
        [f, n, k] = load_model(model_filename)

    # create the collection of sparse matrices Fj
    F = [dok_matrix((n, n**i), dtype=np.float64) for i in range(1, k+1)]

    # read the powers appearing in each monomial
    dictionary_f = [fi.dict() for fi in f];

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

#===============================================
# Functions to solve ODE's
#===============================================

def solve_linearized_ode(AN=None, x0=None, N=2, tini=0, T=1, NPOINTS=400):
    """
    Solve Carleman linearized 1st order ODE using dense matrix-vector multiplications.
    """
    def initial_state_kron(x0, N):
        from carlin.transformation import kron_power

        y0 = kron_power(x0, 1)
        for i in range(2, N+1):
            y0 += kron_power(x0, i)
        return vector(y0)

    # transform to x0, x0^[2], ..., x0^[N]
    y0 = initial_state_kron(x0, N)

    # time domain
    from numpy import linspace
    t_dom = linspace(tini, T, num=NPOINTS)

    # compute solution
    from sage.functions.log import exp
    sol = [exp(AN*ti)*y0 for ti in t_dom]

    return sol
