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

import numpy as np

import scipy
from scipy import inf
import scipy.sparse as sp
import scipy.sparse.linalg

# this is a dependency for load
from sage.rings.integer import Integer   

# Sage objects: Rings and Polynomials
from sage.rings.all import RR, QQ
from sage.rings.real_double import RDF
from sage.rings.polynomial.polynomial_ring import polygens 

#==========================
# Functions to load a model
#==========================

def load_model(model_filename):
    r""" Read an input ODE system.

    INPUTS:

    - ``model_filename`` : string with the model filename

    OUTPUTS:

    - ``f`` : polynomial vector field. Each component belongs to the polynomial ring `\QQ[x_1,\ldots,x_n]`

    - ``n`` : dimension of f

    - ``k`` : degree of f

    TO-DO:

    - Accept file that is not polynomial and try to convert it to polynomial form. 
    See ``automatic_recastic.ipynb`` notebook.

    """
    from sage.structure.sage_object import load
    
    # should define n and f
    load(model_filename)

    k = max( [fi.degree() for fi in f] )

    return [f, n, k]


def get_Fj_from_model(model_filename=None, f=None, n=None, k=None):
    r""" Transform an input model of a polynomial vector field into standard form as a sum of Kronecker products.
    
    
    The model can be given either as an external file (model_filename), or as the tuple ``(f, n, k)``.

    INPUTS:

    - ``model_filename`` : string containing the filename

    OUTPUTS:

    - ``F`` : F is a list of sparse matrices `F1, ..., Fk`. These are formatted in dok (dictionary-of-keys) form.

    - ``n`` : dimension of the state-space of the system

    - ``k`` : degree of the system

    NOTES:

    - There was a problem with sum(1) with Sage's sum, that happens for the scalar case
      (n=1). In that case we can use: from scipy import sum
      However, now that case is handled separately.
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
    F = [sp.dok_matrix((n, n**i), dtype=np.float64) for i in range(1, k+1)]

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

    elif (n==1): #the scalar case is treated separately

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
    