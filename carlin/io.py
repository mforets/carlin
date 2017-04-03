r"""
This is the utils module for Carleman linearization.
It contains:
- functions to load a model
- auxiliary mathematical functions

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

import scipy.sparse as sp
from scipy.sparse import kron, eye
import scipy.sparse.linalg
from scipy.io import loadmat, savemat

#===============================================
# Functions to load a model
#===============================================

def load_model(model_filename):
    r""" [f, n, k] = load(model_filename)
    Read an input system.

    INPUTS:

    - "model_filename" : string containin the filename

    OUTPUTS:

    - "f" : polynomial vector field. Each component belongs to the polynomial ring QQ[x1,...,xn]

    - "n" : dimension of f.

    - "k" : degree of f.

    TO-DO:

    - Accept file that is not polynomial and try to convert it to polynomial form. See automatic_recastic.ipynb notebook.

    """

    # should define n and f
    load(model_filename)

    k = max( [fi.degree() for fi in f] )

    return [f, n, k]


def get_Fj_from_model(model_filename=None, f=None, n=None, k=None):
    r""" [F, n, k] = get_Fj_from_model(...)
    Transform an input model of a polynomial vector field into standard form as a sum of Kronecker products.
    The model can be given either as an external file (model_filename), or as the tuple (f, n, k).

    INPUTS:

    - "model_filename" : string containing the filename

    OUTPUTS:

    - "F" : F is a list of sparse matrices F1, ..., Fk. These are formatted in dok (dictionary-of-keys) form.

    - "n" : dimension of the state-space of the system

    - "k" : degree of the system

    EXAMPLE:

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
    F = [sp.dok_matrix((n,n^i), dtype=np.float64) for i in [1..k]]

    # read the powers appearing in each monomial
    dictionary_f = [fi.dict() for fi in f];

    if (n>1):

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
    