r"""
This algorithm implementss the time-triggered scheme to use with Carleman linarization. 

AUTHOR:

- Marcelo Forets (June 2017 at VERIMAG - UGA)
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

from carlin.io import solve_ode_exp
from carlin.transformation import get_Fj_from_model, truncated_matrix

from sage.plot.graphics import Graphics
from sage.plot.plot import plot
from sage.plot.plot import list_plot

def solve_time_triggered(AN, N, x0, tini, T, NRESETS, NPOINTS):
    r"""
    Time-triggered algorithm to use with Carleman linarization.

    INPUT:

    - ``AN`` -- sparse matrix

    - ``N`` -- integer, truncation order

    - ``x0`` -- list, initial condition

    - ``tini`` -- initial time

    - ``T`` -- final time  

    - ``NRESETS`` -- integer, fixed number of resets  

    - ``NPOINTS`` -- integer, number of computation points

    OUTPUT:

    Solution as a collection of lists, each list corresponding to the solution 
    of a given chunk.
    """
    tdom = srange(tini, T, (T-tini)/(NPOINTS-1)*1., include_endpoint=True)

    # number of samples in each chunk
    CHUNK_SIZE = int(NPOINTS/(NRESETS+1))

    # number of variables
    n = AN.shape[0]

    sol_tot = []
    tdom_k = tdom[0:CHUNK_SIZE]
    x0_k = x0
    sol_chunk_k = solve_ode_exp(AN, x0=x0_k, N=N, tini=tdom_k[0], T=tdom_k[-1], NPOINTS=CHUNK_SIZE)
    # sol_chunk_k contain the solution in high-dim, so we shall save the projection to low dim
    sol_tot.append(sol_chunk_k[:, 0:n+1])

    for i in range(1, NRESETS+1):
        tdom_k = tdom[CHUNK_SIZE*(i)-1:CHUNK_SIZE*(i+1)]
        x0_k = list(sol_chunk_k[-1, 0:2])
        sol_chunk_k = solve_ode_exp(AN, x0=x0_k, N=N, tini=0, T=tdom_k[-1]-tdom_k[0], NPOINTS=CHUNK_SIZE+1)
        sol_tot.append(sol_chunk_k[:, 0:n+1])
    return sol_tot

def plot_time_triggered(model, N, x0, tini, T, NRESETS, NPOINTS, j, **kwargs):
    r"""
    Solve and plot the time-triggered algorithm for Carleman linarization.

    INPUT:

    - ``model`` -- PolynomialODE

    - ``N`` -- integer, truncation order

    - ``x0`` -- list, initial condition

    - ``tini`` -- initial time

    - ``T`` -- final time  

    - ``NRESETS`` -- integer, fixed number of resets  

    - ``NPOINTS`` -- integer, number of computation points

    - ``j`` -- integer in `0\ldots n-1`, variable to plot against time 

    OUTPUT:

    Solution as a collection of lists, each list corresponding to the solution 
    of a given chunk.
    NOTES:

    Other optional keyword argument include:

    - ``color`` -- color to be plotted
    """
    G = Graphics()

    if 'color' in kwargs:
        color=kwargs['color']
    else:
        color='blue'

    Fj = get_Fj_from_model(model.funcs(), model.dim(), model.degree())
    AN = truncated_matrix(N, *Fj, input_format="Fj_matrices")
    solution = solve_time_triggered(AN, N, x0, tini, T, NRESETS, NPOINTS)

    # number of samples in each chunk
    CHUNK_SIZE = int(NPOINTS/(NRESETS+1))

    tdom = srange(tini, T, (T-tini)/(NPOINTS-1)*1., include_endpoint=True)
    tdom_k = tdom[0:CHUNK_SIZE]

    G += list_plot(zip(tdom_k, solution[0][:, j]), plotjoined=True, 
               linestyle="dashed", color=color, legend_label="$N="+str(N)+", r="+str(NRESETS)+"$")

    for i in range(1, NRESETS+1):
        # add point at switching time?
        G += point([tdom_k[0], x0_k[1]], size=25, marker='x', color=color)

        # add solution for this chunk
        G += list_plot(zip(tdom_k, solution[i][:, j]), plotjoined=True, linestyle="dashed", color=color)

    # add numerical solution of the nonlinear ODE
    # solution of the nonlinear ODE
    S = model.solve(x0=x0, tini=tini, T=T, NPOINTS=NPOINTS)
    x_t = lambda i : S.interpolate_solution(i)
    G += plot(x_t(j), tini, T, axes_labels = ["$t$", "$x_{"+str(j)+"}$"], gridlines=True, color="black")

    return G
