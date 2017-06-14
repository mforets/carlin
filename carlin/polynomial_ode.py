r"""
A class to represent a system of polynomial ordinary differential equations (ODEs).

AUTHOR:

- Marcelo Forets (May 2017 at VERIMAG - UGA)
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

from sage.structure.sage_object import SageObject
from sage.plot.plot import list_plot

class PolynomialODE(SageObject):
    """
    This class represents a finite set of polynomial ODEs.
    """
    def __init__(self, f=None, n=None, k=None):
        self._funcs = f
        self._dim = n
        self._degree = k

    def __repr__(self):
        return "A Polynomial ODE in n = %s variables"%self._dim

    def funcs(self):
        """
        Get the right-hand side functions list.
        """
        return self._funcs

    def dim(self):
        """
        Get the dimesion (number of variables).
        """
        return self._dim

    def degree(self):
        """
        Get the degree (max over the degree of all polynomials).
        """
        return self._degree
    deg=degree

    def solve(self, x0=None, tini=0, T=1, NPOINTS=100):
        """
        Solve the polynomial ODE using GSL.

        INPUT:

        - ``model`` -- PolynomialODE, defining the tuple `(f, n, k)`

        - ``N`` -- integer; truncation order

        - ``x0`` -- vector; initial condition

        - ``tini`` -- initial time of simulation

        - ``T`` -- final time of simulation

        - ``NPOINTS`` -- number of points sampled

        - ``xcoord`` -- (default: `0`), x-coordinate in plot

        - ``ycoord`` -- (default: `1`), y coordinate in plot        

        EXAMPLES:

        Let us solve the scalar equation `x' = -2x + 3`::
    
            sage: from carlin.polynomial_ode import PolynomialODE
            sage: x = polygen(QQ, 'x')
            sage: P = PolynomialODE([-2*x+3], 1, 1)
            sage: P.solve(x0=[0], tini=0, T=4)
            
        Let us compute and plot the osolution of the vanderpol ODE::

            sage: from carlin.library import vanderpol
            sage: S = vanderpol(1, 1).solve(x0=[0.5, 1.])
            sage: x1x2 = [S.solution[i][1] for i in range(len(S.solution))]
            sage: LPLOT = list_plot(x1x2, plotjoined=True)
            
        
        """
        from sage.calculus.ode import ode_solver
        from sage.misc.misc_c import prod

        S = ode_solver()

        if x0 is None:
            raise ValueError("to solve, you should specify the initial condition")

        def funcs(t, x):
            f = []
            if self._dim == 1:
                fid = self._funcs[0].dict()
                f = sum([fid[fiex] * x[0]**fiex for fiex in fid.keys()])
                return [f]
            elif self._dim > 1:
                for i, fi in enumerate(self._funcs):
                    fid = fi.dict()
                    row_i = sum([fid[fiex] * prod([x[i]**ai for i, ai in enumerate(fiex)]) for fiex in fid.keys()])
                    f.append(row_i)
                return f
        S.function = funcs

        def jac(t, x):
            n = self._dim
            # jacobian
            dfi_dyj = [[self._funcs[i].derivative(x[j]) for j in range(n)] for i in range(n)]

            # we consider autonomous systems
            dfi_dt = [0]*n

            return dfi_dyj + dfi_dt
        S.jacobian = jac

        # choose integration algorithm and solve
        S.algorithm = "rk4"
        S.ode_solve(y_0=x0, t_span=[tini, T], num_points=NPOINTS)
        return S

    def plot_solution(self, x0=None, tini=0, T=1, NPOINTS=100, xcoord=0, ycoord=1, plotjoined=True, **kwargs):
        """
        Solve and plot for the given coordinates.
    
        INPUT:

        - ``x0`` -- vector; initial condition

        - ``tini`` -- initial time of simulation

        - ``T`` -- final time of simulation

        - ``NPOINTS`` -- number of points sampled

        - ``xcoord`` -- (default: `0`), x-coordinate in plot

        - ``ycoord`` -- (default: `1`), y coordinate in plot
        """
        S = self.solve(x0=x0, tini=tini, T=T, NPOINTS=NPOINTS)
        sol_xy = [(S_ti[1][xcoord], S_ti[1][ycoord]) for S_ti in S.solution]
        return list_plot(sol_xy, **kwargs)