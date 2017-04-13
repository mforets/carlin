.. nodoctest

Context
~~~~~~~~~

Carleman linearization is an established method in mathematical nonlinear control. 
It consists in embedding a nonlinear system of differential equations 

.. MATH::

    $$
    x'(t)=f(x(t))+u(t)g(x(t))
    $$

of finite dimension into a system of bilinear differential equations 

.. MATH::

    $$
    y'(t)=Ay(t)+u(t)By(t)
    $$

of *infinite dimension*. By truncating the obtained bilinear system at 
finite orders, one obtains a systematic way of creating arbitrary-order approximation 
of the solutions of the nonlinear system. 

Polynomial ODE's
~~~~~~~~~~~~~~~~~

An important subclass of nonlinear systems are polynomial differential equations.
Indeed, many systems can be rewritten as polynomial vector fields by introducing
more variables and, in fact, any polynomial system can be reduced to a second-order
polynomial one. Consider the initial-value problem (IVP)

.. MATH::

    \begin{equation}
    \left\{
    \begin{aligned}
    x'(t) &= F_1 x + F_2 x^{[2]} + \ldots +  F_k x^{[k]}, \\
     x(0) &= x_0 \in \mathbb{R}^n,\qquad t \in I.
    \end{aligned}
    \right.
    \label{eq:S_ode}
    \end{equation}
    
We assume that the matrix-valued functions `F_j \in \mathbb{R}^{n \times n^j}` are independent of `t`. 
Here 

.. MATH::

    \begin{equation*}
    x^{[i]} :=  \underset{\text{i times}}{\underbrace{x\otimes \cdots \otimes x}}
    \end{equation*}
    
denotes the `i`-th *Kronecker power* of `x \in \mathbb{R}^n`, a convenient notation to express 
all possible products of elements of a vector up to a given order. 

Carleman embedding
~~~~~~~~~~~~~~~~~~~~

It can be shown that for all `i \in \mathbb{N}`, `y_i := x^{[i]}` satisfies the infinite-dimensional IVP

.. MATH::

    \begin{equation}
    \left\{
    \begin{aligned}
    y'(t) &= \mathcal{A} y(t), \\
    y_i(0) &= (x(0))^{[i]} = x_0^{[i]},\qquad \forall i \in \mathbb{N},
    \end{aligned}
    \right.
    \label{eq:InfSysMatrix}
    \end{equation}
    
where `y := (y_1,y_2,\ldots)^\transp` and `\mathcal{A}` is the infinite-dimensional block upper-triangular matrix

.. MATH::

    \begin{equation*}
    \mathcal{A} := \begin{pmatrix}
    A_1^1 & A_2^1 & A_3^1 & \ldots & A_{k}^1 & 0 & 0 & \ldots \\ 
    0& A_2^2 & A_3^2 & \ldots & A_{k}^2 & A_{k+1}^2&0 & \ldots \\
    0 & 0 & A_3^3 & \ldots  & A_{k}^3 & A_{k+1}^3  & A_{k+2}^3 & \ldots \\   
    \vdots &  \vdots &  \vdots &    & \vdots &\vdots & \vdots &\\
    \end{pmatrix}. \label{eq:Adefi}
    \end{equation*}
    
This particular structure can be exploited both from a theoretical and from a practical point of view.
Finally, we use 

.. MATH::

    \begin{equation*}
    x(t) := y_1(t),\qquad \hat{x}(t) := \hat{y}_1(t) 
    \end{equation*}
    
to denote the solution of the exact and truncated systems respectively, projected into `\mathbb{R}^n`. 
The associated *error* is

.. MATH::

    \begin{equation*}
    \varepsilon(t) := x(t) - \hat{x}(t).
    \end{equation*}