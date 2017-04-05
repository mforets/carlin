# Van der pol oscillator
# ======================

# dimension of state-space
n=2

# define the vector of symbolic variables
x = polygens(QQ, ['x'+str(1+i) for i in range(n)])

# vector field (n-dimensional)
f = [None] * n

f[0] = x[1]
f[1] = x[1] - x[0]^2*x[1] - x[0]