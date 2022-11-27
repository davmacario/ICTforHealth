import sub.minimization as mymin
import numpy as np

Np = 100
Nf = 4

np.random.seed(315054)

A = np.random.randn(Np, Nf)
w = np.random.randn(Nf, 1)
y = A@w
m = mymin.SolveGrad(y, A)
m.run(gamma=1e-5, Nit=200)
# Methods inherited by SolveMinProbl
m.print_result('LLS')
m.plot_w_hat('LLS')

# Method exclusive to SolveGrad
m.plot_err()
