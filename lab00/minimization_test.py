import sub.minimization as mymin
import numpy as np

Np = 100
Nf = 4

np.random.seed(315054)

A = np.random.randn(Np, Nf)
w = np.random.randn(Nf,)
y = A@w
m = mymin.SolveLLS(y, A)
m.run()
m.print_result('LLS')
m.plot_w_hat('LLS')
