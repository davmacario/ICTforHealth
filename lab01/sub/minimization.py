import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(1)
np.random.seed(315054)

class SolveMinProbl:
    """
    This class is used to solve quadratic minimization problems of the LLS type
    """

    def __init__(self, y=np.ones((3,)), A=np.eye(3)):
        self.matr = A
        self.Np = y.shape[0]
        self.Nf = A.shape[1]
        self.vect = y
        self.sol = np.zeros((self.Nf, ), dtype=float)
        return

    def plot_w_hat(self, title='Solution'):
        w_hat = self.sol
        n = np.arange(self.Nf)
        plt.figure()
        plt.plot(n, w_hat)
        plt.xlabel('n')
        plt.ylabel('$\hat{w}(n)$')
        plt.title(title)
        plt.grid()
        plt.show()
        return

    def print_result(self, title):
        print(title, ': ')
        print('The optimum weight vector is: ')
        print(self.sol)
        return

# Define class SolveLLS which inherits from SolveMinProbl (Same methods work)


class SolveLLS(SolveMinProbl):
    """
    ---------------------------------------------------------
    This class is used to solve minimization problems by 
    means of the LLS method
    ---------------------------------------------------------
    The result used is w_hat = (A^T*A)^{-1}*A^T*y
    ---------------------------------------------------------
    """

    def run(self):
        A = self.matr
        y = self.vect
        w_hat = np.linalg.inv(A.T@A)@(A.T@y)
        self.sol = w_hat
        self.min = np.linalg.norm(A@w_hat-y)**2
        return


# Define class SolveGrad which inherits from SolveMinProbl (Same methods work)
class SolveGrad(SolveMinProbl):
    """
    -----------------------------------------------
    This class is used to solve iteratively 
    minimization problems using the gradient method
    -----------------------------------------------
    The termination condition is given by a maximum 
    number of iterations whose default value is 100
    -----------------------------------------------
    """

    def run(self, gamma=1e-3, Nit=100):
        """ 
        ----------------------------------------------------------
        This method is used to run the Gradient Algorithm 
        having defined the problem to solve
        ----------------------------------------------------------
        Parameters:
        - gamma: algorithm hyperparameter (multiplies the gradient 
          at each iteration)
        - Nit: number of iterations (stopping condition)
        ----------------------------------------------------------
        The solution is then stored in the 'sol' 
        attribute (defined in SolveMinProbl)
        ----------------------------------------------------------
        """
        # Define vector self.err to store the values of the error
        # at each iteration (each row contains n. iter and error value)
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect

        w = np.random.rand(self.Nf, 1)

        for it in range(Nit):
            grad = 2*(A.T@((A@w) - y))
            w = w - gamma*grad

            # First element of self.err is the iteration number,
            # second one is the actual error
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm((A@w) - y)**2

        self.sol = w
        self.min = self.err[it, 1]

    def plot_err(self, title='Square error', logy=0, logx=0):
        """
        This function allows to plot the error 
        value at each iteration of the gradient method
        """
        err = self.err
        plt.figure()

        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1])
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 1):
            plt.semilogx(err[:, 1], err[:, 1])
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
        return


class SteepestDescent(SolveMinProbl):
    """
    This class is used to solve minimization problems using the steepest descent method
    """

    def run(self, stoppingCondition='iterations', Nit=20, eps=1e-8):
        """ 
        -------------------------------------------------------------------------
        Solve the minimization problem with the Steepest Descent algorithm
        -------------------------------------------------------------------------
        2 stopping conditions:
        - 'iterations' (default): stop at a specific inumber of iterations (Nit)
        - 'epsilon': stop when the norm of the gradient vector is lower than a 
          specific value (eps)
        -------------------------------------------------------------------------
        Nit: number of iterations (default: 20)
        eps: maximum gradient norm (default: 1e-10)
        -------------------------------------------------------------------------
        """
        A = self.matr
        y = self.vect
        Np = self.Np
        Nf = self.Nf

        # In the quadreatic problem, the Hessian matrix is constant
        Hess = 2*(A.T@A)

        # Initial w: random
        w = np.random.rand(self.Nf,)

        if stoppingCondition == 'epsilon':
            grad = np.ones((Nf,))
            errList = []
            iter = 0
            while np.linalg.norm(grad) >= eps:
                grad = 2*A.T@(A@(w.reshape(len(w), 1)) - y.reshape(len(y), 1))

                # gamma = (np.linalg.norm(grad)**2)/((grad.T@Hess)@grad)
                numer = np.linalg.norm(grad)**2
                denom = float((grad.T@Hess)@grad)

                gamma = numer/denom

                w = w - gamma*(grad.reshape(len(grad),))

                newrow = [iter, np.linalg.norm(
                    A@(w.reshape(len(w), 1)) - y.reshape(len(y), 1))**2]
                errList.append(newrow)

                iter += 1
            # By appending to a list and only then creating an array the code runs noticeably faster
            self.err = np.array(errList)
        
        #elif stoppingCondition == 'iterations':
        else:       # This is the default solution
            self.err = np.zeros((Nit, 2), dtype=float)
            for it in range(Nit):

                grad = 2*A.T@(A@(w.reshape(len(w), 1)) - y.reshape(len(y), 1))

                # gamma = (np.linalg.norm(grad)**2)/((grad.T@Hess)@grad)
                numer = np.linalg.norm(grad)**2
                denom = float((grad.T@Hess)@grad)

                gamma = numer/denom

                w = w - gamma*(grad.reshape(len(grad),))

                newrow = [it, np.linalg.norm(
                    A@(w.reshape(len(w), 1)) - y.reshape(len(y), 1))**2]

                self.err[it, :] = np.copy(newrow)

        self.sol = w.reshape(len(w),)
        self.min = self.err[-1, 1]

        return self.sol

    def plot_err(self, title='Square error', logy=0, logx=0):
        """
        This function allows to plot the error 
        value at each iteration of the Steepest Descent
        method
        """
        err = self.err
        plt.figure()

        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1])
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1])
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
        return


# %% This section will only run if the file 'minimization.py'
# is run "directly" (i.e., if it is called as % python3 minimization.py)
if __name__ == '__main__':
    Np = 100
    Nf = 4

    A = np.random.randn(Np, Nf)
    w = np.random.randn(Nf,)
    y = A@w
    m = SteepestDescent(y, A)
    m.run(stoppingCondition='epsilon')
    m.print_result('Steepest Descent')
    m.plot_w_hat('Steepest Descent')
