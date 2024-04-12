import numpy as np
from scipy import optimize

class KernelSVC:
    def __init__(self, C, kernel, epsilon=1e-3):
        self.type = "non-linear"
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None

    def fit(self, X, y):
        N = len(y)
        Y = np.diag(y)
        K = self.kernel(X, X)
        b = np.array(N*[0] + N*[self.C])
        A = np.concatenate([-np.eye(N), np.eye(N)])
        
        def loss(alpha):
            return .5 * alpha.T @ Y @ K @ Y @ alpha - alpha.sum()

        def grad_loss(alpha):
            return Y @ K @ Y @ alpha - np.ones_like(alpha)

        fun_eq = lambda alpha: y.T @ alpha
        jac_eq = lambda alpha: y
        fun_ineq = lambda alpha: b - A @ alpha
        jac_ineq = lambda alpha: - A

        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac': jac_ineq})
        
        optRes = optimize.minimize(
            fun=lambda alpha: loss(alpha),
            x0=np.ones(N),
            method="SLSQP",
            jac=lambda alpha: grad_loss(alpha),
            constraints=constraints,
        )
        self.alpha = optRes.x

        support = np.where(self.alpha > self.epsilon)[0]
        self.support = X[support, :]  
        self.b = (y - K @ (self.alpha * y)).mean()   
        self.norm_f = np.sqrt(self.alpha @ Y @ K @ Y @ self.alpha)  
        self.alpha = (self.alpha * y)[support]

    def separating_function(self, x):
        return self.kernel(x, self.support) @ (self.alpha)

    def predict(self, X):
        """Predict y values in {-1, 1}"""
        d = self.separating_function(X)
        return 2 * (d + self.b > 0) - 1