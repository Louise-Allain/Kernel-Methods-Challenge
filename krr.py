import numpy as np 

class KernelRR:
    
    def __init__(self,kernel,lmbda):
        self.lmbda = lmbda                    
        self.kernel = kernel    
        self.alpha = None 
        self.b = None
        self.support = None
        self.type='ridge'
        
    def fit(self, X, y):
        self.support = X
        N = len(X)
        K = self.kernel(X, X)
        self.alpha = np.linalg.inv(K + self.lmbda * np.eye(N)) @ y
        self.b = np.mean(y - K @ self.alpha)
        
    def regression_function(self,x):
        return self.kernel(x, self.support) @ self.alpha

    def predict(self, X):
        """ Predict y values in {-1, 1} """
        return self.regression_function(X)+self.b