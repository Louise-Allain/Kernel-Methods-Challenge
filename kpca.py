import numpy as np

class KernelPCA:
    
    def __init__(self,kernel, r=2):                             
        self.kernel = kernel          # <---
        self.alpha = None # Matrix of shape N times d representing the d eingenvectors alpha corresp
        self.lmbda = None # Vector of size d representing the top d eingenvalues
        self.support = None # Data points where the features are evaluated
        self.r =r ## Number of principal components

    def compute_PCA(self, X):
        # assigns the vectors
        self.support = X
        N = len(X)
        K = self.kernel(X, X)
        one_n = np.ones((N, N)) / N
        G = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        eigenvalues, eigenvectors = np.linalg.eigh(G / N)
        indices = np.argsort(eigenvalues)
        self.lmbda = eigenvalues[indices][: self.r]
        self.alpha = eigenvectors[indices][: self.r]
        
    def transform(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        G = self.kernel(x, self.support)
        return G @ self.alpha.T