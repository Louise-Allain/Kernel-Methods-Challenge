import numpy as np
from scipy.spatial.distance import cdist

class Gaussian:
    def __init__(self, sigma=1.):
        self.sigma = sigma
        
    def kernel(self, X, Y):
        return np.exp(-.5 * cdist(X, Y)**2 / self.sigma**2)

class Linear:
    def kernel(self, X, Y):
        return X @ Y.T

class Polynomial:
    def __init__(self, degree=2, coef0=1):
        self.degree = degree
        self.coef0 = coef0
    
    def kernel(self, X, Y):
        return (np.dot(X, Y.T) + self.coef0)**self.degree

class Sigmoid:
    def __init__(self, gamma=0.01, coef0=0.0):
        self.gamma = gamma
        self.coef0 = coef0
    
    def kernel(self, X, Y):
        return np.tanh(self.gamma * np.dot(X, Y.T) + self.coef0)

class Laplacian:
    def __init__(self, sigma=1.):
        self.sigma = sigma
    
    def kernel(self, X, Y):
        dist_matrix = cdist(X, Y, 'cityblock') 
        return np.exp(-dist_matrix / self.sigma)

class PolynomialAnova:
    def __init__(self, degree=2, gamma=1.0, coef0=1.0):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
    
    def kernel(self, X, Y):
        return (self.gamma * np.dot(X, Y.T) + self.coef0)**self.degree
    
class Chi2():
    def __init__(self, gamma=1):
         self.gamma = gamma
    
    def kernel(self, X, Y):
        
        n, d = X.shape
        m = Y.shape
        
        X_reshaped = X[:, np.newaxis, :]
        Y_reshaped = Y[np.newaxis, :, :]
        
        return np.sum((X_reshaped - Y_reshaped) ** 2 / (X_reshaped + Y_reshaped + 1e-10), axis=2) / 2