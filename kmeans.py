import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        '''Computes the centroids
        params : 
            - X the dataset'''
        # Random initializaion
        centroids_indices = np.random.choice(X.shape[0], size=self.n_clusters, replace=False)
        self.centroids = X[centroids_indices]
    
        for _ in range(self.max_iter):
            # Update centroids
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)
            updated_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
 
            # Check for convergence
            if np.allclose(self.centroids, updated_centroids):
                break

            self.centroids = updated_centroids