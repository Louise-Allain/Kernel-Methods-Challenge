from itertools import combinations
import numpy as np
from tqdm import tqdm

from krr import KernelRR
from svc import KernelSVC

class MulticlassOvR:
    
    def __init__(self, kernel, model, lmbda = None, C = None):
        self.model = model
        self.C = C
        self.lmbda = lmbda                    
        self.kernel = kernel    
        self.classifiers = {}  
        
    def fit(self, X, y):
        classes = np.unique(y)
        for c in classes:
            y_binary = np.where(y == c, 1, -1)  
            if self.model == 'krr':
                classifier = KernelRR(kernel = self.kernel, lmbda = self.lmbda)
            elif self.model == 'svc':
                classifier = KernelSVC(kernel = self.kernel, C = self.C)
            else:
                print('Incorrect model')
            classifier.fit(X, y_binary)
            self.classifiers[c] = classifier
        
    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.classifiers)))
        for c, classifier in self.classifiers.items():
            predictions[:, c] = classifier.predict(X)
        return np.argmax(predictions, axis=1)

class MulticlassOvO:
    
    def __init__(self, kernel, model, lmbda=None, C=None):
        self.lmbda = lmbda 
        self.model = model  
        self.C = C            
        self.kernel = kernel
        self.class_pairs = None
        self.classifiers = {} 
        
    def fit(self, X, y):
        classes = np.unique(y)
        self.class_pairs = list(combinations(classes, 2))
        
        for c1, c2 in tqdm(self.class_pairs):
            X_subset = X[(y == c1) | (y == c2)]
            y_binary = np.where(y[(y == c1) | (y == c2)] == c1, 1, -1)
            
            if self.model == 'krr':
                classifier = KernelRR(kernel=self.kernel, lmbda = self.lmbda)
            elif self.model == 'svc':
                classifier = KernelSVC(kernel=self.kernel, C = self.C)
            else:
                print('Invalid model')
            classifier.fit(X_subset, y_binary)
            
            self.classifiers[(c1, c2)] = classifier
        
    def predict(self, X):
        predictions = np.zeros((X.shape[0], 10))
        for i, (c1, c2) in enumerate(self.class_pairs):
            classifier = self.classifiers[(c1, c2)]
            pred = classifier.predict(X)
            for i, p in enumerate(pred):
                if np.sign(p) == 1:
                    predictions[i, c1] += 1
                else:
                    predictions[i, c2] += 1
                        
        predicted_classes = np.argmax(predictions, axis=1)
        
        return predicted_classes