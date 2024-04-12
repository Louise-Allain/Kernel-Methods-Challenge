import numpy as np
import pandas as pd

from kernel import *
from kpca import KernelPCA
from multiclass import MulticlassOvR, MulticlassOvO
from preprocessing import preprocessing

# Data path
X_path = 'data-challenge-kernel-methods-2023-2024\Xtr.csv'
Y_path = 'data-challenge-kernel-methods-2023-2024\Ytr.csv'
test_path = 'data-challenge-kernel-methods-2023-2024\Xte.csv'

# Preprocessing choice
data_augmentation = True
preprocessing_method = 'hog' #hog, sift, grayscaling, data_augmentation or none
PCA = False

# hog parameters
orientations = 8
pixels_per_cells = 4

# sift parameters
nfeatures = 0
nOctaveLayers = 4
contrastThreshold = 0.05
edgeThreshold = 10
sigma_sift = 1
n_centroids = 50

# PCA parameters
r = 100
degree_pca = 6
gamma_pca = 1
coef0_pca = 0
sigma_pca = 1
# Linear(), Polynomial(degree = degree_pca, coef0 = coef0_pca), Gaussian(sigma = sigma_pca), 
# Sigmoid(gamma = gamma_pca, coef0 = coef0_pca), Laplacian(sigma = sigma_pca)
# PolynomialAnova(degree = degree_pca, gamma = gamma_pca, coef0 = coef0_pca)
kernel_pca = PolynomialAnova(degree = degree_pca, gamma = gamma_pca, coef0 = coef0_pca)

# model choice and parameters
model = 'krr' #krr, svc
paradigm = 'ovo' #ovo, ovr
lmbda_model = 1 # if model = krr
C_model = 1 # if model = svm

# kernel choice and parameters
degree = 6
gamma = 1
coef0 = 0
sigma = 1
# Linear(), Polynomial(degree = degree, coef0 = coef0), Gaussian(sigma = sigma), 
# Sigmoid(gamma = gamma, coef0 = coef0), Laplacian(sigma = sigma)
# PolynomialAnova(degree = degree, gamma = gamma, coef0 = coef0)
kernel_model = PolynomialAnova(degree = degree, gamma = gamma, coef0 = coef0)

# True if you want to perform cross validation
cv = False

if __name__ == '__main__':

    # load the dataset
    print('Data loading...')
    X_train = np.array(pd.read_csv(X_path,header=None,sep=',',usecols=range(3072))) 
    X_test = np.array(pd.read_csv(test_path,header=None,sep=',',usecols=range(3072))) 
    Y_train = np.array(pd.read_csv(Y_path,sep=',',usecols=[1])).squeeze() 

    # preprocessing
    if preprocessing_method:
        X_train, Y_train, X_test = preprocessing(X_train, Y_train, X_test, preprocessing_method, data_augmentation, orientations = orientations, 
                      pixels_per_cells = pixels_per_cells, nfeatures = nfeatures, nOctaveLayers = nOctaveLayers, 
                      contrastThreshold = contrastThreshold, edgeThreshold = edgeThreshold, sigma = sigma_sift, 
                      n_centroids = n_centroids)
    
    if PCA:
        print('Principal Component Analysis...')
        pca = KernelPCA(kernel_pca.kernel, r=r)
        pca.compute_PCA(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    
    if paradigm == 'ovo':
        classifier = MulticlassOvO(model = model, lmbda=lmbda_model, C = C_model, kernel = kernel_model.kernel)
    elif paradigm == 'ovr':
        classifier = MulticlassOvR(model = model, lmbda = lmbda_model, C = C_model, kernel = kernel_model.kernel)
    else :
        print('Incorrect paradigm')

    if cv:
        print('Cross validation...')
        acc = []
        len_batch = int(X_train.shape[0] / 5)
        for batch in range(5):
            X_te, y_te = X_train[len_batch*batch:(len_batch*(batch+1))], Y_train[len_batch*batch:(len_batch*(batch+1))]
            X_tr, y_tr = np.delete(X_train, np.arange(len_batch*batch,(len_batch*(batch+1))), 0), np.delete(Y_train, np.arange(len_batch*batch,(len_batch*(batch+1))), 0)
            if data_augmentation:
                X_te = np.array([X_te[i] for i in range(0, len_batch, 2)])
                y_te = np.array([y_te[i] for i in range(0, len_batch, 2)])
            classifier.fit(X_tr, y_tr)
            y_pred = classifier.predict(X_te)
            accuracy = sum(1 for true, pred in zip(y_te, y_pred) if true == pred) / 1000
            print('Batch ', batch, 'accuracy :', accuracy)
            acc += [accuracy]
        print('Mean accuracy :', np.mean(acc))


    print('Classification...')
    print('     Training...')
    classifier.fit(X_train, Y_train)

    print('     Predicting...')
    Y_test_pred = classifier.predict(X_test) 

    Y_test_pred = {'Prediction' : Y_test_pred} 
    dataframe = pd.DataFrame(Y_test_pred) 
    dataframe.index += 1 
    dataframe.to_csv('Yte.csv',index_label='Id')