import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

from kmeans import KMeans

def bag_of_visual_words(X, centroids):
    '''Compute the histogram of the centroids for SIFT
    @param :
        - X the descriptors
        - centroids the pre-computed centroids
    @return :
        - the histogram of centroids of the descriptors'''
    n = X.shape[0]
    c = centroids.shape[0]
    histogram = np.zeros(c, dtype=int)
    
    for i in range(n):
        cluster = np.argmin(np.linalg.norm(X[i] - centroids, axis=1), axis=0)
        histogram[cluster] += 1
    return histogram

def sift_preprocessing(X, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, n_centroids):
    '''Operates the SIFT preprocessing
    @params :
        - X the dataset
        - nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold the SIFT parameters
        - n_centroids the number of centroids
    @return :
        - new features'''
    descriptors = []
    sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma, centroids = None)
    
    print('     Descriptors extraction...')
    for x in X:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        descriptors += [sift.detectAndCompute(x, None)[1]]
    
    if not centroids:
        print('     Centroids computation...')
        all_descriptors = np.concatenate(descriptors)
        kmeans = KMeans(n_centroids=n_centroids)
        kmeans.fit(all_descriptors)
        centroids = kmeans.centroids

    print('     Bag of Visual Words...')
    bovw = []
    for x in descriptors:
        bovw += [bag_of_visual_words(x, centroids)]
    return bovw, centroids

def preprocessing(X, Y, X_t, preprocessing_method, data_augmentation, orientations=8, pixels_per_cells=4, nfeatures=0, 
                  nOctaveLayers=4, contrastThreshold=0.05, edgeThreshold=10, sigma=1, n_centroids = 50):

    X_train = []
    X_test = []
    Y_train = Y

    n = X.shape[0]
    m, M = np.min(X), np.max(X)

    # Reshaping
    for x in X:
        img = []
        for i in range(1024):
            img += [[(x[i]-m)/(M-m), (x[i+1024]-m)/(M-m), (x[i+1024*2]-m)/(M-m)]]
        img = np.reshape(img, (32, 32, 3))
        X_train += [img]
    
    for x in X_t:
        img = []
        for i in range(1024):
            img += [[(x[i]-m)/(M-m), (x[i+1024]-m)/(M-m), (x[i+1024*2]-m)/(M-m)]]
        img = np.reshape(img, (32, 32, 3))
        X_test += [img]

    # Data augmentation
    if data_augmentation:
        print("Data augmentation...")
        X_train_augmented = []
        Y_train_augmented = []
        for i in range(n):
            x = X_train[i]
            y = Y[i]
            X_train_augmented += [x]
            X_train_augmented += [cv2.flip(x, 1)]
            Y_train_augmented += [y]
            Y_train_augmented += [y]
        X_train = X_train_augmented
        Y_train = Y_train_augmented

    if preprocessing_method == 'grayscaling':
        print('Grayscaling...')
        i = 0
        for i, x in enumerate(X_train):
            X_train[i] = cv2.cvtColor(np.array(x, dtype=np.float32), cv2.COLOR_RGB2GRAY)
        
        for i, x in enumerate(X_test):
            X_test[i] = cv2.cvtColor(np.array(x, dtype=np.float32), cv2.COLOR_RGB2GRAY)

    elif preprocessing_method == 'hog':
        print('Histogram of Gradients...')
        for i, x in enumerate(X_train):
            X_train[i] = hog(x, orientations=orientations, pixels_per_cell=(pixels_per_cells, pixels_per_cells), channel_axis=-1)
        for i, x in enumerate(X_test):
            X_test[i] = hog(x, orientations=orientations, pixels_per_cell=(pixels_per_cells, pixels_per_cells), channel_axis=-1)

    elif preprocessing_method == 'sift':
        print('Scale-Invariant Feature Transform...')
        X_train, centroids = sift_preprocessing(X_train, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, n_centroids)
        X_test, _ = sift_preprocessing(X_test, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, n_centroids, centroids = centroids)
    
    elif preprocessing_method != 'data_augmentation':
        print("Unvalid preprocessing method.")

    return np.array(X_train), np.array(Y_train), np.array(X_test)