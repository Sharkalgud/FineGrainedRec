import numpy as np

def mypdist(X):
    #only does euclidean distance
    #rows are instances
    #assums this is a numpy matrix
    X_rsum = np.sum(X ^ 2, axis = 1)
    D = 
