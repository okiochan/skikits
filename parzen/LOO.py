import numpy as np
import matplotlib.pyplot as plt
import classification_map as Map
from sklearn import neighbors, datasets

def LOO(X, Y, maxK, classifier):
    n = X.shape[0]
    ks = []
    error = []
    
    for k in range (1, maxK +1) :
        now_error=0
        for id in range (n) :
            x = X[id]
            y = int(Y[id])
            newX = np.delete(X, id, axis=0)
            newY = np.delete(Y, id, axis=0)
            retClass = classifier(x,k,newX,newY)
            if retClass != y :
                now_error += 1
        ks.append(k)
        error.append(now_error)

    return ks, error
