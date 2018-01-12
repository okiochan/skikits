import numpy as np
import matplotlib.pyplot as plt
import classification_map as Map
import LOO
from sklearn import neighbors, datasets
import sklearn
    
def KNNski(x, r, X,Y, W='uniform'):
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.RadiusNeighborsClassifier(r, weights=W, metric='seuclidean')
    clf.fit(X, Y)
    x = np.atleast_2d(x)
    return clf.predict(x)
      
#--------------------- main --------------------

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
Y = iris.target

# ks, err = LOO.LOO(X, Y, maxK=140, classifier = KNNski)
# bestR = ks[np.argmin(err)]

# print("error for RadiusNeighborsClassifier")
# print( err[np.argmin(err)] / X.shape[0])

# plt.plot(ks, err)
# plt.show()

clf = neighbors.KNeighborsClassifier(6, weights='uniform')
clf.fit(X, Y)
def classifier(x, y):
    input = np.atleast_2d(np.float_([x,y]))
    return clf.predict(input)

Map.plot(classifier, X, Y, ticks=200)

