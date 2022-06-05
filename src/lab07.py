import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

def makeMGArr(start, finish):
    xVal = np.linspace(start,finish)
    yVal = np.linspace(start,finish)
    mgArr = []
    for i in xVal:
        for j in yVal:
            mgArr.append([i, j])
    return mgArr

#2.
class knn:
    def __init__(self, nneighbors = 1):
        self.nneighbors = nneighbors

    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, Xarr):
        resultArray = []
        for i in range(len(Xarr)):
            distanceArr = np.zeros(shape=(len(self.X)), dtype=object)
            for j in range(len(self.X)):
                distance = np.sqrt((Xarr[i][0]-self.X[j][0])**2+(Xarr[i][1]-self.X[j][1])**2)
                distanceArr[j] = distance
            ind = np.argsort(distanceArr)
            classCnt = 0
            for j in range(self.nneighbors):
                classCnt += self.y[ind[j]]
            if self.nneighbors - classCnt < classCnt:
                resultArray.append(0)
            else:
                resultArray.append(1) 
        return resultArray

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

#3.1)
X,y = make_classification(
    n_samples = 100,
    n_features= 2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=3
)

#3.2)
knnObj = knn(1)
knnObj.fit(X,y)
predicted=knnObj.predict(makeMGArr(-3,3))

#3.3)
XX, YY = np.meshgrid(np.linspace(-3,3), np.linspace(-3,3))
matplotlib.pyplot.scatter(X[:,0],X[:,1],c=y)
matplotlib.pyplot.contour(XX, YY, np.reshape(predicted,(50,50),order='F'))
matplotlib.pyplot.show()

#3.4)
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'])
knn2 = KNeighborsClassifier(n_neighbors=2)
knn2.fit(X_train, y_train)
y_prediction = knn2.predict(X_test)
print(np.mean(y_test == y_prediction))

#3.5)
pca = PCA(n_components=2)
Xr = pca.fit(iris.data).transform(iris.data)
#Xr = pca.inverse_transform(Xr)

knnObj2 = KNeighborsClassifier(n_neighbors=1)
knnObj2.fit(Xr, iris.target)
predicted = knnObj2.predict(makeMGArr(-6,6))

XX, YY = np.meshgrid(np.linspace(-6,6), np.linspace(-6,6))
matplotlib.pyplot.scatter(Xr[:,0],Xr[:,1],c=iris.target)
matplotlib.pyplot.contour(XX, YY, np.reshape(predicted,(50,50),order='F'))
matplotlib.pyplot.show()

#3.6)
def crossValidLeaveOneOut(X, y, nneighbors):
    result = []
    X=list(X)
    y=list(y)
    knnObj = KNeighborsClassifier(n_neighbors=nneighbors)
    for i in range(len(X)):
        toPredictX = X.pop(0)
        toPredictY = y.pop(0)
        knnObj.fit(X,y)
        result.append(knnObj.predict([toPredictX]) == toPredictY)
        X.append(toPredictX)
        y.append(toPredictY)
    return np.sum(result)/len(result)

print(np.array([['nneighbours=1',crossValidLeaveOneOut(X, y, 1)],['nneighbours=2',crossValidLeaveOneOut(X, y, 2)],['nneighbours=3',crossValidLeaveOneOut(X, y, 3)],['nneighbours=4',crossValidLeaveOneOut(X, y, 4)],['nneighbours=5',crossValidLeaveOneOut(X, y, 5)]]))

knnBrute = KNeighborsClassifier(n_neighbors=5,algorithm='brute')
knnKdTree = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree')
for i in range(4):
    X,y = make_classification(
    n_samples = np.power(10,i+1),
    n_features= 2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=3
    )
    
    start=time.time()
    knnBrute.fit(X,y)
    knnBrute.predict(makeMGArr(-3,3))
    end=time.time()

    print('brute-force dla',np.power(10,i+1), 'elementow:',end-start,'s')

    start=time.time()
    knnKdTree.fit(X,y)
    knnKdTree.predict(makeMGArr(-3,3))
    end=time.time()

    print('kd-drzewa dla',np.power(10,i+1), 'elementow:',end-start,'s')

