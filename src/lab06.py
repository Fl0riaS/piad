import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib
from matplotlib import pyplot

def wiPCA(x,p):
    w,v=np.linalg.eig(np.cov(x.T))
    indexes=np.argsort(w[::-1])
    w=w[indexes]
    v=v[:,indexes]
    return np.dot(x,v[:,0:p])

def wiPCA2(x,p):
    w,v=np.linalg.eig(np.cov(x.T))
    indexes=np.argsort(w[::-1])
    w=w[indexes]
    #v=v[:,indexes]
    return w


#1a)
objects200=np.dot(np.random.randn(2,2),np.random.randn(2,100)).T
#1b)
matplotlib.pyplot.scatter(objects200[:,0],objects200[:,1],color="green")
#matplotlib.pyplot.show();

#1c)
pca = PCA(n_components=1)
#Xr = pca.fit(objects200.T).transform(objects200.T)
Xr=wiPCA(objects200,1)
w=Xr*wiPCA2(objects200,1).T

matplotlib.pyplot.scatter(w[:,0],w[:,1],color="red")
matplotlib.pyplot.show()

#2a)
iris = datasets.load_iris()
pca = PCA(n_components=2)
Xr = pca.fit(iris.data).transform(iris.data)
y=iris.target

matplotlib.pyplot.scatter(Xr[:,0],Xr[:,1],c=y)
matplotlib.pyplot.show()

#2b)
Xr=wiPCA(iris.data,2)
#2c)
matplotlib.pyplot.scatter(Xr[:,0],Xr[:,1],c=y)
matplotlib.pyplot.show()

#3a)
digits=datasets.load_digits()
#3b)
Xr = wiPCA(digits.data,2)
#3c)
pca = PCA().fit(digits.data)
matplotlib.pyplot.plot(np.cumsum(wiPCA2(digits.data,64)))
matplotlib.pyplot.show()

#3d)
matplotlib.pyplot.scatter(Xr[:,0],Xr[:,1],c=digits.target)
matplotlib.pyplot.show()

print('end')