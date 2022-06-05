import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster
import scipy as sp

#region notatki
#fig9
# odległość, przypisać do najbliższego
# przysunąć centroidy na środek klastra(średnia po punktach niebieskich)
# znowu odległość i przypisać
# ileś razy(aż centroidy nie zmieniają miejsca/ilość iteracji)

#C - współrzędne pkt
#Cx - wektor z przynależnością do klastra
#4. nie trzeba
#2. tylko dla numerycznych
#endregion


def kmeans(x, k, iter):
    #random centroids
    C = []
    for _ in range(k):
        rn = np.random.randint(0, len(x))
        C.append(x[rn])
    C = np.array(C)

    #main loop
    distances = sp.spatial.distance.cdist(x, C, 'euclidean')
    P = [np.argmin(i) for i in distances]
    for i in range(iter):
        centroids = []
        for j in range(k):
            centroids.append(x[P == j, :].mean(axis=0))
        centroids = np.vstack(centroids)
        distances = sp.spatial.distance.cdist(x, C, 'euclidean')
        P = [np.argmin(i) for i in distances]
    return P, C

#ladowanie z pliku
csv = pd.read_csv('autos.csv')
X = [csv['length'], csv['width'], csv['city-mpg']]
X = np.transpose(np.array(X))

#kmeans
result, cetroids = kmeans(X, 3, 1000)

pca = PCA(n_components=2)
Xr = pca.fit_transform(X)

#wykres
#for i in np.unique(result):
#    plt.scatter(X[result == i , 0] , X[result == i , 1] , label = i)
#plt.scatter(np.transpose(cetroids)[0], np.transpose(cetroids)[1], color='red')
plt.scatter(Xr[:,0],Xr[:,1],c=result)
#plt.legend()
plt.show()

print('end')