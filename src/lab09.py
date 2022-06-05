import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
import scipy
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.mixture import GaussianMixture

#region notatki
#lab9
#4. avg=none
#9. bez convex hulla i 3d
#7. tylko dla aglomeracyjnych!
#kmeansy i gmm wbudowane 
#3 cześć dodatkowa(bez 8,9)
#endregion


#region 1)
iris=datasets.load_iris()
X=iris.data
Y=iris.target
#endregion
#region 2)
agloSingle = AgglomerativeClustering(linkage = 'single',n_clusters=3).fit(X)#Odległość między dwoma klastrami jest minimalną odległością między obserwacją w jednym klastrze a obserwacją w innym klastrze. Sprawdza się, gdy klastry są wyraźnie oddzielone. 
agloAvg = AgglomerativeClustering(linkage = 'average',n_clusters=3).fit(X) #Odległość między dwoma klastrami jest średnią odległością między obserwacją w jednym klastrze a obserwacją w innym klastrze. 
agloCmplt = AgglomerativeClustering(linkage = 'complete',n_clusters=3).fit(X)#Odległość między dwoma klastrami jest maksymalną odległością między obserwacją w jednym klastrze a obserwacją w innym klastrze. Może być wrażliwy na występowanie outlier’ów.
agloWard = AgglomerativeClustering(linkage = 'ward',n_clusters=3).fit(X)#Odległość między dwoma klastrami jest sumą kwadratów odchyleń od punktów do centroidów. Ten sposób dąży do zminimalizowania sumy kwadratów wewnątrz klastra. 
#endregion
#region 3)
def findperm(clusters, Yreal, Ypred):
    perm=[]
    for i in range(clusters):
        idx = Ypred == i
        newlabel=scipy.stats.mode(Yreal[idx])[0]
        perm.append(newlabel)
    return[perm[label] for label in Ypred]
#znajduje permutacje wyniku zgodną z targetem
# endregion
#region 4)
print(jaccard_score(Y,findperm(150,Y,agloSingle.labels_),average=None))
print(jaccard_score(Y,findperm(150,Y,agloAvg.labels_),average=None))
print(jaccard_score(Y,findperm(150,Y,agloCmplt.labels_),average=None))
print(jaccard_score(Y,findperm(150,Y,agloWard.labels_),average=None))
#endregion
#region 5)
def findDifferences(arr1, arr2):
    res = []
    for i in range(len(arr1)):
        res.append(arr1[i]==arr2[i])
    return res

pca = PCA(n_components=2)
Xr = pca.fit_transform(X)

#single
plt.scatter(Xr[:,0],Xr[:,1],c=Y)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=agloSingle.labels_)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=findDifferences(Y,findperm(150,Y,agloSingle.labels_)))
plt.show()

#avg
plt.scatter(Xr[:,0],Xr[:,1],c=Y)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=agloAvg.labels_)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=findDifferences(Y,findperm(150,Y,agloAvg.labels_)))
plt.show()

#ward
plt.scatter(Xr[:,0],Xr[:,1],c=Y)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=agloWard.labels_)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=findDifferences(Y,findperm(150,Y,agloWard.labels_)))
plt.show()

#complete
plt.scatter(Xr[:,0],Xr[:,1],c=Y)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=agloCmplt.labels_)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=findDifferences(Y,findperm(150,Y,agloCmplt.labels_)))
plt.show()

#kmeans
kmeans=KMeans(3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(Xr[:,0],Xr[:,1],c=Y)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=y_kmeans)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=findDifferences(Y,findperm(150,Y,y_kmeans)))
plt.show()

gauss=GaussianMixture(3)
gauss.fit(X)
y_gauss = gauss.predict(X)

plt.scatter(Xr[:,0],Xr[:,1],c=Y)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=y_gauss)
plt.show()
plt.scatter(Xr[:,0],Xr[:,1],c=findDifferences(Y,findperm(150,Y,y_gauss)))
plt.show()
#endregion
#region 6)
dendrogram(linkage(X, 'single'))
plt.show()

dendrogram(linkage(X, 'average'))
plt.show()

dendrogram(linkage(X, 'complete'))
plt.show()

dendrogram(linkage(X, 'ward'))
plt.show()
#endregion
#region 9)
zooCsv = pd.read_csv('zoo.csv').drop(['animal','type'], axis='columns')

dendrogram(linkage(zooCsv, 'single'))
plt.show()

dendrogram(linkage(zooCsv, 'average'))
plt.show()

dendrogram(linkage(zooCsv, 'complete'))
plt.show()

dendrogram(linkage(zooCsv, 'ward'))
plt.show()
#endregion