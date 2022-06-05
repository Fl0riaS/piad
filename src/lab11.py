from audioop import avg
from statistics import mean
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
import scipy
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, jaccard_score, recall_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
from matplotlib import rcParams
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import auc

def findDifferences(arr1, arr2):
    res = []
    for i in range(len(arr1)):
        res.append(arr1[i]==arr2[i])
    return res

def makeMGArr(start, finish):
    xVal = np.linspace(start,finish)
    yVal = np.linspace(start,finish)
    mgArr = []
    for i in xVal:
        for j in yVal:
            mgArr.append([i, j])
    return mgArr

#1)
X,y = make_classification(n_samples=1501, n_features=2, n_clusters_per_class=1, n_informative=2, n_redundant=0, n_classes=4)

plt.scatter(X[:,0],X[:,1],c=y)
#plt.show()

#2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#3)
roc_auc=np.zeros(8)
f1=np.zeros(8)
prec=np.zeros(8)
recall=np.zeros(8)
accu=np.zeros(8)

svcLinearOvO=OneVsOneClassifier(SVC(kernel='linear', probability=True))
svcRbfOvO=OneVsOneClassifier(SVC(kernel='rbf', probability=True))
LogRegOvO=OneVsOneClassifier(LogisticRegression())
PercOvO=OneVsOneClassifier(Perceptron())
svcLinearOvR=OneVsRestClassifier(SVC(kernel='linear', probability=True))
svcRbfOvR=OneVsRestClassifier(SVC(kernel='rbf', probability=True))
LogRegOvR=OneVsRestClassifier(LogisticRegression())
PercOvR=OneVsRestClassifier(Perceptron())
methodsList = [svcLinearOvO,svcRbfOvO,LogRegOvO,PercOvO,svcLinearOvR,svcRbfOvR,LogRegOvR,PercOvR]

for i in range(len(methodsList)):
  methodsList[i].fit(X_train,y_train)
  y_pred=methodsList[i].predict(X_test)

  fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))
  ax1.scatter(X_test[:,0],X_test[:,1],c=y_test)
  ax2.scatter(X_test[:,0],X_test[:,1],c=y_pred)
  ax3.scatter(X_test[:,0],X_test[:,1],c=findDifferences(y_test,y_pred))
  plt.show()

  accu[i]=metrics.accuracy_score(y_test,y_pred)
  recall[i]=metrics.recall_score(y_test,y_pred,average='macro')
  prec[i]=metrics.precision_score(y_test,y_pred,average='macro')
  f1[i]=metrics.f1_score(y_test,y_pred,average='macro')
  # if i<5:
  #   roc_auc[i]=metrics.roc_auc_score(y_test,y_pred,multi_class='ovo',average='macro')
  # else:
  #   roc_auc[i]=metrics.roc_auc_score(y_test,y_pred,multi_class='ovr')

  y_pred_dec=methodsList[i].decision_function(X_test)

  fpr0, tpr0, _ = roc_curve(y_test==0, y_pred_dec[:,0])
  fpr1, tpr1, _ = roc_curve(y_test==1, y_pred_dec[:,1])
  fpr2, tpr2, _ = roc_curve(y_test==2, y_pred_dec[:,2])
  fpr3, tpr3, _ = roc_curve(y_test==3, y_pred_dec[:,3])

  auc0=auc(fpr0,tpr0)
  auc1=auc(fpr1,tpr1)
  auc2=auc(fpr2,tpr2)
  auc3=auc(fpr3,tpr3)

  roc_auc[i]=np.mean([auc0, auc1, auc2, auc3])

  plt.plot(fpr0, tpr0, label="ROC curve (area = %0.2f)" % auc0)
  plt.plot(fpr1, tpr1, label="ROC curve (area = %0.2f)" % auc1)
  plt.plot(fpr2, tpr2, label="ROC curve (area = %0.2f)" % auc2)
  plt.plot(fpr3, tpr3, label="ROC curve (area = %0.2f)" % auc3)
  plt.plot([0, 1], [0, 1], color="red", linestyle="--")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.legend(loc="lower right")
  plt.show()

  predictedMeshGrid = methodsList[i].predict(makeMGArr(-4,4))
  XX, YY = np.meshgrid(np.linspace(-4,4), np.linspace(-4,4))
  plt.scatter(X_test[:,0],X_test[:,1],c=y_pred)
  plt.contour(XX, YY, np.reshape(predictedMeshGrid,(50,50),order='F'))
  plt.show()


dict = {'accuracy-score': accu, 'recall-score': recall, 'precision-score': prec, 'f1-score': f1, 'roc_auc': roc_auc}
df=pd.DataFrame(data=dict)
dfTrans = df.transpose()

rcParams.update({'figure.autolayout': True})
dfTrans.plot(kind="bar")
plt.legend(['OVO SVC lin','OVO SVC rbf','OVO LogisticRegression','OVO Perceptron','OVR SVC lin','OVR SVC rbf','OVR LogisticRegression','OVR Perceptron'])
plt.show()

print ('end')