from audioop import avg
from statistics import mean
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
import scipy
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

#region notatki
#3. bez random state
#zadanie 1 na 4(ocena)
#sprawozdanie 2 strony maks, pdf

#krzywa roc determinuje jak dobrze algorytm determinuje przynależność, czym dalej od linii(0.5 chyba) tym lepiej
#endregion

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
X,y = make_classification(
    n_samples = 100,
    n_classes = 2,
    n_clusters_per_class=2,
    n_features=2,
    n_informative=2,
    n_repeated=0,
    n_redundant=0
)

#2)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

#3)
gaus = GaussianNB()
discr = QuadraticDiscriminantAnalysis()
neigh = KNeighborsClassifier()
svc = SVC(gamma='auto', probability=True)
tree = DecisionTreeClassifier()
arr=[gaus, discr, neigh, svc, tree]

testTime=np.zeros((5,100))
trainTime=np.zeros((5,100))
roc_auc=np.zeros((5,100))
f1=np.zeros((5,100))
prec=np.zeros((5,100))
recall=np.zeros((5,100))
accu=np.zeros((5,100))

for i in range(100):
    #split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    #gauss
    start=time.time()
    gaus.fit(X_train, y_train)
    end=time.time()
    trainTime[0,i]=end-start

    start=time.time()
    y_pred=gaus.predict(X_test)
    end=time.time()
    testTime[0,i]=end-start

    accu[0,i]=metrics.accuracy_score(y_test,y_pred)
    recall[0,i]=metrics.recall_score(y_test,y_pred)
    prec[0,i]=metrics.precision_score(y_test,y_pred)
    f1[0,i]=metrics.f1_score(y_test,y_pred)
    roc_auc[0,i]=metrics.roc_auc_score(y_test,y_pred)

    if i==99:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))
        ax1.scatter(X_test[:,0],X_test[:,1],c=y_test)
        ax2.scatter(X_test[:,0],X_test[:,1],c=y_pred)
        ax3.scatter(X_test[:,0],X_test[:,1],c=findDifferences(y_test,y_pred))
        plt.show()

        y_pred_proba=gaus.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="ROC curve (area = %0.2f)" % roc_auc[0,i],)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

        predictedMeshGrid = gaus.predict(makeMGArr(-4,4))
        XX, YY = np.meshgrid(np.linspace(-4,4), np.linspace(-4,4))
        plt.scatter(X_test[:,0],X_test[:,1],c=y_pred)
        plt.contour(XX, YY, np.reshape(predictedMeshGrid,(50,50),order='F'))
        plt.show()



    #discr
    start=time.time()
    discr.fit(X_train, y_train)
    end=time.time()
    trainTime[1,i]=end-start

    start=time.time()
    y_pred=discr.predict(X_test)
    end=time.time()
    testTime[1,i]=end-start

    accu[1,i]=metrics.accuracy_score(y_test,y_pred)
    recall[1,i]=metrics.recall_score(y_test,y_pred)
    prec[1,i]=metrics.precision_score(y_test,y_pred)
    f1[1,i]=metrics.f1_score(y_test,y_pred)
    roc_auc[1,i]=metrics.roc_auc_score(y_test,y_pred)

    if i==99:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))
        ax1.scatter(X_test[:,0],X_test[:,1],c=y_test)
        ax2.scatter(X_test[:,0],X_test[:,1],c=y_pred)
        ax3.scatter(X_test[:,0],X_test[:,1],c=findDifferences(y_test,y_pred))
        plt.show()

        y_pred_proba=discr.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="ROC curve (area = %0.2f)" % roc_auc[0,i],)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

        predictedMeshGrid = discr.predict(makeMGArr(-4,4))
        XX, YY = np.meshgrid(np.linspace(-4,4), np.linspace(-4,4))
        plt.scatter(X_test[:,0],X_test[:,1],c=y_pred)
        plt.contour(XX, YY, np.reshape(predictedMeshGrid,(50,50),order='F'))
        plt.show()

    #neigh
    start=time.time()
    neigh.fit(X_train, y_train)
    end=time.time()
    trainTime[2,i]=end-start

    start=time.time()
    y_pred=neigh.predict(X_test)
    end=time.time()
    testTime[2,i]=end-start

    accu[2,i]=metrics.accuracy_score(y_test,y_pred)
    recall[2,i]=metrics.recall_score(y_test,y_pred)
    prec[2,i]=metrics.precision_score(y_test,y_pred)
    f1[2,i]=metrics.f1_score(y_test,y_pred)
    roc_auc[2,i]=metrics.roc_auc_score(y_test,y_pred)

    if i==99:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))
        ax1.scatter(X_test[:,0],X_test[:,1],c=y_test)
        ax2.scatter(X_test[:,0],X_test[:,1],c=y_pred)
        ax3.scatter(X_test[:,0],X_test[:,1],c=findDifferences(y_test,y_pred))
        plt.show()

        y_pred_proba=neigh.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="ROC curve (area = %0.2f)" % roc_auc[0,i],)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

        predictedMeshGrid = neigh.predict(makeMGArr(-4,4))
        XX, YY = np.meshgrid(np.linspace(-4,4), np.linspace(-4,4))
        plt.scatter(X_test[:,0],X_test[:,1],c=y_pred)
        plt.contour(XX, YY, np.reshape(predictedMeshGrid,(50,50),order='F'))
        plt.show()

    #svc
    start=time.time()
    svc.fit(X_train, y_train)
    end=time.time()
    trainTime[3,i]=end-start

    start=time.time()
    y_pred=svc.predict(X_test)
    end=time.time()
    testTime[3,i]=end-start

    accu[3,i]=metrics.accuracy_score(y_test,y_pred)
    recall[3,i]=metrics.recall_score(y_test,y_pred)
    prec[3,i]=metrics.precision_score(y_test,y_pred)
    f1[3,i]=metrics.f1_score(y_test,y_pred)
    roc_auc[3,i]=metrics.roc_auc_score(y_test,y_pred)

    if i==99:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))
        ax1.scatter(X_test[:,0],X_test[:,1],c=y_test)
        ax2.scatter(X_test[:,0],X_test[:,1],c=y_pred)
        ax3.scatter(X_test[:,0],X_test[:,1],c=findDifferences(y_test,y_pred))
        plt.show()

        y_pred_proba=svc.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="ROC curve (area = %0.2f)" % roc_auc[0,i],)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

        predictedMeshGrid = svc.predict(makeMGArr(-4,4))
        XX, YY = np.meshgrid(np.linspace(-4,4), np.linspace(-4,4))
        plt.scatter(X_test[:,0],X_test[:,1],c=y_pred)
        plt.contour(XX, YY, np.reshape(predictedMeshGrid,(50,50),order='F'))
        plt.show()

    #tree
    start=time.time()
    tree.fit(X_train, y_train)
    end=time.time()
    trainTime[4,i]=end-start

    start=time.time()
    y_pred=tree.predict(X_test)
    end=time.time()
    testTime[4,i]=end-start

    accu[4,i]=metrics.accuracy_score(y_test,y_pred)
    recall[4,i]=metrics.recall_score(y_test,y_pred)
    prec[4,i]=metrics.precision_score(y_test,y_pred)
    f1[4,i]=metrics.f1_score(y_test,y_pred)
    roc_auc[4,i]=metrics.roc_auc_score(y_test,y_pred)

    if i==99:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))
        ax1.scatter(X_test[:,0],X_test[:,1],c=y_test)
        ax2.scatter(X_test[:,0],X_test[:,1],c=y_pred)
        ax3.scatter(X_test[:,0],X_test[:,1],c=findDifferences(y_test,y_pred))
        plt.show()

        y_pred_proba=tree.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="ROC curve (area = %0.2f)" % roc_auc[0,i],)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

        predictedMeshGrid = tree.predict(makeMGArr(-4,4))
        XX, YY = np.meshgrid(np.linspace(-4,4), np.linspace(-4,4))
        plt.scatter(X_test[:,0],X_test[:,1],c=y_pred)
        plt.contour(XX, YY, np.reshape(predictedMeshGrid,(50,50),order='F'))
        plt.show()

#dataFrame creation
dict = {'testTime': [mean(testTime[0,:])*500,mean(testTime[1,:])*500,mean(testTime[2,:])*500,mean(testTime[3,:])*500,mean(testTime[4,:])*500], 'trainTime': [mean(trainTime[0,:])*500,mean(trainTime[1,:])*500,mean(trainTime[2,:])*500,mean(trainTime[3,:])*500,mean(trainTime[4,:])*500], 'roc_auc': [mean(roc_auc[0,:]),mean(roc_auc[1,:]),mean(roc_auc[2,:]),mean(roc_auc[3,:]),mean(roc_auc[4,:])], 'f1_score': [mean(f1[0,:]),mean(f1[1,:]),mean(f1[2,:]),mean(f1[3,:]),mean(f1[4,:])], 'precision_score': [mean(prec[0,:]),mean(prec[1,:]),mean(prec[2,:]),mean(prec[3,:]),mean(prec[4,:])], 'recall_score': [mean(recall[0,:]),mean(recall[1,:]),mean(recall[2,:]),mean(recall[3,:]),mean(recall[4,:])], 'accuracy_score': [mean(accu[0,:]),mean(accu[1,:]),mean(accu[2,:]),mean(accu[3,:]),mean(accu[4,:])]}
df = pd.DataFrame(data=dict)
dfTrans = df.transpose()

#dataFrame visualization
rcParams.update({'figure.autolayout': True})
dfTrans.plot(kind="bar")
plt.legend(['Gaus','disc','neigh','svc','tree'])
plt.show()

print('end')