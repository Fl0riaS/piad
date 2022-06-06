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
import sounddevice as sd
import soundfile as sf

#zadanie 4 dodatkowe

#1)
s, fs = sf.read('./src/files/jestemstudentem.wav', dtype='float32')

s=s[:,0]

plt.plot(np.arange(0,len(s))/fs, s)
plt.show()

#2.1)
def Ej(x):
  result = 0
  for i in range(len(x)):
    result += x[i]**2
  return result

def Zj(x):
  result = 0
  for i in range(1,len(x)):
    #print(x[i-1]*x[i])
    if x[i-1]*x[i] < 0:
      result += 1
  return result

splited_size = int(int(fs / 100) * 5)
splited = [s[x:x+splited_size] for x in range(0, len(s), splited_size)]

energy = []
zero_move = []

for splited_item in splited:
    energy.append(Ej(splited_item))
    zero_move.append(Zj(splited_item))

#2.2)
energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
zero_move = (zero_move - np.min(zero_move)) / (np.max(zero_move) - np.min(zero_move))

fig, ax1 = plt.subplots(1, 1)
ax1.set_title('window 20ms')
ax1.plot(np.arange(0,len(s)), s, color='green')
ax1cp = ax1.twiny()
ax1cp.plot(energy, color='red')
ax1cp.plot(zero_move, color='blue')
plt.show()

#2.4)
