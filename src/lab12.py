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
import scipy.fftpack
import librosa

#zadanie 4 dodatkowe

#1)
s, fs = sf.read('./src/files/jestemstudentem.wav', dtype='float32')

s=s[:,0]

plt.plot(np.arange(0,len(s))/fs, s)
plt.show()

#region zadanie 2
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

splited_size = int(int(fs / 1000) * 20)
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
splited_size = int(int(fs / 1000) * 5)
splited = [s[x:x+splited_size] for x in range(0, len(s), splited_size)]

energy = []
zero_move = []

for splited_item in splited:
    energy.append(Ej(splited_item))
    zero_move.append(Zj(splited_item))

energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
zero_move = (zero_move - np.min(zero_move)) / (np.max(zero_move) - np.min(zero_move))

fig, ax1 = plt.subplots(1, 1)
ax1.set_title('window 5ms')
ax1.plot(np.arange(0,len(s)), s, color='green')
ax1cp = ax1.twiny()
ax1cp.plot(energy, color='red')
ax1cp.plot(zero_move, color='blue')
plt.show()

splited_size = int(int(fs / 1000) * 50)
splited = [s[x:x+splited_size] for x in range(0, len(s), splited_size)]

energy = []
zero_move = []

for splited_item in splited:
    energy.append(Ej(splited_item))
    zero_move.append(Zj(splited_item))

energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
zero_move = (zero_move - np.min(zero_move)) / (np.max(zero_move) - np.min(zero_move))

fig, ax1 = plt.subplots(1, 1)
ax1.set_title('window 50ms')
ax1.plot(np.arange(0,len(s)), s, color='green')
ax1cp = ax1.twiny()
ax1cp.plot(energy, color='red')
ax1cp.plot(zero_move, color='blue')
plt.show()

#2.5)
splited_size = int(int(fs / 1000) * 20)
splited = [s[x:x+splited_size] for x in range(0, len(s), splited_size)]

energy = []
zero_move = []

for splited_item in splited:
    energy.append(Ej(splited_item))
    zero_move.append(Zj(splited_item))

energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
zero_move = (zero_move - np.min(zero_move)) / (np.max(zero_move) - np.min(zero_move))

fig, ax1 = plt.subplots(1, 1)
ax1.set_title('window 20ms + nakladanie 50%')
ax1.plot(np.arange(0,len(s)), s, color='green')
ax1cp = ax1.twiny()
ax1cp.plot(energy, color='red')
ax1cp.plot(zero_move, color='blue')
plt.show()
#endregion

#3.1)
samogloska = s[int(fs*1.00) - 1024:int(fs*1.00) + 1024]
x_samogloska = [int(fs*1.00) - 1024,int(fs*1.00) + 1024]

#3.2)
hamm = np.hamming(len(samogloska))
zamaskowane_okno = samogloska*hamm

#3.3)
amplituda = np.log(np.abs(scipy.fftpack.fft(zamaskowane_okno, 40000)))

#3.4)
fig = plt.figure(constrained_layout=True, figsize=(12, 6))

spec = fig.add_gridspec(3, 4)
ax1 = fig.add_subplot(spec[0, :])
ax1.set_title('25ms')
ax1.plot((np.arange(0,len(s))/fs)*1000, s, color='green')
ax1cp = ax1.twiny()
ax1cp.plot(energy, color='red')
ax1cp.plot(zero_move, color='blue')
ax1cp.axes.get_xaxis().set_visible(False)

x_samogloska = np.array(x_samogloska) / fs * 1000 - 1000
x_samogloska = np.array(x_samogloska).astype(int)
x_samogloska = np.linspace(x_samogloska[0], x_samogloska[1], 6).astype(int)
ax2 = fig.add_subplot(spec[1, 0])
ax2.set_xticklabels(x_samogloska)
ax2.plot(samogloska, color='green')

ax3 = fig.add_subplot(spec[1, 1])
ax3.plot(hamm)

ax4 = fig.add_subplot(spec[1, 2])
ax4.plot(zamaskowane_okno)

ax5 = fig.add_subplot(spec[1, 3])
ax5.set_xticklabels([0, 0, 1, 2, 3, 4])
ax5.plot(amplituda, color='red')

ax6 = fig.add_subplot(spec[2, :])
ax6.plot(amplituda[0:10000], color='red')

plt.show()

#4.2)
a = list(librosa.lpc(samogloska, 20))

#4.4)
tempsize = (2048 - np.shape(a)[0])
temp = [0] * tempsize
a.extend(temp)

#4.5)
widmo = -1 * np.log(np.abs(scipy.fftpack.fft(a)))
widmo = widmo[0:len(widmo) // 2]

fig = plt.figure(constrained_layout=True, figsize=(12, 6))

#4.6)
spec = fig.add_gridspec(3, 4)
ax1 = fig.add_subplot(spec[0, :])
ax1.set_title('25ms')
ax1.plot((np.arange(0,len(s))/fs)*1000, s, color='green')
ax1cp = ax1.twiny()
ax1cp.plot(energy, color='red')
ax1cp.plot(zero_move, color='blue')
ax1cp.axes.get_xaxis().set_visible(False)

x_samogloska = np.array(x_samogloska) / fs * 1000 - 1000
x_samogloska = np.array(x_samogloska).astype(int)
x_samogloska = np.linspace(x_samogloska[0], x_samogloska[1], 6).astype(int)
ax2 = fig.add_subplot(spec[1, 0])
ax2.set_xticklabels(x_samogloska)
ax2.plot(samogloska, color='green')

ax3 = fig.add_subplot(spec[1, 1])
ax3.plot(hamm)

ax4 = fig.add_subplot(spec[1, 2])
ax4.plot(zamaskowane_okno)

ax5 = fig.add_subplot(spec[1, 3])
ax5.set_xticklabels([0, 0, 1, 2, 3, 4])
ax5.plot(amplituda, color='red')

ax6 = fig.add_subplot(spec[2, :])
ax6.plot(amplituda[0:6000], color='blue')
ax6cp = ax6.twiny()
ax6cp.plot(widmo - 5, color='red')

plt.show()