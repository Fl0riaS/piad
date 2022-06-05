from itertools import groupby
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats

f = open("demofile2.txt", "a")
f.write("Now the file has more content!")
f.close()

#Tworzenie tabeli
d = {'data': ['2020-03-01', '2020-03-01', '2020-03-01', '2020-03-01', '2020-03-01'], 'A': np.random.rand(5), 'B': np.random.rand(5), 'C': np.random.rand(5)}
df = pd.DataFrame(data=d)

#Wygeneruj tabele
intDict = {'A': np.random.randint(1,100,20), 'B': np.random.randint(1,100,20),'C': np.random.randint(1,100,20)}
intDF = pd.DataFrame(data=intDict)
intDF.index.name='ID'
print(intDF.head(3))
print(intDF.tail(3))
print(intDF.index.name)
#wyswietl bez indexow
print(intDF.iloc[np.random.randint(0,19,5)])
print(intDF.loc[:,'A'])
print(intDF.loc[:,'A':'B'])
print(intDF.iloc[0:3,0:2])
print(intDF.iloc[5,])
print(intDF.iloc[[0,5,6,7],[1,2]])

#describe
test=intDF.describe()
print(test.T.iloc[2,2])
test2=list(test.T.columns)
print(intDF.describe()>0)
#srednia po wierszach
print(stats.describe(intDF, axis=1).mean)
#srednia po kolumnach
print(stats.describe(intDF).mean)

#concat(laczenie df)
concatDict1={'A': np.random.randint(1,100,5), 'B': np.random.randint(1,100,5),'C': np.random.randint(1,100,5)}
concatDF1=pd.DataFrame(data=concatDict1)
concatDict2={'A': np.random.randint(1,100,5), 'B': np.random.randint(1,100,5),'C': np.random.randint(1,100,5)}
concatDF2=pd.DataFrame(data=concatDict2)
concatDFResult=pd.concat([concatDF1,concatDF2])
concatTransposed=pd.DataFrame.transpose(concatDFResult)


#sortowanie
sortDict={'X': [5, 4, 3, 2, 1], 'Y': ['a', 'b', 'a', 'b', 'b']}
sortDF = pd.DataFrame(data=sortDict, index=np.arange(5))
sortDF.index.name='id' 
print(sortDF.sort_values(by=['X']))
print(sortDF.sort_values(by=['Y'],ascending=False))


#Wypelnianie danych
wdDF=pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])
wdDF.index.name='id'
print(wdDF)
#Wypelnianie kolumny B jedynkami
wdDF['B']=1
#Wpisanie 10 w wierzu 1 kolumnie 2
wdDF.iloc[1,2]=10
#Dla wartosci mniejszych od zera liczba przeciwna
wdDF[wdDF<0]=-wdDF
print(wdDF)


#Uzupelnianie danych
df=pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])
df.index.name='id'
#Wpisywanie NaN w podanych miejscach
df.iloc[[0, 3], 1] = np.nan
#Zamiana NaN na liczbe
df.fillna(0, inplace=True)
#zamiana danych na inne
df.iloc[[0, 3], 1] = np.nan
df=df.replace(to_replace=np.nan,value=-9999)
#wypisanie miejsc z nullami
df.iloc[[0, 3], 1] = np.nan
print(pd.isnull(df))


#Grupowanie danych
slownik = {'Day': ['Mon', 'Tue', 'Mon', 'Tue', 'Mon'], 'Fruit': ['Apple','Apple', 'Banana', 'Banana', 'Apple'], 'Pound': [10, 15, 50, 40, 5], 'Pro-fit':[20, 30, 25, 20, 10]}
df3 = pd.DataFrame(slownik)
print(df3)
#grupuje wedlug dni i sumuje wartosci
print(df3.groupby('Day').sum())
#grupuje wedlug dwoch kolumn i sumuje wartosci
print(df3.groupby(['Day','Fruit']).sum())


#zadania
#1)
dict={'x': [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']}
df = pd.DataFrame(dict)
print(df.groupby('y').mean())
#2)
print(pd.DataFrame.value_counts(df))
#3)
autosCsv=pd.read_csv('./src/files/autos.csv')
#autosCsv=np.loadtxt('./src/files/autos.csv', delimiter=',',dtype='str')
#4)
print(autosCsv[['make','city-mpg','highway-mpg']].groupby('make').mean().mean(axis=1))
#5)
print(autosCsv[['Unnamed: 0','fuel-type']].groupby('fuel-type').count())
#6)
autosCsvPolyfit1=np.polyfit(np.array(autosCsv['city-mpg']),autosCsv['length'],1)
autosCsvPolyfit2=np.polyfit(np.array(autosCsv['city-mpg']),autosCsv['length'],2)
#7) wspołczynnik korelacji
zad7Cor,x= sp.stats.pearsonr(autosCsv['width'], autosCsv['wheel-base'])
print(zad7Cor)
#8)
slope, intercept, r, p, stderr = sp.stats.linregress(autosCsv['width'], autosCsv['wheel-base'])
fig, ax = plt.subplots()
ax.plot(autosCsv['width'], autosCsv['wheel-base'], '*')
ax.plot(autosCsv['width'], intercept + slope * autosCsv['width'])
plt.show()
#9)
#scipy.stats.gaussian_kde()

#10)

#11)



print('end')