import numpy as np

b=np.array([[1,2,3,4,5], [6,7,8,9,10]])

#TABLICE
#transpozycja macierzy
bT=np.transpose(b)
#arange
bArange=np.arange(0,100,1)
print(bArange)
#linspace
bLinspace=np.linspace(0,2,num=10)
print(bLinspace)
#arange 0-100 skok 5
bArange2=np.arange(0,101,5)

#LICZBY LOSOWE
#rozklad normalny, 20 losowych zaokraglonych do 2 po przecinku
mRand1=np.around(np.random.randn(20), 2)
#100 liczb calkowitych zakres 1-1000
mRand2=np.random.randint(1,1000,100)
#zeros i ones
mZeros=np.zeros((3,2))
mOnes=np.ones((3,2))
#zadanie tablice
aZadTablice=10*np.random.rand(10)
bZadTablice=np.ndarray.astype(aZadTablice,int)
cZadTablice=np.around(aZadTablice,0)
cZadTablice=np.ndarray.astype(cZadTablice,int)

#SELEKCJA DANYCH
sdArray=np.array([[1,2,3,4,5], [6,7,8,9,10]],dtype=np.int32)
sdDim=np.ndim(sdArray)
sdSize=np.size(sdArray)
sdValue2=sdArray[0,1]
sdValue4=sdArray[0,3]
sdRow=sdArray[0,]
sdColumn=sdArray[:,1]
sdMatrix=np.random.randint(0,100,[20,7])
print(sdMatrix[:,0:4])

#OPERACJE MATEMATYCZNE I LOGICZNE
omilMatrixA=10*np.random.rand(3,3)
omilMatrixB=10*np.random.rand(3,3)
omilAdd=omilMatrixA+omilMatrixB
omilMultiply=omilMatrixA*omilMatrixB
omilDivide=omilMatrixA/omilMatrixB
omilPower=np.power(omilMatrixA,omilMatrixB)
print(omilMatrixA<=4)
print(np.logical_and(omilMatrixA>=1,omilMatrixA<=4))
omilDiag=np.sum(np.diagonal(omilMatrixB))
omilDiag2=np.trace(omilMatrixB)

#DANE STATYSTYCZNE
dsSum=np.sum(omilMatrixA)
dsMin=np.min(omilMatrixA)
dsMax=np.max(omilMatrixA)
dsStd=np.std(omilMatrixA)
dsMeanColumns=np.mean(omilMatrixA,axis=0)
dsMeanRows=np.mean(omilMatrixA,axis=1)

#RZUTOWANIE WYMIAROW
rwArray=np.arange(0,50,1)
rwReshape=np.reshape(rwArray,((10,5)))
rwResize=np.resize(rwArray,((10,5)))
rwRavel=np.ravel(rwReshape)
rwArray2=np.arange(0,5,1)
rwArray3=np.arange(0,4,1)


#SORTOWANIE DANYCH
sortArray=np.random.randn(5,5)
sortArray=np.sort(sortArray,axis=1)
sortArray[::-1].sort(axis=0)
sortMatrix=np.array([(1,"MZ","mazowieckie"),(2,"ZP","zachodniopomorskie"),(3,"ML","małopolskie")])
sortMatrix=sortMatrix.reshape(3,3)
sortMatrix = sortMatrix[sortMatrix[:,1].argsort()]
print(sortMatrix[2,2])

#ZADANIA PODSUMOWUJACE
#1)
zad1Matrix=np.random.randint(10, size=(5,10))
zad1Trace=np.trace(zad1Matrix)
print(np.diag(zad1Matrix))
#2)
zad2Array1=np.random.randn(10)
zad2Array2=np.random.randn(10)
zad2ResultArray=zad2Array1*zad2Array2
#3)
zad3Array1=np.random.randint(1,100,10)
zad3Array2=np.random.randint(1,100,10)
zad3ResultArray=np.reshape(zad3Array1,((2,5)))+np.reshape(zad3Array2,((2,5)))
#4)
zad4Matrix1=np.random.randint(10, size=(4,5))
zad4Matrix2=np.random.randint(10, size=(5,4))
zad4ResultMatrix=np.lib.pad(zad4Matrix1,((0,1),(0,0)),'constant', constant_values=(0))+np.lib.pad(zad4Matrix2,((0,0),(0,1)),'constant', constant_values=(0))
#5)
zad5ResultMatrix=np.lib.pad(zad4Matrix1,((0,1),(0,0)),'constant', constant_values=(0))[:,[2,3]]*np.lib.pad(zad4Matrix2,((0,0),(0,1)),'constant', constant_values=(0))[:,[2,3]]
#6)
zad6ArrayNormal=np.random.normal(size=(5,5))
zad6ArrayUniform=np.random.uniform(size=(5,5))
#srednia
zad6Mean1=np.mean(zad6ArrayNormal)
zad6Mean2=np.mean(zad6ArrayUniform)
#odchylenie standardowe
zad6Std1=np.std(zad6ArrayNormal)
zad6Std2=np.std(zad6ArrayUniform)
#wariancja
zad6Var1=np.var(zad6ArrayNormal)
zad6Var2=np.var(zad6ArrayUniform)
#7)
zad7Matrix1=np.random.normal(size=(5,5))
zad7Matrix2=np.random.normal(size=(5,5))
zad7MultiplyStar=zad7Matrix1*zad7Matrix2
zad7Multiply=np.dot(zad7Matrix1,zad7Matrix2)
#dot to iloczyn wektorowy, uzywany do obliczania kata miedzy wektorami
#9)
zad9Array1=np.array([1,2,3,4,5])
zad9Array2=np.array([6,7,8,9,10])
zad9Vstack=np.vstack((zad9Array1,zad9Array2))
zad9Hstack=np.hstack((zad9Array1,zad9Array2))
#Vstack laczy macierze wierszami a hstack kolumnami

print('end')

