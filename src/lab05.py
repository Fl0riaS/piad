from math import log2
import pandas as pd

#1
def freq(x, prob=True):
    return x.value_counts(normalize=prob)

#2
def freq2(x, prob=True):
    return x.value_counts(normalize=prob)

#3
def entropy(x):
    y=freq(x)
    result=0
    for i in range(2):
        result+=y[i]*log2(y[i])
    return -result
        
def entropy2(x):
    y=freq2(x)
    print(y)
    result=0
    for i in range(2):
        for j in range(2):
            result+=y[i][j]*log2(y[i][j])
    return -result

def infogain(x,y,z):
    return entropy(x)+entropy(y)-entropy2(z)
    



#4
zoo=pd.read_csv('./src/files/zoo.csv')
#print(freq(zoo[['milk']]))
#print(freq(zoo[['milk']],prob=False))
print(freq2(zoo[['predator','toothed']]))
print(freq2(zoo[['predator','toothed']],prob=False))
#print(entropy(zoo[['milk']]))
print(entropy2(zoo[['predator','toothed']]))
print(infogain(zoo[['hair']],zoo[['aquatic']],zoo[['hair','aquatic']]))

print('end')