from turtle import end_fill
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
import scipy as sp

#DYSKRETYZACJA
#1
def dyssin(f,fs):
    t=np.arange(0,1,1/fs)
    s=np.sin(2*np.pi*f*t)
    return t,s

#3
fs=21
t,s=dyssin(10,fs)
plt.plot(t,s)
plt.show()

#4
#Twierdzenie o próbkowaniu, twierdzenie Nyquista–Shannona

#5
#zjawisko aliasingu

#6
img = mpimg.imread('bricks.png')
plt.imshow(img,interpolation='nearest')
plt.show()
plt.imshow(img,interpolation='lanczos')

#7
imgplot = plt.imshow(img)
plt.show()

#KWANTYZACJA
#2 
print(img.ndim)
#3
print(img[0,0].size)
#4
xsize=int(img[0].size/4)
ysize=int(img.size/4/xsize)
for x in range(ysize):
    for y in range(xsize):
        img[x,y,0]=0.21*img[x,y,0]+0.72*img[x,y,1]+0.07*img[x,y,2]
        img[x,y,1]=0.21*img[x,y,0]+0.72*img[x,y,1]+0.07*img[x,y,2]
        img[x,y,2]=0.21*img[x,y,0]+0.72*img[x,y,1]+0.07*img[x,y,2]

#for x in range(ysize):
#    for y in range(xsize):
#        img[x,y,0]=(img[x,y,0]+img[x,y,1]+img[x,y,2])/3
#        img[x,y,1]=(img[x,y,0]+img[x,y,1]+img[x,y,2])/3
#        img[x,y,2]=(img[x,y,0]+img[x,y,1]+img[x,y,2])/3

#for x in range(ysize):
#    for y in range(xsize):
#        img[x,y,0]=(max(img[x,y,0],img[x,y,1],img[x,y,2])+min(img[x,y,0],img[x,y,1],img[x,y,2]))/2
#        img[x,y,1]=(max(img[x,y,0],img[x,y,1],img[x,y,2])+min(img[x,y,0],img[x,y,1],img[x,y,2]))/2
#        img[x,y,2]=(max(img[x,y,0],img[x,y,1],img[x,y,2])+min(img[x,y,0],img[x,y,1],img[x,y,2]))/2

imgplot = plt.imshow(img)
plt.show()

#5
plt.hist(img.flatten())
histogram, binEdges = np.histogram(img, bins=256, range=(0, 1))
plt.plot(binEdges[0:-1], histogram)
plt.show()


#6
#plt.hist(img.flatten(), bins=16, range=(0, 1))
histogram, binEdges = np.histogram(img, bins=16, range=(0, 1))
plt.plot(binEdges[0:-1], histogram)
plt.show()
#7
binlenght=np.size(binEdges)
imgreduced=img.copy()
for x in range(ysize):
    for y in range(xsize):
      for i in range(binlenght-1):
          if imgreduced[x,y,0]<binEdges[i+1]:
            imgreduced[x,y,0]=(binEdges[i]+binEdges[i+1])/2
            imgreduced[x,y,1]=(binEdges[i]+binEdges[i+1])/2
            imgreduced[x,y,2]=(binEdges[i]+binEdges[i+1])/2
            break
        
imgplot = plt.imshow(imgreduced)
plt.show()
            


#BINARYZACJA
#1,2
img2 = mpimg.imread('cosnagradiencie.png')

xsize=int(img2[0,:,:].size/3)
ysize=int(img2.size/3/xsize)
for x in range(ysize):
    for y in range(xsize):
        img2[x,y,0]=0.21*img2[x,y,0]+0.72*img2[x,y,1]+0.07*img2[x,y,2]
        img2[x,y,1]=0.21*img2[x,y,0]+0.72*img2[x,y,1]+0.07*img2[x,y,2]
        img2[x,y,2]=0.21*img2[x,y,0]+0.72*img2[x,y,1]+0.07*img2[x,y,2]

imgplot = plt.imshow(img2)
plt.show()

histogram, binEdges = np.histogram(img2, bins=256, range=(0, 1))
plt.plot(binEdges[0:-1], histogram)
plt.show()



#3
def binariseImage(value, binEdges):
    for i in range(np.size(binEdges)):
        if binEdges[i] > value:
            return i/np.size(binEdges)

def binariseImage2(image,value):
    xsize=int(image[0].size/3)
    ysize=int(image.size/3/xsize)
    for x in range(ysize):
        for y in range(xsize):
            if image[x,y,0]<value:
                image[x,y,0]=0
                image[x,y,1]=0
                image[x,y,2]=0
            else:
                image[x,y,0]=1
                image[x,y,1]=1
                image[x,y,2]=1
    return image

#test=binariseImage(0.3,binEdges)
#print(test)
#mini = sp.signal.argrelextrema(histogram,np.less)
img3=binariseImage2(img2,0.3)
imgplot3 = plt.imshow(img3)
plt.show()






print('end')
