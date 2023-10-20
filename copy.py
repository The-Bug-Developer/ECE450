################################################################
#                                                              #
# Zachary DeLuca                                               #
# ECE 450                                                      #
# Exam 2                                                       #
# Due: Oct 06                                                  #
#                                                              #
################################################################
import numpy as np                                             #
import matplotlib . pyplot as plt                              #
import scipy as sp                                             #
import scipy . signal as sig                                   #
import pandas as pd                                            #       
import time                                                    #
import math                                                    #
import cmath                                                   #
from scipy . fftpack import fft , fftshift                     #
################################################################

from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf

s = 1

num = [0,0,s**2]
den = [0,1,1]
den = np.convolve(den,[1,1,1])



system = sig.lti(num,den)
w,Hmag,Hphase = sig.bode(system)

plt.figure(figsize= (10,5))
plt.title('Fifth Order Low Pass',size = 20)
plt.axis([0.1,10,0,1])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.yticks([0,0.1,0.5,0.707,1])
plt.grid(which='both')
plt.semilogx(w,10**(0.05*Hmag),'k')

dt = 0.001
NN = 50000
TT = np.arange(0,NN*dt,dt)
y=np.zeros(NN)
f=np.zeros(NN)

A,B,C,D = sig.tf2ss(num,den)
x = np.zeros(np.shape(B))

omega = 1
for i in range(NN):
    f[i] = sin(omega*i*dt)
    
plt.figure(figsize = (10,5))
plt.suptitle('Omega = 1',size = 15)
plt.subplot(2,1,1)
plt.title('Input')
plt.grid()
plt.plot(TT,f,'k')

for i in range(NN):
    x = x + dt*A.dot(x) + dt*B*f[i]          
    y[i] = C.dot(x) + D*f[i]
    
plt.subplot(2,1,2)
plt.title('Output')
plt.grid()
plt.plot(TT,y,'k')