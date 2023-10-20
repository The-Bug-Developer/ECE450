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

wc = 1e3
W = 300

n = 2

num = [0,0,W**2]
den =[1,2*cos(1*pi/(2*n))*W,W**2]

system = sig.lti(num,den)
w,Hmag,Hphase = sig.bode(system,10000)

plt.figure(figsize= (10,5))
plt.subplot(2,1,1)
plt.title('Second Order Low Pass',size = 12)
#plt.axis([0.1,10000,0,1])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.yticks([0,0.1,0.95,0.707,1])
plt.grid(which='both')
plt.semilogx(w,10**(0.05*Hmag),'k')
plt.savefig('Bode.png',dpi=300)


num = [0,0,W**2,0,0]
den = [1,1.41*W,(2*wc**2+W**2),1.41*W*wc**2,wc**4]

system = sig.lti(num,den)
w,Hmag,Hphase = sig.bode(system,1000)


plt.subplot(2,1,2)
plt.title('Bandpass',size = 12)
plt.axis([wc/10,10*wc,0,1])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.yticks([0,0.1,0.95,0.8,1])
plt.xticks([920,1080,700,1300])
plt.grid(which='both')
plt.semilogx(w,10**(0.05*Hmag),'k')
plt.savefig('Bode.png',dpi=300)

dt = 0.0001
NN = 500000
TT = np.arange(0,NN*dt,dt)
y=np.zeros(NN)
f=np.zeros(NN)

A,B,C,D = sig.tf2ss(num,den)
x = np.zeros(np.shape(B))

omega = 800

for i in range(NN):
    f[i] = sin(omega*i*dt)

Omega = str(omega)
plt.figure(figsize = (10,5))
plt.suptitle('$\omega =$ '+Omega,size = 15)
plt.subplot(2,1,1)
plt.title('Input')
plt.axis([0,100/omega,-1.5,1.5])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.yticks([0,0.1,0.5,0.707,1])
plt.grid(which='both')
plt.plot(TT,f,'k')

for i in range(NN):
    x = x + dt*A.dot(x) + dt*B*f[i]          
    y[i] = C.dot(x) + D*f[i]
    
plt.subplot(2,1,2)
plt.title('Output')
plt.axis([0,100/omega,-1.5,1.5])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.yticks([0,0.1,0.95,0.707,1])
plt.grid(which='both')
plt.plot(TT,y,'k')
plt.savefig('0.95.png',dpi=300)

omega = 1000

for i in range(NN):
    f[i] = sin(omega*i*dt)

Omega = str(omega)
plt.figure(figsize = (10,5))
plt.suptitle('$\omega =$ '+Omega,size = 15)
plt.subplot(2,1,1)
plt.title('Input')
plt.axis([0,100/omega,-1.5,1.5])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.yticks([0,0.1,0.5,0.707,1])
plt.grid(which='both')
plt.plot(TT,f,'k')

for i in range(NN):
    x = x + dt*A.dot(x) + dt*B*f[i]          
    y[i] = C.dot(x) + D*f[i]
    
plt.subplot(2,1,2)
plt.title('Output')
plt.axis([0,100/omega,-1.5,1.5])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.yticks([0,0.1,0.95,0.707,1])
plt.grid(which='both')
plt.plot(TT,y,'k')
plt.savefig('0.95.png',dpi=300)

omega = 1200

for i in range(NN):
    f[i] = sin(omega*i*dt)

Omega = str(omega)
plt.figure(figsize = (10,5))
plt.suptitle('$\omega =$ '+Omega,size = 15)
plt.subplot(2,1,1)
plt.title('Input')
plt.axis([0,100/omega,-1.5,1.5])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.yticks([0,0.1,0.5,0.707,1])
plt.grid(which='both')
plt.plot(TT,f,'k')

for i in range(NN):
    x = x + dt*A.dot(x) + dt*B*f[i]          
    y[i] = C.dot(x) + D*f[i]
    
plt.subplot(2,1,2)
plt.title('Output')
plt.axis([0,100/omega,-1.5,1.5])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.yticks([0,0.1,0.95,0.707,1])
plt.grid(which='both')
plt.plot(TT,y,'k')
plt.savefig('0.95.png',dpi=300)
