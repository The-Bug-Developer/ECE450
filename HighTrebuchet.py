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

W = 1e3
Hp = 0.98
eps = sqrt(1/Hp**2 - 1)
n = 2

alpha = 1/eps + (sqrt(1 + 1/eps**2))

a = 0.5*(alpha**(1/n) - alpha**(-1/n))
b = 0.5*(alpha**(1/n) + alpha**(-1/n))

theta = 45*(pi/180)
d1 = [1, 2*a*cos(theta), (a**2)*cos(theta)**2 + (b**2)*sin(theta)**2]

K = d1[2]*Hp

num = [0, 0, K/d1[2], 0, 0]
den = [1, (d1[1]/d1[2])*W, (1/d1[2])*W**2]

system = sig.lti(num,den)
w,Hmag,Hphase = sig.bode(system,1000)

plt.figure(figsize= (10,5))
plt.title('Second Order High Pass',size = 20)
plt.axis([W/10,W*10,0,1.5])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.yticks([0,0.1,0.8,0.707,1])
plt.grid(which='both')
plt.semilogx(w,10**(0.05*Hmag),'k')
plt.savefig('Bode.png',dpi=300)

#dt = 0.001
#NN = 50000
#TT = np.arange(0,NN*dt,dt)
#y=np.zeros(NN)
#f=np.zeros(NN)
#
#A,B,C,D = sig.tf2ss(num,den)
#x = np.zeros(np.shape(B))
#
#
#
#omega = 90
#
#for i in range(NN):
#    f[i] = sin(omega*i*dt)
#
#Omega = str(omega)
#plt.figure(figsize = (10,5))
#plt.suptitle('$\omega =$ '+Omega,size = 15)
#plt.subplot(2,1,1)
#plt.title('Input')
#plt.axis([0.1,50/s,-1,1])
#plt.xlabel('$\omega$ rad/s')
#plt.ylabel('|H|')
#plt.yticks([0,0.1,0.5,0.707,1])
#plt.grid(which='both')
#plt.plot(TT,f,'k')
#
#for i in range(NN):
#    x = x + dt*A.dot(x) + dt*B*f[i]          
#    y[i] = C.dot(x) + D*f[i]
#    
#plt.subplot(2,1,2)
#plt.title('Output')
#plt.axis([0.1,50/s,-1,1])
#plt.xlabel('$\omega$ rad/s')
#plt.ylabel('|H|')
#plt.yticks([0,0.8])
#plt.grid(which='both')
#plt.plot(TT,y,'k')
#plt.savefig('0.95.png',dpi=300)

