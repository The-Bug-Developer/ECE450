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

#s = 1
Hp = 0.9
Hs = 0.3
ws = 1
wp= 0.2
eps = sqrt(Hs**2/(1-Hs**2))
alpha  = 1/eps + sqrt(1+1/eps**2)

na = np.arccosh(sqrt(1/((eps**2)*(1-Hp**2))))
nb = 1/np.arccosh(1/wp)
n = na*nb
print(n)

a = 0.5*(alpha**(1/n)-alpha**(-1/n))
b = 0.5*(alpha**(1/n)+alpha**(-1/n))

theta = 45*pi/180


s1 = a*cos(theta)+1j*b*sin(theta)
q1 = 1/s1
q1c = np.conjugate(q1)

n1 = [1,0,(1/(cos(1*pi/4)))**2]
d1 = [1,np.real(q1+q1c),np.real(q1*q1c)]
K= d1[2]/n1[2]

system = sig.lti(n1,d1)
w,Hmag,Hphase = sig.bode(system,1000)

plt.figure(figsize= (10,5))
plt.title('Third Order Low Pass',size = 20)
plt.axis([1,100,0,2])
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

