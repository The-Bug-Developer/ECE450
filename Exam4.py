################################################################
#                                                              #
# Zachary DeLuca                                               #
# ECE 450                                                      #
# Exam 4                                                     #
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

s = 50e3

tol = 0.01

num = [0,0,s**3]
den = [0,1,s]
den = np.convolve(den,[1,2*cos(1*pi/3)*s,s**2])


system = sig.lti(num,den)
w,Hmag,Hphase = sig.bode(system)

plt.figure(figsize= (8,8))
plt.subplot(2,1,1)
plt.title('Third Order Low Pass',size = 20)
plt.axis([s/10,s*10,0,1])
plt.axvline(s,color ="RED",linestyle='dotted')
plt.axvline(s*4,color ="RED",linestyle='dotted')
#plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.xticks([s,s*4])
plt.yticks([0,0.1,0.95,0.707,1])
plt.grid(which='both')
plt.semilogx(w,10**(0.05*Hmag),'k')
plt.savefig('Bode.png',dpi=300)

plt.subplot(2,1,2)
plt.title('Third Order Low Pass Phase',size = 12)
plt.axis([s/10,s*10,0,1])
plt.axvline(s,color ="RED",linestyle='dotted')
plt.axvline(s*4,color ="RED",linestyle='dotted')
plt.xlabel('$\omega$ rad/s')
plt.ylabel('$\phi$')
plt.xticks([s,s*4])
plt.yticks([0,-45,-90,-180,-270])
plt.grid(which='both')
plt.semilogx(w,Hphase,'k')
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
#lookup = 0.707
#output = 1.00
#
#for i in range(len(Hmag)):
#    if( 10**(0.05*Hmag[i]) >= (lookup*(1-tol))  and 10**(0.05*Hmag[i])  <= (lookup*(1+tol))):
#        output = w[i]
#        #print('True',10**(0.05*Hmag[i]) ,lookup,output,w[i])
#        break;
#
#omega = output
#for i in range(NN):
#    f[i] = sin(omega*i*dt)
#
#Omega = str(omega)
#plt.figure(figsize = (10,5))
#plt.suptitle('$\omega =$ '+Omega,size = 15)
#plt.subplot(2,1,1)
#plt.title('Input')
#plt.axis([0.1,50,-1,1])
#plt.xlabel('$\omega$ rad/s')
#plt.ylabel('|H|')
#plt.yticks([0,0.01,0.5,0.707,1])
#plt.xticks([50000,200000])
#plt.grid(which='both')
#plt.plot(TT,f,'k')
#
#for i in range(NN):
#    x = x + dt*A.dot(x) + dt*B*f[i]          
#    y[i] = C.dot(x) + D*f[i]
#    
#plt.subplot(2,1,2)
#plt.title('Output')
#plt.axis([0.1,50,-1,1])
#plt.xlabel('$\omega$ rad/s')
#plt.ylabel('|H|')
#plt.yticks([0,0.1,0.95,0.707,1])
#plt.grid(which='both')
#plt.plot(TT,y,'k')
#plt.savefig('0.95.png',dpi=300)
