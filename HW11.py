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

s = 500

Hp = 0.95
Hs = 0.15
ws = 1
wp = 0.5
eps = sqrt(Hs**2/(1-Hs**2))
alpha  = 1/eps + sqrt(1+1/eps**2)

n = np.arccosh(sqrt(1/(eps**2*(1-Hp**2))))*1/np.arccosh(1/wp)
n=3
print("n=",n)
a = 0.5*(alpha**(1/n)-alpha**(-1/n))
b = 0.5*(alpha**(1/n)+alpha**(-1/n))

thet = pi/6

c = 2*a*cos(thet)
f =a**2*cos(thet)**2+b**2*sin(thet)**2

den = np.convolve([1,s/a],[1,c*s/f,s**2/f])

k = (0.15*(1+a)*(1-c+f))*7/3

num = np.convolve([0,0,0,k+0.08],[7/3,0,s**2,0])


print(num)
print(den)

system = sig.lti(num,den)
w,Hmag,Hphase = sig.bode(system,10000,1000000)

plt.figure(figsize = (8,5))
plt.subplot(2,1,1)
plt.axis([0.1,1e4,0,1.2])
plt.semilogx(w,10**(0.05*Hmag),'k')
plt.title("High Filter")
plt.grid()

# plt.subplot(2,1,2)
# plt.semilogx(w,10**(0.05*Hphase),'k')
# plt.title("Bessel Phase")
# plt.grid()

# dt = 0.001
# NN = 50000
# TT = np.arange(0,NN*dt,dt)
# y = np.zeros(NN)
# f = y
# A,B,C,D = sig.tf2ss(num,den)
# x = np.zeros(np.shape(B))

# omega = 3

# for n in range(NN):
#    aaa = (((n-2000)/30)**2)
#    f[n] = cos((omega*(n-2000))*dt)*exp(-aaa*dt)
    
# #for i in range(NN):
# #    f[i] = sin(omega*i*dt)

# plt.figure(figsize = (8,5))
# plt.subplot(2,1,1)
# plt.title('F')
# plt.axis([0,10,-1.5,1.5])
# plt.plot(TT,f,'k')
# plt.grid()


# for i in range(NN):
#     x = x + dt*A.dot(x) + dt*B*f[i]          
#     y[i] = C.dot(x) + D*f[i]
    
# plt.subplot(2,1,2)
# plt.title('Output')
# plt.axis([0,10,-1.5,1.5])
# plt.xlabel('$\omega$ rad/s')
# plt.ylabel('|H|')
# #plt.yticks([0,0.8,0.1])
# plt.grid(which='both')
# plt.plot(TT,y,'k')
# plt.savefig('0.95.png',dpi=300)