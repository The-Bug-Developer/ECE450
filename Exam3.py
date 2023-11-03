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

Hp = 0.9
Hs = 0.1
wp = 400
ws = 200

""" Butterworth Check"""
c = 235

wp = (200/c)
Hp2 = (0.9)**2

Np = math.log10(1/Hp2 -1)/(2*math.log10(wp))

ws = 400/c
Hp2 = (0.1)**2

Ns = math.log10(1/Hp2 -1)/(2*math.log10(ws))
print("Butterworth Checker")
print('np =',Np)
print('ns =',Ns)

"""Trebuchet Filter Check """
ws = 1
wp = 0.5
eps = sqrt(Hs**2/(1-Hs**2))
alpha  = 1/eps + sqrt(1+1/eps**2)

na = np.arccosh(sqrt(1/(eps**2*(1-Hp**2))))
nb = 1/np.arccosh(1/wp)

n = int(na*nb+1)
print("Trebuchet Checker")
print("n=",n)
a = 0.5*(alpha**(1/n)-alpha**(-1/n))
b = 0.5*(alpha**(1/n)+alpha**(-1/n))

""" Low Pass Trebuchet Time"""
theta = pi-pi/3

s1 = a*cos(theta)+1j*b*sin(theta)
q1 = 1/s1
first = np.real(q1+np.conjugate(q1))
zero = np.real(q1*np.conjugate(q1))

d1 = [1,first,zero]

n1 = [1,0,(1/cos(pi/6))**2]
k = d1[2]/n1[2]

den = np.convolve([1,1/a],d1)
k = d1[2]/n1[2]/a
num = np.convolve([k],n1)


"""Low Pass Graphing"""
w = np.linspace(1,1e4,num=10000)
system = sig . lti (num , den)
w , Hmag , Hphase = sig . bode (system,10000)


plt.figure ( figsize = (14 , 7) )
plt.semilogx (w ,10**(0.05* Hmag ) ,'k') # Plot Amplitude instead of dB
plt.title ( 'Low Pass Stage')
#plt.axis ([0.1 ,10 ,0 ,1])
plt.yticks ([0 , 0.1 , 0.3, 0.5 , 0.9 , 1])

# plt.xlabel ('$\omega$ [rad/s]')
plt.ylabel (r'| H | ')
plt.axvline(0.5, color='PURPLE',linestyle='dotted') # cutoff frequency
plt.axvline(1, color='PURPLE',linestyle='dotted') # cutoff frequency
plt.axhline(0.1,color ="RED",linestyle='dotted')
plt.axhline(0.9,color ="BLUE",linestyle='dotted')
plt.xticks ([0.1 ,0.8 ,1 ,2 ,10])
plt.xlabel ('$\omega$ [rad/s]')
plt.savefig('Figure_1.png',dpi=600)
plt.grid ( which = 'both')


""" High Pass Trebuchet Time"""
s = 200
den = [1,(den[2]/den[3]),(den[1]/den[3]),(1/den[3])]
num = [1,0,1/n1[2],0]


"""High Pass Graphing"""
system = sig . lti (num , den)
w , Hmag , Hphase = sig . bode (system,1000)


plt.figure ( figsize = (14 , 7) )
plt.semilogx (w ,10**(0.05* Hmag ) ,'k') # Plot Amplitude instead of dB
plt.title ( 'High Pass Stage')
plt.axvline(2, color='PURPLE',linestyle='dotted') # cutoff frequency
plt.axvline(1, color='PURPLE',linestyle='dotted') # cutoff frequency
plt.axhline(0.1,color ="RED",linestyle='dotted')
plt.axhline(0.9,color ="BLUE",linestyle='dotted')
plt.yticks ([0 , 0.1 , 0.3, 0.5 , 0.9 , 1])
# plt.xlabel ('$\omega$ [rad/s]')
plt.ylabel (r'| H | ')
plt.xticks ([0.1 ,0.8 ,1 ,2 ,10])
plt.xlabel ('$\omega$ [rad/s]')
plt.savefig('Figure_2.png',dpi=600)
plt.grid ( which = 'both')

print(num)
print(den)

num = [1,0,num[2]*s**2,0]
den = [1,den[1]*s,den[2]*s**2,den[3]*s**3]

system = sig . lti (num , den)
w , Hmag , Hphase = sig . bode (system,1000)

plt.figure ( figsize = (14 , 7) )
plt.semilogx (w ,10**(0.05* Hmag ) ,'k') # Plot Amplitude instead of dB
plt.title ( 'Shifted High Pass Stage')
plt.axhline(0.1,color ="RED",linestyle='dotted')
plt.axhline(0.9,color ="BLUE",linestyle='dotted')
plt.axvline(400, color='PURPLE',linestyle='dotted') # cutoff frequency
plt.axvline(200, color='PURPLE',linestyle='dotted') # cutoff frequency
plt.yticks ([0 , 0.1 , 0.3, 0.5 , 0.9 , 1])
# plt.xlabel ('$\omega$ [rad/s]')
plt.ylabel (r'| H | ')
plt.xticks ([0.1 ,0.8 ,1 ,2 ,10,100,1000])
plt.xlabel ('$\omega$ [rad/s]')
plt.savefig('Figure_3.png',dpi=600)
plt.grid ( which = 'both')

plt.figure ( figsize = (14 , 7) )
plt.semilogx (w ,10**(0.05* Hmag ) ,'k') # Plot Amplitude instead of dB
plt.title ( 'Shifted High Pass Stage 2')
plt.axis ([100 ,1000 ,0 ,1.2])
plt.yticks ([0 , 0.1 ,0.5 , 0.9 , 1])
plt.axhline(0.1,color ="RED",linestyle='dotted')
plt.axhline(0.9,color ="BLUE",linestyle='dotted')
plt.axvline(400, color='PURPLE',linestyle='dotted') # cutoff frequency
plt.axvline(200, color='PURPLE',linestyle='dotted') # cutoff frequency
plt.xlabel ('$\omega$ [rad/s]')
plt.ylabel (r'| H | ')
plt.xticks ([100,200,400,1000])
plt.grid ( which = 'both')
plt.savefig('Figure_4.png',dpi=600)
plt.show()

print(num)
print(den)
""" Sinusoidal Input """
dt = 0.0001
NN = 1000
TT = np.arange(0,NN*dt,dt)
y = np.zeros(NN)
f = np.zeros(NN)
A, B, C, D = sig.tf2ss(num,den)
x = np.zeros(np.shape(B))

omega = 200
for n in range(NN):
    f[n] = sin(omega*n*dt)
    
for m in range(NN):
    y[m] =10**(0.05* Hmag[860] )*sin(omega*m*dt+Hphase[860])

plt . figure (figsize = (10 , 5))
plt.subplot(211)
plt.title('Time Domain Output')
plt.plot(TT,f,'k')
plt.plot(TT,y,'b.')
plt.yticks([-1, -.2, 0, .2, 1])
plt.axis([0, NN*dt, -1, 1])
plt.grid()
plt.text(0,1.1,'$\omega$ = {}'.format(round(omega,1)), fontsize=12)
plt.xlabel('t (sec)')
plt.savefig('Figure_5.png',dpi=600)
plt.legend()
plt.show()

""" Sinusoidal Input """
dt = 0.0001
NN = 1000
TT = np.arange(0,NN*dt,dt)
y = np.zeros(NN)
f = np.zeros(NN)
A, B, C, D = sig.tf2ss(num,den)
x = np.zeros(np.shape(B))

omega = 400
for n in range(NN):
    f[n] = sin(omega*n*dt)
    
for m in range(NN):
    y[m] =10**(0.05* Hmag[920] )*sin(omega*m*dt+Hphase[920])

plt . figure (figsize = (10 , 5))
plt.subplot(211)
plt.title('Time Domain Output')
plt.plot(TT,f,'k')
plt.plot(TT,y,'b.')
plt.yticks([-1, -.9, 0, .9, 1])
plt.axis([0, NN*dt, -1, 1])
plt.grid()
plt.text(0,1.1,'$\omega$ = {}'.format(round(omega,1)), fontsize=12)
plt.xlabel('t (sec)')
plt.savefig('Figure_6.png',dpi=600)
plt.show()