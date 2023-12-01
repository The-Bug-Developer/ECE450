#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:48:29 2023

@author: void
"""

""" Example 9.1.1"""
import matplotlib.pyplot as plt
import numpy as np
from math import pi, exp, cos, sin, log, sqrt
import cmath
NN = 10000
phi = np.linspace(0,2*pi,NN)
dt = .01
z = np.zeros(NN, dtype = complex)
H = np.zeros(NN, dtype = complex)
a = 1
c = 1
T = 0.1
k= 1
"""---- One pole filter (Eq, 9.1.3) ------------"""
for n in range(0,NN):
    z = cmath.exp(1j*phi[n])
    try:
        H[n] = k*(a*T*z**2-a*T*z)/((c*T**2+1)*z**2-(a*T+2)*z+1)
    except:
        print('woops')
plt.subplot(211)
plt.semilogx((180/pi)*phi,20*np.log10(H),'k')
#plt.plot((180/pi)*phi,abs(H))
plt.axis([0.1, 100, -40, 10])
plt.ylabel('|H| ')
plt.yticks([-40,-20,-3,0])
plt.axvline(5.7,color='k')
plt.text(3,-15,'$\phi$ = {}'.format(round(5.7,3)),fontsize=12)
plt.axvline(57,color='k')
plt.text(30,-15,'$\phi$ = {}'.format(round(57,3)),fontsize=12)
plt.title('Problem_9-3-1')
plt.grid(which='both')
aaa = np.angle(H)
#for n in range(NN):
# if aaa[n] > pi:
# aaa[n] = aaa[n] - 2*pi
plt.subplot(212)
plt.semilogx((180/pi)*phi,(180/pi)*aaa,'k')
#plt.plot((180/pi)*phi,(180/pi)*aaa,'k')
plt.ylabel('/H (degrees)')
plt.axis([1, 100, -90, 0])
plt.yticks([-90,-45,0])
#plt.axis([.1,100, -90,90])
#plt.yticks([-90,-45,0])
#plt.xticks([1,5.7,10,100])
plt.axvline(5.7,color='k')
plt.axvline(57,color='k')
#plt.text(.3,-70,'$\phi$ = {}'.format(round(5.7,2)),fontsize=12)
plt.grid(which='both')
plt.xlabel('$\phi$ (degrees)')
plt.savefig('H_zbode.png',dpi=300)
plt.show()
"""This is the time-domain simulation of section 9.1.3"""
MM = 5000
x = np.zeros(MM)
y = np.zeros(MM)
w1 = 5
w2 = 100
for k in range(1,MM):
    try:
        x[k] = 0.5*sin(0.1*dt*k) + 0.5*sin(10*dt*k)+sin(dt*k)
        y[k] = (a*T)*x[k]-(a*T)*x[k-1]-y[k-2]+((a*T)+2)*y[k-1]+(c*T**2+1)*y[k]
    except:
        print(k)
plt.subplot(2,1,1)
plt.plot(x,'k')
plt.ylabel('x[k]')
plt.title('Problem_9-1-1')
plt.grid()
plt.subplot(2,1,2)
plt.plot(y,'k')
plt.ylabel('y[k]')
#plt.axis([0,MM,-1.2,1.2])
plt.yticks([-1,-.7,0,.7,1])
plt.grid()
plt.savefig('Time_plot.png')
plt.show()