""" My_zbode.py.  """ 

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf
import cmath 

NN = 5000
phi = np.linspace(0,2*pi,NN)
z = np.zeros(NN, dtype = complex)
H = np.zeros(NN, dtype = complex)

num1 = [1.15,0]
num2 = [0.866,0]

den1 = [1,0.818]
den2 = [1,3.07,.141]
num = np.convolve(num1,num2)
den = np.convolve(den1,den2)

for n in range(0,NN):
    z = cmath.exp(1j*phi[n])
    H[n] = (0.1*sqrt(3/4)*z/(z-.905))*(0.866*z/(z**2-0.307*z+0.141))
        
plt.figure(figsize= (14,8))
plt.subplot(211)
#plt.plot((180/pi)*phi,abs(H),'k')
plt.semilogx((180/pi)*phi,20*np.log10(H),'k')
plt.axis([1,100, -20, 10])
plt.ylabel('|G| dB')
plt.yticks([ -20,-3,0])
plt.axvline(5.7,color ="RED",linestyle='dotted')
plt.axvline(4*5.7,color ="RED",linestyle='dotted')
plt.text(2.8,-10,'$\phi$ = {}'.format(round(5.7,1)),fontsize=12)
plt.text(2.7*4,-5,'$\phi$ = {}'.format(round(5.7*4,1)),fontsize=12)
plt.title('Z Domain Filter')
plt.grid(which='both')

aaa = np.angle(H)
#for n in range(NN):
#    if aaa[n] > 0:
#        aaa[n] = aaa[n] - 2*pi

plt.subplot(212)
#plt.plot((180/pi)*phi,(180/pi)*aaa,'k')
plt.semilogx((180/pi)*phi,(180/pi)*aaa,'k')
plt.ylabel('/G (degrees)')
plt.axis([1,100, -90,0])
plt.yticks([-90,-45,0,-270])
plt.axvline(5.7,color ="RED",linestyle='dotted')
plt.axvline(4*5.7,color ="RED",linestyle='dotted')
plt.text(2.8,-75,'$\phi$ = {}'.format(round(5.7,2)),fontsize=12)
plt.text(2.7*4,-150,'$\phi$ = {}'.format(round(5.7*4,1)),fontsize=12)
plt.grid(which='both')
plt.xlabel('$\phi$ (degrees)')
plt.savefig('H_zbode.png',dpi=300)
plt.show()

