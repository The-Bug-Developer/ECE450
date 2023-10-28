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
Hp = 0.944
Hs = 0.1
ws = 1.2
eps = sqrt(1/Hp**2-1)
alpha  = 1/eps + sqrt(1+1/eps**2)

n = (np.arccosh((1/eps)*sqrt((1/abs(Hs)**2)-1))*(1/(np.arccosh(ws))))

a = 0.5*(alpha**(1/n)+alpha**(-1/n))
b = 0.5*(alpha**(1/n)+alpha**(-1/n))

d1 = [1,a]
d2 = [1,-2*a*cos(pi/n),a**2*cos(pi/n)**2+b**2*sin(pi/n)**2]
d3 = [a**2*cos(pi/n)**2+b**2*sin(pi/n)**2,-2*a*cos(pi/n),1]


den = d2
K= den[2]*Hp

num = [0,0,]

system = sig.lti(num,den)
w,Hmag,Hphase = sig.bode(system,1000)

plt.figure(figsize= (10,5))
plt.title('Third Order Low Pass',size = 20)
plt.axis([s/10,s*10,0,1])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
plt.yticks([0,0.1,0.8,0.707,1])
plt.grid(which='both')
plt.semilogx(w,10**(0.05*Hmag),'k')
plt.savefig('Bode.png',dpi=300)
