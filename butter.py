"""Butter Filter """

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf

Hp = 1 
Hs = 2
wp = 1
ws = 1
k=1

d1 = np.convolve([0,0,0,0,1,6,15,15],[0,7])
d4 =d1+np.convolve([0,0,0,0,1,3,3],[1,0,0])
# for i in range(len(d4)-1):
#     try:
#         if(d4[len(d4)-i]):
#             k = k * d4[len(d4)-i]
#     except:
#         print('Failed')
#         k=k
k =d4[len(d4)-1]
n1 = [0,0,0,0,k*Hs]

num= n1
den= d4

system = sig.lti(num,den)
w,Hmag,Hphase = sig.bode(system)

plt.figure(figsize = (8,5))
plt.subplot(2,1,1)
plt.semilogx(w,10**(0.05*Hmag),'k')
plt.title("Bessel Filter")
plt.grid()

plt.subplot(2,1,2)
plt.semilogx(w,10**(0.05*Hphase),'k')
plt.title("Bessel Phase")
plt.grid()

dt = 0.001
NN = 50000
TT = np.arange(0,NN*dt,dt)
y = np.zeros(NN)
f = y
A,B,C,D = sig.tf2ss(num,den)
x = np.zeros(np.shape(B))

omega = 3

for n in range(NN):
   aaa = (((n-2000)/30)**2)
   f[n] = cos((omega*(n-2000))*dt)
    
#for i in range(NN):
#    f[i] = sin(omega*i*dt)

plt.figure(figsize = (8,5))
plt.subplot(2,1,1)
plt.title('F')
plt.axis([0,10,-1.5,1.5])
plt.plot(TT,f,'k')
plt.grid()


for i in range(NN):
    x = x + dt*A.dot(x) + dt*B*f[i]          
    y[i] = C.dot(x) + D*f[i]
    
plt.subplot(2,1,2)
plt.title('Output')
plt.axis([0,10,-1.5,1.5])
plt.xlabel('$\omega$ rad/s')
plt.ylabel('|H|')
#plt.yticks([0,0.8,0.1])
plt.grid(which='both')
plt.plot(TT,y,'k')
plt.savefig('0.95.png',dpi=300)