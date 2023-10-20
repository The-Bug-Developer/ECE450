"""Butter Filter """

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf

num= [0,0,1]
den= [1,1.41,1]

system = sig.lti(num,den)
w,Hmag,Hphase = sig.bode(system)

plt.subplot(2,1,1)
plt.semilogx(w,10**(0.05*Hmag),'k')
plt.title("Butter Filter")
plt.grid()

dt = 0.001
NN = 50000
TT = np.arrange(0,NN*dt,dt)
y = np.zeros(NN)
f = y
A,B,C,D = sig.tf2ss(num,den)
x = np.zeros(np.shape(B))

omega = 1
for i in range(NN):
    