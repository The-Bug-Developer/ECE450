""" Bode_comp.py.  """

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf

gain = 0
lead = 0
lag = 0
poles = 0 


Range = 1000

fi = 30
phim = (pi/180)*fi
alpha = (1+sin(phim))/(1-sin(phim))
dbshift = -10*np.log10(alpha)

n1 = [0,0 ,10]
d1 = [1, 3, -10]


if gain:
    K = 1
    num = K*n1
    den = d1
else:
    num = n1
    den = d1


w = np.linspace(0.1,100,num=1000)
system = sig.lti(n1,d1)
w, Hmag, Hphase = sig.bode(system,w)

wm = 1

for i in range(len(Hmag)):
    if(dbshift>0):
        if((Hmag[i]>dbshift*0.95) and (Hmag[i]<dbshift*1.15)): 
            wm = w[i]
            #print(Hmag[i],dbshift,'True')
            break
        else: 
            #print(Hmag[i],dbshift,'False')
            wm =wm 
    if(dbshift<0):
        if((Hmag[i]<dbshift*0.95) and (Hmag[i]>dbshift*1.15)): 
            wm = w[i]
            #print(Hmag[i],dbshift,'True')
            break
        else: 
            #print(Hmag[i],dbshift,'False')
            wm =wm 

p = sqrt(alpha)*wm
z = wm/sqrt(alpha)

# First Gc
n2 = [1,z]
d2 = [1,p]

#Lead addition
if lead:
    K = p/z
    num = K*np.convolve(num,n2)
    den = np.convolve(den,d2)
else:
    num = num
    den = den

#Lag addition
if lag:
    p = 2.0
    z = p/100.0
    n2 = [1,p]
    d2 = [1,z]
    num = K*np.convolve(num,n2)
    den = np.convolve(den,d2)
else:
    num = num
    den = den

if poles:
    n = 5
    n2 = np.convolve([1,n],[1,n])
    d2 = [1,0,0]
    num = K*np.convolve(num,n2)
    den = np.convolve(den,d2)
else:
    num = num
    den = den


w = np.linspace(0.1,Range,num=1000)
system = sig.lti(num,den)
w, Hmag, Hphase = sig.bode(system,w)
gm, pm, wg, wp = margin(Hmag,Hphase,w)
# wp  freq for phase margin at gain crossover (gain = 1)
# pm phase maring
plt.figure(figsize=(14,30))
plt.subplot(8,2,(1,2))
plt.semilogx(w,Hmag,'k')
plt.semilogx(w,Hmag,'k')
plt.axis([ .1, Range, -60, 20])
#plt.xticks([1,10,30,100,1000])
plt.ylabel('|H| dB',size = 12)
plt.text(.3,-40,'$\omega$p = {}'.format(round(wp,1)),fontsize=12)
plt.title('Bode Comp')
plt.grid(which='both')

for n in range(Range):
    if Hphase[n] > 0:
        Hphase[n] = Hphase[n] - 360

plt.subplot(8,2,(3,4))
plt.semilogx(w,Hphase,'k')
plt.axis([ .1, Range, -180,0])
plt.yticks([-180,-90,0])


plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('Phase (degrees)',size=12)
plt.text(.3,-150,'pm = {}'.format(round(pm,0)),fontsize=12)
plt.grid(which='both')



""" Time portion """

dt = 0.05
NN = 500
TT = np.arange(0,NN*dt,dt)
step = np.zeros(NN)
ramp = np.zeros(NN)
parabola = np.zeros(NN)
errS = np.zeros(NN)
errR = np.zeros(NN)
errP = np.zeros(NN)

for i in range(NN):
    step[i] = 1.0
    ramp[i] = (dt*i)
    parabola[i] = (dt*i)**(2)
    
denCL = np.add(num,den)

t1, y1, x1 = sig.lsim((num,denCL),step,TT)
t2, y2, x2 = sig.lsim((num,denCL),ramp,TT)
t3, y3, x3 = sig.lsim((num,denCL),parabola,TT)

for i in range(NN):
    errS[i] = step[i] - y1[i]
    errR[i] = ramp[i] - y2[i]
    errP[i] = parabola[i]  - y3[i]    

plt.subplot(8,2,5)
plt.plot(TT,y1,'k--',label='y1(t)')
plt.plot(TT,step,'k',label='u(t)')
plt.axis([0,2,0,1.5])
plt.ylabel('step')
plt.xlabel('t (sec)')
#plt.yticks([0,.9,1.1,1.5])
plt.legend()
plt.grid()

plt.subplot(8,2,6)
plt.plot(TT,errS,'k',label='error')
plt.legend()
plt.axis([0,2,-.5,1])
#plt.yticks([0,0.02,.05,.1])
plt.grid()
plt.savefig('position.png')



plt.subplot(8,2,7)

plt.plot(TT,y2,'k--',label='y2(t)')
plt.plot(TT,ramp,'k',label='r(t)')
plt.xlabel('t (sec)')
plt.ylabel('ramp')
plt.legend()
plt.grid()

plt.subplot(8,2,8)
plt.plot(TT,errR,'k',label='error')
plt.legend(loc=4)
plt.xlabel('t (sec)')
plt.axis([0,2,0,.2])
plt.yticks([0,.05,.2])
plt.grid()
plt.savefig('velocity.png')

plt.show()

