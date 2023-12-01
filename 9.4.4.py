
""" Find_FIR.py """
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt

fac = 1
bound = 150
dt = .002/fac
NN = int(100*fac)

N2 = int(NN/2)
x = np.zeros(NN)
y = np.zeros(NN)

def bandaid(first,H):
    I = H
    fend = first + 0
    tol = 3
    for n in range(N2):
        I[n] = exp(-.5*((n-first)/tol)**2)
    return I


TT = np.linspace(0,dt*(NN-1),NN)
DF = 1/(dt*NN)
FF = np.linspace(0,DF*(NN-1),NN)
f1 = 20
f2 = 100
f3 = 180
f4 = 300

freq1 = 2*pi*f1
freq2 = 2*pi*f2
freq3 = 2*pi*f3
freq4 = 2*pi*f4

for n in range(NN):
    x[n] = np.sin(2*pi*100*n*dt)+0.5*np.random.normal(0,1)

plt.figure(figsize=[10,12])
plt.subplot(423)
plt.plot(TT,x,'k')
plt.axis([0,NN*dt,-5.5,5.5])
# plt.text(5,-15,'$\phi$ = {}'.format(round(14,3)),fontsize=12)

plt.ylabel('a). x[k]')
plt.xlabel('T (sec)')
plt.grid()
X = (1/NN)*np.fft.fft(x)
""" Create the filter """
H = np.zeros(NN)
""" Rectangular Low pass """
# for n in range(8):
#     H[n] = 1
""" Low pass """
#for n in range(4):
# H[n] = 1
#for n in range(4,16):
# H[n] = exp(-.5*((n-4)/4)**2)
""" Band pass """
A = bandaid(20,np.zeros(NN))
H = A

plt.subplot(4,2,(1,2))
plt.title('9.4.4')
plt.plot(FF,abs(X),'k',label='X')
plt.plot(FF,A,'r--',label='A')
plt.legend(loc='upper right')
plt.ylabel('b). H(w),X(w)')
plt.xlabel('Freq (Hz)')
plt.grid()
plt.axis([0,bound,0,1.1])
plt.xticks([100])

""" High pass """
#for n in range(15):
# H[n] = exp(-.5*((n-15)/4)**2)
#for n in range(15,N2+2):
# H[n] = 1
""" Reflect the positive frequencies to the right side """
for n in range(1,N2-1):
    H[NN-n] = H[n]
Y = H*X
plt.subplot(424)
plt.plot(FF,abs(X),'k',label='X')
plt.plot(FF,H,'k--',label='H')
plt.legend(loc='upper right')
plt.ylabel('b). H(w),X(w)')
plt.xlabel('Freq (Hz)')
plt.grid()
plt.axis([0,bound,0,1.1])
plt.xticks([100])

h = np.fft.ifft(H)

plt.subplot(425)
plt.plot(h.real,'k')
plt.xlabel('k')
plt.ylabel('c). h[k]')
plt.grid()
plt.axis([0,NN,-.1,.2])

M = 8
hh = np.zeros(NN)

""" Move the filter to the left side """
for n in range(M):
    hh[n+M] = h[n].real
    hh[M-n] = hh[M+n]

plt.subplot(426)
plt.plot(hh,'ok')
plt.axis([0 ,2*M,-.15,.25])
plt.xlabel('k')
plt.ylabel('d). hh[k]')
plt.grid()

""" Convolve hh and x """

yy=np.convolve(hh,x)

y = np.zeros(NN)
for n in range(NN):
    y[n] = yy[n+M]

plt.subplot(427)
plt.plot(TT,y,'k')
plt.ylabel('e). y[k]')
plt.xlabel('T (sec)')
plt.grid()

Y = (1/NN)*np.fft.fft(y)

plt.subplot(428)
plt.plot(FF,abs(Y),'k')
plt.axis([0,bound,-.1,1])
plt.xticks([100])
plt.ylabel('f). Y[w]')
plt.xlabel('Freq (Hz)')
plt.grid()
plt.savefig('9.4.4.png',dpi=300)
plt.show()