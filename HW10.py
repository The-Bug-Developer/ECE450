import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt, log10
from control import margin
from control import tf

#""" Problem 8.3.3. """
#Hp = 0.93
#eps = sqrt(1/Hp**2 - 1)
#n = 3
#
#alpha = 1/eps + (sqrt(1 + 1/eps**2))
#
#a = 0.5*(alpha**(1/n) - alpha**(-1/n))
#b = 0.5*(alpha**(1/n) + alpha**(-1/n))
#
#theta = 0*(pi/180)
#d1 = [1, a]
#
#theta = 60*(pi/180)
#d2 = [1, 2*a*cos(theta), (a**2)*cos(theta)**2 + (b**2)*sin(theta)**2]
#
#den = np.convolve(d1,d2)
#K = d1[1]*d2[2]
#num = [0, 0, 0, 0, K]
#
#print("epsilon = ", eps)
#print("alpha = ", alpha)
#print("a = ", a)
#print("b = ", b)
#print("K = ", K)
#print("d1 = ", d1)
#print("d2 = ", d2)
#print("num = ", num)
#print("den = ", den)
#
#system = sig . lti ( num , den )
#w , Hmag , Hphase = sig . bode ( system )
#
#
#plt.figure ( figsize = (10 , 7) )
#plt.subplot (211)
#plt.semilogx (w ,10**(0.05* Hmag ) ,'k') # Plot Amplitude instead of dB
#plt.title ( '8.3.3.')
#plt.axis ([0.1 ,10 ,0 ,1.1])
#plt.yticks ([0 , 0.1 , 0.5 , 0.707 , 1])
#plt.grid ( which = 'both')
#plt.xlabel ('$\omega$ [rad/s]')
#plt.ylabel (r'| H | ')
#plt.xticks ([0.1 ,0.8 ,1 ,2 ,10])
#plt.show()
#
#dt = 0.002
#NN = 20000
#TT = np.arange(0,NN*dt,dt)
#y = np.zeros(NN)
#f = np.zeros(NN)
#A, B, C, D = sig.tf2ss(num,den)
#x = np.zeros(np.shape(B))
#
#""" Sinusoidal Input """
#omega = 3
#for n in range(NN):
#    f[n] = sin(omega*n*dt)
#    
#for m in range(NN):
#    x = x + dt*A.dot(x) + dt*B*f[m]
#    y[m] = C.dot(x) + D*f[m]
#
#plt . figure ( figsize = (10 , 5) )
#plt.subplot(211)
#plt.title('Time Domain Output')
#plt.plot(TT,f,'k')
#plt.plot(TT,y,'r--')
#plt.yticks([-1, -.707, 0, .707, 1])
#plt.axis([0, NN*dt, -1, 1])
#plt.grid()
#plt.text(3.5,.707,'$\omega$ = {}'.format(round(omega,1)), fontsize=12)
#plt.xlabel('t (sec)')
#plt.show()

""" Problem 8.3.5. """
W = 1e3
Hp = 0.98
eps = sqrt(1/Hp**2 - 1)
n = 2

alpha = 1/eps + (sqrt(1 + 1/eps**2))

a = 0.5*(alpha**(1/n) - alpha**(-1/n))
b = 0.5*(alpha**(1/n) + alpha**(-1/n))

theta = 45*(pi/180)
d1 = [1, 2*a*cos(theta), (a**2)*cos(theta)**2 + (b**2)*sin(theta)**2]

#den = np.convolve(d1,d2)
K = d1[2]*Hp
#num = [0, 0, 1, 0, 0]
#den = [1, 880, 0.5*(1e6)]

num = [0, 0, K/d1[2], 0, 0]
den = [1, (d1[1]/d1[2])*W, (1/d1[2])*W**2]

print("epsilon = ", eps)
print("alpha = ", alpha)
print("a = ", a)
print("b = ", b)
print("K = ", K)
print("d1 = ", d1)
print("num = ", num)
print("den = ", den)

plt.figure ( figsize = (10 , 7) )
w = np.linspace(10,1e4,num=1000)
system = sig.lti(num,den)
w, Hmag, Hphase = sig.bode(system,w)
gm, pm, wg, wp = margin(Hmag,Hphase,w)
# wp freq for phase margin at gain crossover (gain = 1)
# pm phase maring
plt.subplot(211)
plt.semilogx(w,Hmag,'k')
plt.semilogx(w,Hmag,'k')
#plt.axis([ .1, 1e2, -60, 20])
#plt.xticks([1,10,30,100,1000])
plt.ylabel('|H| dB',size = 12)
plt.text(10,-40,'$\omega$p = {}'.format(round(wp,1)),fontsize=12)
plt.title('8.3.5. Bode')
plt.grid(which='both')
plt.show()

system = sig . lti (num , den)
w , Hmag , Hphase = sig . bode ( system )


plt.figure ( figsize = (10 , 7) )
plt.subplot (211)
plt.semilogx (w ,10**(0.05* Hmag ) ,'k') # Plot Amplitude instead of dB
plt.title ( '8.3.5.')
plt.axis ([10,1e4 ,0 ,1.1])
plt.yticks ([0 , 0.1 , 0.5 , 0.707 , 1])
plt.grid ( which = 'both')
plt.xlabel ('$\omega$ [rad/s]')
plt.ylabel (r'| H | ')
plt.xticks ([10, 100, 1e3, 1e4])
plt.show()

dt = 0.0001
NN = 1000
TT = np.arange(0,NN*dt,dt)
y = np.zeros(NN)
f = np.zeros(NN)
A, B, C, D = sig.tf2ss(num,den)
x = np.zeros(np.shape(B))

""" Sinusoidal Input """
omega = 1000
for n in range(NN):
    f[n] = sin(omega*n*dt)
    
for m in range(NN):
    x = x + dt*A.dot(x) + dt*B*f[m]
    y[m] = C.dot(x) + D*f[m]

plt . figure ( figsize = (10 , 5) )
plt.subplot(211)
plt.title('Time Domain Output')
plt.plot(TT,f,'k')
plt.plot(TT,y,'r--')
plt.yticks([-1, -.707, 0, .707, 1])
plt.axis([0, NN*dt, -1, 1])
plt.grid()
plt.text(0,1.1,'$\omega$ = {}'.format(round(omega,1)), fontsize=12)
plt.xlabel('t (sec)')
plt.show()

# """ Problem 8.3.6. """
# W = 500
# wc = 5e3

# Hp = 0.95
# eps = sqrt(1/Hp**2 - 1)
# n = 2

# alpha = 1/eps + (sqrt(1 + 1/eps**2))

# a = 0.5*(alpha**(1/n) - alpha**(-1/n))
# b = 0.5*(alpha**(1/n) + alpha**(-1/n))

# print("epsilon = ", eps)
# print("alpha = ", alpha)
# print("a = ", a)
# print("b = ", b)

# theta = 45*(pi/180)
# d1 = [1, 2*a*cos(theta), (a**2)*cos(theta)**2 + (b**2)*sin(theta)**2]

# K = d1[2]*Hp

# num = [0, 0, K*W**2, 0, 0]
# den = [1, d1[1]*W, ((d1[2]*W**2)+(2*wc**2)), d1[1]*W*wc**2, wc**4]

# print("epsilon = ", eps)
# print("alpha = ", alpha)
# print("a = ", a)
# print("b = ", b)
# print("K = ", K)
# print("d1 = ", d1)
# print("num = ", num)
# print("den = ", den)


# plt.figure ( figsize = (10 , 7) )
# w = np.linspace(1e3,1e5,num=1000)
# system = sig.lti(num,den)
# w, Hmag, Hphase = sig.bode(system,w)
# gm, pm, wg, wp = margin(Hmag,Hphase,w)
# # wp freq for phase margin at gain crossover (gain = 1)
# # pm phase maring
# plt.subplot(211)
# plt.semilogx(w,Hmag,'k')
# #plt.semilogx(w,Hmag,'k')
# plt.axis([ 1e3, 1e4, -40, 5])
# plt.xticks([1000, 4750,5e3, 5250, 1e4])
# plt.ylabel('|H| dB',size = 12)
# plt.text(1000,0,'$\omega$p = {}'.format(round(wp,1)),fontsize=12)
# plt.title('8.3.6. Bode')
# plt.grid(which='both')
# plt.show()

# system = sig . lti (num , den)
# w , Hmag , Hphase = sig . bode ( system )


# plt.figure ( figsize = (10 , 7) )
# plt.subplot (211)
# plt.semilogx (w ,10**(0.05* Hmag ) ,'k') # Plot Amplitude instead of dB
# plt.title ( '8.3.6.')
# plt.axis ([1e3,1e4 ,0 ,1.1])
# plt.yticks ([0 , 0.1 , 0.5 , 0.707 , 1])
# plt.grid ( which = 'both')
# plt.xlabel ('$\omega$ [rad/s]')
# plt.ylabel (r'| H | ')
# #plt.xticks ([10, 100, 1e3, 1e4])
# plt.show()

# dt = 0.0001
# NN = 250
# TT = np.arange(0,NN*dt,dt)
# y = np.zeros(NN)
# f = np.zeros(NN)
# A, B, C, D = sig.tf2ss(num,den)
# x = np.zeros(np.shape(B))

# """ Sinusoidal Input """
# omega = 5000
# for n in range(NN):
#     f[n] = sin(omega*n*dt)
    
# for m in range(NN):
#     x = x + dt*A.dot(x) + dt*B*f[m]
#     y[m] = C.dot(x) + D*f[m]

# plt . figure ( figsize = (10 , 5) )
# plt.subplot(211)
# plt.title('Time Domain Output')
# plt.plot(TT,f,'k')
# plt.plot(TT,y,'r--')
# plt.yticks([-1, -.707, 0, .707, 1])
# plt.axis([0, NN*dt, -1, 1])
# plt.grid()
# plt.text(0,1.1,'$\omega$ = {}'.format(round(omega,1)), fontsize=12)
# plt.xlabel('t (sec)')
# #plt.ylabel('f')
# plt.show()

# """Problem 8.4.2. """

# Hs = 0.1
# eps = np.sqrt((Hs**2)/(1-Hs**2))
# n = 6

# alpha = 1/eps + (sqrt(1 + 1/eps**2))

# a = 0.5*(alpha**(1/n) - alpha**(-1/n))
# b = 0.5*(alpha**(1/n) + alpha**(-1/n))

# theta = 15*(pi/180)
# s1 = a*cos(theta) + 1j*b*sin(theta)
# q1 = 1/s1
# q1c = np.conjugate(q1)
# d1 = [1, np.real(q1+q1c), np.real(q1*q1c)]

# theta = 45*(pi/180)
# s2 = a*cos(theta) + 1j*b*sin(theta)
# q2 = 1/s2
# q2c = np.conjugate(q2)
# d2 = [1, np.real(q2+q2c), np.real(q2*q2c)]

# theta = 75*(pi/180)
# s3 = a*cos(theta) + 1j*b*sin(theta)
# q3 = 1/s3
# q3c = np.conjugate(q3)
# d3 = [1, np.real(q3+q3c), np.real(q3*q3c)]

# d12 = np.convolve(d1,d2)
# den = np.convolve(d12,d3)

# w1 = 1/cos(pi/(2*n))
# w3 = 1/cos(3*pi/(2*n))
# w5 = 1/cos(5*pi/(2*n))

# n1 = [1, 0, w1**2]
# n2 = [1, 0, w3**2]
# n3 = [1, 0, w5**2]

# K = d1[2]*d2[2]*d3[2]/(n1[2]*n2[2]*n3[2])

# n12 = np.convolve(n1,n2)
# num = K*np.convolve(n12,n3)

# print("epsilon = ", eps)
# print("alpha = ", alpha)
# print("a = ", a)
# print("b = ", b)
# print("K = ", K)
# print("num = ", num)
# print("den = ", den)

# system = sig . lti ( num , den )
# w , Hmag , Hphase = sig . bode ( system )


# plt.figure ( figsize = (10 , 7) )
# plt.subplot (211)
# plt.semilogx (w ,10**(0.05* Hmag ) ,'k') # Plot Amplitude instead of dB
# plt.title ( '8.4.2.')
# #plt.axis ([0.1 ,10 ,0 ,1])
# plt.yticks ([0 , 0.1 , 0.3, 0.5 , 0.707 , 1])
# plt.grid ( which = 'both')
# plt.xlabel ('$\omega$ [rad/s]')
# plt.ylabel (r'| H | ')
# plt.xticks ([0.1 ,0.8 ,1 ,2 ,10])
# plt.show()

# dt = 0.001
# NN = 50000
# TT = np.arange(0,NN*dt,dt)
# y = np.zeros(NN)
# f = np.zeros(NN)
# A, B, C, D = sig.tf2ss(num,den)
# x = np.zeros(np.shape(B))

# """ Sinusoidal Input """
# omega = 1
# for n in range(NN):
#     f[n] = sin(omega*n*dt)
    
# for m in range(NN):
#     x = x + dt*A.dot(x) + dt*B*f[m]
#     y[m] = C.dot(x) + D*f[m]

# plt . figure ( figsize = (10 , 5) )
# plt.subplot(211)
# plt.title('Time Domain Output')
# plt.plot(TT,f,'k')
# plt.plot(TT,y,'r--')
# plt.yticks([-1, -.707, 0, .707, 1])
# plt.axis([0, NN*dt, -1, 1])
# plt.grid()
# plt.text(0,1.1,'$\omega$ = {}'.format(round(omega,1)), fontsize=12)
# plt.xlabel('t (sec)')
# #plt.ylabel('f')
# plt.show()