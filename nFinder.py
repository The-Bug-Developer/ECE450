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


wp = 0.85
Hp2 = (0.85)**2

np = math.log10(1/Hp2 -1)/(2*math.log10(wp))


ws = 600/c
Hp2 = (0.2)**2

ns = math.log10(1/Hp2 -1)/(2*math.log10(ws))
print('np =',np)
print('ns =',ns)
