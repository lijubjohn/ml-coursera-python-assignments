#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:19:29 2018

@author: liju
"""


import os
import numpy as np
#import matplotlib as pyplot
from matplotlib import pyplot
import mpl_toolkits.mplot3d as Axes3D

#A = np.eye(5)
#print(A)

#m = y.size
#print(m)

def plotData(x,y):
    figure = pyplot.figure();
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Profit in $10,000')
    pyplot.xlabel('Population of City in 10,000s')
    
 

data = np.loadtxt(os.path.join('Data','ex1data1.txt'),delimiter=',')
x,y = data[:,0],data[:,1]
plotData(x,y)