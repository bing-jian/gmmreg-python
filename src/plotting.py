#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ._config import read_sections


def display2Dpointset(A):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.grid(True)
    ax.plot(A[:,0],A[:,1],'yo',markersize=8,mew=1)
    labels = plt.getp(plt.gca(), 'xticklabels')
    plt.setp(labels, color='k', fontweight='bold')
    labels = plt.getp(plt.gca(), 'yticklabels')
    plt.setp(labels, color='k', fontweight='bold')
    for i,x in enumerate(A):
        ax.annotate('%d'%(i+1), xy = x, xytext = x + 0)
    ax.set_axis_off()
    plt.show()
    #fig.show()


def display2Dpointsets(A, B, ax = None):
    """ display a pair of 2D point sets """
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.plot(A[:,0],A[:,1],'yo',markersize=8,mew=1)
    ax.plot(B[:,0],B[:,1],'b+',markersize=8,mew=1)
    #pylab.setp(pylab.gca(), 'xlim', [-0.15,0.6])
    labels = plt.getp(plt.gca(), 'xticklabels')
    plt.setp(labels, color='k', fontweight='bold')
    labels = plt.getp(plt.gca(), 'yticklabels')
    plt.setp(labels, color='k', fontweight='bold')


def display3Dpointsets(A,B,ax):
    #ax.plot3d(A[:,0],A[:,1],A[:,2],'yo',markersize=10,mew=1)
    #ax.plot3d(B[:,0],B[:,1],B[:,2],'b+',markersize=10,mew=1)
    ax.scatter(A[:,0],A[:,1],A[:,2], c = 'y', marker = 'o')
    ax.scatter(B[:,0],B[:,1],B[:,2], c = 'b', marker = '+')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def displayABC(A,B,C):
    fig = plt.figure()
    dim = A.shape[1]
    if dim==2:
        ax = plt.subplot(121)
        display2Dpointsets(A, B, ax)
        ax = plt.subplot(122)
        display2Dpointsets(C, B, ax)
    if dim==3:
        plot1 = plt.subplot(1,2,1)
        ax = Axes3D(fig, rect = plot1.get_position())
        display3Dpointsets(A,B,ax)
        plot2 = plt.subplot(1,2,2)
        ax = Axes3D(fig, rect = plot2.get_position())
        display3Dpointsets(C,B,ax)
    plt.show()


def display_pts(f_config):
    files = read_sections(f_config)['FILES']
    m = np.loadtxt(files['model'])
    s = np.loadtxt(files['scene'])
    t = np.loadtxt(files['transformed_model'])
    displayABC(m,s,t)
