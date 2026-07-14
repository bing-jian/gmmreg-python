#!/usr/bin/env python
#coding=utf-8

import time
import subprocess

from numpy import loadtxt

import gmmreg

import plotting

def test(f_config, display=True):
    model, scene, after_tps = gmmreg.run_config(f_config)
    if display:
        plotting.displayABC(model, scene, after_tps)


def run_executable(gmmreg_exe, f_config, method, display=True):
    t1 = time.time()
    subprocess.call([gmmreg_exe, f_config, method])
    t2 = time.time()
    print("Elasped time: {} seconds".format(t2 - t1))
    if display:
        display_pts(f_config)


def display_pts(f_config):
    files = gmmreg.read_sections(f_config)['FILES']
    m = loadtxt(files['model'])
    s = loadtxt(files['scene'])
    t = loadtxt(files['transformed_model'])
    plotting.displayABC(m,s,t)


import sys
if __name__ == '__main__':
    test(sys.argv[1])
