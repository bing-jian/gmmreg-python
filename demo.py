#!/usr/bin/env python
#coding=utf-8

import time
import subprocess
from six.moves import configparser

from numpy import loadtxt

import gmmreg

import plotting

def test(f_config, display=True):
    model, scene, after_tps = gmmreg.run_config(f_config)
    if display:
        plotting.displayABC(model, scene, after_tps)


def run_executable(gmmreg_exe, f_config, method, display=True):
    cmd = '%s %s %s'%(gmmreg_exe, f_config, method)
    t1 = time.time()
    subprocess.call(cmd, shell=True)
    t2 = time.time()
    print("Elasped time: {} seconds".format(t2 - t1))
    if display:
        display_pts(f_config)


def display_pts(f_config):
    c = configparser.ConfigParser()
    c.read(f_config)
    section_common = 'FILES'
    mf = c.get(section_common, 'model')
    sf = c.get(section_common, 'scene')
    tf = c.get(section_common, 'transformed_model')

    m = loadtxt(mf)
    s = loadtxt(sf)
    t = loadtxt(tf)
    plotting.displayABC(m,s,t)


import sys
if __name__ == '__main__':
    test(sys.argv[1])
