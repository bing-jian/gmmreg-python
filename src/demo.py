#!/usr/bin/env python
#coding=utf-8

import sys
import time
import subprocess

from numpy import loadtxt

from ._run_config import run_config
from ._config import read_sections
from . import plotting

def test(f_config, display=True):
    model, scene, after_tps = run_config(f_config)
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
    files = read_sections(f_config)['FILES']
    m = loadtxt(files['model'])
    s = loadtxt(files['scene'])
    t = loadtxt(files['transformed_model'])
    plotting.displayABC(m,s,t)


def main():
    if len(sys.argv) != 2:
        print("Usage: gmmreg-demo <config.ini|config.yaml>", file=sys.stderr)
        sys.exit(1)
    test(sys.argv[1])


if __name__ == '__main__':
    main()
