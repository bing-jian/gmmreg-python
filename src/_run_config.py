#!/usr/bin/env python
#coding=utf-8

import time
from six.moves import configparser
import numpy as np

from ._core import *

def run_config(f_config):
    section_common = 'FILES'
    section_option = 'GMMREG_OPT'

    c = configparser.ConfigParser()
    c.read(f_config)
    model_file = c.get(section_common, 'model')
    scene_file = c.get(section_common, 'scene')
    model = np.loadtxt(model_file)
    scene = np.loadtxt(scene_file)
    try:
        ctrl_pts_file = c.get(section_common, 'ctrl_pts')
        ctrl_pts = np.loadtxt(ctrl_pts_file)
    except:
        ctrl_pts = model
    level = int(c.get(section_option, 'level'))
    option_str = c.get(section_option, 'sigma')
    scales = [float(s) for s in option_str.split(' ')]
    option_str = c.get(section_option, 'lambda')
    lambdas = [float(s) for s in option_str.split(' ')]

    option_str = c.get(section_option, 'max_function_evals')
    iters = [int(s) for s in option_str.split(' ')]

    normalize_flag = int(c.get(section_option, 'normalize'))
    if normalize_flag==1:
        [model, c_m, s_m] = normalize(model)
        [scene, c_s, s_s] = normalize(scene)
        [ctrl_pts, c_c, s_c] = normalize(ctrl_pts)
    t1 = time.time()
    after_tps = run_multi_level(model,scene,ctrl_pts,level,scales,lambdas,iters)
    if normalize_flag==1:
        model = denormalize(model,c_m,s_m)
        scene = denormalize(scene,c_s,s_s)
        after_tps = denormalize(after_tps,c_s,s_s)
    t2 = time.time()
    print("Elasped time: {} seconds".format(t2-t1))
    return model, scene, after_tps
