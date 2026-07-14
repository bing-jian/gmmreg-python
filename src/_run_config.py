#!/usr/bin/env python
#coding=utf-8

import time
import numpy as np

from ._config import read_sections, as_list
from ._core import *

def run_config(f_config):
    sections = read_sections(f_config)
    files = sections['FILES']
    opt = sections['GMMREG_OPT']

    model = np.loadtxt(files['model'])
    scene = np.loadtxt(files['scene'])
    try:
        ctrl_pts = np.loadtxt(files['ctrl_pts'])
    except (KeyError, OSError, ValueError):
        ctrl_pts = model
    level = int(opt['level'])
    scales = as_list(opt['sigma'], float)
    lambdas = as_list(opt['lambda'], float)
    iters = as_list(opt['max_function_evals'], int)

    normalize_flag = int(opt['normalize'])
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
