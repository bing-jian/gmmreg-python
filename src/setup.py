#!/usr/bin/env python
#coding=utf-8

"""
python setup.py build  -c mingw32
python setup.py install --skip-build
"""

from os.path import join
from distutils.core import setup, Extension
import numpy

module_pycvgmi = Extension('gmmreg._extension',
                            define_macros=[('MAJOR_VERSION', '1'),
                                           ('MINOR_VERSION', '0')],
                            sources=[join('c_extension', 'py_extension.c'),
                                     join('c_extension', 'GaussTransform.c'),
                                     join('c_extension', 'DistanceMatrix.c')],
                            include_dirs=[numpy.get_include()])


setup (name='gmmreg',
       version='1.0',
       description='Robust Point Set Registration using Gaussian Mixture Models.',
       author='Bing Jian',
       author_email='bing.jian@gmail.com',
       url='https://github.com/bing-jian/gmmreg-python',
       long_description='''
              Python package for robust (2D/3D) point set (rigid/non-rigid) registration using Gaussian mixture models.
              ''',
       package_dir={'gmmreg': ''},
       packages=['gmmreg'],
       ext_modules=[module_pycvgmi])
