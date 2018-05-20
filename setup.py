#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 14:19:42 2017

@author: Sebastijan Mrak <smrak@gmail.com>
"""

from setuptools import setup


setup(name='pkrasi',
      description='Poker Flat all sky imager tools',
      author='Sebastijan Mrak',
      url='https://github.com/aldebaran1/pkrasi.git',
      install_requires=['dascutils', 'scipy'],
      packages=['pkrasi']

)