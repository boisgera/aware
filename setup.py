#!/usr/bin/env python
# coding: utf-8

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = "_aware",
  ext_modules = cythonize("_aware.pyx"),
)


