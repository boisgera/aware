#!/usr/bin/env python
# coding: utf-8

from setuptools import setup
from Cython.Build import cythonize

setup(
  name = "aware",
  py_modules = ["aware"],
  scripts = ["aware"],
  ext_modules = cythonize("_aware.pyx"),
)


